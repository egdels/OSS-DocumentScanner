package docquad;

/**
 * Deterministic postprocessor for DocQuadNet-256.
 * <p>
 * Dual-path corner detection:
 * <ul>
 *   <li><b>CORNERS path:</b> Argmax (or 3×3 refined) peaks from {@code corner_heatmaps} [1,4,64,64]</li>
 *   <li><b>MASK path:</b> PCA-based oriented rectangle from {@code mask_logits} [1,1,64,64]</li>
 * </ul>
 * A penalty-based decision selects the better quad.
 * <p>
 * No Android dependencies.
 */
public final class DocQuadPostprocessor {

    public enum ChosenSource { CORNERS, MASK }

    public enum PeakMode { ARGMAX, REFINE_3X3 }

    private DocQuadPostprocessor() {
    }

    // ── Public API ──

    public static Result postprocess(float[][][][] cornerHeatmaps, float[][][][] maskLogits) {
        return postprocess(cornerHeatmaps, maskLogits, null, PeakMode.ARGMAX);
    }

    public static Result postprocess(float[][][][] cornerHeatmaps, float[][][][] maskLogits,
                                     DocQuadLetterbox lb) {
        return postprocess(cornerHeatmaps, maskLogits, lb, PeakMode.ARGMAX);
    }

    public static Result postprocess(float[][][][] cornerHeatmaps, float[][][][] maskLogits,
                                     DocQuadLetterbox lb, PeakMode peakMode) {
        if (peakMode == null) throw new IllegalArgumentException("peakMode must not be null");

        double[][] corners256 = corners64ToCorners256(cornerHeatmaps, peakMode);
        MaskStats ms = computeMaskStats(maskLogits);

        QuadFromMask qm = quadFromMask256(maskLogits, corners256);

        PathChoice pc = choosePath(corners256, qm.quad256, qm.usedFallback, maskLogits);
        double[][] chosenQuad256 = pc.chosenQuad256;
        ChosenSource chosenSource = pc.chosenSource;

        double[][] cornersOriginal = null;
        double[][] quadOriginal = null;
        double[][] chosenOriginal = null;
        if (lb != null) {
            cornersOriginal = mapCorners256ToOriginal(corners256, lb);
            quadOriginal = mapCorners256ToOriginal(qm.quad256, lb);
            chosenOriginal = (chosenSource == ChosenSource.MASK) ? quadOriginal : cornersOriginal;
        }
        return new Result(
                corners256, cornersOriginal,
                ms.maskProbGt05Count, ms.maskProbMean,
                qm.quad256, quadOriginal, qm.usedFallback,
                chosenQuad256, chosenOriginal, chosenSource,
                pc.penaltyCorners, pc.penaltyMask
        );
    }

    // ── Path choice ──

    private static final double HARD_PENALTY_THRESHOLD = 1e5;
    private static final double AGREEMENT_MAX_CORNER_DIST = 32.0;
    private static final double MASK_SCORE_MARGIN = 50.0;

    static PathChoice choosePath(double[][] quadCorners256, double[][] quadFromMask256,
                                 boolean quadFromMaskUsedFallback, float[][][][] maskLogits) {
        requireShapeMask(maskLogits);

        double pAGeom = quadPenaltyGeometry(quadCorners256);
        double pA = pAGeom + maskDisagreementPenaltyForCorners(quadCorners256, maskLogits);

        if (quadFromMaskUsedFallback)
            return new PathChoice(quadCorners256, ChosenSource.CORNERS, pA, Double.POSITIVE_INFINITY);

        double pB = quadPenaltyGeometry(quadFromMask256);

        if (pAGeom >= HARD_PENALTY_THRESHOLD && pB < HARD_PENALTY_THRESHOLD)
            return new PathChoice(quadFromMask256, ChosenSource.MASK, pA, pB);
        if (pB >= HARD_PENALTY_THRESHOLD)
            return new PathChoice(quadCorners256, ChosenSource.CORNERS, pA, pB);

        double maxCornerDist = maxCornerDistance(quadCorners256, quadFromMask256);
        if (maxCornerDist > AGREEMENT_MAX_CORNER_DIST)
            return new PathChoice(quadCorners256, ChosenSource.CORNERS, pA, pB);

        if (pB < pAGeom - MASK_SCORE_MARGIN)
            return new PathChoice(quadFromMask256, ChosenSource.MASK, pA, pB);

        return new PathChoice(quadCorners256, ChosenSource.CORNERS, pA, pB);
    }

    private static double maxCornerDistance(double[][] quad1, double[][] quad2) {
        if (quad1 == null || quad2 == null || quad1.length != 4 || quad2.length != 4)
            return Double.MAX_VALUE;
        double maxDist = 0.0;
        for (int i = 0; i < 4; i++) {
            double dx = quad1[i][0] - quad2[i][0];
            double dy = quad1[i][1] - quad2[i][1];
            double dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > maxDist) maxDist = dist;
        }
        return maxDist;
    }

    // ── Geometry penalty ──

    private static double quadPenaltyGeometry(double[][] quad256) {
        if (quad256 == null || quad256.length != 4) return 1e6;
        for (int i = 0; i < 4; i++) {
            if (quad256[i] == null || quad256[i].length != 2) return 1e6;
            if (!Double.isFinite(quad256[i][0]) || !Double.isFinite(quad256[i][1])) return 1e6;
        }

        double penalty = 0.0;
        final double w = 256.0, h = 256.0, tol = 2.0, hard = 16.0;
        final double kSoft = 10.0, kHard = 1000.0;

        double oobSum = DocQuadScore.oobSum(quad256, w, h, tol);
        if (oobSum > 0.0) penalty += oobSum * kSoft;
        double oobMax = DocQuadScore.oobMax(quad256, w, h, tol);
        if (oobMax > hard) penalty += 1e5 + (oobMax - hard) * kHard;

        if (DocQuadScore.selfIntersects(quad256)) penalty += 1e6;
        if (!DocQuadScore.isConvex(quad256)) penalty += 1e6;
        double areaAbs = DocQuadScore.areaAbs(quad256);
        if (!(areaAbs > 1.0)) penalty += 1e6;

        double edgeMin = DocQuadScore.edgeLengthMin(quad256);
        double edgeMax = DocQuadScore.edgeLengthMax(quad256);
        if (edgeMin < 8.0) penalty += (8.0 - edgeMin) * 1000.0;
        double r = edgeMax / Math.max(edgeMin, 1e-9);
        if (r > 25.0) penalty += (r - 25.0) * 100.0;

        return penalty;
    }

    private static double maskDisagreementPenaltyForCorners(double[][] quadCorners256,
                                                            float[][][][] maskLogits) {
        double[][] quad64 = new double[4][2];
        for (int i = 0; i < 4; i++) {
            quad64[i][0] = quadCorners256[i][0] / 4.0;
            quad64[i][1] = quadCorners256[i][1] / 4.0;
        }

        int[] grid = {0, 8, 16, 24, 32, 40, 48, 56};
        int disagree = 0;
        float[][] m = maskLogits[0][0];

        for (int gy : grid) {
            for (int gx : grid) {
                double px = gx + 0.5, py = gy + 0.5;
                boolean inQuad = pointInPolyInclusive(quad64, px, py);
                boolean inMask = sigmoid(m[gy][gx]) > 0.5;
                if (inQuad != inMask) disagree++;
            }
        }
        return disagree * 10.0;
    }

    private static boolean pointInPolyInclusive(double[][] poly, double px, double py) {
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            if (onSegment(poly[i][0], poly[i][1], poly[j][0], poly[j][1], px, py, 1e-9))
                return true;
        }
        boolean inside = false;
        for (int i = 0, j = 3; i < 4; j = i++) {
            double xi = poly[i][0], yi = poly[i][1];
            double xj = poly[j][0], yj = poly[j][1];
            boolean intersect = ((yi > py) != (yj > py))
                    && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    private static double orient(double ax, double ay, double bx, double by, double cx, double cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    private static boolean onSegment(double ax, double ay, double bx, double by,
                                     double px, double py, double eps) {
        if (Math.abs(orient(ax, ay, bx, by, px, py)) > eps) return false;
        return (Math.min(ax, bx) - eps <= px && px <= Math.max(ax, bx) + eps)
                && (Math.min(ay, by) - eps <= py && py <= Math.max(ay, by) + eps);
    }

    // ── Corner extraction ──

    public static double[][] corners64ToCorners256(float[][][][] cornerHeatmaps, PeakMode peakMode) {
        if (peakMode == null) throw new IllegalArgumentException("peakMode must not be null");
        if (peakMode == PeakMode.ARGMAX) return argmaxCorners64ToCorners256(cornerHeatmaps);
        if (peakMode == PeakMode.REFINE_3X3) return refineCorners64ToCorners256_3x3(cornerHeatmaps);
        throw new IllegalArgumentException("unsupported peakMode: " + peakMode);
    }

    /**
     * Argmax per channel (TL,TR,BR,BL), mapping 64→256: x256 = (ix + 0.5) * 4.0
     */
    public static double[][] argmaxCorners64ToCorners256(float[][][][] cornerHeatmaps) {
        requireShapeCorners(cornerHeatmaps);
        double[][] corners256 = new double[4][2];
        for (int c = 0; c < 4; c++) {
            float best = -Float.MAX_VALUE;
            int bestX = 0, bestY = 0;
            float[][] hm = cornerHeatmaps[0][c];
            for (int y = 0; y < 64; y++) {
                float[] row = hm[y];
                for (int x = 0; x < 64; x++) {
                    if (row[x] > best) { best = row[x]; bestX = x; bestY = y; }
                }
            }
            corners256[c][0] = (bestX + 0.5) * 4.0;
            corners256[c][1] = (bestY + 0.5) * 4.0;
        }
        return corners256;
    }

    /**
     * Subpixel refinement: 3×3 weighted centroid around argmax peak.
     * Weights: exp(logit − maxLogitInWindow). Mapping: x256 = x64 * 4.0
     */
    public static double[][] refineCorners64ToCorners256_3x3(float[][][][] cornerHeatmaps) {
        requireShapeCorners(cornerHeatmaps);
        double[][] corners256 = new double[4][2];
        for (int c = 0; c < 4; c++) {
            float best = -Float.MAX_VALUE;
            int bestX = 0, bestY = 0;
            float[][] hm = cornerHeatmaps[0][c];

            for (int y = 0; y < 64; y++) {
                float[] row = hm[y];
                for (int x = 0; x < 64; x++) {
                    if (row[x] > best) { best = row[x]; bestX = x; bestY = y; }
                }
            }

            int x0 = Math.max(0, bestX - 1), x1 = Math.min(63, bestX + 1);
            int y0 = Math.max(0, bestY - 1), y1 = Math.min(63, bestY + 1);

            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                    if (hm[y][x] > maxLogit) maxLogit = hm[y][x];

            double sumW = 0.0, sumX = 0.0, sumY = 0.0;
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    double w = Math.exp(hm[y][x] - maxLogit);
                    sumW += w;
                    sumX += w * (x + 0.5);
                    sumY += w * (y + 0.5);
                }
            }

            double x64, y64;
            if (sumW == 0.0 || !Double.isFinite(sumW)) {
                x64 = bestX + 0.5;
                y64 = bestY + 0.5;
            } else {
                x64 = sumX / sumW;
                y64 = sumY / sumW;
            }
            corners256[c][0] = x64 * 4.0;
            corners256[c][1] = y64 * 4.0;
        }
        return corners256;
    }

    // ── Mask → Quad (PCA) ──

    /**
     * Extracts an oriented rectangle from the binary mask via PCA.
     * Fallback: returns fallbackCorners256 if mask is empty/degenerate.
     */
    public static QuadFromMask quadFromMask256(float[][][][] maskLogits, double[][] fallbackCorners256) {
        requireShapeMask(maskLogits);
        if (fallbackCorners256 == null || fallbackCorners256.length != 4)
            throw new IllegalArgumentException("fallbackCorners256 must be double[4][2]");
        for (int i = 0; i < 4; i++)
            if (fallbackCorners256[i] == null || fallbackCorners256[i].length != 2)
                throw new IllegalArgumentException("fallbackCorners256 must be double[4][2]");

        float[][] m = maskLogits[0][0];
        int maskCount = 0;
        double sumX = 0.0, sumY = 0.0;

        for (int y = 0; y < 64; y++) {
            float[] row = m[y];
            for (int x = 0; x < 64; x++) {
                if (sigmoid(row[x]) > 0.5) {
                    maskCount++;
                    sumX += (x + 0.5);
                    sumY += (y + 0.5);
                }
            }
        }
        if (maskCount == 0) return new QuadFromMask(fallbackCorners256, true);

        double cx = sumX / maskCount, cy = sumY / maskCount;
        if (!Double.isFinite(cx) || !Double.isFinite(cy))
            return new QuadFromMask(fallbackCorners256, true);

        // Covariance
        double sxx = 0.0, sxy = 0.0, syy = 0.0;
        for (int y = 0; y < 64; y++) {
            float[] row = m[y];
            for (int x = 0; x < 64; x++) {
                if (sigmoid(row[x]) > 0.5) {
                    double dx = (x + 0.5) - cx, dy = (y + 0.5) - cy;
                    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
                }
            }
        }
        sxx /= maskCount; sxy /= maskCount; syy /= maskCount;

        double trace = sxx + syy;
        if (!Double.isFinite(trace) || trace < 1e-12)
            return new QuadFromMask(fallbackCorners256, true);

        double det = sxx * syy - sxy * sxy;
        double disc = Math.sqrt(Math.max(0.0, trace * trace / 4.0 - det));
        double lambda1 = trace / 2.0 + disc;

        final double eps = 1e-12;
        double v1x, v1y;
        if (Math.abs(sxy) > eps) { v1x = lambda1 - syy; v1y = sxy; }
        else if (sxx >= syy) { v1x = 1.0; v1y = 0.0; }
        else { v1x = 0.0; v1y = 1.0; }

        double n = Math.hypot(v1x, v1y);
        if (n == 0.0 || !Double.isFinite(n))
            return new QuadFromMask(fallbackCorners256, true);
        v1x /= n; v1y /= n;
        double v2x = -v1y, v2y = v1x;

        double uMin = Double.POSITIVE_INFINITY, uMax = Double.NEGATIVE_INFINITY;
        double vMin = Double.POSITIVE_INFINITY, vMax = Double.NEGATIVE_INFINITY;
        for (int y = 0; y < 64; y++) {
            float[] row = m[y];
            for (int x = 0; x < 64; x++) {
                if (sigmoid(row[x]) > 0.5) {
                    double px = (x + 0.5) - cx, py = (y + 0.5) - cy;
                    double u = px * v1x + py * v1y;
                    double v = px * v2x + py * v2y;
                    if (u < uMin) uMin = u; if (u > uMax) uMax = u;
                    if (v < vMin) vMin = v; if (v > vMax) vMax = v;
                }
            }
        }

        if (!(Double.isFinite(uMin) && Double.isFinite(uMax)
                && Double.isFinite(vMin) && Double.isFinite(vMax)))
            return new QuadFromMask(fallbackCorners256, true);
        if (uMax - uMin < 1e-12 || vMax - vMin < 1e-12)
            return new QuadFromMask(fallbackCorners256, true);

        double[][] quad64 = new double[4][2];
        quad64[0][0] = cx + uMax * v1x + vMax * v2x;
        quad64[0][1] = cy + uMax * v1y + vMax * v2y;
        quad64[1][0] = cx + uMin * v1x + vMax * v2x;
        quad64[1][1] = cy + uMin * v1y + vMax * v2y;
        quad64[2][0] = cx + uMin * v1x + vMin * v2x;
        quad64[2][1] = cy + uMin * v1y + vMin * v2y;
        quad64[3][0] = cx + uMax * v1x + vMin * v2x;
        quad64[3][1] = cy + uMax * v1y + vMin * v2y;

        quad64 = canonicalizeQuadOrderV1(quad64);

        double[][] quad256 = new double[4][2];
        for (int i = 0; i < 4; i++) {
            quad256[i][0] = quad64[i][0] * 4.0;
            quad256[i][1] = quad64[i][1] * 4.0;
        }
        return new QuadFromMask(quad256, false);
    }

    // ── Canonicalization ──

    private static double[][] canonicalizeQuadOrderV1(double[][] pts) {
        if (pts == null || pts.length != 4)
            throw new IllegalArgumentException("pts must be double[4][2]");
        double cx = 0.0, cy = 0.0;
        for (int i = 0; i < 4; i++) {
            if (pts[i] == null || pts[i].length != 2)
                throw new IllegalArgumentException("pts must be double[4][2]");
            cx += pts[i][0]; cy += pts[i][1];
        }
        cx /= 4.0; cy /= 4.0;

        int[] ordered = {0, 1, 2, 3};
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                int a = ordered[i], b = ordered[j];
                double angA = Math.atan2(pts[a][1] - cy, pts[a][0] - cx);
                double angB = Math.atan2(pts[b][1] - cy, pts[b][0] - cx);
                if (angB < angA || (angB == angA && b < a)) {
                    ordered[i] = b; ordered[j] = a;
                }
            }
        }

        int tlPos = 0;
        double bestSum = Double.POSITIVE_INFINITY;
        for (int k = 0; k < 4; k++) {
            double s = pts[ordered[k]][0] + pts[ordered[k]][1];
            if (s < bestSum || (s == bestSum && k < tlPos)) { bestSum = s; tlPos = k; }
        }

        double[][] out = new double[4][2];
        for (int i = 0; i < 4; i++) {
            int src = ordered[(tlPos + i) % 4];
            out[i][0] = pts[src][0]; out[i][1] = pts[src][1];
        }
        return out;
    }

    // ── Mask stats ──

    public static MaskStats computeMaskStats(float[][][][] maskLogits) {
        requireShapeMask(maskLogits);
        float[][] m = maskLogits[0][0];
        int count = 0;
        double sumProb = 0.0;
        for (int y = 0; y < 64; y++) {
            float[] row = m[y];
            for (int x = 0; x < 64; x++) {
                double prob = sigmoid(row[x]);
                sumProb += prob;
                if (prob > 0.5) count++;
            }
        }
        return new MaskStats(count, sumProb / (64.0 * 64.0));
    }

    // ── Coordinate mapping ──

    public static double[][] mapCorners256ToOriginal(double[][] corners256, DocQuadLetterbox lb) {
        if (corners256 == null || corners256.length != 4)
            throw new IllegalArgumentException("corners256 must be double[4][2]");
        if (lb == null) throw new IllegalArgumentException("lb must not be null");
        double[][] out = new double[4][2];
        for (int i = 0; i < 4; i++) {
            if (corners256[i] == null || corners256[i].length != 2)
                throw new IllegalArgumentException("corners256 must be double[4][2]");
            double[] p = lb.inverse(corners256[i][0], corners256[i][1]);
            out[i][0] = p[0]; out[i][1] = p[1];
        }
        return out;
    }

    // ── Helpers ──

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static void requireShapeMask(float[][][][] maskLogits) {
        if (maskLogits == null || maskLogits.length != 1
                || maskLogits[0] == null || maskLogits[0].length != 1
                || maskLogits[0][0] == null || maskLogits[0][0].length != 64
                || maskLogits[0][0][0] == null || maskLogits[0][0][0].length != 64)
            throw new IllegalArgumentException("mask_logits must have shape [1][1][64][64]");
        for (int y = 0; y < 64; y++)
            if (maskLogits[0][0][y] == null || maskLogits[0][0][y].length != 64)
                throw new IllegalArgumentException("mask_logits must have shape [1][1][64][64]");
    }

    private static void requireShapeCorners(float[][][][] cornerHeatmaps) {
        if (cornerHeatmaps == null || cornerHeatmaps.length != 1
                || cornerHeatmaps[0] == null || cornerHeatmaps[0].length != 4)
            throw new IllegalArgumentException("corner_heatmaps must have shape [1][4][64][64]");
        for (int c = 0; c < 4; c++) {
            if (cornerHeatmaps[0][c] == null || cornerHeatmaps[0][c].length != 64
                    || cornerHeatmaps[0][c][0] == null || cornerHeatmaps[0][c][0].length != 64)
                throw new IllegalArgumentException("corner_heatmaps must have shape [1][4][64][64]");
            for (int y = 0; y < 64; y++)
                if (cornerHeatmaps[0][c][y] == null || cornerHeatmaps[0][c][y].length != 64)
                    throw new IllegalArgumentException("corner_heatmaps must have shape [1][4][64][64]");
        }
    }

    // ── Result types ──

    public static final class Result {
        public final double[][] corners256;
        public final double[][] cornersOriginal;
        public final int maskProbGt05Count;
        public final double maskProbMean;
        public final double[][] quadFromMask256;
        public final double[][] quadFromMaskOriginal;
        public final boolean quadFromMaskUsedFallback;
        public final double[][] chosenQuad256;
        public final double[][] chosenQuadOriginal;
        public final ChosenSource chosenSource;
        public final double penaltyCorners;
        public final double penaltyMask;

        public Result(double[][] corners256, double[][] cornersOriginal,
                      int maskProbGt05Count, double maskProbMean,
                      double[][] quadFromMask256, double[][] quadFromMaskOriginal,
                      boolean quadFromMaskUsedFallback,
                      double[][] chosenQuad256, double[][] chosenQuadOriginal,
                      ChosenSource chosenSource,
                      double penaltyCorners, double penaltyMask) {
            this.corners256 = corners256;
            this.cornersOriginal = cornersOriginal;
            this.maskProbGt05Count = maskProbGt05Count;
            this.maskProbMean = maskProbMean;
            this.quadFromMask256 = quadFromMask256;
            this.quadFromMaskOriginal = quadFromMaskOriginal;
            this.quadFromMaskUsedFallback = quadFromMaskUsedFallback;
            this.chosenQuad256 = chosenQuad256;
            this.chosenQuadOriginal = chosenQuadOriginal;
            this.chosenSource = chosenSource;
            this.penaltyCorners = penaltyCorners;
            this.penaltyMask = penaltyMask;
        }
    }

    static final class PathChoice {
        final double[][] chosenQuad256;
        final ChosenSource chosenSource;
        final double penaltyCorners;
        final double penaltyMask;

        PathChoice(double[][] chosenQuad256, ChosenSource chosenSource,
                   double penaltyCorners, double penaltyMask) {
            this.chosenQuad256 = chosenQuad256;
            this.chosenSource = chosenSource;
            this.penaltyCorners = penaltyCorners;
            this.penaltyMask = penaltyMask;
        }
    }

    public static final class QuadFromMask {
        public final double[][] quad256;
        public final boolean usedFallback;

        public QuadFromMask(double[][] quad256, boolean usedFallback) {
            this.quad256 = quad256;
            this.usedFallback = usedFallback;
        }
    }

    public static final class MaskStats {
        public final int maskProbGt05Count;
        public final double maskProbMean;

        public MaskStats(int maskProbGt05Count, double maskProbMean) {
            this.maskProbGt05Count = maskProbGt05Count;
            this.maskProbMean = maskProbMean;
        }
    }
}
