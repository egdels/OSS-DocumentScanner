package docquad;

/**
 * Deterministic geometry utilities for quad scoring and validation.
 * <p>
 * Coordinates are typically in 256-space (double), corner order: TL, TR, BR, BL.
 * No Android dependencies.
 */
public final class DocQuadScore {

    private DocQuadScore() {
    }

    /** Absolute area of the quad (Shoelace formula). */
    public static double areaAbs(double[][] quad) {
        requireQuad(quad);
        double s = 0.0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            s += quad[i][0] * quad[j][1] - quad[j][0] * quad[i][1];
        }
        return Math.abs(0.5 * s);
    }

    public static double perimeter(double[][] quad) {
        requireQuad(quad);
        double p = 0.0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            p += dist(quad[i][0], quad[i][1], quad[j][0], quad[j][1]);
        }
        return p;
    }

    public static double edgeLengthMin(double[][] quad) {
        requireQuad(quad);
        double m = Double.POSITIVE_INFINITY;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            double d = dist(quad[i][0], quad[i][1], quad[j][0], quad[j][1]);
            if (d < m) m = d;
        }
        return m;
    }

    public static double edgeLengthMax(double[][] quad) {
        requireQuad(quad);
        double m = 0.0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            double d = dist(quad[i][0], quad[i][1], quad[j][0], quad[j][1]);
            if (d > m) m = d;
        }
        return m;
    }

    public static double aspectLike(double[][] quad) {
        double min = edgeLengthMin(quad);
        double max = edgeLengthMax(quad);
        return max / Math.max(min, 1e-9);
    }

    /** Returns true if the quad's non-adjacent edges intersect (self-intersecting / bowtie). */
    public static boolean selfIntersects(double[][] quad) {
        requireQuad(quad);
        return segmentsIntersect(
                quad[0][0], quad[0][1], quad[1][0], quad[1][1],
                quad[2][0], quad[2][1], quad[3][0], quad[3][1]
        ) || segmentsIntersect(
                quad[1][0], quad[1][1], quad[2][0], quad[2][1],
                quad[3][0], quad[3][1], quad[0][0], quad[0][1]
        );
    }

    /** Returns true if the quad is strictly convex (no collinear edges). */
    public static boolean isConvex(double[][] quad) {
        requireQuad(quad);
        final double eps = 1e-9;
        int sign = 0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            int k = (i + 2) % 4;
            double cross = orient(
                    quad[i][0], quad[i][1],
                    quad[j][0], quad[j][1],
                    quad[k][0], quad[k][1]
            );
            if (Math.abs(cross) <= eps) return false;
            int s = cross > 0.0 ? 1 : -1;
            if (sign == 0) {
                sign = s;
            } else if (s != sign) {
                return false;
            }
        }
        return true;
    }

    /**
     * Sum of out-of-bounds distances for all 4 corners against frame [0..w-1] × [0..h-1],
     * expanded by tolPx.
     */
    public static double oobSum(double[][] quad, double w, double h, double tolPx) {
        requireQuad(quad);
        if (!(w > 0.0) || !(h > 0.0) || !Double.isFinite(tolPx) || tolPx < 0.0)
            throw new IllegalArgumentException("invalid bounds/tol");

        double left = -tolPx, top = -tolPx;
        double right = (w - 1.0) + tolPx, bottom = (h - 1.0) + tolPx;

        double s = 0.0;
        for (int i = 0; i < 4; i++) {
            s += oob1d(quad[i][0], left, right) + oob1d(quad[i][1], top, bottom);
        }
        return s;
    }

    /**
     * Maximum out-of-bounds distance over all 4 corners.
     */
    public static double oobMax(double[][] quad, double w, double h, double tolPx) {
        requireQuad(quad);
        if (!(w > 0.0) || !(h > 0.0) || !Double.isFinite(tolPx) || tolPx < 0.0)
            throw new IllegalArgumentException("invalid bounds/tol");

        double left = -tolPx, top = -tolPx;
        double right = (w - 1.0) + tolPx, bottom = (h - 1.0) + tolPx;

        double m = 0.0;
        for (int i = 0; i < 4; i++) {
            double v = oob1d(quad[i][0], left, right) + oob1d(quad[i][1], top, bottom);
            if (v > m) m = v;
        }
        return m;
    }

    // ── internal helpers ──

    private static void requireQuad(double[][] quad) {
        if (quad == null || quad.length != 4)
            throw new IllegalArgumentException("quad must be double[4][2]");
        for (int i = 0; i < 4; i++)
            if (quad[i] == null || quad[i].length != 2)
                throw new IllegalArgumentException("quad must be double[4][2]");
    }

    private static double oob1d(double v, double min, double max) {
        if (v < min) return min - v;
        if (v > max) return v - max;
        return 0.0;
    }

    private static double dist(double ax, double ay, double bx, double by) {
        return Math.hypot(bx - ax, by - ay);
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

    private static boolean segmentsIntersect(double ax, double ay, double bx, double by,
                                             double cx, double cy, double dx, double dy) {
        final double eps = 1e-9;
        double o1 = orient(ax, ay, bx, by, cx, cy);
        double o2 = orient(ax, ay, bx, by, dx, dy);
        double o3 = orient(cx, cy, dx, dy, ax, ay);
        double o4 = orient(cx, cy, dx, dy, bx, by);

        int s1 = sign(o1, eps), s2 = sign(o2, eps);
        int s3 = sign(o3, eps), s4 = sign(o4, eps);

        if (s1 == 0 && onSegment(ax, ay, bx, by, cx, cy, eps)) return true;
        if (s2 == 0 && onSegment(ax, ay, bx, by, dx, dy, eps)) return true;
        if (s3 == 0 && onSegment(cx, cy, dx, dy, ax, ay, eps)) return true;
        if (s4 == 0 && onSegment(cx, cy, dx, dy, bx, by, eps)) return true;

        return (s1 * s2 < 0) && (s3 * s4 < 0);
    }

    private static int sign(double v, double eps) {
        if (v > eps) return 1;
        if (v < -eps) return -1;
        return 0;
    }
}
