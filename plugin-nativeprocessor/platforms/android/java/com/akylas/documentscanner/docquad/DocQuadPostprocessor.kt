package com.akylas.documentscanner.docquad

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Deterministic postprocessor for DocQuadNet-256.
 *
 * Dual-path corner detection:
 * - **CORNERS path:** Argmax (or 3×3 refined) peaks from `corner_heatmaps [1,4,64,64]`
 * - **MASK path:** PCA-based oriented rectangle from `mask_logits [1,1,64,64]`
 *
 * A penalty-based decision selects the better quad.
 *
 * No Android dependencies.
 */
object DocQuadPostprocessor {

    enum class ChosenSource { CORNERS, MASK }

    enum class PeakMode { ARGMAX, REFINE_3X3 }

    // ── Public API ──

    @JvmStatic
    fun postprocess(cornerHeatmaps: Array<Array<Array<FloatArray>>>, maskLogits: Array<Array<Array<FloatArray>>>): Result {
        return postprocess(cornerHeatmaps, maskLogits, null, PeakMode.ARGMAX)
    }

    @JvmStatic
    fun postprocess(
        cornerHeatmaps: Array<Array<Array<FloatArray>>>,
        maskLogits: Array<Array<Array<FloatArray>>>,
        lb: DocQuadLetterbox?
    ): Result {
        return postprocess(cornerHeatmaps, maskLogits, lb, PeakMode.ARGMAX)
    }

    @JvmStatic
    fun postprocess(
        cornerHeatmaps: Array<Array<Array<FloatArray>>>,
        maskLogits: Array<Array<Array<FloatArray>>>,
        lb: DocQuadLetterbox?,
        peakMode: PeakMode
    ): Result {
        val corners256 = corners64ToCorners256(cornerHeatmaps, peakMode)
        val ms = computeMaskStats(maskLogits)

        val qm = quadFromMask256(maskLogits, corners256)

        val pc = choosePath(corners256, qm.quad256, qm.usedFallback, maskLogits)
        val chosenQuad256 = pc.chosenQuad256
        val chosenSource = pc.chosenSource

        var cornersOriginal: Array<DoubleArray>? = null
        var quadOriginal: Array<DoubleArray>? = null
        var chosenOriginal: Array<DoubleArray>? = null
        if (lb != null) {
            cornersOriginal = mapCorners256ToOriginal(corners256, lb)
            quadOriginal = mapCorners256ToOriginal(qm.quad256, lb)
            chosenOriginal = if (chosenSource == ChosenSource.MASK) quadOriginal else cornersOriginal
        }
        return Result(
            corners256, cornersOriginal,
            ms.maskProbGt05Count, ms.maskProbMean,
            qm.quad256, quadOriginal, qm.usedFallback,
            chosenQuad256, chosenOriginal, chosenSource,
            pc.penaltyCorners, pc.penaltyMask
        )
    }

    // ── Path choice ──

    private const val HARD_PENALTY_THRESHOLD = 1e5
    private const val AGREEMENT_MAX_CORNER_DIST = 32.0
    private const val MASK_SCORE_MARGIN = 50.0

    internal fun choosePath(
        quadCorners256: Array<DoubleArray>,
        quadFromMask256: Array<DoubleArray>,
        quadFromMaskUsedFallback: Boolean,
        maskLogits: Array<Array<Array<FloatArray>>>
    ): PathChoice {
        requireShapeMask(maskLogits)

        val pAGeom = quadPenaltyGeometry(quadCorners256)
        val pA = pAGeom + maskDisagreementPenaltyForCorners(quadCorners256, maskLogits)

        if (quadFromMaskUsedFallback)
            return PathChoice(quadCorners256, ChosenSource.CORNERS, pA, Double.POSITIVE_INFINITY)

        val pB = quadPenaltyGeometry(quadFromMask256)

        if (pAGeom >= HARD_PENALTY_THRESHOLD && pB < HARD_PENALTY_THRESHOLD)
            return PathChoice(quadFromMask256, ChosenSource.MASK, pA, pB)
        if (pB >= HARD_PENALTY_THRESHOLD)
            return PathChoice(quadCorners256, ChosenSource.CORNERS, pA, pB)

        val maxCornerDist = maxCornerDistance(quadCorners256, quadFromMask256)
        if (maxCornerDist > AGREEMENT_MAX_CORNER_DIST)
            return PathChoice(quadCorners256, ChosenSource.CORNERS, pA, pB)

        if (pB < pAGeom - MASK_SCORE_MARGIN)
            return PathChoice(quadFromMask256, ChosenSource.MASK, pA, pB)

        return PathChoice(quadCorners256, ChosenSource.CORNERS, pA, pB)
    }

    private fun maxCornerDistance(quad1: Array<DoubleArray>?, quad2: Array<DoubleArray>?): Double {
        if (quad1 == null || quad2 == null || quad1.size != 4 || quad2.size != 4)
            return Double.MAX_VALUE
        var maxDist = 0.0
        for (i in 0 until 4) {
            val dx = quad1[i][0] - quad2[i][0]
            val dy = quad1[i][1] - quad2[i][1]
            val dist = sqrt(dx * dx + dy * dy)
            if (dist > maxDist) maxDist = dist
        }
        return maxDist
    }

    // ── Geometry penalty ──

    private fun quadPenaltyGeometry(quad256: Array<DoubleArray>?): Double {
        if (quad256 == null || quad256.size != 4) return 1e6
        for (i in 0 until 4) {
            if (quad256[i].size != 2) return 1e6
            if (!quad256[i][0].isFinite() || !quad256[i][1].isFinite()) return 1e6
        }

        var penalty = 0.0
        val w = 256.0; val h = 256.0; val tol = 2.0; val hard = 16.0
        val kSoft = 10.0; val kHard = 1000.0

        val oobSum = DocQuadScore.oobSum(quad256, w, h, tol)
        if (oobSum > 0.0) penalty += oobSum * kSoft
        val oobMax = DocQuadScore.oobMax(quad256, w, h, tol)
        if (oobMax > hard) penalty += 1e5 + (oobMax - hard) * kHard

        if (DocQuadScore.selfIntersects(quad256)) penalty += 1e6
        if (!DocQuadScore.isConvex(quad256)) penalty += 1e6
        val areaAbs = DocQuadScore.areaAbs(quad256)
        if (!(areaAbs > 1.0)) penalty += 1e6

        val edgeMin = DocQuadScore.edgeLengthMin(quad256)
        val edgeMax = DocQuadScore.edgeLengthMax(quad256)
        if (edgeMin < 8.0) penalty += (8.0 - edgeMin) * 1000.0
        val r = edgeMax / max(edgeMin, 1e-9)
        if (r > 25.0) penalty += (r - 25.0) * 100.0

        return penalty
    }

    private fun maskDisagreementPenaltyForCorners(
        quadCorners256: Array<DoubleArray>,
        maskLogits: Array<Array<Array<FloatArray>>>
    ): Double {
        val quad64 = Array(4) { DoubleArray(2) }
        for (i in 0 until 4) {
            quad64[i][0] = quadCorners256[i][0] / 4.0
            quad64[i][1] = quadCorners256[i][1] / 4.0
        }

        val grid = intArrayOf(0, 8, 16, 24, 32, 40, 48, 56)
        var disagree = 0
        val m = maskLogits[0][0]

        for (gy in grid) {
            for (gx in grid) {
                val px = gx + 0.5; val py = gy + 0.5
                val inQuad = pointInPolyInclusive(quad64, px, py)
                val inMask = sigmoid(m[gy][gx].toDouble()) > 0.5
                if (inQuad != inMask) disagree++
            }
        }
        return disagree * 10.0
    }

    private fun pointInPolyInclusive(poly: Array<DoubleArray>, px: Double, py: Double): Boolean {
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            if (onSegment(poly[i][0], poly[i][1], poly[j][0], poly[j][1], px, py, 1e-9))
                return true
        }
        var inside = false
        var j = 3
        for (i in 0 until 4) {
            val xi = poly[i][0]; val yi = poly[i][1]
            val xj = poly[j][0]; val yj = poly[j][1]
            val intersect = ((yi > py) != (yj > py))
                    && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            if (intersect) inside = !inside
            j = i
        }
        return inside
    }

    private fun orient(ax: Double, ay: Double, bx: Double, by: Double, cx: Double, cy: Double): Double {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    }

    private fun onSegment(
        ax: Double, ay: Double, bx: Double, by: Double,
        px: Double, py: Double, eps: Double
    ): Boolean {
        if (abs(orient(ax, ay, bx, by, px, py)) > eps) return false
        return (min(ax, bx) - eps <= px && px <= max(ax, bx) + eps)
                && (min(ay, by) - eps <= py && py <= max(ay, by) + eps)
    }

    // ── Corner extraction ──

    @JvmStatic
    fun corners64ToCorners256(cornerHeatmaps: Array<Array<Array<FloatArray>>>, peakMode: PeakMode): Array<DoubleArray> {
        return when (peakMode) {
            PeakMode.ARGMAX -> argmaxCorners64ToCorners256(cornerHeatmaps)
            PeakMode.REFINE_3X3 -> refineCorners64ToCorners256_3x3(cornerHeatmaps)
        }
    }

    /** Argmax per channel (TL,TR,BR,BL), mapping 64→256: x256 = (ix + 0.5) * 4.0 */
    @JvmStatic
    fun argmaxCorners64ToCorners256(cornerHeatmaps: Array<Array<Array<FloatArray>>>): Array<DoubleArray> {
        requireShapeCorners(cornerHeatmaps)
        val corners256 = Array(4) { DoubleArray(2) }
        for (c in 0 until 4) {
            var best = -Float.MAX_VALUE
            var bestX = 0; var bestY = 0
            val hm = cornerHeatmaps[0][c]
            for (y in 0 until 64) {
                val row = hm[y]
                for (x in 0 until 64) {
                    if (row[x] > best) { best = row[x]; bestX = x; bestY = y }
                }
            }
            corners256[c][0] = (bestX + 0.5) * 4.0
            corners256[c][1] = (bestY + 0.5) * 4.0
        }
        return corners256
    }

    /** Subpixel refinement: 3×3 weighted centroid around argmax peak. */
    @JvmStatic
    fun refineCorners64ToCorners256_3x3(cornerHeatmaps: Array<Array<Array<FloatArray>>>): Array<DoubleArray> {
        requireShapeCorners(cornerHeatmaps)
        val corners256 = Array(4) { DoubleArray(2) }
        for (c in 0 until 4) {
            var best = -Float.MAX_VALUE
            var bestX = 0; var bestY = 0
            val hm = cornerHeatmaps[0][c]

            for (y in 0 until 64) {
                val row = hm[y]
                for (x in 0 until 64) {
                    if (row[x] > best) { best = row[x]; bestX = x; bestY = y }
                }
            }

            val x0 = max(0, bestX - 1); val x1 = min(63, bestX + 1)
            val y0 = max(0, bestY - 1); val y1 = min(63, bestY + 1)

            var maxLogit = Double.NEGATIVE_INFINITY
            for (y in y0..y1)
                for (x in x0..x1)
                    if (hm[y][x] > maxLogit) maxLogit = hm[y][x].toDouble()

            var sumW = 0.0; var sumX = 0.0; var sumY = 0.0
            for (y in y0..y1) {
                for (x in x0..x1) {
                    val w = exp(hm[y][x] - maxLogit)
                    sumW += w
                    sumX += w * (x + 0.5)
                    sumY += w * (y + 0.5)
                }
            }

            val x64: Double
            val y64: Double
            if (sumW == 0.0 || !sumW.isFinite()) {
                x64 = bestX + 0.5
                y64 = bestY + 0.5
            } else {
                x64 = sumX / sumW
                y64 = sumY / sumW
            }
            corners256[c][0] = x64 * 4.0
            corners256[c][1] = y64 * 4.0
        }
        return corners256
    }

    // ── Mask → Quad (PCA) ──

    @JvmStatic
    fun quadFromMask256(maskLogits: Array<Array<Array<FloatArray>>>, fallbackCorners256: Array<DoubleArray>): QuadFromMask {
        requireShapeMask(maskLogits)
        require(fallbackCorners256.size == 4) { "fallbackCorners256 must be double[4][2]" }
        for (i in 0 until 4)
            require(fallbackCorners256[i].size == 2) { "fallbackCorners256 must be double[4][2]" }

        val m = maskLogits[0][0]
        var maskCount = 0
        var sumX = 0.0; var sumY = 0.0

        for (y in 0 until 64) {
            val row = m[y]
            for (x in 0 until 64) {
                if (sigmoid(row[x].toDouble()) > 0.5) {
                    maskCount++
                    sumX += (x + 0.5)
                    sumY += (y + 0.5)
                }
            }
        }
        if (maskCount == 0) return QuadFromMask(fallbackCorners256, true)

        val cx = sumX / maskCount; val cy = sumY / maskCount
        if (!cx.isFinite() || !cy.isFinite())
            return QuadFromMask(fallbackCorners256, true)

        // Covariance
        var sxx = 0.0; var sxy = 0.0; var syy = 0.0
        for (y in 0 until 64) {
            val row = m[y]
            for (x in 0 until 64) {
                if (sigmoid(row[x].toDouble()) > 0.5) {
                    val dx = (x + 0.5) - cx; val dy = (y + 0.5) - cy
                    sxx += dx * dx; sxy += dx * dy; syy += dy * dy
                }
            }
        }
        sxx /= maskCount; sxy /= maskCount; syy /= maskCount

        val trace = sxx + syy
        if (!trace.isFinite() || trace < 1e-12)
            return QuadFromMask(fallbackCorners256, true)

        val det = sxx * syy - sxy * sxy
        val disc = sqrt(max(0.0, trace * trace / 4.0 - det))
        val lambda1 = trace / 2.0 + disc

        val eps = 1e-12
        var v1x: Double; var v1y: Double
        if (abs(sxy) > eps) { v1x = lambda1 - syy; v1y = sxy }
        else if (sxx >= syy) { v1x = 1.0; v1y = 0.0 }
        else { v1x = 0.0; v1y = 1.0 }

        val n = kotlin.math.hypot(v1x, v1y)
        if (n == 0.0 || !n.isFinite())
            return QuadFromMask(fallbackCorners256, true)
        v1x /= n; v1y /= n
        val v2x = -v1y; val v2y = v1x

        var uMin = Double.POSITIVE_INFINITY; var uMax = Double.NEGATIVE_INFINITY
        var vMin = Double.POSITIVE_INFINITY; var vMax = Double.NEGATIVE_INFINITY
        for (y in 0 until 64) {
            val row = m[y]
            for (x in 0 until 64) {
                if (sigmoid(row[x].toDouble()) > 0.5) {
                    val px = (x + 0.5) - cx; val py = (y + 0.5) - cy
                    val u = px * v1x + py * v1y
                    val v = px * v2x + py * v2y
                    if (u < uMin) uMin = u; if (u > uMax) uMax = u
                    if (v < vMin) vMin = v; if (v > vMax) vMax = v
                }
            }
        }

        if (!(uMin.isFinite() && uMax.isFinite() && vMin.isFinite() && vMax.isFinite()))
            return QuadFromMask(fallbackCorners256, true)
        if (uMax - uMin < 1e-12 || vMax - vMin < 1e-12)
            return QuadFromMask(fallbackCorners256, true)

        var quad64 = Array(4) { DoubleArray(2) }
        quad64[0][0] = cx + uMax * v1x + vMax * v2x
        quad64[0][1] = cy + uMax * v1y + vMax * v2y
        quad64[1][0] = cx + uMin * v1x + vMax * v2x
        quad64[1][1] = cy + uMin * v1y + vMax * v2y
        quad64[2][0] = cx + uMin * v1x + vMin * v2x
        quad64[2][1] = cy + uMin * v1y + vMin * v2y
        quad64[3][0] = cx + uMax * v1x + vMin * v2x
        quad64[3][1] = cy + uMax * v1y + vMin * v2y

        quad64 = canonicalizeQuadOrderV1(quad64)

        val quad256 = Array(4) { DoubleArray(2) }
        for (i in 0 until 4) {
            quad256[i][0] = quad64[i][0] * 4.0
            quad256[i][1] = quad64[i][1] * 4.0
        }
        return QuadFromMask(quad256, false)
    }

    // ── Canonicalization ──

    private fun canonicalizeQuadOrderV1(pts: Array<DoubleArray>): Array<DoubleArray> {
        require(pts.size == 4) { "pts must be double[4][2]" }
        var cx = 0.0; var cy = 0.0
        for (i in 0 until 4) {
            require(pts[i].size == 2) { "pts must be double[4][2]" }
            cx += pts[i][0]; cy += pts[i][1]
        }
        cx /= 4.0; cy /= 4.0

        val ordered = intArrayOf(0, 1, 2, 3)
        for (i in 0 until 4) {
            for (j in i + 1 until 4) {
                val a = ordered[i]; val b = ordered[j]
                val angA = atan2(pts[a][1] - cy, pts[a][0] - cx)
                val angB = atan2(pts[b][1] - cy, pts[b][0] - cx)
                if (angB < angA || (angB == angA && b < a)) {
                    ordered[i] = b; ordered[j] = a
                }
            }
        }

        var tlPos = 0
        var bestSum = Double.POSITIVE_INFINITY
        for (k in 0 until 4) {
            val s = pts[ordered[k]][0] + pts[ordered[k]][1]
            if (s < bestSum || (s == bestSum && k < tlPos)) { bestSum = s; tlPos = k }
        }

        val out = Array(4) { DoubleArray(2) }
        for (i in 0 until 4) {
            val src = ordered[(tlPos + i) % 4]
            out[i][0] = pts[src][0]; out[i][1] = pts[src][1]
        }
        return out
    }

    // ── Mask stats ──

    @JvmStatic
    fun computeMaskStats(maskLogits: Array<Array<Array<FloatArray>>>): MaskStats {
        requireShapeMask(maskLogits)
        val m = maskLogits[0][0]
        var count = 0
        var sumProb = 0.0
        for (y in 0 until 64) {
            val row = m[y]
            for (x in 0 until 64) {
                val prob = sigmoid(row[x].toDouble())
                sumProb += prob
                if (prob > 0.5) count++
            }
        }
        return MaskStats(count, sumProb / (64.0 * 64.0))
    }

    // ── Coordinate mapping ──

    @JvmStatic
    fun mapCorners256ToOriginal(corners256: Array<DoubleArray>, lb: DocQuadLetterbox): Array<DoubleArray> {
        require(corners256.size == 4) { "corners256 must be double[4][2]" }
        val out = Array(4) { DoubleArray(2) }
        for (i in 0 until 4) {
            require(corners256[i].size == 2) { "corners256 must be double[4][2]" }
            val p = lb.inverse(corners256[i][0], corners256[i][1])
            out[i][0] = p[0]; out[i][1] = p[1]
        }
        return out
    }

    // ── Helpers ──

    private fun sigmoid(x: Double): Double {
        return 1.0 / (1.0 + exp(-x))
    }

    private fun requireShapeMask(maskLogits: Array<Array<Array<FloatArray>>>) {
        require(
            maskLogits.size == 1
                    && maskLogits[0].size == 1
                    && maskLogits[0][0].size == 64
                    && maskLogits[0][0][0].size == 64
        ) { "mask_logits must have shape [1][1][64][64]" }
        for (y in 0 until 64)
            require(maskLogits[0][0][y].size == 64) { "mask_logits must have shape [1][1][64][64]" }
    }

    private fun requireShapeCorners(cornerHeatmaps: Array<Array<Array<FloatArray>>>) {
        require(
            cornerHeatmaps.size == 1
                    && cornerHeatmaps[0].size == 4
        ) { "corner_heatmaps must have shape [1][4][64][64]" }
        for (c in 0 until 4) {
            require(
                cornerHeatmaps[0][c].size == 64
                        && cornerHeatmaps[0][c][0].size == 64
            ) { "corner_heatmaps must have shape [1][4][64][64]" }
            for (y in 0 until 64)
                require(cornerHeatmaps[0][c][y].size == 64) { "corner_heatmaps must have shape [1][4][64][64]" }
        }
    }

    // ── Result types ──

    data class Result(
        val corners256: Array<DoubleArray>,
        val cornersOriginal: Array<DoubleArray>?,
        val maskProbGt05Count: Int,
        val maskProbMean: Double,
        val quadFromMask256: Array<DoubleArray>,
        val quadFromMaskOriginal: Array<DoubleArray>?,
        val quadFromMaskUsedFallback: Boolean,
        val chosenQuad256: Array<DoubleArray>,
        val chosenQuadOriginal: Array<DoubleArray>?,
        val chosenSource: ChosenSource,
        val penaltyCorners: Double,
        val penaltyMask: Double
    )

    internal data class PathChoice(
        val chosenQuad256: Array<DoubleArray>,
        val chosenSource: ChosenSource,
        val penaltyCorners: Double,
        val penaltyMask: Double
    )

    data class QuadFromMask(
        val quad256: Array<DoubleArray>,
        val usedFallback: Boolean
    )

    data class MaskStats(
        val maskProbGt05Count: Int,
        val maskProbMean: Double
    )
}
