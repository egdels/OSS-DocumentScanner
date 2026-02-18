package com.akylas.documentscanner.docquad

/**
 * Letterbox transformation: maps a source rectangle into a destination rectangle
 * while preserving aspect ratio, centering with black bars.
 *
 * Immutable. No Android dependencies.
 */
class DocQuadLetterbox private constructor(
    val srcW: Int,
    val srcH: Int,
    val dstW: Int,
    val dstH: Int,
    val scale: Double,
    val offsetX: Double,
    val offsetY: Double
) {

    /** Forward: source coordinates → destination (256-space) coordinates. */
    fun forward(x: Double, y: Double): DoubleArray {
        return doubleArrayOf(x * scale + offsetX, y * scale + offsetY)
    }

    /** Inverse: destination (256-space) coordinates → source coordinates. */
    fun inverse(x: Double, y: Double): DoubleArray {
        return doubleArrayOf((x - offsetX) / scale, (y - offsetY) / scale)
    }

    companion object {
        /** Creates a letterbox mapping from (srcW×srcH) → (dstW×dstH). */
        @JvmStatic
        fun create(srcW: Int, srcH: Int, dstW: Int, dstH: Int): DocQuadLetterbox {
            require(srcW > 0 && srcH > 0) { "srcW/srcH must be > 0" }
            require(dstW > 0 && dstH > 0) { "dstW/dstH must be > 0" }

            val s = minOf(dstW.toDouble() / srcW, dstH.toDouble() / srcH)
            val newW = srcW * s
            val newH = srcH * s
            val ox = (dstW - newW) / 2.0
            val oy = (dstH - newH) / 2.0
            return DocQuadLetterbox(srcW, srcH, dstW, dstH, s, ox, oy)
        }

        /** Creates a letterbox mapping from (srcW×srcH) → (256×256). */
        @JvmStatic
        fun create(srcW: Int, srcH: Int): DocQuadLetterbox {
            return create(srcW, srcH, 256, 256)
        }
    }
}
