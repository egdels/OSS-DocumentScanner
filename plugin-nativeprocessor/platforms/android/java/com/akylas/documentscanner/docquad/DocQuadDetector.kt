package com.akylas.documentscanner.docquad

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF

/**
 * High-level detector: Bitmap in → document corner coordinates out.
 *
 * Usage example:
 * ```
 *   val detector = DocQuadDetector()
 *   val corners = detector.detect(bitmap, context)
 *   // corners = { {xTL,yTL}, {xTR,yTR}, {xBR,yBR}, {xBL,yBL} }
 *   // in original bitmap pixel coordinates, or null on failure.
 * ```
 *
 * **Android dependency:** Uses `Bitmap`, `Canvas`, `Context`.
 */
class DocQuadDetector @JvmOverloads constructor(
    private val modelAssetPath: String = DEFAULT_MODEL_ASSET_PATH,
    private val injectedRunner: DocQuadOrtRunner? = null
) {

    /** Use a pre-loaded runner (e.g. for live camera). */
    constructor(injectedRunner: DocQuadOrtRunner) : this(DEFAULT_MODEL_ASSET_PATH, injectedRunner)

    /**
     * Detects document corners in the given bitmap.
     *
     * @param src the input image (any size)
     * @param ctx Android context (for asset loading)
     * @return 4×2 array {{xTL,yTL},{xTR,yTR},{xBR,yBR},{xBL,yBL}} in original pixel coords,
     *         or `null` on failure
     */
    fun detect(src: Bitmap?, ctx: Context?): Array<DoubleArray>? {
        if (src == null || ctx == null) return null

        var in256: Bitmap? = null
        try {
            val srcW = src.width
            val srcH = src.height
            if (srcW <= 0 || srcH <= 0) return null

            val lb = DocQuadLetterbox.create(srcW, srcH, DocQuadOrtRunner.IN_W, DocQuadOrtRunner.IN_H)
            in256 = renderLetterbox256(src, lb)
            val input = bitmapToNchwFloat01(in256)

            val outputs = if (injectedRunner != null) {
                injectedRunner.run(input)
            } else {
                DocQuadOrtRunner.getInstance(ctx, modelAssetPath).run(input)
            }

            val r = DocQuadPostprocessor.postprocess(
                outputs.cornerHeatmaps, outputs.maskLogits,
                lb, DocQuadPostprocessor.PeakMode.REFINE_3X3
            )

            if (r.chosenQuadOriginal == null || r.chosenQuadOriginal.size != 4)
                return null
            if (!isValidQuad(r.chosenQuadOriginal, srcW, srcH))
                return null

            return r.chosenQuadOriginal
        } catch (t: Throwable) {
            return null
        } finally {
            in256?.let {
                if (!it.isRecycled) {
                    try { it.recycle() } catch (_: Throwable) {}
                }
            }
        }
    }

    // ── Preprocessing ──

    companion object {
        const val DEFAULT_MODEL_ASSET_PATH = "docquad/docquadnet256_trained_opset17.ort"

        /** Renders the source bitmap into a 256×256 letterboxed bitmap (black bars). */
        private fun renderLetterbox256(src: Bitmap, lb: DocQuadLetterbox): Bitmap {
            val out = Bitmap.createBitmap(DocQuadOrtRunner.IN_W, DocQuadOrtRunner.IN_H, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(out)
            canvas.drawColor(Color.BLACK)

            val left = lb.offsetX.toFloat()
            val top = lb.offsetY.toFloat()
            val right = (lb.offsetX + lb.srcW.toDouble() * lb.scale).toFloat()
            val bottom = (lb.offsetY + lb.srcH.toDouble() * lb.scale).toFloat()
            canvas.drawBitmap(src, null, RectF(left, top, right, bottom), null)
            return out
        }

        /** Converts a 256×256 ARGB bitmap to NCHW float32 array (RGB, 0..1). */
        private fun bitmapToNchwFloat01(bmp: Bitmap): FloatArray {
            val w = bmp.width; val h = bmp.height
            require(w == 256 && h == 256) { "bitmap must be 256x256" }

            val hw = h * w
            val out = FloatArray(3 * hw)
            val px = IntArray(hw)
            bmp.getPixels(px, 0, w, 0, 0, w, h)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val c = px[y * w + x]
                    val idx = y * w + x
                    out[idx]          = ((c shr 16) and 0xFF) / 255.0f // R
                    out[hw + idx]     = ((c shr  8) and 0xFF) / 255.0f // G
                    out[2 * hw + idx] = ( c         and 0xFF) / 255.0f // B
                }
            }
            return out
        }

        private fun isValidQuad(c: Array<DoubleArray>?, w: Int, h: Int): Boolean {
            if (c == null || c.size != 4) return false
            for (i in 0 until 4) {
                if (c[i].size != 2) return false
                val x = c[i][0]; val y = c[i][1]
                if (!x.isFinite() || !y.isFinite()) return false
                if (x < -w * 0.25 || x > w * 1.25) return false
                if (y < -h * 0.25 || y > h * 1.25) return false
            }
            return true
        }
    }
}
