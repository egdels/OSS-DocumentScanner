package docquad;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;

/**
 * High-level detector: Bitmap in → document corner coordinates out.
 * <p>
 * Usage example:
 * <pre>
 *   DocQuadDetector detector = new DocQuadDetector();
 *   double[][] corners = detector.detect(bitmap, context);
 *   // corners = { {xTL,yTL}, {xTR,yTR}, {xBR,yBR}, {xBL,yBL} }
 *   // in original bitmap pixel coordinates, or null on failure.
 * </pre>
 * <p>
 * <b>Android dependency:</b> Uses {@code Bitmap}, {@code Canvas}, {@code Context}.
 */
public final class DocQuadDetector {

    public static final String DEFAULT_MODEL_ASSET_PATH = "docquad/docquadnet256_trained_opset17.ort";

    private final String modelAssetPath;
    private final DocQuadOrtRunner injectedRunner;

    public DocQuadDetector() {
        this(DEFAULT_MODEL_ASSET_PATH, null);
    }

    public DocQuadDetector(String modelAssetPath) {
        this(modelAssetPath, null);
    }

    /** Use a pre-loaded runner (e.g. for live camera). */
    public DocQuadDetector(DocQuadOrtRunner injectedRunner) {
        this(DEFAULT_MODEL_ASSET_PATH, injectedRunner);
    }

    private DocQuadDetector(String modelAssetPath, DocQuadOrtRunner injectedRunner) {
        this.modelAssetPath = modelAssetPath;
        this.injectedRunner = injectedRunner;
    }

    /**
     * Detects document corners in the given bitmap.
     *
     * @param src the input image (any size)
     * @param ctx Android context (for asset loading)
     * @return 4×2 array {{xTL,yTL},{xTR,yTR},{xBR,yBR},{xBL,yBL}} in original pixel coords,
     *         or {@code null} on failure
     */
    public double[][] detect(Bitmap src, Context ctx) {
        if (src == null || ctx == null) return null;

        Bitmap in256 = null;
        try {
            int srcW = src.getWidth();
            int srcH = src.getHeight();
            if (srcW <= 0 || srcH <= 0) return null;

            DocQuadLetterbox lb = DocQuadLetterbox.create(srcW, srcH, DocQuadOrtRunner.IN_W, DocQuadOrtRunner.IN_H);
            in256 = renderLetterbox256(src, lb);
            float[] input = bitmapToNchwFloat01(in256);

            DocQuadOrtRunner.Outputs outputs;
            if (injectedRunner != null) {
                outputs = injectedRunner.run(input);
            } else {
                outputs = DocQuadOrtRunner.getInstance(ctx, modelAssetPath).run(input);
            }

            DocQuadPostprocessor.Result r = DocQuadPostprocessor.postprocess(
                    outputs.cornerHeatmaps(), outputs.maskLogits(),
                    lb, DocQuadPostprocessor.PeakMode.REFINE_3X3);

            if (r == null || r.chosenQuadOriginal == null || r.chosenQuadOriginal.length != 4)
                return null;
            if (!isValidQuad(r.chosenQuadOriginal, srcW, srcH))
                return null;

            return r.chosenQuadOriginal;
        } catch (Throwable t) {
            return null;
        } finally {
            if (in256 != null && !in256.isRecycled()) {
                try { in256.recycle(); } catch (Throwable ignore) {}
            }
        }
    }

    // ── Preprocessing ──

    /** Renders the source bitmap into a 256×256 letterboxed bitmap (black bars). */
    private static Bitmap renderLetterbox256(Bitmap src, DocQuadLetterbox lb) {
        Bitmap out = Bitmap.createBitmap(DocQuadOrtRunner.IN_W, DocQuadOrtRunner.IN_H, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        canvas.drawColor(Color.BLACK);

        float left = (float) lb.offsetX;
        float top = (float) lb.offsetY;
        float right = (float) (lb.offsetX + (double) lb.srcW * lb.scale);
        float bottom = (float) (lb.offsetY + (double) lb.srcH * lb.scale);
        canvas.drawBitmap(src, null, new RectF(left, top, right, bottom), null);
        return out;
    }

    /** Converts a 256×256 ARGB bitmap to NCHW float32 array (RGB, 0..1). */
    private static float[] bitmapToNchwFloat01(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        if (w != 256 || h != 256) throw new IllegalArgumentException("bitmap must be 256x256");

        int hw = h * w;
        float[] out = new float[3 * hw];
        int[] px = new int[hw];
        bmp.getPixels(px, 0, w, 0, 0, w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int c = px[y * w + x];
                int idx = y * w + x;
                out[idx]          = ((c >> 16) & 0xFF) / 255.0f; // R
                out[hw + idx]     = ((c >>  8) & 0xFF) / 255.0f; // G
                out[2 * hw + idx] = ( c        & 0xFF) / 255.0f; // B
            }
        }
        return out;
    }

    private static boolean isValidQuad(double[][] c, int w, int h) {
        if (c == null || c.length != 4) return false;
        for (int i = 0; i < 4; i++) {
            if (c[i] == null || c[i].length != 2) return false;
            double x = c[i][0], y = c[i][1];
            if (!Double.isFinite(x) || !Double.isFinite(y)) return false;
            if (x < -w * 0.25 || x > w * 1.25) return false;
            if (y < -h * 0.25 || y > h * 1.25) return false;
        }
        return true;
    }
}
