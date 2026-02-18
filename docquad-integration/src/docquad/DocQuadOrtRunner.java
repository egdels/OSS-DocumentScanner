package docquad;

import ai.onnxruntime.*;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Runs inference on the DocQuadNet-256 ONNX model using ONNX Runtime.
 * <p>
 * Input:  {@code float[3 * 256 * 256]} in NCHW layout, RGB, values 0..1<br>
 * Output: {@code corner_heatmaps [1,4,64,64]} + {@code mask_logits [1,1,64,64]}
 * <p>
 * Thread-safe singleton access via {@link #getInstance(Context, String)}.
 * <p>
 * <b>Android dependency:</b> Uses {@code Context} for asset loading and {@code Log} for logging.
 * Adapt these two points when porting to a non-Android environment.
 */
public final class DocQuadOrtRunner implements AutoCloseable {

    private static final String TAG = "DocQuadOrtRunner";

    public static final int IN_H = 256;
    public static final int IN_W = 256;
    public static final int OUT_H = 64;
    public static final int OUT_W = 64;

    private static volatile DocQuadOrtRunner instance;
    private static final Object LOCK = new Object();
    private static final Executor DEFAULT_EXECUTOR = Executors.newSingleThreadExecutor();

    private final OrtEnvironment env;
    private final OrtSession session;

    private DocQuadOrtRunner(Context context, String modelAssetPath) throws Exception {
        this.env = OrtEnvironment.getEnvironment();
        File modelFile = copyAssetToCache(context, modelAssetPath);

        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
            try { opts.addNnapi(); Log.i(TAG, "NNAPI EP enabled"); }
            catch (Throwable t) { Log.i(TAG, "NNAPI not available: " + t.getMessage()); }
            try { opts.addXnnpack(Collections.emptyMap()); Log.i(TAG, "XNNPACK EP enabled"); }
            catch (Throwable t) { Log.i(TAG, "XNNPACK not available: " + t.getMessage()); }
            this.session = env.createSession(modelFile.getAbsolutePath(), opts);
        }
        Log.d(TAG, "Model loaded from " + modelFile.getAbsolutePath());
    }

    /** Returns the singleton instance (thread-safe, lazy init). */
    public static DocQuadOrtRunner getInstance(Context context, String modelAssetPath) throws Exception {
        if (instance == null) {
            synchronized (LOCK) {
                if (instance == null) {
                    instance = new DocQuadOrtRunner(context, modelAssetPath);
                }
            }
        }
        return instance;
    }

    /** Async variant using default single-thread executor. */
    public static CompletableFuture<DocQuadOrtRunner> getInstanceAsync(Context context, String modelAssetPath) {
        return getInstanceAsync(context, modelAssetPath, DEFAULT_EXECUTOR);
    }

    /** Async variant with custom executor. */
    public static CompletableFuture<DocQuadOrtRunner> getInstanceAsync(Context context, String modelAssetPath, Executor executor) {
        return CompletableFuture.supplyAsync(() -> {
            try { return getInstance(context, modelAssetPath); }
            catch (Exception e) { throw new java.util.concurrent.CompletionException(e); }
        }, executor);
    }

    /** Returns true if the singleton is already loaded. */
    public static boolean isInstanceLoaded() {
        return instance != null;
    }

    /** Releases the singleton and closes the ONNX session. */
    public static void releaseInstance() {
        synchronized (LOCK) {
            if (instance != null) {
                try { instance.close(); } catch (Exception e) { Log.w(TAG, "Error closing: " + e.getMessage()); }
                instance = null;
            }
        }
    }

    /**
     * Runs inference.
     *
     * @param inputNchw float array of length {@code 3 * 256 * 256}, NCHW, RGB, 0..1
     * @return model outputs (corner heatmaps + mask logits)
     */
    public Outputs run(float[] inputNchw) throws Exception {
        if (inputNchw == null || inputNchw.length != 3 * IN_H * IN_W)
            throw new IllegalArgumentException("inputNchw must have length " + (3 * IN_H * IN_W));

        long[] inputShape = {1, 3, IN_H, IN_W};
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputNchw), inputShape);
             OrtSession.Result results = session.run(Collections.singletonMap("input", input))) {

            float[][][][] maskLogits = getRequiredFloat4d(results, "mask_logits");
            float[][][][] cornerHeatmaps = getRequiredFloat4d(results, "corner_heatmaps");
            return new Outputs(maskLogits, cornerHeatmaps);
        }
    }

    @Override
    public void close() throws Exception {
        if (session != null) session.close();
    }

    // ── Asset handling ──

    private static File copyAssetToCache(Context context, String assetPath) throws IOException {
        AssetManager am = context.getAssets();
        String baseName = new File(assetPath).getName();
        long versionCode;
        try {
            android.content.pm.PackageInfo pi =
                    context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
            versionCode = pi.getLongVersionCode();
        } catch (Exception e) { versionCode = -1L; }

        String versionedName = versionCode + "_" + baseName;
        File outFile = new File(context.getCacheDir(), versionedName);
        if (!outFile.exists()) {
            Log.i(TAG, "Copying asset " + assetPath + " to cache as " + versionedName);
            try (InputStream is = am.open(assetPath);
                 FileOutputStream fos = new FileOutputStream(outFile)) {
                byte[] buffer = new byte[256 * 1024];
                int len;
                while ((len = is.read(buffer)) != -1) fos.write(buffer, 0, len);
            }
            deleteStaleModelFiles(context.getCacheDir(), baseName, versionedName);
        }
        return outFile;
    }

    private static void deleteStaleModelFiles(File cacheDir, String baseName, String currentName) {
        File[] staleFiles = cacheDir.listFiles((dir, name) ->
                name.endsWith("_" + baseName) && !name.equals(currentName));
        if (staleFiles != null) {
            for (File stale : staleFiles) {
                if (stale.delete()) Log.i(TAG, "Deleted stale: " + stale.getName());
                else Log.w(TAG, "Failed to delete stale: " + stale.getName());
            }
        }
    }

    // ── Output extraction ──

    private static float[][][][] getRequiredFloat4d(OrtSession.Result results, String outputName) throws OrtException {
        Optional<OnnxValue> ov = results.get(outputName);
        if (ov.isEmpty()) throw new IllegalStateException("Missing output: " + outputName);
        OnnxValue val = ov.get();
        if (!(val instanceof OnnxTensor)) throw new IllegalStateException(outputName + " is not a tensor");
        Object raw = ((OnnxTensor) val).getValue();
        if (!(raw instanceof float[][][][]))
            throw new IllegalStateException(outputName + " is not float[][][][]");
        return (float[][][][]) raw;
    }

    /** Model outputs container. */
    public static final class Outputs {
        private final float[][][][] maskLogits;
        private final float[][][][] cornerHeatmaps;

        public Outputs(float[][][][] maskLogits, float[][][][] cornerHeatmaps) {
            this.maskLogits = maskLogits;
            this.cornerHeatmaps = cornerHeatmaps;
        }

        public float[][][][] maskLogits() { return maskLogits; }
        public float[][][][] cornerHeatmaps() { return cornerHeatmaps; }
    }
}
