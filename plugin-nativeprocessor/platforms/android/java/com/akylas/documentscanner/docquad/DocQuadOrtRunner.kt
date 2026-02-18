package com.akylas.documentscanner.docquad

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import android.content.Context
import android.util.Log

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/**
 * Runs inference on the DocQuadNet-256 ONNX model using ONNX Runtime.
 *
 * Input:  `float[3 * 256 * 256]` in NCHW layout, RGB, values 0..1
 * Output: `corner_heatmaps [1,4,64,64]` + `mask_logits [1,1,64,64]`
 *
 * Thread-safe singleton access via [getInstance].
 *
 * **Android dependency:** Uses `Context` for asset loading and `Log` for logging.
 */
class DocQuadOrtRunner private constructor(context: Context, modelAssetPath: String) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelFile = copyAssetToCache(context, modelAssetPath)

        val opts = OrtSession.SessionOptions()
        try {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(maxOf(1, Runtime.getRuntime().availableProcessors() / 2))
            try { opts.addNnapi(); Log.i(TAG, "NNAPI EP enabled") }
            catch (t: Throwable) { Log.i(TAG, "NNAPI not available: ${t.message}") }
            try { opts.addXnnpack(emptyMap()); Log.i(TAG, "XNNPACK EP enabled") }
            catch (t: Throwable) { Log.i(TAG, "XNNPACK not available: ${t.message}") }
            session = env.createSession(modelFile.absolutePath, opts)
        } finally {
            opts.close()
        }
        Log.d(TAG, "Model loaded from ${modelFile.absolutePath}")
    }

    /**
     * Runs inference.
     *
     * @param inputNchw float array of length `3 * 256 * 256`, NCHW, RGB, 0..1
     * @return model outputs (corner heatmaps + mask logits)
     */
    fun run(inputNchw: FloatArray): Outputs {
        require(inputNchw.size == 3 * IN_H * IN_W) { "inputNchw must have length ${3 * IN_H * IN_W}" }

        val inputShape = longArrayOf(1, 3, IN_H.toLong(), IN_W.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(inputNchw), inputShape).use { input ->
            session.run(mapOf("input" to input)).use { results ->
                val maskLogits = getRequiredFloat4d(results, "mask_logits")
                val cornerHeatmaps = getRequiredFloat4d(results, "corner_heatmaps")
                return Outputs(maskLogits, cornerHeatmaps)
            }
        }
    }

    override fun close() {
        session.close()
    }

    // ── Asset handling ──

    companion object {
        private const val TAG = "DocQuadOrtRunner"

        const val IN_H = 256
        const val IN_W = 256
        const val OUT_H = 64
        const val OUT_W = 64

        @Volatile
        private var instance: DocQuadOrtRunner? = null
        private val LOCK = Any()
        private val DEFAULT_EXECUTOR: Executor = Executors.newSingleThreadExecutor()

        /** Returns the singleton instance (thread-safe, lazy init). */
        @JvmStatic
        @Throws(Exception::class)
        fun getInstance(context: Context, modelAssetPath: String): DocQuadOrtRunner {
            if (instance == null) {
                synchronized(LOCK) {
                    if (instance == null) {
                        instance = DocQuadOrtRunner(context, modelAssetPath)
                    }
                }
            }
            return instance!!
        }

        /** Async variant using default single-thread executor. */
        @JvmStatic
        fun getInstanceAsync(context: Context, modelAssetPath: String): CompletableFuture<DocQuadOrtRunner> {
            return getInstanceAsync(context, modelAssetPath, DEFAULT_EXECUTOR)
        }

        /** Async variant with custom executor. */
        @JvmStatic
        fun getInstanceAsync(context: Context, modelAssetPath: String, executor: Executor): CompletableFuture<DocQuadOrtRunner> {
            return CompletableFuture.supplyAsync({
                try { getInstance(context, modelAssetPath) }
                catch (e: Exception) { throw java.util.concurrent.CompletionException(e) }
            }, executor)
        }

        /** Returns true if the singleton is already loaded. */
        @JvmStatic
        fun isInstanceLoaded(): Boolean = instance != null

        /** Releases the singleton and closes the ONNX session. */
        @JvmStatic
        fun releaseInstance() {
            synchronized(LOCK) {
                instance?.let {
                    try { it.close() } catch (e: Exception) { Log.w(TAG, "Error closing: ${e.message}") }
                    instance = null
                }
            }
        }

        @Throws(IOException::class)
        private fun copyAssetToCache(context: Context, assetPath: String): File {
            val am = context.assets
            val baseName = File(assetPath).name
            val versionCode: Long = try {
                val pi = context.packageManager.getPackageInfo(context.packageName, 0)
                pi.longVersionCode
            } catch (e: Exception) { -1L }

            val versionedName = "${versionCode}_$baseName"
            val outFile = File(context.cacheDir, versionedName)
            if (!outFile.exists()) {
                Log.i(TAG, "Copying asset $assetPath to cache as $versionedName")
                am.open(assetPath).use { inputStream ->
                    FileOutputStream(outFile).use { fos ->
                        val buffer = ByteArray(256 * 1024)
                        var len: Int
                        while (inputStream.read(buffer).also { len = it } != -1) {
                            fos.write(buffer, 0, len)
                        }
                    }
                }
                deleteStaleModelFiles(context.cacheDir, baseName, versionedName)
            }
            return outFile
        }

        private fun deleteStaleModelFiles(cacheDir: File, baseName: String, currentName: String) {
            val staleFiles = cacheDir.listFiles { _, name ->
                name.endsWith("_$baseName") && name != currentName
            }
            staleFiles?.forEach { stale ->
                if (stale.delete()) Log.i(TAG, "Deleted stale: ${stale.name}")
                else Log.w(TAG, "Failed to delete stale: ${stale.name}")
            }
        }

        private fun getRequiredFloat4d(results: OrtSession.Result, outputName: String): Array<Array<Array<FloatArray>>> {
            val ov = results.get(outputName)
            check(ov.isPresent) { "Missing output: $outputName" }
            val value = ov.get()
            check(value is OnnxTensor) { "$outputName is not a tensor" }
            val raw = value.value
            check(raw is Array<*>) { "$outputName is not float[][][][]" }
            @Suppress("UNCHECKED_CAST")
            return raw as Array<Array<Array<FloatArray>>>
        }
    }

    /** Model outputs container. */
    data class Outputs(
        val maskLogits: Array<Array<Array<FloatArray>>>,
        val cornerHeatmaps: Array<Array<Array<FloatArray>>>
    )
}
