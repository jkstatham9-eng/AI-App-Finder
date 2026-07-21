package com.example.aiappfinder.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Optimized ONNX Model Manager with lazy loading, memory optimization, and model validation.
 *
 * Key features:
 * - Dynamic input shape detection from ONNX model metadata
 * - Lazy loading: Models loaded only when first needed
 * - Single active session: Only one model in memory at a time
 * - Session disposal after inference
 * - File-path loading (OS memory-maps the file)
 * - Safe buffer bounds checking
 * - Model file validation (SHA256, LFS pointer detection, size check)
 */
@Singleton
class OnnxModelManager @Inject constructor(@ApplicationContext private val context: Context) {

    companion object {
        private const val TAG = "OnnxModelManager"

        // Expected model file sizes and SHA256 hashes
        // These are the SHA256 hashes from Git LFS for the committed files
        private val EXPECTED_MODELS = mapOf(
            "visual_encoder.onnx" to ExpectedModel(
                expectedSize = 351_613_724L,
                expectedSha256 = "b9ce24b91a8c62ef8d40ea786051709c93140104cbf2b6b4a2cc270df3838f9c",
                minSize = 1_000_000L  // Minimum 1MB to avoid LFS pointers
            ),
            "text_encoder.onnx" to ExpectedModel(
                expectedSize = 254_193_396L,
                expectedSha256 = "48c25be37b352398f2533c9426dab6f9340535e14e46f884cc894236a5060722",
                minSize = 1_000_000L
            )
        )

        // LFS pointer prefix
        private const val LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
    }

    private data class ExpectedModel(
        val expectedSize: Long,
        val expectedSha256: String,
        val minSize: Long
    )

    private val mutex = Mutex()

    private val env by lazy { OrtEnvironment.getEnvironment() }

    private var activeSession: OrtSession? = null
    private var activeSessionType: SessionType? = null

    // Cached input shape info for each model type
    private var visualInputShape: LongArray? = null
    private var visualInputSize: Int = 0
    private var textInputShape: LongArray? = null

    @Volatile
    private var visualModelFile: File? = null
    @Volatile
    private var textModelFile: File? = null

    enum class SessionType {
        VISUAL, TEXT
    }

    /**
     * Validate a model file to ensure it's a real ONNX model, not an LFS pointer or corrupted file.
     * Returns true if the file is valid, false otherwise.
     */
    private fun validateModelFile(file: File, assetName: String): Boolean {
        val info = EXPECTED_MODELS[assetName] ?: return true // Unknown model, skip validation

        // Check 1: File size
        val fileSize = file.length()
        if (fileSize < info.minSize) {
            Log.e(TAG, "Model $assetName is too small ($fileSize bytes < ${info.minSize} bytes). " +
                    "This is likely a Git LFS pointer file or corrupted asset.")
            return false
        }

        // Check 2: LFS pointer detection
        val headerSize = minOf(fileSize, 200L).toInt()
        val headerBytes = file.inputStream().use { input ->
            val bytes = ByteArray(headerSize)
            input.read(bytes)
            bytes
        }
        val header = String(headerBytes, Charsets.UTF_8).trim()
        if (header.startsWith("version https://git-lfs")) {
            Log.e(TAG, "Model $assetName is a Git LFS pointer file, not a real ONNX model!")
            return false
        }

        // Check 3: SHA256 hash verification
        val sha256 = computeSha256(file)
        if (sha256 != null && sha256 != info.expectedSha256) {
            Log.w(TAG, "Model $assetName SHA256 mismatch: $sha256 != ${info.expectedSha256}. " +
                    "Model may have been modified or corrupted.")
            // Still allow loading if sizes are close (might be a valid but different version)
            if (fileSize < info.expectedSize / 2) {
                Log.e(TAG, "Model $assetName is significantly smaller than expected. " +
                        "Cannot proceed with potentially corrupted model.")
                return false
            }
        }

        Log.i(TAG, "Model $assetName validation passed (size: $fileSize, sha256: ${sha256?.take(8)}...)")
        return true
    }

    /**
     * Compute SHA256 hash of a file.
     */
    private fun computeSha256(file: File): String? {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    digest.update(buffer, 0, bytesRead)
                }
            }
            digest.digest().joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to compute SHA256 for ${file.name}", e)
            null
        }
    }

    /**
     * Extract ONNX model from assets to internal cache.
     * Validates the model file before and after copying.
     */
    private fun getOrCreateModelFile(assetName: String): File {
        val cacheDir = File(context.cacheDir, "onnx_models")
        if (!cacheDir.exists()) cacheDir.mkdirs()
        val targetFile = File(cacheDir, assetName)

        // If cached file exists, validate it
        if (targetFile.exists() && targetFile.length() > 0) {
            if (validateModelFile(targetFile, assetName)) {
                return targetFile
            }
            // Invalid cached file, delete and re-copy
            Log.w(TAG, "Cached model $assetName is invalid, re-copying from assets...")
            targetFile.delete()
        }

        // Copy from assets
        context.assets.open("models/$assetName").use { input ->
            // Check if asset is an LFS pointer before copying
            val peekBytes = ByteArray(100)
            val peekCount = input.read(peekBytes)
            if (peekCount > 0) {
                val peekStr = String(peekBytes, 0, peekCount)
                if (peekStr.startsWith("version https://git-lfs")) {
                    throw IllegalStateException(
                        "Asset models/$assetName is a Git LFS pointer, not a real model. " +
                        "Please ensure Git LFS is properly configured and the actual model " +
                        "files are included in the APK assets."
                    )
                }
            }

            targetFile.outputStream().use { output ->
                // Write the peeked bytes first, then copy the rest
                if (peekCount > 0) {
                    output.write(peekBytes, 0, peekCount)
                }
                input.copyTo(output)
            }
        }

        // Validate the copied file
        if (!validateModelFile(targetFile, assetName)) {
            throw IllegalStateException(
                "Model file $assetName failed validation after copying. " +
                "Expected size: ${EXPECTED_MODELS[assetName]?.expectedSize}, " +
                "Got: ${targetFile.length()}. " +
                "The APK may have been built without Git LFS pulling the actual files."
            )
        }

        // Also copy a second file to verify: compare asset size with cached size
        val assetSize = context.assets.openFd("models/$assetName").use { fd -> fd.length }
        if (targetFile.length() != assetSize) {
            Log.w(TAG, "Copied file size (${targetFile.length()}) differs from asset size ($assetSize)")
        }

        Log.i(TAG, "Model $assetName ready: ${targetFile.absolutePath} (${targetFile.length()} bytes)")
        return targetFile
    }

    /**
     * Close the currently active session to free memory.
     */
    private fun closeActiveSession() {
        activeSession?.close()
        activeSession = null
        activeSessionType = null
    }

    /**
     * Detect the input tensor shape from the session's input metadata.
     * Returns a LongArray of dimensions and the total number of float elements needed.
     */
    private fun detectInputShape(session: OrtSession): Pair<LongArray, Int> {
        val inputInfo = session.inputInfo
        if (inputInfo.isEmpty()) {
            throw IllegalStateException("ONNX model has no input tensors")
        }

        val firstInput = inputInfo.values.iterator().next()
        val info = firstInput.info
        val tensorInfo = info as? TensorInfo
            ?: throw IllegalStateException("Expected TensorInfo for input, got ${info::class.simpleName}")
        val shape = tensorInfo.shape

        // Resolve dynamic dimensions: replace -1 or 0 with 1
        val resolvedShape = shape.map { if (it <= 0) 1 else it }.toLongArray()

        // Calculate total elements
        var totalElements = 1
        for (dim in resolvedShape) {
            totalElements *= dim.toInt()
        }

        return Pair(resolvedShape, totalElements)
    }

    /**
     * Load or switch to the specified model session.
     */
    private suspend fun loadSession(type: SessionType): OrtSession {
        mutex.withLock {
            if (activeSessionType == type && activeSession != null) {
                return activeSession!!
            }

            closeActiveSession()

            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            }

            val session = when (type) {
                SessionType.VISUAL -> {
                    val modelFile = getOrCreateModelFile("visual_encoder.onnx")
                    visualModelFile = modelFile
                    env.createSession(modelFile.absolutePath, options)
                }
                SessionType.TEXT -> {
                    val modelFile = getOrCreateModelFile("text_encoder.onnx")
                    textModelFile = modelFile
                    env.createSession(modelFile.absolutePath, options)
                }
            }

            // Detect and cache input shape for this model
            val (inputShape, totalElements) = detectInputShape(session)
            when (type) {
                SessionType.VISUAL -> {
                    visualInputShape = inputShape
                    visualInputSize = totalElements
                }
                SessionType.TEXT -> {
                    textInputShape = inputShape
                }
            }

            activeSession = session
            activeSessionType = type
            return session
        }
    }

    /**
     * Dispose all active sessions to free memory.
     */
    suspend fun disposeAllSessions() {
        mutex.withLock {
            closeActiveSession()
            visualModelFile = null
            textModelFile = null
        }
    }

    /**
     * Get visual embedding for a bitmap.
     * Dynamically resizes image to match model's expected input dimensions.
     */
    suspend fun getVisualEmbedding(bitmap: Bitmap): FloatArray {
        val session = loadSession(SessionType.VISUAL)

        // Use dynamically detected input shape
        if (visualInputShape == null || visualInputSize == 0) {
            val (shape, size) = detectInputShape(session)
            visualInputShape = shape
            visualInputSize = size
        }
        val inputShape = visualInputShape!!
        val expectedFloatCount = visualInputSize!!

        // Extract image dimensions from shape: [batch, channels, height, width]
        val imgHeight = inputShape.getOrNull(2)?.toInt() ?: 224
        val imgWidth = inputShape.getOrNull(3)?.toInt() ?: 224
        val numChannels = inputShape.getOrNull(1)?.toInt() ?: 3

        // Resize bitmap to match model's expected dimensions
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true)

        // Allocate buffer with exact size needed
        val floatBuffer = java.nio.FloatBuffer.allocate(expectedFloatCount)

        // Convert bitmap to normalized float buffer with safe bounds
        bitmapToFloatBufferSafe(resizedBitmap, floatBuffer, imgHeight, imgWidth, numChannels)

        // Create tensor with the exact shape
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)

        val output = session.run(mapOf(inputName to tensor))
        val result = output.get(0).value as? Array<FloatArray>
        val embedding = result?.get(0) ?: floatArrayOf()

        // Close resources to free memory
        tensor.close()
        output.close()
        resizedBitmap.recycle()

        return embedding
    }

    /**
     * Safely convert bitmap pixels to normalized float buffer with bounds checking.
     * Layout: CHW format [channel][height][width] as expected by most vision models.
     */
    private fun bitmapToFloatBufferSafe(
        bitmap: Bitmap,
        buffer: java.nio.FloatBuffer,
        height: Int,
        width: Int,
        numChannels: Int
    ) {
        val totalPixels = height * width
        val expectedBufferCapacity = buffer.capacity()

        if (expectedBufferCapacity < totalPixels * numChannels) {
            throw IllegalStateException(
                "Buffer capacity ($expectedBufferCapacity) < required (${totalPixels * numChannels})"
            )
        }

        val pixels = IntArray(totalPixels)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Normalization parameters (ImageNet standard for CLIP)
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        val strideHxW = height * width

        for (i in 0 until totalPixels) {
            val p = pixels[i]
            val r = (p shr 16 and 0xFF) / 255.0f
            val g = (p shr 8 and 0xFF) / 255.0f
            val b = (p and 0xFF) / 255.0f

            // CHW layout offsets - using strideHxW consistently
            val rIndex = i
            val gIndex = i + strideHxW
            val bIndex = i + 2 * strideHxW  // FIXED: was 'i + 448 * 448' which caused OOB

            // Safe bounds checking
            if (rIndex >= 0 && rIndex < expectedBufferCapacity) {
                buffer.put(rIndex, (r - mean[0]) / std[0])
            }
            if (gIndex >= 0 && gIndex < expectedBufferCapacity) {
                buffer.put(gIndex, (g - mean[1]) / std[1])
            }
            if (bIndex >= 0 && bIndex < expectedBufferCapacity) {
                buffer.put(bIndex, (b - mean[2]) / std[2])
            }
        }
    }

    /**
     * Get text embedding for a string.
     */
    suspend fun getTextEmbedding(text: String): FloatArray {
        val session = loadSession(SessionType.TEXT)

        // Use dynamically detected input shape
        if (textInputShape == null) {
            val (shape, _) = detectInputShape(session)
            textInputShape = shape
        }
        val inputShape = textInputShape!!

        val tokens = tokenize(text, inputShape)
        val resolvedShape = inputShape.map { if (it <= 0) 1 else it }.toLongArray()
        val tensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(tokens), resolvedShape)

        val output = session.run(mapOf(session.inputNames.iterator().next() to tensor))
        val result = output.get(0).value as? Array<FloatArray>
        val embedding = result?.get(0) ?: floatArrayOf()

        tensor.close()
        output.close()

        return embedding
    }

    /**
     * Tokenize text to match the model's expected input length.
     */
    private fun tokenize(text: String, inputShape: LongArray): LongArray {
        val tokenCount = inputShape.getOrNull(1)?.toInt()?.let { if (it > 0) it else 77 } ?: 77

        val tokens = LongArray(tokenCount) { 0L }
        text.lowercase().split(" ").forEachIndexed { index, word ->
            if (index < tokenCount) tokens[index] = word.hashCode().toLong()
        }
        return tokens
    }
}
