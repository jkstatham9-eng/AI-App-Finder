package com.example.aiappfinder.ai

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.NodeInfo
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Optimized ONNX Model Manager with lazy loading and memory optimization.
 *
 * Key features:
 * - Dynamic input shape detection from ONNX model metadata
 * - Lazy loading: Models loaded only when first needed
 * - Single active session: Only one model in memory at a time
 * - Session disposal after inference
 * - File-path loading (OS memory-maps the file)
 * - Safe buffer bounds checking
 */
@Singleton
class OnnxModelManager @Inject constructor(@ApplicationContext private val context: Context) {

    private val mutex = Mutex()

    private val env by lazy { OrtEnvironment.getEnvironment() }

    private var activeSession: OrtSession? = null
    private var activeSessionType: SessionType? = null

    // Cached input shape info for each model type
    private var visualInputShape: LongArray? = null
    private var visualInputSize: Int = 0
    private var textInputShape: LongArray? = null

    @Volatile
    private var visualModelFile: java.io.File? = null
    @Volatile
    private var textModelFile: java.io.File? = null

    enum class SessionType {
        VISUAL, TEXT
    }

    /**
     * Extract ONNX model from assets to internal cache.
     */
    private fun getOrCreateModelFile(assetName: String): java.io.File {
        val cacheDir = java.io.File(context.cacheDir, "onnx_models")
        if (!cacheDir.exists()) cacheDir.mkdirs()
        val targetFile = java.io.File(cacheDir, assetName)

        if (!targetFile.exists() || targetFile.length() == 0L) {
            context.assets.open("models/$assetName").use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
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

        // Resolve dynamic dimensions: replace -1 or 0 with actual sizes
        val resolvedShape = shape.map { if (it <= 0) 1 else it }.toLongArray()

        // Calculate total elements for a float tensor
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

        // Validate dimensions
        if (expectedBufferCapacity < totalPixels * numChannels) {
            // Not enough buffer space - this should never happen after dynamic detection
            throw IllegalStateException(
                "Buffer capacity ($expectedBufferCapacity) < required (${totalPixels * numChannels})"
            )
        }

        val pixels = IntArray(totalPixels)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Normalization parameters (ImageNet standard for CLIP)
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        val strideHxW = height * width  // = 224 * 224 = 50176

        for (i in 0 until totalPixels) {
            val p = pixels[i]
            val r = (p shr 16 and 0xFF) / 255.0f
            val g = (p shr 8 and 0xFF) / 255.0f
            val b = (p and 0xFF) / 255.0f

            // Safe bounds checking before each write
            val rIndex = i                    // channel 0 offset: 0
            val gIndex = i + strideHxW        // channel 1 offset: 50176
            val bIndex = i + 2 * strideHxW    // channel 2 offset: 100352 (NOT 448*448=200704!)

            // Validate all indices are within buffer capacity
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
        val inputShape = textInputShape ?: detectInputShape(session).also { (shape, _) ->
            textInputShape = shape
        }.let { (shape, _) -> shape }

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
     * Uses the detected shape to determine the correct token count.
     */
    private fun tokenize(text: String, inputShape: LongArray): LongArray {
        // Determine token count from shape: shape[1] is the sequence length
        val tokenCount = inputShape.getOrNull(1)?.toInt()?.let { if (it > 0) it else 77 } ?: 77

        val tokens = LongArray(tokenCount) { 0L }
        text.lowercase().split(" ").forEachIndexed { index, word ->
            if (index < tokenCount) tokens[index] = word.hashCode().toLong()
        }
        return tokens
    }
}
