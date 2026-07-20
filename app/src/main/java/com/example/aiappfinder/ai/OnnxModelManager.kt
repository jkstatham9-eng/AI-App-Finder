package com.example.aiappfinder.ai

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class OnnxModelManager @Inject constructor(private val context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private var visualSession: OrtSession? = null
    private var textSession: OrtSession? = null

    init {
        // Models should be placed in assets/models/
        // For production, these would be loaded from assets
        // loadModels()
    }

    private fun loadModels() {
        val visualModel = context.assets.open("models/visual_encoder.onnx").readBytes()
        visualSession = env.createSession(visualModel)

        val textModel = context.assets.open("models/text_encoder.onnx").readBytes()
        textSession = env.createSession(textModel)
    }

    fun getVisualEmbedding(bitmap: Bitmap): FloatArray {
        if (visualSession == null) loadModels()
        
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val floatBuffer = bitmapToFloatBuffer(resizedBitmap)
        
        val inputName = visualSession?.inputNames?.iterator()?.next() ?: return floatArrayOf()
        val tensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 224, 224))
        
        val output = visualSession?.run(mapOf(inputName to tensor))
        val result = output?.get(0)?.value as? Array<FloatArray>
        return result?.get(0) ?: floatArrayOf()
    }

    fun getTextEmbedding(text: String): FloatArray {
        if (textSession == null) loadModels()
        
        // Tokenization logic would go here (Simplified for structure)
        val tokens = tokenize(text)
        val inputName = textSession?.inputNames?.iterator()?.next() ?: return floatArrayOf()
        val tensor = OnnxTensor.createTensor(env, tokens, longArrayOf(1, tokens.size.toLong()))
        
        val output = textSession?.run(mapOf(inputName to tensor))
        val result = output?.get(0)?.value as? Array<FloatArray>
        return result?.get(0) ?: floatArrayOf()
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        
        // Normalize: (pixel / 255.0 - mean) / std
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        for (i in 0 until 224 * 224) {
            val p = pixels[i]
            buffer.put(i, ((p shr 16 and 0xFF) / 255.0f - mean[0]) / std[0])
            buffer.put(i + 224 * 224, ((p shr 8 and 0xFF) / 255.0f - mean[1]) / std[1])
            buffer.put(i + 448 * 448, ((p and 0xFF) / 255.0f - mean[2]) / std[2])
        }
        return buffer
    }

    private fun tokenize(text: String): LongArray {
        // Simplified tokenizer - In a real app, use a BPE tokenizer library.
        // For this buildable version, we provide a placeholder that matches the expected input size (77).
        val tokens = LongArray(77) { 0L }
        // Basic mapping for common words to demonstrate functionality
        text.lowercase().split(" ").forEachIndexed { index, word ->
            if (index < 77) tokens[index] = word.hashCode().toLong()
        }
        return tokens
    }
}
