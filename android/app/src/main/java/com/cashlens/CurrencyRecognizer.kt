package com.cashlens

import android.content.Context
import android.graphics.Bitmap
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class RecognitionResult(
    val currency: String,
    val denomination: String,
    val face: String,
    val confidence: Float,
    val label: String
)

private data class LabelsFile(@SerializedName("classes") val classes: List<String>)

class CurrencyRecognizer(context: Context) : AutoCloseable {

    private val encoder: Interpreter
    private val classifier: Interpreter
    private val labels: List<String>

    // Input size expected by encoder (MobileNetV2 224x224)
    private val IMAGE_SIZE = 224
    private val EMBEDDING_SIZE = 256

    init {
        encoder = Interpreter(loadModelFile(context, "banknote_encoder.tflite"))
        classifier = Interpreter(loadModelFile(context, "all_currencies_classifier.tflite"))
        val json = context.assets.open("labels.json").bufferedReader().readText()
        labels = Gson().fromJson(json, LabelsFile::class.java).classes
    }

    fun recognize(bitmap: Bitmap): RecognitionResult {
        val scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)

        // Preprocess: normalize to [-1, 1] as MobileNetV2 expects
        val inputBuffer = ByteBuffer.allocateDirect(1 * IMAGE_SIZE * IMAGE_SIZE * 3 * 4)
            .apply { order(ByteOrder.nativeOrder()) }

        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {
                val px = scaled.getPixel(x, y)
                inputBuffer.putFloat(((px shr 16 and 0xFF) / 127.5f) - 1f) // R
                inputBuffer.putFloat(((px shr 8  and 0xFF) / 127.5f) - 1f) // G
                inputBuffer.putFloat(((px        and 0xFF) / 127.5f) - 1f) // B
            }
        }

        // Run encoder: image -> 256-dim embedding
        val embeddingBuffer = Array(1) { FloatArray(EMBEDDING_SIZE) }
        encoder.run(inputBuffer, embeddingBuffer)

        // Run classifier: embedding -> class probabilities
        val classBuffer = Array(1) { FloatArray(labels.size) }
        val embeddingInput = ByteBuffer.allocateDirect(1 * EMBEDDING_SIZE * 4)
            .apply {
                order(ByteOrder.nativeOrder())
                embeddingBuffer[0].forEach { putFloat(it) }
            }
        classifier.run(embeddingInput, classBuffer)

        // Parse result
        val probs = classBuffer[0]
        val bestIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val bestLabel = labels[bestIdx]          // e.g. "TRY_100_1"
        val parts = bestLabel.split("_")
        val currency = parts[0]                  // TRY
        val face = parts.last()                  // 1 or 2
        val denom = parts.drop(1).dropLast(1).joinToString("_")  // 100

        return RecognitionResult(
            currency = currency,
            denomination = denom,
            face = if (face == "1") "Ön" else "Arka",
            confidence = probs[bestIdx],
            label = bestLabel
        )
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val fd = context.assets.openFd(filename)
        val stream = FileInputStream(fd.fileDescriptor)
        return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    override fun close() {
        encoder.close()
        classifier.close()
    }
}
