package org.tensorflow.lite.examples.transfer.api


import android.content.Context
import android.content.res.AssetManager
import org.tensorflow.lite.examples.transfer.api.Constants
import org.tensorflow.lite.examples.transfer.api.LiteModelWrapper
import org.tensorflow.lite.examples.transfer.api.ModelLoader
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Handles loading various parts of the model stored as a directory under Android assets.  */
open class AssetModelLoader(context: Context, private val directoryName: String) : ModelLoader {
    private val assetManager: AssetManager

    /**
     * Create a loader for a transfer learning model under given directory.
     *
     * @param directoryName path to model directory in assets tree.
     */
    init {
        assetManager = context.assets
    }

    @Throws(IOException::class)
    protected fun loadMappedFile(filePath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd("${directoryName}/${filePath}")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    override fun loadInitializeModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile(Constants.INITIALIZE_TFLITE))
    }

    @Throws(IOException::class)
    override fun loadBaseModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile(Constants.BOTTLENECK_TFLITE))
    }

    @Throws(IOException::class)
    override fun loadTrainModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile(Constants.TRAIN_HEAD_TFLITE))
    }

    @Throws(IOException::class)
    override fun loadInferenceModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile(Constants.INFERENCE_TFLITE))
    }

    @Throws(IOException::class)
    override fun loadOptimizerModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile(Constants.OPTIMIZER_TFLITE))
    }

}