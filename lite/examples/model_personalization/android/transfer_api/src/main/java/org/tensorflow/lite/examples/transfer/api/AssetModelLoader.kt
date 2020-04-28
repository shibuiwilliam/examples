/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.lite.examples.transfer.api

import android.content.Context
import android.content.res.AssetManager
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Handles loading various parts of the model stored as a directory under Android assets.  */
open class AssetModelLoader(context: Context, private val directoryName: String) : ModelLoader {
    private val assetManager: AssetManager

    @Throws(IOException::class)
    override fun loadInitializeModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile("initialize.tflite"))
    }

    @Throws(IOException::class)
    override fun loadBaseModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile("bottleneck.tflite"))
    }

    @Throws(IOException::class)
    override fun loadTrainModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile("train_head.tflite"))
    }

    @Throws(IOException::class)
    override fun loadInferenceModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile("inference.tflite"))
    }

    @Throws(IOException::class)
    override fun loadOptimizerModel(): LiteModelWrapper {
        return LiteModelWrapper(loadMappedFile("optimizer.tflite"))
    }

    @Throws(IOException::class)
    protected fun loadMappedFile(filePath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(directoryName + "/" + filePath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Create a loader for a transfer learning model under given directory.
     *
     * @param directoryName path to model directory in assets tree.
     */
    init {
        assetManager = context.assets
    }
}