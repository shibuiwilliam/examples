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

import java.io.Closeable
import java.nio.ByteBuffer

/**
 * A wrapper for TFLite model that generates bottlenecks from images.
 */
internal class LiteBottleneckModel(private val modelWrapper: LiteModelWrapper?) : Closeable {

    /**
     * Passes a single image through the bottleneck model.
     * @param image image RGB data.
     * @param outBottleneck where to store the bottleneck. A new buffer is allocated if null.
     * @return bottleneck data. This is either [outBottleneck], or a newly allocated buffer.
     */
    @Synchronized
    fun generateBottleneck(image: ByteBuffer, outBottleneck: ByteBuffer?): ByteBuffer? {
        var outBottleneck = outBottleneck
        if (outBottleneck == null) {
            outBottleneck = ByteBuffer.allocateDirect(numBottleneckFeatures * FLOAT_BYTES)
        }
        modelWrapper!!.interpreter.run(image, outBottleneck)
        image.rewind()
        outBottleneck!!.rewind()
        return outBottleneck
    }

    val numBottleneckFeatures: Int
        get() = modelWrapper!!.interpreter.getOutputTensor(0).numElements()

    val bottleneckShape: IntArray
        get() = modelWrapper!!.interpreter.getOutputTensor(0).shape()

    override fun close() {
        modelWrapper!!.close()
    }

    companion object {
        private const val FLOAT_BYTES = 4
    }

}