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
import java.nio.ByteOrder
import java.util.*

internal class LiteInferenceModel(private val modelWrapper: LiteModelWrapper?, private val numClasses: Int) : Closeable {
    fun runInference(bottleneck: ByteBuffer?, modelParameters: Array<ByteBuffer?>): FloatArray {
        val predictionsBuffer = ByteBuffer.allocateDirect(numClasses * FLOAT_BYTES)
        predictionsBuffer.order(ByteOrder.nativeOrder())
        val outputs: MutableMap<Int, Any> = TreeMap()
        outputs[0] = predictionsBuffer
        val inputs = arrayOfNulls<Any>(modelParameters.size + 1)
        inputs[0] = bottleneck
        System.arraycopy(modelParameters, 0, inputs, 1, modelParameters.size)
        modelWrapper!!.interpreter.runForMultipleInputsOutputs(inputs, outputs)
        bottleneck!!.rewind()
        for (buffer in modelParameters) {
            buffer!!.rewind()
        }
        predictionsBuffer.rewind()
        val predictions = FloatArray(numClasses)
        for (classIdx in 0 until numClasses) {
            predictions[classIdx] = predictionsBuffer.float
        }
        return predictions
    }

    override fun close() {
        modelWrapper!!.close()
    }

    companion object {
        private const val FLOAT_BYTES = 4
    }

}