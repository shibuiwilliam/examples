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

/**
 * A wrapper for TFLite model that calculates the gradients of trainable layers.
 */
internal class LiteTrainHeadModel(private val modelWrapper: LiteModelWrapper?) : Closeable {

    /**
     * Performs single training pass (forward + backward).
     *
     * @param bottleneckBatch image bottlenecks.
     * @param classBatch ground truth labels for images.
     * @param modelParameters current model trainable parameter values.
     * @param modelGradients where to store model trainable parameter gradients.
     * @return loss
     */
    fun calculateGradients(
            bottleneckBatch: ByteBuffer,
            classBatch: ByteBuffer,
            modelParameters: Array<ByteBuffer?>,
            modelGradients: Array<ByteBuffer?>): Float {
        require(modelParameters.size == modelGradients.size) {
            String.format(
                    "Parameter array size (%d) is different from gradient array size (%d)",
                    modelParameters.size,
                    modelGradients.size)
        }
        require(modelWrapper!!.interpreter.outputTensorCount == modelParameters.size + 1) {
            String.format(
                    "Model expected %d parameter tensors, but got %d",
                    modelWrapper.interpreter.inputTensorCount - 1,
                    modelParameters.size)
        }
        val lossBuffer = ByteBuffer.allocateDirect(Constants.FLOAT_BYTES)
        lossBuffer.order(ByteOrder.nativeOrder())
        val outputs: MutableMap<Int, Any?> = TreeMap()
        outputs[0] = lossBuffer
        for (outputIndex in 1 until modelWrapper.interpreter.outputTensorCount) {
            outputs[outputIndex] = modelGradients[outputIndex - 1]
        }
        val inputs = arrayOfNulls<Any>(modelParameters.size + 2)
        inputs[0] = bottleneckBatch
        inputs[1] = classBatch
        System.arraycopy(modelParameters, 0, inputs, 2, modelParameters.size)
        modelWrapper.interpreter.runForMultipleInputsOutputs(inputs, outputs)
        bottleneckBatch.rewind()
        classBatch.rewind()
        for (buffer in modelParameters) {
            buffer!!.rewind()
        }
        for (buffer in modelGradients) {
            buffer!!.rewind()
        }
        lossBuffer.rewind()
        return lossBuffer.float
    }

    val batchSize: Int
        get() = modelWrapper!!.interpreter.getInputTensor(0).shape()[0]

    val parameterSizes: IntArray
        get() {
            val parameterSizes = IntArray(modelWrapper!!.interpreter.inputTensorCount - 2)
            for (inputIndex in 2 until modelWrapper.interpreter.inputTensorCount) {
                parameterSizes[inputIndex - 2] = modelWrapper.interpreter.getInputTensor(inputIndex).numElements()
            }
            return parameterSizes
        }

    val parameterShapes: Array<IntArray?>
        get() {
            val interpreter = modelWrapper!!.interpreter
            val parameterShapes = arrayOfNulls<IntArray>(interpreter.inputTensorCount - 2)
            for (inputIndex in 2 until interpreter.inputTensorCount) {
                val inputTensor = interpreter.getInputTensor(inputIndex)
                parameterShapes[inputIndex - 2] = IntArray(inputTensor.numDimensions())
                System.arraycopy(
                        inputTensor.shape(),
                        0,
                        parameterShapes[inputIndex - 2]!!,
                        0,
                        inputTensor.numDimensions())
            }
            return parameterShapes
        }

    override fun close() {
        modelWrapper!!.close()
    }
}