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
import java.util.*

/** A wrapper for TFLite optimizer model.  */
class LiteOptimizerModel internal constructor(private val modelWrapper: LiteModelWrapper?) : Closeable {

    /**
     * Performs a single optimizer step.
     *
     * @param currentParams current values of model trainable parameters.
     * @param gradients trainable parameter gradients.
     * @param optimizerState current mutable optimizer state.
     * @param newParams where to store new parameter values.
     * @param newOptimizerState where to store new mutable optimizer state.
     */
    fun performStep(
            currentParams: Array<ByteBuffer?>,
            gradients: Array<ByteBuffer?>,
            optimizerState: Array<ByteBuffer?>?,
            newParams: Array<ByteBuffer?>,
            newOptimizerState: Array<ByteBuffer?>?) {
        val inputs = arrayOfNulls<Any>(currentParams.size + gradients.size)
        System.arraycopy(currentParams, 0, inputs, 0, currentParams.size)
        System.arraycopy(gradients, 0, inputs, currentParams.size, gradients.size)
        val outputs: MutableMap<Int, Any?> = TreeMap()
        for (paramIdx in newParams.indices) {
            outputs[paramIdx] = newParams[paramIdx]
        }
        modelWrapper!!.interpreter.runForMultipleInputsOutputs(inputs, outputs)
        for (buffer in currentParams) {
            buffer!!.rewind()
        }
        for (buffer in gradients) {
            buffer!!.rewind()
        }
        for (buffer in newParams) {
            buffer!!.rewind()
        }
    }

    /**
     * Reads the sizes of the mutable optimizer state elements.
     *
     * @return sizes of optimizer state elements.
     */
    fun stateElementSizes(): IntArray {
        // The generic optimizer model signature is:
        // *variables, *gradients, *optim_state -> *new_variables, *new_optim_state
        // There is no metadata included that would contain the number of variables
        // for the model, but we can easily infer it using the fact that
        // len(variables) == len(gradients) == len(new_variables) == number of variables.
        val numVariables = (modelWrapper!!.interpreter.inputTensorCount
                - modelWrapper.interpreter.outputTensorCount)
        val result = IntArray(modelWrapper.interpreter.inputTensorCount - numVariables * 2)
        for (inputIdx in numVariables * 2 until modelWrapper.interpreter.inputTensorCount) {
            result[inputIdx - numVariables * 2] = modelWrapper.interpreter.getInputTensor(inputIdx).numElements()
        }
        return result
    }

    override fun close() {
        modelWrapper!!.close()
    }

    companion object {
        private const val FLOAT_BYTES = 4
    }

}