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

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*

/** Gradient calculation test for [LiteTrainHeadModel].  */
@RunWith(AndroidJUnit4::class)
class LiteTrainHeadModelTest {
    private interface Supplier<T> {
        fun get(): T
    }
    private class CSupplier(val t: Float): Supplier<kotlin.Float>{
        override fun get(): kotlin.Float {
            return t
        }
    }

    @Test
    @Throws(IOException::class)
    fun shouldCalculateGradientsCorrectly() {
        val model = LiteTrainHeadModel(
                AssetModelLoader(InstrumentationRegistry.getInstrumentation().context, "model")
                        .loadTrainModel())
        val bottlenecks = generateRandomByteBuffer(
//                BATCH_SIZE * BOTTLENECK_SIZE, Supplier<Float> { random.nextFloat() })
        BATCH_SIZE * BOTTLENECK_SIZE, CSupplier(random.nextFloat()) )
        val classes = generateRandomByteBuffer(BATCH_SIZE * NUM_CLASSES, CSupplier(0f))
        val parameterShapes = model.parameterShapes
        val parameters = arrayOfNulls<ByteBuffer>(parameterShapes.size)
        for (parameterIdx in parameterShapes.indices) {
            parameters[parameterIdx] = generateRandomByteBuffer(parameterShapes[parameterIdx])
        }

        // Fill with one-hot.
        for (sampleIdx in 0 until BATCH_SIZE) {
            val sampleClass = random.nextInt(NUM_CLASSES)
            classes.putFloat((sampleIdx * NUM_CLASSES + sampleClass) * FLOAT_BYTES, 1f)
        }
        val gradients = arrayOfNulls<ByteBuffer>(parameterShapes.size)
        for (parameterIdx in parameterShapes.indices) {
            gradients[parameterIdx] = generateRandomByteBuffer(parameterShapes[parameterIdx])
        }
        val loss = model.calculateGradients(bottlenecks, classes, parameters, gradients)
        for (parameterIdx in parameters.indices) {
            val parameter = parameters[parameterIdx]
            val analyticalGrads = gradients[parameterIdx]
            val numElementsToCheck = Math.min(product(parameterShapes[parameterIdx]), MAX_ELEMENTS_TO_CHECK)
            for (elementIdx in 0 until numElementsToCheck) {
                val analyticalGrad = analyticalGrads!!.getFloat(elementIdx * FLOAT_BYTES)
                val originalParam = parameter!!.getFloat(elementIdx * FLOAT_BYTES)
                parameter.putFloat(elementIdx * FLOAT_BYTES, originalParam + DELTA_PARAM)
                val newLoss = model.calculateGradients(bottlenecks, classes, parameters, gradients)
                val numericalGrad = (newLoss - loss) / DELTA_PARAM
                Assert.assertTrue(String.format("Numerical gradient %.5f is different from analytical %.5f "
                        + "(loss = %.5f -> %.5f)",
                        numericalGrad, analyticalGrad, loss, newLoss),
                        Math.abs(numericalGrad - analyticalGrad) < EPS)
                parameter.putFloat(elementIdx * FLOAT_BYTES, originalParam)
            }
        }
    }

    companion object {
        private const val FLOAT_BYTES = 4
        private const val BATCH_SIZE = 20
        private const val BOTTLENECK_SIZE = 7 * 7 * 1280
        private const val NUM_CLASSES = 5

        // Only first N elements of every parameter will be checked to save time.
        private const val MAX_ELEMENTS_TO_CHECK = 30
        private const val DELTA_PARAM = 3e-3f
        private const val EPS = 1e-2f
        private val random = Random(32)
        private fun generateRandomByteBuffer(tensorShape: IntArray?): ByteBuffer {
            val stdDev: Float
            stdDev = if (tensorShape!!.size >= 2) {
                Math.sqrt(2.0 / (tensorShape[0] + tensorShape[1])).toFloat()
            } else {
                0f
            }
            return generateRandomByteBuffer(
                    product(tensorShape), CSupplier(random.nextGaussian().toFloat() * stdDev) )
        }

        private fun generateRandomByteBuffer(numElements: Int, initializer: Supplier<Float>): ByteBuffer {
            val result = ByteBuffer.allocateDirect(numElements * FLOAT_BYTES)
            result.order(ByteOrder.nativeOrder())
            for (idx in 0 until numElements) {
                result.putFloat(initializer.get())
            }
            result.rewind()
            return result
        }

        private fun product(array: IntArray?): Int {
            var result = 1
            for (element in array!!) {
                result *= element
            }
            return result
        }
    }
}