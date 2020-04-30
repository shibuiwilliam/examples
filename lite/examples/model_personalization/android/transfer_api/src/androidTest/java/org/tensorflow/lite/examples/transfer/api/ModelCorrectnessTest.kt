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

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer
import org.tensorflow.lite.examples.transfer.api.ZipUtils.readAllZipFiles
import java.io.BufferedReader
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.charset.Charset
import java.util.*
import java.util.concurrent.ExecutionException

/** End-to-end model correctness test.  */
@RunWith(AndroidJUnit4::class)
class ModelCorrectnessTest {
    private class Sample internal constructor(var imagePath: String, var className: String)

    @Test
    @Throws(IOException::class)
    fun shouldLearnToClassifyImages() {
        val zipFiles = readAllZipFiles(
                InstrumentationRegistry.getInstrumentation().context, "test_data.zip")
        val model = TransferLearningModel(
                AssetModelLoader(
                        InstrumentationRegistry.getInstrumentation().context, "model"),
                Arrays.asList("daisy", "dandelion", "roses", "sunflowers", "tulips"))
        println("Going to add the samples.")
        for (sample in readSampleList(zipFiles["train.txt"])) {
            try {
                model.addSample(jpgBytesToRgb(zipFiles[sample.imagePath]), sample.className).get()
            } catch (e: InterruptedException) {
                return
            } catch (e: ExecutionException) {
                throw RuntimeException("Could not add training sample", e.cause)
            }
        }
        println("Finished adding the samples.")
        class CLossConsumer(): LossConsumer{
            override fun onLoss(epoch: Int, loss: Float) {
                System.out.printf("Epoch %d: loss = %.5f\n", epoch, loss)
            }
        }
        val cLossConsumer = CLossConsumer()
        try {
            model
                    .train(
                            NUM_EPOCHS,
                            cLossConsumer)
                    .get()
        } catch (e: ExecutionException) {
            throw RuntimeException(e.cause)
        } catch (e: InterruptedException) {
            // Exit peacefully.
        }
        var correct = 0
        var total = 0
        for (sample in readSampleList(zipFiles["val.txt"])) {
            val predictions = model.predict(jpgBytesToRgb(zipFiles[sample.imagePath]))
            if (predictions!![0]!!.className == sample.className) {
                correct++
            }
            total++
        }
        val accuracy = correct / total.toFloat()
        System.out.printf("Accuracy is %.5f\n", accuracy)
        Assert.assertTrue(String.format("Accuracy is %.5f, expected at least %.5f", accuracy, TARGET_ACCURACY),
                accuracy >= TARGET_ACCURACY)
    }

    private fun readSampleList(sampleListBytes: ByteArray?): Iterable<Sample> {
        val linesReader = BufferedReader(InputStreamReader(
                ByteArrayInputStream(sampleListBytes), Charset.defaultCharset()))
        return label@ Iterable<Sample> {
            object : Iterator<Sample> {
                private var hasBufferedLine = false
                private var nextLine: String? = null
                override fun hasNext(): Boolean {
                    maybeReadToBuffer()
                    return@label nextLine != null
                }

                override fun next(): Sample {
                    maybeReadToBuffer()
                    hasBufferedLine = false
                    val parts = nextLine!!.split(",").dropLastWhile { it.isEmpty() }.toTypedArray()
                    return@label Sample(parts[0], parts[1])
                }

                private fun maybeReadToBuffer() {
                    if (!hasBufferedLine) {
                        try {
                            nextLine = linesReader.readLine()
                            hasBufferedLine = true
                        } catch (e: IOException) {
                            throw RuntimeException(e)
                        }
                    }
                }
            }
        }
    }

    companion object {
        private const val LOWER_BYTE_MASK = 0xFF
        private const val NUM_EPOCHS = 20
        private const val TARGET_ACCURACY = 0.70f

        @Throws(IOException::class)
        private fun jpgBytesToRgb(jpgBytes: ByteArray?): FloatArray {
            val inputStream = ByteArrayInputStream(jpgBytes)
            val image = BitmapFactory.decodeStream(inputStream)
            val result = FloatArray(image.width * image.height * 3)
            var nextIdx = 0
            for (y in 0 until image.height) {
                for (x in 0 until image.width) {
                    val rgb = image.getPixel(x, y)
                    val r = (rgb shr 16 and LOWER_BYTE_MASK) * (1 / 255f)
                    val g = (rgb shr 8 and LOWER_BYTE_MASK) * (1 / 255f)
                    val b = (rgb and LOWER_BYTE_MASK) * (1 / 255f)
                    result[nextIdx++] = r
                    result[nextIdx++] = g
                    result[nextIdx++] = b
                }
            }
            return result
        }
    }
}