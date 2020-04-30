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

/** Smoke tests for [LiteBottleneckModel].  */
@RunWith(AndroidJUnit4::class)
class LiteBottleneckModelTest {
    @Test
    @Throws(IOException::class)
    fun shouldGenerateSaneBottlenecks() {
        val model = LiteBottleneckModel(
                AssetModelLoader(InstrumentationRegistry.getInstrumentation().context, "model")
                        .loadBaseModel())
        val image = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGE_CHANNELS * FLOAT_BYTES)
        for (idx in 0 until IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGE_CHANNELS) {
            image.putFloat(IMAGE_FILL)
        }
        image.rewind()
        val bottleneck = model.generateBottleneck(image, null)
        var nonZeroCount = 0
        for (idx in 0 until NUM_BOTTLENECK_FEATURES) {
            val feature = bottleneck!!.float
            if (Math.abs(feature) > EPS) {
                nonZeroCount++
            }
        }
        Assert.assertTrue(nonZeroCount > 0)
    }

    companion object {
        private const val FLOAT_BYTES = 4
        private const val NUM_BOTTLENECK_FEATURES = 7 * 7 * 1280
        private const val IMAGE_SIZE = 224
        private const val NUM_IMAGE_CHANNELS = 3
        private const val IMAGE_FILL = 0.3f
        private const val EPS = 1e-8f
    }
}