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
package org.tensorflow.lite.examples.transfer

import android.util.Log
import java.util.*

/**
 * A simple class for measuring execution time in various contexts.
 */
internal class LoggingBenchmark(private val tag: String) {
    private val totalImageTime: MutableMap<String, Long> = HashMap()
    private val stageTime: MutableMap<String, MutableMap<String, Long>> = HashMap()
    private val stageStartTime: MutableMap<String, MutableMap<String, Long>> = HashMap()
    fun startStage(imageId: String, stageName: String) {
        if (!ENABLED) {
            return
        }
        val stageStartTimeForImage: MutableMap<String, Long>
        if (!stageStartTime.containsKey(imageId)) {
            stageStartTimeForImage = HashMap()
            stageStartTime[imageId] = stageStartTimeForImage
        } else {
            stageStartTimeForImage = stageStartTime[imageId]!!
        }
        val timeNs = System.nanoTime()
        stageStartTimeForImage[stageName] = timeNs
    }

    fun endStage(imageId: String, stageName: String) {
        if (!ENABLED) {
            return
        }
        val endTime = System.nanoTime()
        val startTime = stageStartTime[imageId]!![stageName]!!
        val duration = endTime - startTime
        if (!stageTime.containsKey(imageId)) {
            stageTime[imageId] = HashMap()
        }
        stageTime[imageId]!![stageName] = duration
        if (!totalImageTime.containsKey(imageId)) {
            totalImageTime[imageId] = 0L
        }
        totalImageTime[imageId] = totalImageTime[imageId]!! + duration
    }

    fun finish(imageId: String) {
        if (!ENABLED) {
            return
        }
        val msg = StringBuilder()
        for ((key, value) in stageTime[imageId]!!) {
            msg.append(String.format(Locale.getDefault(),
                    "%s: %.2fms | ", key, value / 1.0e6))
        }
        msg.append(String.format(Locale.getDefault(),
                "TOTAL: %.2fms", totalImageTime[imageId]!! / 1.0e6))
        Log.d(tag, msg.toString())
    }

    companion object {
        private const val ENABLED = false
    }

}