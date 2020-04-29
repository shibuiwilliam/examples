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

import androidx.lifecycle.*
import androidx.lifecycle.Observer
import java.util.*

/**
 * Holds information about view model of CameraFragment. This information is preserved across
 * configuration changes and automatically restored..
 */
class CameraFragmentViewModel : ViewModel() {
    /**
     * Current state of training.
     */
    enum class TrainingState {
        NOT_STARTED, STARTED, PAUSED
    }


    /**
     * Whether capture mode is enabled.
     */
    @kotlin.jvm.JvmField
    var captureMode = MutableLiveData(false)
    private val confidence = MutableLiveData<MutableMap<String, Float>>(TreeMap())
    /** Number of samples in a single training batch.  */
    val trainBatchSize = MutableLiveData(0)
    private val numSamples = MutableLiveData<MutableMap<String, Int>>(TreeMap())
    private val trainingState = MutableLiveData(TrainingState.NOT_STARTED)
    private val lastLoss = MutableLiveData<Float>()

    /**
     * Whether "you can switch to inference mode now" snackbar has been shown before.
     */
    @kotlin.jvm.JvmField
    val inferenceSnackbarWasDisplayed = MutableLiveData(false)

    /**
     * Name of the class with the highest confidence score.
     */
    var firstChoice: LiveData<String>? = null
        get() {
            if (field == null) {
                field = Transformations.map(confidence) { map: Map<String, Float> ->
                    if (map.isEmpty()) {
                        return@map null
                    }
                    mapEntriesDecreasingValue(map)[0].key
                }
            }
            return field
        }
        private set

    /**
     * Name of the class with the second highest confidence score.
     */
    var secondChoice: LiveData<String>? = null
        get() {
            if (field == null) {
                field = Transformations.map(confidence) { map: Map<String, Float> ->
                    if (map.size < 2) {
                        return@map null
                    }
                    mapEntriesDecreasingValue(map)[1].key
                }
            }
            return field
        }
        private set

    /**
     * A single integer representing the total number of samples added for all classes.
     */
    var totalSamples: LiveData<Int>? = null
        get() {
            if (field == null) {
                field = Transformations.map(getNumSamples()) { map: Map<String, Int> ->
                    var total = 0
                    for (number in map.values) {
                        total += number
                    }
                    total
                }
            }
            return field
        }
        private set

    /**
     * Number of samples needed to complete a single batch.
     */
    var neededSamples: LiveData<Int>? = null
        get() {
            if (field == null) {
                val result = MediatorLiveData<Int>()
                result.addSource(
                        totalSamples as LiveData<Int>,
                        Observer { totalSamples: Int -> result.setValue(Math.max(0, trainBatchSize.value!! - totalSamples)) })
                result.addSource(
                        trainBatchSize
                ) { trainBatchSize: Int -> result.setValue(Math.max(0, trainBatchSize - totalSamples!!.value!!)) }
                field = result
            }
            return field
        }
        private set

    fun setCaptureMode(newValue: Boolean) {
        captureMode.postValue(newValue)
    }

    /**
     * Number of added samples for each class.
     */
    fun getNumSamples(): LiveData<MutableMap<String, Int>> {
        return numSamples
    }

    fun increaseNumSamples(className: String) {
        val map = numSamples.value!!
        val currentNumber: Int
        currentNumber = if (map.containsKey(className)) map[className]!! else 0

        map[className] = currentNumber + 1
        numSamples.postValue(map)
    }

    /**
     * Confidence values for each class from inference.
     */
    fun getConfidence(): LiveData<MutableMap<String, Float>> {
        return confidence
    }

    fun setConfidence(className: String, confidenceScore: Float) {
        val map = confidence.value!!
        map[className] = confidenceScore
        confidence.postValue(map)
    }

    fun setTrainBatchSize(newValue: Int) {
        trainBatchSize.postValue(newValue)
    }

    /** Whether model training is not yet started, already started, or temporarily paused.  */
    fun getTrainingState(): LiveData<TrainingState> {
        return trainingState
    }

    fun setTrainingState(newValue: TrainingState) {
        trainingState.postValue(newValue)
    }

    /**
     * Last training loss value reported by the training routine.
     */
    fun getLastLoss(): LiveData<Float> {
        return lastLoss
    }

    fun setLastLoss(newLoss: Float) {
        lastLoss.postValue(newLoss)
    }

    fun markInferenceSnackbarWasCalled() {
        inferenceSnackbarWasDisplayed.postValue(true)
    }

    companion object {
        private fun mapEntriesDecreasingValue(map: Map<String, Float>): List<Map.Entry<String, Float>> {
            val entryList: List<Map.Entry<String, Float>> = ArrayList(map.entries)
            Collections.sort(entryList) { e1: Map.Entry<String, Float>, e2: Map.Entry<String, Float> -> -java.lang.Float.compare(e1.value, e2.value) }
            return entryList
        }
    }
}