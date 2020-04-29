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

import android.content.Context
import android.os.ConditionVariable
import org.tensorflow.lite.examples.transfer.AssetModelLoader
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer
import java.io.Closeable
import java.util.*
import java.util.concurrent.ExecutionException
import java.util.concurrent.Future

/**
 * App-layer wrapper for [TransferLearningModel].
 *
 *
 * This wrapper allows to run training continuously, using start/stop API, in contrast to
 * run-once API of [TransferLearningModel].
 */
class TransferLearningModelWrapper internal constructor(context: Context?) : Closeable {
    private val model: TransferLearningModel
    private val shouldTrain = ConditionVariable()

    @Volatile
    private var lossConsumer: LossConsumer? = null

    init {
        model = TransferLearningModel(
                AssetModelLoader(context!!, "model"),
                Arrays.asList("1", "2", "3", "4"))
        Thread(Runnable {
            while (!Thread.interrupted()) {
                shouldTrain.block()
                try {
                    model.train(1, lossConsumer).get()
                } catch (e: ExecutionException) {
                    throw RuntimeException("Exception occurred during model training", e.cause)
                } catch (e: InterruptedException) {
                    // no-op
                }
            }
        }).start()
    }

    // This method is thread-safe.
    fun addSample(image: FloatArray?, className: String?): Future<Void?> {
        return model.addSample(image!!, className!!)
    }

    // This method is thread-safe, but blocking.
    fun predict(image: FloatArray?): Array<TransferLearningModel.Prediction?>? {
        return model.predict(image!!)
    }

    val trainBatchSize: Int
        get() = model.trainBatchSize

    /**
     * Start training the model continuously until [disableTraining][.disableTraining] is
     * called.
     *
     * @param lossConsumer callback that the loss values will be passed to.
     */
    fun enableTraining(lossConsumer: LossConsumer?) {
        this.lossConsumer = lossConsumer
        shouldTrain.open()
    }

    /**
     * Stops training the model.
     */
    fun disableTraining() {
        shouldTrain.close()
    }

    /** Frees all model resources and shuts down all background threads.  */
    override fun close() {
        model.close()
    }

    companion object {
        const val IMAGE_SIZE = 224
    }

}