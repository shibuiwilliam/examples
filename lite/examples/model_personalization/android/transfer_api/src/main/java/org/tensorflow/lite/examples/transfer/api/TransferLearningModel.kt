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
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.GatheringByteChannel
import java.nio.channels.ScatteringByteChannel
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReadWriteLock
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.locks.ReentrantReadWriteLock

/**
 * Represents a "partially" trainable model that is based on some other,
 * base model.
 */
class TransferLearningModel(modelLoader: ModelLoader, classes: Collection<String>) : Closeable {
    /**
     * Prediction for a single class produced by the model.
     */
    class Prediction(val className: String, val confidence: Float)

    private class TrainingSample internal constructor(var bottleneck: ByteBuffer?, var className: String)

    /**
     * Consumer interface for training loss.
     */
    interface LossConsumer {
        fun onLoss(epoch: Int, loss: Float)
    }

    private val bottleneckShape: IntArray?
    private val classes: MutableMap<String, Int>
    private val classesByIdx: Array<String>
    private var initializeModel: LiteInitializeModel? = null
    private var bottleneckModel: LiteBottleneckModel? = null
    private var trainHeadModel: LiteTrainHeadModel? = null
    private var inferenceModel: LiteInferenceModel? = null
    private var optimizerModel: LiteOptimizerModel? = null
    private val trainingSamples: MutableList<TrainingSample?> = ArrayList()
    private var modelParameters: Array<ByteBuffer?>

    // Where to store the optimizer outputs.
    private var nextModelParameters: Array<ByteBuffer?>
    private var optimizerState: Array<ByteBuffer?>

    // Where to store the updated optimizer state.
    private var nextOptimizerState: Array<ByteBuffer?>

    // Where to store training inputs.
    private val trainingBatchBottlenecks: ByteBuffer
    private val trainingBatchClasses: ByteBuffer

    // A zero-filled buffer of the same size as `trainingBatchClasses`.
    private val zeroBatchClasses: ByteBuffer

    // Where to store calculated gradients.
    private val modelGradients: Array<ByteBuffer?>

    // Where to store bottlenecks produced during inference.
    private val inferenceBottleneck: ByteBuffer

    // Used to spawn background threads.
    private val executor = Executors.newFixedThreadPool(NUM_THREADS)

    // This lock guarantees that only one thread is performing training at any point in time.
    // It also protects the sample collection from being modified while in use by a training
    // thread.
    private val trainingLock: Lock = ReentrantLock()

    // This lock guards access to trainable parameters.
    private val parameterLock: ReadWriteLock = ReentrantReadWriteLock()

    // This lock allows [close] method to assure that no threads are performing inference.
    private val inferenceLock: Lock = ReentrantLock()

    // Set to true when [close] has been called.
    @Volatile
    private var isTerminating = false


    init {
        classesByIdx = classes.toTypedArray()
        this.classes = TreeMap()
        for (classIdx in classes.indices) {
            this.classes[classesByIdx[classIdx]] = classIdx
        }
        try {
            initializeModel = LiteInitializeModel(modelLoader.loadInitializeModel())
            bottleneckModel = LiteBottleneckModel(modelLoader.loadBaseModel())
            trainHeadModel = LiteTrainHeadModel(modelLoader.loadTrainModel())
            inferenceModel = LiteInferenceModel(modelLoader.loadInferenceModel(), classes.size)
            optimizerModel = LiteOptimizerModel(modelLoader.loadOptimizerModel())
        } catch (e: IOException) {
            throw RuntimeException("Couldn't read underlying models for TransferLearningModel", e)
        }
        bottleneckShape = bottleneckModel!!.bottleneckShape
        val modelParameterSizes = trainHeadModel!!.parameterSizes
        modelParameters = arrayOfNulls(modelParameterSizes.size)
        modelGradients = arrayOfNulls(modelParameterSizes.size)
        nextModelParameters = arrayOfNulls(modelParameterSizes.size)
        for (parameterIndex in modelParameterSizes.indices) {
            val bufferSize = modelParameterSizes[parameterIndex] * Constants.FLOAT_BYTES
            modelParameters[parameterIndex] = allocateBuffer(bufferSize)
            modelGradients[parameterIndex] = allocateBuffer(bufferSize)
            nextModelParameters[parameterIndex] = allocateBuffer(bufferSize)
        }
        initializeModel!!.initializeParameters(modelParameters)
        val optimizerStateElementSizes = optimizerModel!!.stateElementSizes()
        optimizerState = arrayOfNulls(optimizerStateElementSizes.size)
        nextOptimizerState = arrayOfNulls(optimizerStateElementSizes.size)
        for (elemIdx in optimizerState.indices) {
            val bufferSize = optimizerStateElementSizes[elemIdx] * Constants.FLOAT_BYTES
            optimizerState[elemIdx] = allocateBuffer(bufferSize)
            nextOptimizerState[elemIdx] = allocateBuffer(bufferSize)
            fillBufferWithZeros(optimizerState[elemIdx])
        }
        trainingBatchBottlenecks = allocateBuffer(trainBatchSize * numBottleneckFeatures() * Constants.FLOAT_BYTES)
        val batchClassesNumElements = trainBatchSize * classes.size
        trainingBatchClasses = allocateBuffer(batchClassesNumElements * Constants.FLOAT_BYTES)
        zeroBatchClasses = allocateBuffer(batchClassesNumElements * Constants.FLOAT_BYTES)
        for (idx in 0 until batchClassesNumElements) {
            zeroBatchClasses.putFloat(0f)
        }
        zeroBatchClasses.rewind()
        inferenceBottleneck = allocateBuffer(numBottleneckFeatures() * Constants.FLOAT_BYTES)
    }

    /**
     * Adds a new sample for training.
     *
     * Sample bottleneck is generated in a background thread, which resolves the returned Future
     * when the bottleneck is added to training samples.
     *
     * @param image image RGB data.
     * @param className ground truth label for image.
     */
    fun addSample(image: FloatArray, className: String): Future<Void?> {
        checkNotTerminating()
        require(classes.containsKey(className)) {
            String.format(
                    "Class \"%s\" is not one of the classes recognized by the model", className)
        }
        return executor.submit<Void?> {
            val imageBuffer = allocateBuffer(image.size * Constants.FLOAT_BYTES)
            for (f in image) {
                imageBuffer.putFloat(f)
            }
            imageBuffer.rewind()
            if (Thread.interrupted()) {
                return@submit null
            }
            val bottleneck = bottleneckModel!!.generateBottleneck(imageBuffer, null)
            trainingLock.lockInterruptibly()
            try {
                trainingSamples.add(TrainingSample(bottleneck, className))
            } finally {
                trainingLock.unlock()
            }
            null
        }
    }

    /**
     * Trains the model on the previously added data samples.
     *
     * @param numEpochs number of epochs to train for.
     * @param lossConsumer callback to receive loss values, may be null.
     * @return future that is resolved when training is finished.
     */
    fun train(numEpochs: Int, lossConsumer: LossConsumer?): Future<Void> {
        checkNotTerminating()
        if (trainingSamples.size < trainBatchSize) {
            throw RuntimeException(String.format(
                    "Too few samples to start training: need %d, got %d",
                    trainBatchSize, trainingSamples.size))
        }
        return executor.submit<Void> {
            trainingLock.lock()
            try {
                epochLoop@ for (epoch in 0 until numEpochs) {
                    var totalLoss = 0f
                    var numBatchesProcessed = 0
                    for (batch in trainingBatches()) {
                        if (Thread.interrupted()) {
                            break@epochLoop
                        }
                        trainingBatchClasses.put(zeroBatchClasses)
                        trainingBatchClasses.rewind()
                        zeroBatchClasses.rewind()
                        for (sampleIdx in batch.indices) {
                            val sample = batch[sampleIdx]
                            trainingBatchBottlenecks.put(sample.bottleneck)
                            sample.bottleneck!!.rewind()

                            // Fill trainingBatchClasses with one-hot.
                            val position = (sampleIdx * classes.size + classes[sample.className]!!) * Constants.FLOAT_BYTES
                            trainingBatchClasses.putFloat(position, 1f)
                        }
                        trainingBatchBottlenecks.rewind()
                        val loss = trainHeadModel!!.calculateGradients(
                                trainingBatchBottlenecks,
                                trainingBatchClasses,
                                modelParameters,
                                modelGradients)
                        totalLoss += loss
                        numBatchesProcessed++
                        optimizerModel!!.performStep(
                                modelParameters,
                                modelGradients,
                                optimizerState,
                                nextModelParameters,
                                nextOptimizerState)
                        var swapBufferArray: Array<ByteBuffer?>

                        // Swap optimizer state with its next version.
                        swapBufferArray = optimizerState
                        optimizerState = nextOptimizerState
                        nextOptimizerState = swapBufferArray

                        // Swap model parameters with their next versions.
                        parameterLock.writeLock().lock()
                        try {
                            swapBufferArray = modelParameters
                            modelParameters = nextModelParameters
                            nextModelParameters = swapBufferArray
                        } finally {
                            parameterLock.writeLock().unlock()
                        }
                    }
                    val avgLoss = totalLoss / numBatchesProcessed
                    lossConsumer?.onLoss(epoch, avgLoss)
                }
                return@submit null
            } finally {
                trainingLock.unlock()
            }
        }
    }

    /**
     * Runs model inference on a given image.
     * @param image image RGB data.
     * @return predictions sorted by confidence decreasing. Can be null if model is terminating.
     */
    fun predict(image: FloatArray): Array<Prediction?>? {
        checkNotTerminating()
        inferenceLock.lock()
        return try {
            if (isTerminating) {
                return null
            }
            val imageBuffer = allocateBuffer(image.size * Constants.FLOAT_BYTES)
            for (f in image) {
                imageBuffer.putFloat(f)
            }
            imageBuffer.rewind()
            val bottleneck = bottleneckModel!!.generateBottleneck(imageBuffer, inferenceBottleneck)
            val confidences: FloatArray?
            parameterLock.readLock().lock()
            confidences = try {
                inferenceModel!!.runInference(bottleneck, modelParameters)
            } finally {
                parameterLock.readLock().unlock()
            }
            val predictions = arrayOfNulls<Prediction>(classes.size)
            for (classIdx in 0 until classes.size) {
                predictions[classIdx] = Prediction(classesByIdx[classIdx], confidences!![classIdx])
            }
//            Arrays.sort(predictions) { a: Prediction, b: Prediction -> -java.lang.Float.compare(a.confidence, b.confidence) }
            predictions.sortBy { it -> it!!.confidence }
            predictions
        } finally {
            inferenceLock.unlock()
        }
    }

    /**
     * Writes the current values of the model parameters to a writable channel.
     *
     * The written values can be restored later using [.loadParameters],
     * under condition that the same underlying model is used.
     *
     * @param outputChannel where to write the parameters.
     * @throws IOException if an I/O error occurs.
     */
    @Throws(IOException::class)
    fun saveParameters(outputChannel: GatheringByteChannel) {
        parameterLock.readLock().lock()
        try {
            outputChannel.write(modelParameters)
            for (buffer in modelParameters) {
                buffer!!.rewind()
            }
        } finally {
            parameterLock.readLock().unlock()
        }
    }

    /**
     * Overwrites the current model parameter values with the values read from a channel.
     *
     * The channel should contain values previously written by
     * [.saveParameters] for the same underlying model.
     *
     * @param inputChannel where to read the parameters from.
     * @throws IOException if an I/O error occurs.
     */
    @Throws(IOException::class)
    fun loadParameters(inputChannel: ScatteringByteChannel) {
        parameterLock.writeLock().lock()
        try {
            inputChannel.read(modelParameters)
            for (buffer in modelParameters) {
                buffer!!.rewind()
            }
        } finally {
            parameterLock.writeLock().unlock()
        }
    }

    /** Training model expected batch size.  */
    val trainBatchSize: Int
        get() = trainHeadModel!!.batchSize

    /**
     * Constructs an iterator that iterates over training sample batches.
     * @return iterator over batches.
     */
    private fun trainingBatches(): Iterable<List<TrainingSample>> {
        if (!trainingLock.tryLock()) {
            throw RuntimeException("Thread calling trainingBatches() must hold the training lock")
        }
        trainingLock.unlock()
        Collections.shuffle(trainingSamples)
        return label@ Iterable<List<TrainingSample>> {
            object : Iterator<List<TrainingSample>> {
                private var nextIndex = 0
                override fun hasNext(): Boolean {
                    return@label nextIndex < trainingSamples.size
                }

                override fun next(): List<TrainingSample> {
                    val fromIndex = nextIndex
                    val toIndex = nextIndex + trainBatchSize
                    nextIndex = toIndex
//                    if (toIndex >= trainingSamples.size) {
//                        // To keep batch size consistent, last batch may include some elements from the
//                        // next-to-last batch.
//                        return@label trainingSamples.subList(
//                                trainingSamples.size - trainBatchSize, trainingSamples.size)
//                    } else {
//                        return@label trainingSamples.subList(fromIndex, toIndex)
//                    }
                    return@label if(toIndex >= trainingSamples.size)
                        trainingSamples.subList(trainingSamples.size - trainBatchSize, trainingSamples.size)
                    else trainingSamples.subList(fromIndex, toIndex)
                }
            }
        }
    }

    private fun checkNotTerminating() {
        check(!isTerminating) { "Cannot operate on terminating model" }
    }

    private fun numBottleneckFeatures(): Int {
        var result = 1
        for (size in bottleneckShape!!) {
            result *= size
        }
        return result
    }

    /**
     * Terminates all model operation safely. Will block until current inference request is finished
     * (if any).
     *
     *
     * Calling any other method on this object after [close] is not allowed.
     */
    override fun close() {
        isTerminating = true
        executor.shutdownNow()

        // Make sure that all threads doing inference are finished.
        inferenceLock.lock()
        try {
            val ok = executor.awaitTermination(5, TimeUnit.SECONDS)
            if (!ok) {
                throw RuntimeException("Model thread pool failed to terminate")
            }
            initializeModel!!.close()
            bottleneckModel!!.close()
            trainHeadModel!!.close()
            inferenceModel!!.close()
            optimizerModel!!.close()
        } catch (e: InterruptedException) {
            // no-op
        } finally {
            inferenceLock.unlock()
        }
    }

    companion object {
        // Setting this to a higher value allows to calculate bottlenecks for more samples while
        // adding them to the bottleneck collection is blocked by an active training thread.
        private val NUM_THREADS = Math.max(1, Runtime.getRuntime().availableProcessors() - 1)
        private fun allocateBuffer(capacity: Int): ByteBuffer {
            val buffer = ByteBuffer.allocateDirect(capacity)
            buffer.order(ByteOrder.nativeOrder())
            return buffer
        }

        private fun fillBufferWithZeros(buffer: ByteBuffer?) {
            val bufSize = buffer!!.capacity()
            val chunkSize = Math.min(1024, bufSize)
            val zerosChunk = allocateBuffer(chunkSize)
            for (idx in 0 until chunkSize) {
                zerosChunk.put(0.toByte())
            }
            zerosChunk.rewind()
            for (chunkIdx in 0 until bufSize / chunkSize) {
                buffer.put(zerosChunk)
            }
            for (idx in 0 until bufSize % chunkSize) {
                buffer.put(0.toByte())
            }
        }
    }

}