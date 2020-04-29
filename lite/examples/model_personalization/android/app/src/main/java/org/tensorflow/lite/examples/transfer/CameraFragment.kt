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

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.DisplayMetrics
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.*
import android.widget.TextView
import androidx.camera.core.*
import androidx.camera.core.CameraX.LensFacing
import androidx.camera.core.ImageAnalysis.ImageReaderMode
import androidx.camera.core.Preview.OnPreviewOutputUpdateListener
import androidx.camera.core.Preview.PreviewOutput
import androidx.databinding.BindingAdapter
import androidx.databinding.DataBindingUtil
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProviders
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.snackbar.Snackbar
import org.tensorflow.lite.examples.transfer.CameraFragmentViewModel
import org.tensorflow.lite.examples.transfer.CameraFragmentViewModel.TrainingState
import org.tensorflow.lite.examples.transfer.ImageUtils.convertYUV420ToARGB8888
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer
import org.tensorflow.lite.examples.transfer.databinding.CameraFragmentBinding
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.ExecutionException

/**
 * The main fragment of the classifier.
 *
 * Camera functionality (through CameraX) is heavily based on the official example:
 * https://github.com/android/camera/tree/master/CameraXBasic.
 */
class CameraFragment : Fragment() {
    private var viewFinder: TextureView? = null
    private var viewFinderRotation: Int? = null
    private var bufferDimens = Size(0, 0)
    private var viewFinderDimens = Size(0, 0)
    private var viewModel: CameraFragmentViewModel? = null
    private var tlModel: TransferLearningModelWrapper? = null

    // When the user presses the "add sample" button for some class,
    // that class will be added to this queue. It is later extracted by
    // InferenceThread and processed.
    private val addSampleRequests = ConcurrentLinkedQueue<String>()
    private val inferenceBenchmark = LoggingBenchmark("InferenceBench")

    /**
     * Set up a responsive preview for the view finder.
     */
    private fun startCamera() {
        viewFinderRotation = getDisplaySurfaceRotation(viewFinder!!.display)
        if (viewFinderRotation == null) {
            viewFinderRotation = 0
        }
        val metrics = DisplayMetrics()
        viewFinder!!.display.getRealMetrics(metrics)
        val screenAspectRatio = Rational(metrics.widthPixels, metrics.heightPixels)
        val config = PreviewConfig.Builder()
                .setLensFacing(LENS_FACING)
                .setTargetAspectRatio(screenAspectRatio)
                .setTargetRotation(viewFinder!!.display.rotation)
                .build()
        val preview = Preview(config)
        preview.onPreviewOutputUpdateListener = OnPreviewOutputUpdateListener { previewOutput: PreviewOutput ->
            val parent = viewFinder!!.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)
            viewFinder!!.surfaceTexture = previewOutput.surfaceTexture
            val rotation = getDisplaySurfaceRotation(viewFinder!!.display)
            updateTransform(rotation, previewOutput.textureSize, viewFinderDimens)
        }
        viewFinder!!.addOnLayoutChangeListener { view: View?, left: Int, top: Int, right: Int, bottom: Int, oldLeft: Int, oldTop: Int, oldRight: Int, oldBottom: Int ->
            val newViewFinderDimens = Size(right - left, bottom - top)
            val rotation = getDisplaySurfaceRotation(viewFinder!!.display)
            updateTransform(rotation, bufferDimens, newViewFinderDimens)
        }
        val inferenceThread = HandlerThread("InferenceThread")
        inferenceThread.start()
        val analysisConfig = ImageAnalysisConfig.Builder()
                .setLensFacing(LENS_FACING)
                .setCallbackHandler(Handler(inferenceThread.looper))
                .setImageReaderMode(ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setTargetRotation(viewFinder!!.display.rotation)
                .build()
        val imageAnalysis = ImageAnalysis(analysisConfig)
        imageAnalysis.analyzer = inferenceAnalyzer
        CameraX.bindToLifecycle(this, preview, imageAnalysis)
    }

    private val inferenceAnalyzer = ImageAnalysis.Analyzer { imageProxy: ImageProxy, rotationDegrees: Int ->
        val imageId = UUID.randomUUID().toString()
        inferenceBenchmark.startStage(imageId, "preprocess")
        val rgbImage = prepareCameraImage(yuvCameraImageToBitmap(imageProxy), rotationDegrees)
        inferenceBenchmark.endStage(imageId, "preprocess")

        // Adding samples is also handled by inference thread / use case.
        // We don't use CameraX ImageCapture since it has very high latency (~650ms on Pixel 2 XL)
        // even when using .MIN_LATENCY.
        val sampleClass = addSampleRequests.poll()
        if (sampleClass != null) {
            inferenceBenchmark.startStage(imageId, "addSample")
            try {
                tlModel!!.addSample(rgbImage, sampleClass).get()
            } catch (e: ExecutionException) {
                throw RuntimeException("Failed to add sample to model", e.cause)
            } catch (e: InterruptedException) {
                // no-op
            }
            viewModel!!.increaseNumSamples(sampleClass)
            inferenceBenchmark.endStage(imageId, "addSample")
        } else {
            // We don't perform inference when adding samples, since we should be in capture mode
            // at the time, so the inference results are not actually displayed.
            inferenceBenchmark.startStage(imageId, "predict")
            val predictions = tlModel!!.predict(rgbImage)!!
            inferenceBenchmark.endStage(imageId, "predict")
            for (prediction in predictions) {
                viewModel!!.setConfidence(prediction!!.className, prediction.confidence)
            }
        }
        inferenceBenchmark.finish(imageId)
    }
    val onAddSampleClickListener = View.OnClickListener { view: View ->
        val className: String
        className = if (view.id == R.id.class_btn_1) {
            "1"
        } else if (view.id == R.id.class_btn_2) {
            "2"
        } else if (view.id == R.id.class_btn_3) {
            "3"
        } else if (view.id == R.id.class_btn_4) {
            "4"
        } else {
            throw RuntimeException("Listener called for unexpected view")
        }
        addSampleRequests.add(className)
    }

    /**
     * Fit the camera preview into [viewFinder].
     *
     * @param rotation view finder rotation.
     * @param newBufferDimens camera preview dimensions.
     * @param newViewFinderDimens view finder dimensions.
     */
    private fun updateTransform(rotation: Int?, newBufferDimens: Size, newViewFinderDimens: Size) {
        if (rotation == viewFinderRotation
                && newBufferDimens == bufferDimens
                && newViewFinderDimens == viewFinderDimens) {
            return
        }
        viewFinderRotation = (rotation ?: return)
        bufferDimens = if (newBufferDimens.width == 0 || newBufferDimens.height == 0) {
            return
        } else {
            newBufferDimens
        }
        viewFinderDimens = if (newViewFinderDimens.width == 0 || newViewFinderDimens.height == 0) {
            return
        } else {
            newViewFinderDimens
        }
        Log.d(TAG, String.format("""
    Applying output transformation.
    View finder size: %s.
    Preview output size: %s
    View finder rotation: %s

    """.trimIndent(), viewFinderDimens, bufferDimens, viewFinderRotation))
        val matrix = Matrix()
        val centerX = viewFinderDimens.width / 2f
        val centerY = viewFinderDimens.height / 2f
        matrix.postRotate(-viewFinderRotation!!.toFloat(), centerX, centerY)
        val bufferRatio = bufferDimens.height / bufferDimens.width.toFloat()
        val scaledWidth: Int
        val scaledHeight: Int
        if (viewFinderDimens.width > viewFinderDimens.height) {
            scaledHeight = viewFinderDimens.width
            scaledWidth = Math.round(viewFinderDimens.width * bufferRatio)
        } else {
            scaledHeight = viewFinderDimens.height
            scaledWidth = Math.round(viewFinderDimens.height * bufferRatio)
        }
        val xScale = scaledWidth / viewFinderDimens.width.toFloat()
        val yScale = scaledHeight / viewFinderDimens.height.toFloat()
        matrix.preScale(xScale, yScale, centerX, centerY)
        viewFinder!!.setTransform(matrix)
    }

    override fun onCreate(bundle: Bundle?) {
        super.onCreate(bundle)
        tlModel = TransferLearningModelWrapper(activity)
        viewModel = ViewModelProviders.of(this).get(CameraFragmentViewModel::class.java)
        viewModel!!.setTrainBatchSize(tlModel!!.trainBatchSize)
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, bundle: Bundle?): View? {
        val dataBinding: CameraFragmentBinding = DataBindingUtil.inflate(inflater, R.layout.camera_fragment, container, false)
        dataBinding.lifecycleOwner = viewLifecycleOwner
        dataBinding.vm = viewModel
        val rootView = dataBinding.root
        for (buttonId in intArrayOf(
                R.id.class_btn_1, R.id.class_btn_2, R.id.class_btn_3, R.id.class_btn_4)) {
            rootView.findViewById<View>(buttonId).setOnClickListener(onAddSampleClickListener)
        }
        val chipGroup = rootView.findViewById<View>(R.id.mode_chip_group) as ChipGroup
        if (viewModel!!.captureMode.value!!) {
            (rootView.findViewById<View>(R.id.capture_mode_chip) as Chip).isChecked = true
        } else {
            (rootView.findViewById<View>(R.id.inference_mode_chip) as Chip).isChecked = true
        }
        chipGroup.setOnCheckedChangeListener { group: ChipGroup?, checkedId: Int ->
            if (checkedId == R.id.capture_mode_chip) {
                viewModel!!.setCaptureMode(true)
            } else if (checkedId == R.id.inference_mode_chip) {
                viewModel!!.setCaptureMode(false)
            }
        }
        return dataBinding.root
    }

    override fun onViewCreated(view: View, bundle: Bundle?) {
        super.onViewCreated(view, bundle)
        viewFinder = activity!!.findViewById(R.id.view_finder)
        viewFinder!!.post(Runnable { startCamera() })
    }


    override fun onActivityCreated(bundle: Bundle?) {
        super.onActivityCreated(bundle)
        class CLossConsumer(val cViewModel: CameraFragmentViewModel): LossConsumer{
            override fun onLoss(epoch: Int, loss: Float) {
                cViewModel.setLastLoss(loss)
            }
        }
        val cLossConsumer = CLossConsumer(viewModel!!)
        viewModel!!
                .getTrainingState()
                .observe(
                        viewLifecycleOwner,
                        Observer { trainingState: TrainingState? ->
                            when (trainingState) {
                                TrainingState.STARTED -> {
                                    tlModel!!.enableTraining(cLossConsumer as LossConsumer)

//                                    tlModel!!.enableTraining(LossConsumer { epoch: Int, loss: Float -> viewModel!!.setLastLoss(loss) })
                                    if (!viewModel!!.inferenceSnackbarWasDisplayed.value!!) {
                                        Snackbar.make(
                                                activity!!.findViewById(R.id.classes_bar),
                                                R.string.switch_to_inference_hint,
                                                Snackbar.LENGTH_LONG)
                                                .show()
                                        viewModel!!.markInferenceSnackbarWasCalled()
                                    }
                                }
                                TrainingState.PAUSED -> tlModel!!.disableTraining()
                                TrainingState.NOT_STARTED -> {
                                }
                            }
                        })
    }

    override fun onDestroy() {
        super.onDestroy()
        tlModel!!.close()
        tlModel = null
    }

    companion object {
        private const val LOWER_BYTE_MASK = 0xFF
        private val TAG = CameraFragment::class.java.simpleName
        private val LENS_FACING = LensFacing.BACK
        private fun getDisplaySurfaceRotation(display: Display?): Int? {
            return if (display == null) {
                null
            } else when (display.rotation) {
                Surface.ROTATION_0 -> 0
                Surface.ROTATION_90 -> 90
                Surface.ROTATION_180 -> 180
                Surface.ROTATION_270 -> 270
                else -> null
            }
        }

        private fun yuvCameraImageToBitmap(imageProxy: ImageProxy): Bitmap {
            require(imageProxy.format == ImageFormat.YUV_420_888) { "Expected a YUV420 image, but got " + imageProxy.format }
            val yPlane = imageProxy.planes[0]
            val uPlane = imageProxy.planes[1]
            val width = imageProxy.width
            val height = imageProxy.height
            val yuvBytes = arrayOfNulls<ByteArray>(3)
            val argbArray = IntArray(width * height)
            for (i in imageProxy.planes.indices) {
                val buffer = imageProxy.planes[i].buffer
                yuvBytes[i] = ByteArray(buffer.capacity())
                buffer[yuvBytes[i]]
            }
            convertYUV420ToARGB8888(
                    yuvBytes[0]!!,
                    yuvBytes[1]!!,
                    yuvBytes[2]!!,
                    width,
                    height,
                    yPlane.rowStride,
                    uPlane.rowStride,
                    uPlane.pixelStride,
                    argbArray)
            return Bitmap.createBitmap(argbArray, width, height, Bitmap.Config.ARGB_8888)
        }

        /**
         * Normalizes a camera image to [0; 1], cropping it
         * to size expected by the model and adjusting for camera rotation.
         */
        private fun prepareCameraImage(bitmap: Bitmap, rotationDegrees: Int): FloatArray {
            val modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE
            val paddedBitmap = padToSquare(bitmap)
            val scaledBitmap = Bitmap.createScaledBitmap(
                    paddedBitmap, modelImageSize, modelImageSize, true)
            val rotationMatrix = Matrix()
            rotationMatrix.postRotate(rotationDegrees.toFloat())
            val rotatedBitmap = Bitmap.createBitmap(
                    scaledBitmap, 0, 0, modelImageSize, modelImageSize, rotationMatrix, false)
            val normalizedRgb = FloatArray(modelImageSize * modelImageSize * 3)
            var nextIdx = 0
            for (y in 0 until modelImageSize) {
                for (x in 0 until modelImageSize) {
                    val rgb = rotatedBitmap.getPixel(x, y)
                    val r = (rgb shr 16 and LOWER_BYTE_MASK) * (1 / 255f)
                    val g = (rgb shr 8 and LOWER_BYTE_MASK) * (1 / 255f)
                    val b = (rgb and LOWER_BYTE_MASK) * (1 / 255f)
                    normalizedRgb[nextIdx++] = r
                    normalizedRgb[nextIdx++] = g
                    normalizedRgb[nextIdx++] = b
                }
            }
            return normalizedRgb
        }

        private fun padToSquare(source: Bitmap): Bitmap {
            val width = source.width
            val height = source.height
            val paddingX = if (width < height) (height - width) / 2 else 0
            val paddingY = if (height < width) (width - height) / 2 else 0
            val paddedBitmap = Bitmap.createBitmap(
                    width + 2 * paddingX, height + 2 * paddingY, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(paddedBitmap)
            canvas.drawARGB(0xFF, 0xFF, 0xFF, 0xFF)
            canvas.drawBitmap(source, paddingX.toFloat(), paddingY.toFloat(), null)
            return paddedBitmap
        }

        // Binding adapters:
        @JvmStatic
        @BindingAdapter("captureMode", "inferenceText", "captureText")
        fun setClassSubtitleText(
                view: TextView, captureMode: Boolean, inferenceText: Float?, captureText: Int?) {
            if (captureMode) {
                view.text = if (captureText != null) Integer.toString(captureText) else "0"
            } else {
                view.text = String.format(Locale.getDefault(), "%.2f", inferenceText ?: 0f)
            }
        }

        @JvmStatic
        @BindingAdapter("android:visibility")
        fun setViewVisibility(view: View, visible: Boolean) {
            view.visibility = if (visible) View.VISIBLE else View.GONE
        }

        @JvmStatic
        @BindingAdapter("highlight")
        fun setClassButtonHighlight(view: View, highlight: Boolean) {
            val drawableId: Int
            drawableId = if (highlight) {
                R.drawable.btn_default_highlight
            } else {
                R.drawable.btn_default
            }
            view.background = view.context.getDrawable(drawableId)
        }
    }
}