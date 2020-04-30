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

import android.content.Context
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.*
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

internal object ZipUtils {
    private const val BUFFER_SIZE = 65536

    @JvmStatic
    @Throws(IOException::class)
    fun readAllZipFiles(context: Context, pathToZip: String?): Map<String, ByteArray> {
        val result: MutableMap<String, ByteArray> = HashMap()
        val zin = ZipInputStream(BufferedInputStream(context.assets.open(pathToZip!!)))
        val buffer = ByteArray(BUFFER_SIZE)
        var zipEntry: ZipEntry
        while (zin.nextEntry.also { zipEntry = it } != null) {
            val outputStream = ByteArrayOutputStream()
            val filePath = zipEntry.name
            var readBytes: Int
            while (zin.read(buffer).also { readBytes = it } != -1) {
                outputStream.write(buffer, 0, readBytes)
            }
            result[filePath] = outputStream.toByteArray()
        }
        return result
    }
}