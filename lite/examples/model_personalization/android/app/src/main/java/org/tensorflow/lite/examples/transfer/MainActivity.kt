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

import android.os.Bundle
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import org.tensorflow.lite.examples.transfer.PermissionsFragment.PermissionsAcquiredListener

/**
 * Main activity of the classifier demo app.
 */
class MainActivity : FragmentActivity() {
    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // If we're being restored from a previous state,
        // then we don't need to do anything and should return or else
        // we could end up with overlapping fragments.
        if (savedInstanceState != null) {
            return
        }
        val firstFragment = PermissionsFragment()
        supportFragmentManager
                .beginTransaction()
                .add(R.id.fragment_container, firstFragment)
                .commit()
    }

    override fun onAttachFragment(fragment: Fragment) {
        if (fragment is PermissionsFragment) {
            class CPermissionsAcquiredListener: PermissionsAcquiredListener{
                init{
                    val cameraFragment = CameraFragment()
                    supportFragmentManager
                            .beginTransaction()
                            .replace(R.id.fragment_container, cameraFragment)
                            .commit()
                }
            }
            val cPermissionsAcquiredListener = CPermissionsAcquiredListener()
            fragment.setOnPermissionsAcquiredListener(cPermissionsAcquiredListener)
        }
    }
}