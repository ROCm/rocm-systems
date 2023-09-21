// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup INTERNAL_THREADING Internal Thread Handling
 * @brief Callbacks before and after threads created internally by libraries
 *
 * @{
 */

/**
 * @brief Enumeration for specifying which libraries you want callbacks before and after the library
 * creates an internal thread. These callbacks will be invoked on the thread that is about to create
 * the new thread (not on the newly created thread). In thread-aware tools that wrap pthread_create,
 * this can be used to disable the wrapper before the pthread_create invocation and re-enable the
 * wrapper afterwards. In many cases, tools will want to ignore the thread(s) created by rocprofiler
 * since these threads do not exist in the normal application execution, whereas the internal
 * threads for HSA, HIP, etc. are created in normal application execution; however, the HIP, HSA,
 * etc. internal threads are typically background threads which just monitor kernel completion and
 * are unlikely to contribute to any performance issues.
 */
typedef enum
{
    ROCPROFILER_LIBRARY        = (1 << 0),
    ROCPROFILER_HSA_LIBRARY    = (1 << 1),
    ROCPROFILER_HIP_LIBRARY    = (1 << 2),
    ROCPROFILER_MARKER_LIBRARY = (1 << 3),
    ROCPROFILER_LIBRARY_LAST   = ROCPROFILER_MARKER_LIBRARY,
} rocprofiler_internal_thread_library_t;

/**
 * @brief Callback type before and after internal thread creation. @see
 * rocprofiler_at_internal_thread_create
 *
 */
typedef void (*rocprofiler_internal_thread_library_cb_t)(rocprofiler_internal_thread_library_t,
                                                         void*);

/**
 * @brief Invoke this function to receive callbacks before and after the creation of an internal
 * thread by a library which as invoked on the thread which is creating the internal thread(s).
 * Please note that the postcreate callback is guaranteed to be invoked after the underlying
 * system call to create a new thread but it does not guarantee that the new thread has been
 * started. Please note, that once these callbacks are registered, they cannot be removed so the
 * caller is responsible for ignoring these callbacks if they want to ignore them beyond a certain
 * point in the application.
 *
 * @param precreate [in] Callback invoked immediately before a new internal thread is created
 * @param postcreate [in] Callback invoked immediately after a new internal thread is created
 * @param libs [in] Bitwise-or of libraries, e.g. `ROCPROFILER_LIBRARY | ROCPROFILER_MARKER_LIBRARY`
 * means the callbacks will be invoked whenever rocprofiler and/or the marker library create
 * internal threads but not when the HSA or HIP libraries create internal threads.
 * @param data [in] Data shared between callbacks
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_at_internal_thread_create(rocprofiler_internal_thread_library_cb_t precreate,
                                      rocprofiler_internal_thread_library_cb_t postcreate,
                                      int                                      libs,
                                      void*                                    data);

/**
 * @brief opaque handle to an internal thread identifier which delivers callbacks for buffers
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_callback_thread_t;

/**
 * @brief Create a handle to a unique thread (created by rocprofiler) which, when associated with a
 * particular buffer, will guarantee those buffered results always get delivered on the same thread.
 * This is useful to prevent/control thread-safety issues and/or enable multithreaded processing of
 * buffers with non-overlapping data
 *
 * @param [in] cb_thread_id User-provided pointer to a @ref rocprofiler_callback_thread_t
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_create_callback_thread(rocprofiler_callback_thread_t* cb_thread_id)
    ROCPROFILER_NONNULL(1);

/**
 * @brief By default, all buffered results are delivered on the same thread. Using @ref
 * rocprofiler_create_callback_thread, one or more buffers can be assigned to deliever their results
 * on a unique, dedicated thread.
 *
 * @param [in] buffer_id Buffer identifier
 * @param [in] cb_thread_id Callback thread identifier via @ref rocprofiler_create_callback_thread
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t       buffer_id,
                                   rocprofiler_callback_thread_t cb_thread_id);

/** @} */

ROCPROFILER_EXTERN_C_FINI
