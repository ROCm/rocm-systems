// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef ROCPROFSYS_USER_H_
#define ROCPROFSYS_USER_H_

#if defined(ROCPROFSYS_USER_SOURCE) && (ROCPROFSYS_USER_SOURCE > 0)
#    if !defined(ROCPROFSYS_PUBLIC_API)
#        define ROCPROFSYS_PUBLIC_API __attribute__((visibility("default")))
#    endif
#else
#    if !defined(ROCPROFSYS_PUBLIC_API)
#        define ROCPROFSYS_PUBLIC_API
#    endif
#endif

#include "rocprofiler-systems/categories.h"
#include "rocprofiler-systems/types.h"

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @fn int rocprofsys_user_start_trace(void)
    /// @return rocprofsys_user_error_t value
    /// @brief Enable tracing on this thread and all subsequently created threads
    extern int rocprofsys_user_start_trace(void) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_stop_trace(void)
    /// @return rocprofsys_user_error_t value
    /// @brief Disable tracing on this thread and all subsequently created threads
    extern int rocprofsys_user_stop_trace(void) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_start_thread_trace(void)
    /// @return rocprofsys_user_error_t value
    /// @brief Enable tracing on this specific thread. Does not apply to subsequently
    /// created threads
    extern int rocprofsys_user_start_thread_trace(void) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_stop_thread_trace(void)
    /// @return rocprofsys_user_error_t value
    /// @brief Disable tracing on this specific thread. Does not apply to subsequently
    /// created threads
    extern int rocprofsys_user_stop_thread_trace(void) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_push_region(const char* id)
    /// @param id The string identifier for the region
    /// @return rocprofsys_user_error_t value
    /// @brief Start a user defined region.
    extern int rocprofsys_user_push_region(const char*) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_pop_region(const char* id)
    /// @param id The string identifier for the region
    /// @return rocprofsys_user_error_t value
    /// @brief End a user defined region. In general, user regions should be popped in
    /// the inverse order that they were pushed, i.e. first-in, last-out (FILO). The
    /// timemory backend was designed to accommodate asynchronous tasking, where FILO may
    /// be violated, and will thus compenstate for out-of-order popping, however, the
    /// perfetto backend will not; thus, out-of-order popping will result in different
    /// results in timemory vs. perfetto.
    extern int rocprofsys_user_pop_region(const char*) ROCPROFSYS_PUBLIC_API;

    /// @typedef rocprofsys_annotation rocprofsys_annotation_t
    ///
    /// @fn int rocprofsys_user_push_annotated_region(const char* id,
    ///                                              rocprofsys_annotation_t* annotations,
    ///                                              size_t num_annotations)
    /// @param id The string identifier for the region
    /// @param annotations Array of @ref rocprofsys_annotation instances
    /// @param num_annotations Number of annotations
    /// @return rocprofsys_user_error_t value
    /// @brief Start a user defined region and adds the annotations to the perfetto trace.
    extern int rocprofsys_user_push_annotated_region(const char*,
                                                     rocprofsys_annotation_t*,
                                                     size_t) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_pop_annotated_region(const char* id,
    ///                                             rocprofsys_annotation_t* annotations,
    ///                                             size_t num_annotations)
    /// @param id The string identifier for the region
    /// @param annotations Array of @ref rocprofsys_annotation instances
    /// @param num_annotations Number of annotations
    /// @return rocprofsys_user_error_t value
    /// @brief Stop a user defined region and adds the annotations to the perfetto trace.
    extern int rocprofsys_user_pop_annotated_region(const char*, rocprofsys_annotation_t*,
                                                    size_t) ROCPROFSYS_PUBLIC_API;

    /// mark causal progress
    extern int rocprofsys_user_progress(const char*) ROCPROFSYS_PUBLIC_API;

    /// mark causal progress with annotations
    extern int rocprofsys_user_annotated_progress(const char*, rocprofsys_annotation_t*,
                                                  size_t) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_configure(rocprofsys_user_configure_mode_t mode,
    ///                                  rocprofsys_user_callbacks_t inp,
    ///                                  rocprofsys_user_callbacks_t* out)
    /// @param[in] mode Specifies how the new callbacks are merged with the old
    /// callbacks
    /// @param[in] inp An @ref rocprofsys_user_callbacks instance specifying
    ///            the callbacks which should be invoked by the user API.
    /// @param[out] out Pointer to @ref rocprofsys_user_callbacks which,
    ///             when non-NULL, will be assigned the former callbacks.
    /// @return rocprofsys_user_error_t value
    /// @brief Configure the function pointers invoked by the rocprof-sys user API.
    /// The initial callbacks are set via the rocprof-sys-dl library when it is loaded but
    /// the user can user this feature to turn on/off the user API or customize how the
    /// the user callbacks occur. For example, the user could maintain one set of
    /// callbacks which discard any annotation data or redirect all unannotated user
    /// regions to the annotated user regions with annotations about some global state.
    /// Changing the callbacks is thread-safe but not thread-local.
    extern int rocprofsys_user_configure(
        rocprofsys_user_configure_mode_t mode, rocprofsys_user_callbacks_t inp,
        rocprofsys_user_callbacks_t* out) ROCPROFSYS_PUBLIC_API;

    /// @fn int rocprofsys_user_get_callbacks(int category, void** begin_func, void**
    /// end_func)
    /// @param[in] category An @ref ROCPROFSYS_USER_BINDINGS value
    /// @param[out] begin_func The pointer to the function which corresponds to "starting"
    /// the category, e.g. rocprofsys_user_start_trace or rocprofsys_user_push_region
    /// @param[out] end_func The pointer to the function which corresponds to "ending" the
    /// category, e.g. rocprofsys_user_stop_trace or rocprofsys_user_pop_region
    /// @return rocprofsys_user_error_t value
    /// @brief Get the current function pointers for a given category. The initial values
    /// are assigned by rocprof-sys-dl at start up.
    extern int rocprofsys_user_get_callbacks(rocprofsys_user_callbacks_t*)
        ROCPROFSYS_PUBLIC_API;

    /// @fn const char* rocprofsys_user_error_string(int error_category)
    /// @param error_category ROCPROFSYS_USER_ERROR value
    /// @return String descripting the error code
    /// @brief Return a descriptor for the provided error code
    extern const char* rocprofsys_user_error_string(int) ROCPROFSYS_PUBLIC_API;

#if defined(__cplusplus)
}
#endif

#endif  // ROCPROFSYS_USER_H_
