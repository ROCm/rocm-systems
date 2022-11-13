// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef OMNITRACE_USER_H_
#define OMNITRACE_USER_H_

#if defined(OMNITRACE_USER_SOURCE) && (OMNITRACE_USER_SOURCE > 0)
#    if !defined(OMNITRACE_PUBLIC_API)
#        define OMNITRACE_PUBLIC_API __attribute__((visibility("default")))
#    endif
#else
#    if !defined(OMNITRACE_PUBLIC_API)
#        define OMNITRACE_PUBLIC_API
#    endif
#endif

#include "omnitrace/categories.h"
#include "omnitrace/types.h"

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @fn int omnitrace_user_start_trace(void)
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Enable tracing on this thread and all subsequently created threads
    extern int omnitrace_user_start_trace(void) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_stop_trace(void)
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Disable tracing on this thread and all subsequently created threads
    extern int omnitrace_user_stop_trace(void) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_start_thread_trace(void)
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Enable tracing on this specific thread. Does not apply to subsequently
    /// created threads
    extern int omnitrace_user_start_thread_trace(void) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_stop_thread_trace(void)
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Disable tracing on this specific thread. Does not apply to subsequently
    /// created threads
    extern int omnitrace_user_stop_thread_trace(void) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_push_region(const char* id)
    /// @param id The string identifier for the region
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Start a user defined region.
    extern int omnitrace_user_push_region(const char*) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_pop_region(const char* id)
    /// @param id The string identifier for the region
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief End a user defined region. In general, user regions should be popped in
    /// the inverse order that they were pushed, i.e. first-in, last-out (FILO). The
    /// timemory backend was designed to accommodate asynchronous tasking, where FILO may
    /// be violated, and will thus compenstate for out-of-order popping, however, the
    /// perfetto backend will not; thus, out-of-order popping will result in different
    /// results in timemory vs. perfetto.
    extern int omnitrace_user_pop_region(const char*) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_push_annotated_region(const char* id,
    ///                                              omnitrace_annotation_t* annotations,
    ///                                              size_t num_annotations)
    /// @param id The string identifier for the region
    /// @param annotations Array of @ref omnitrace_annotation_t instances
    /// @param num_annotations Number of annotations
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Start a user defined region and adds the annotations to the perfetto trace.
    extern int omnitrace_user_push_annotated_region(const char*, omnitrace_annotation_t*,
                                                    size_t) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_pop_annotated_region(const char* id,
    ///                                             omnitrace_annotation_t* annotations,
    ///                                             size_t num_annotations)
    /// @param id The string identifier for the region
    /// @param annotations Array of @ref omnitrace_annotation_t instances
    /// @param num_annotations Number of annotations
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Stop a user defined region and adds the annotations to the perfetto trace.
    extern int omnitrace_user_pop_annotated_region(const char*, omnitrace_annotation_t*,
                                                   size_t) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_configure(omnitrace_user_configure_mode_t cfg,
    ///                                  omnitrace_user_callbacks_t new_callbacks,
    ///                                  omnitrace_user_callbacks_t* old_callbacks)
    /// @param[in] config Specifies how the new callbacks are merged with the old
    /// callbacks
    /// @param[in] new_callbacks An @ref omnitrace_user_callbacks_t instance specifying
    ///            the callbacks which should be invoked by the user API.
    /// @param[out] old_callbacks Pointer to @ref omnitrace_user_callbacks_t which,
    ///             when non-NULL, will be assigned the former callbacks.
    /// @return @ref omnitrace_user_error_t value
    /// @brief Configure the function pointers invoked by the omnitrace user API.
    /// The initial callbacks are set via the omnitrace-dl library when it is loaded but
    /// the user can user this feature to turn on/off the user API or customize how the
    /// the user callbacks occur. For example, the user could maintain one set of
    /// callbacks which discard any annotation data or redirect all unannotated user
    /// regions to the annotated user regions with annotations about some global state.
    /// Changing the callbacks is thread-safe but not thread-local.
    extern int omnitrace_user_configure(
        omnitrace_user_configure_mode_t, omnitrace_user_callbacks_t new_callbacks,
        omnitrace_user_callbacks_t* old_callbacks) OMNITRACE_PUBLIC_API;

    /// @fn int omnitrace_user_get_callbacks(int category, void** begin_func, void**
    /// end_func)
    /// @param[in] category An @ref OMNITRACE_USER_BINDINGS value
    /// @param[out] begin_func The pointer to the function which corresponds to "starting"
    /// the category, e.g. omnitrace_user_start_trace or omnitrace_user_push_region
    /// @param[out] end_func The pointer to the function which corresponds to "ending" the
    /// category, e.g. omnitrace_user_stop_trace or omnitrace_user_pop_region
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Get the current function pointers for a given category. The initial values
    /// are assigned by omnitrace-dl at start up.
    extern int omnitrace_user_get_callbacks(omnitrace_user_callbacks_t*)
        OMNITRACE_PUBLIC_API;

    /// @fn const char* omnitrace_user_error_string(int error_category)
    /// @param error_category OMNITRACE_USER_ERROR value
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Return a descriptor for the provided error code
    extern const char* omnitrace_user_error_string(int) OMNITRACE_PUBLIC_API;

#if defined(__cplusplus)
}
#endif

#endif  // OMNITRACE_USER_H_
