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
#define OMNITRACE_USER_H_ 1

#define OMNITRACE_ATTRIBUTE(...)   __attribute__((__VA_ARGS__))
#define OMNITRACE_VISIBILITY(MODE) OMNITRACE_ATTRIBUTE(visibility(MODE))
#define OMNITRACE_PUBLIC_API       OMNITRACE_VISIBILITY("default")
#define OMNITRACE_HIDDEN_API       OMNITRACE_VISIBILITY("hidden")

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @enum OMNITRACE_USER_ERROR
    /// @brief Identifier for errors
    ///
    enum OMNITRACE_USER_ERROR
    {
        OMNITRACE_USER_SUCCESS = 0,                 ///< No error
        OMNITRACE_USER_ERROR_NO_BINDING,            ///< Function pointer was not assigned
        OMNITRACE_USER_ERROR_BAD_FUNCTION_POINTER,  ///< Provided function pointer was
                                                    ///< invalid
        OMNITRACE_USER_ERROR_INVALID_CATEGORY,      ///< Invalid user binding category
        OMNITRACE_USER_ERROR_INTERNAL,  ///< Internal error occurred within libomnitrace
        OMNITRACE_USER_ERROR_LAST
    };

    /// @enum OMNITRACE_USER_BINDINGS
    /// @brief Identifier for function pointer categories
    /// @code{.cpp}
    /// int (*omnitrace_push_region_f)(const char*) = nullptr;
    ///
    /// int custom_push_region(const char* name)
    /// {
    ///     // custom push region prints message before calling internal callback
    ///     printf("Pushing region %s\n", name);
    ///     return (*omnitrace_push_region_f)(name);
    /// }
    ///
    /// int main(int argc, char** argv)
    /// {
    ///     // get the internal callback to start a user-defined region
    ///     omnitrace_user_get_callbacks(OMNITRACE_USER_REGION,
    ///                                  (void**) &omnitrace_push_region_f,
    ///                                  nullptr);
    ///     // assign the custom callback to start a user-defined region
    ///     if(omnitrace_push_region_f)
    ///         omnitrace_user_configure(OMNITRACE_USER_REGION,
    ///                                  (void*) &custom_push_region,
    ///                                  nullptr);
    ///     // ...
    /// }
    ///
    /// @endcode
    enum OMNITRACE_USER_BINDINGS
    {
        OMNITRACE_USER_START_STOP =
            0,  ///< Function pointers which control global start/stop
        OMNITRACE_USER_START_STOP_THREAD,  ///< Function pointers which control per-thread
                                           ///< start/stop
        OMNITRACE_USER_REGION,  ///< Function pointers which generate user-defined regions
        OMNITRACE_USER_BINDINGS_LAST
    };

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

    /// @fn int omnitrace_user_configure(int category, void* begin_func, void* end_func)
    /// @param category An @ref OMNITRACE_USER_BINDINGS value
    /// @param begin_func The pointer to the function which corresponds to "starting" the
    /// category, e.g. omnitrace_user_start_trace or omnitrace_user_push_region
    /// @param end_func The pointer to the function which corresponds to "ending" the
    /// category, e.g. omnitrace_user_stop_trace or omnitrace_user_pop_region
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Configure the function pointers for a given category. This is handled by
    /// omnitrace-dl at start up but the user can specify their own if desired.
    extern int omnitrace_user_configure(int, void*, void*) OMNITRACE_PUBLIC_API;

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
    extern int omnitrace_user_get_callbacks(int, void**, void**) OMNITRACE_PUBLIC_API;

    /// @fn const char* omnitrace_user_error_string(int error_category)
    /// @param error_category OMNITRACE_USER_ERROR value
    /// @return @ref OMNITRACE_USER_ERROR value
    /// @brief Return a descriptor for the provided error code
    extern const char* omnitrace_user_error_string(int) OMNITRACE_PUBLIC_API;

#if defined(__cplusplus)
}
#endif

#endif  // OMNITRACE_USER_H_
