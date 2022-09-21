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

#ifndef OMNITRACE_TYPES_H_
#define OMNITRACE_TYPES_H_ 1

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @enum OMNITRACE_USER_ERROR
    /// @brief Identifier for errors
    ///
    typedef enum OMNITRACE_USER_ERROR
    {
        OMNITRACE_USER_SUCCESS = 0,             ///< No error
        OMNITRACE_USER_ERROR_NO_BINDING,        ///< Function pointer was not assigned
        OMNITRACE_USER_ERROR_BAD_VALUE,         ///< Provided value was invalid
        OMNITRACE_USER_ERROR_INVALID_CATEGORY,  ///< Invalid user binding category
        OMNITRACE_USER_ERROR_INTERNAL,  ///< Internal error occurred within libomnitrace
        OMNITRACE_USER_ERROR_LAST
    } omnitrace_user_error_t;

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
    typedef enum OMNITRACE_USER_BINDINGS
    {
        OMNITRACE_USER_START_STOP =
            0,  ///< Function pointers which control global start/stop
        OMNITRACE_USER_START_STOP_THREAD,  ///< Function pointers which control per-thread
                                           ///< start/stop
        OMNITRACE_USER_REGION,  ///< Function pointers which generate user-defined regions
        OMNITRACE_USER_SAMPLE,  ///< Function pointer which generate samples
        OMNITRACE_USER_BINDINGS_LAST
    } omnitrace_user_bindings_t;

#if defined(__cplusplus)
}
#endif

#endif  // OMNITRACE_TYPES_H_
