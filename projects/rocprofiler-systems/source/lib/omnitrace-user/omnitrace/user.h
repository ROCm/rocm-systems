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
    extern void omnitrace_user_start_trace(void) OMNITRACE_PUBLIC_API;
    extern void omnitrace_user_stop_trace(void) OMNITRACE_PUBLIC_API;
    extern void omnitrace_user_start_thread_trace(void) OMNITRACE_PUBLIC_API;
    extern void omnitrace_user_stop_thread_trace(void) OMNITRACE_PUBLIC_API;
    extern void omnitrace_user_push_region(const char*) OMNITRACE_PUBLIC_API;
    extern void omnitrace_user_pop_region(const char*) OMNITRACE_PUBLIC_API;

    extern void omnitrace_user_configure(void (*)(void), void (*)(void), void (*)(void),
                                         void (*)(void), void (*)(const char*),
                                         void (*)(const char*)) OMNITRACE_PUBLIC_API;

#if defined(__cplusplus)
}
#endif

#endif  // OMNITRACE_USER_H_
