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

#pragma once

#define OMNITRACE_ATTRIBUTE(...)   __attribute__((__VA_ARGS__))
#define OMNITRACE_VISIBILITY(MODE) OMNITRACE_ATTRIBUTE(visibility(MODE))
#define OMNITRACE_PUBLIC_API       OMNITRACE_VISIBILITY("default")
#define OMNITRACE_HIDDEN_API       OMNITRACE_VISIBILITY("hidden")
#define OMNITRACE_WEAK_API         OMNITRACE_ATTRIBUTE(weak)
#define OMNITRACE_INLINE           OMNITRACE_ATTRIBUTE(__always_inline__)

#if defined(__cplusplus)
#    if !defined(OMNITRACE_FOLD_EXPRESSION)
#        define OMNITRACE_FOLD_EXPRESSION(...) ((__VA_ARGS__), ...)
#    endif
#endif
