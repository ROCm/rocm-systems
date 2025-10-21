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

#pragma once

#define CAUSAL_STR2(x) #x
#define CAUSAL_STR(x)  CAUSAL_STR2(x)
#define CAUSAL_LABEL   __FILE__ ":" CAUSAL_STR(__LINE__)

#if defined(USE_OMNI) && USE_OMNI > 0
#    include <rocprofiler-systems/causal.h>
#    define CAUSAL_PROGRESS              ROCPROFSYS_CAUSAL_PROGRESS
#    define CAUSAL_PROGRESS_NAMED(LABEL) ROCPROFSYS_CAUSAL_PROGRESS_NAMED(LABEL)
#    define CAUSAL_BEGIN(LABEL)          ROCPROFSYS_CAUSAL_BEGIN(LABEL)
#    define CAUSAL_END(LABEL)            ROCPROFSYS_CAUSAL_END(LABEL)
#elif defined(USE_COZ) && USE_COZ > 0
#    include <coz.h>
#    define CAUSAL_PROGRESS              COZ_PROGRESS_NAMED(CAUSAL_LABEL)
#    define CAUSAL_PROGRESS_NAMED(LABEL) COZ_PROGRESS_NAMED(LABEL)
#    define CAUSAL_BEGIN(LABEL)          COZ_BEGIN(LABEL)
#    define CAUSAL_END(LABEL)            COZ_END(LABEL)
#else
#    define CAUSAL_PROGRESS
#    define CAUSAL_PROGRESS_NAMED(LABEL)
#    define CAUSAL_BEGIN(LABEL)
#    define CAUSAL_END(LABEL)
#endif
