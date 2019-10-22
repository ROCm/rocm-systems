/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_ROCTRACER_HCC_H_
#define INC_ROCTRACER_HCC_H_

#if HIP_VDI
#define HIP_OP_ID_NUMBER 3
#define HIP_OP_ID_COPY 1
extern "C" {
typedef void (hipInitAsyncActivityCallback_t)(void* id_callback, void* op_callback, void* arg);
typedef bool (hipEnableAsyncActivityCallback_t)(unsigned op, bool enable);
typedef const char* (hipGetOpName_t)(unsigned op);
}
#else  // !HIP_VDI
#if LOCAL_BUILD
#include <hc_prof_runtime.h>
#else
#include <hcc/hc_prof_runtime.h>
#endif
#define HIP_OP_ID_NUMBER hc::HSA_OP_ID_NUMBER
#define HIP_OP_ID_COPY hc::HSA_OP_ID_COPY
typedef decltype(Kalmar::CLAMP::InitActivityCallback) hipInitAsyncActivityCallback_t;
typedef decltype(Kalmar::CLAMP::EnableActivityCallback) hipEnableAsyncActivityCallback_t;
typedef decltype(Kalmar::CLAMP::GetCmdName) hipGetOpName_t;
#endif  // !HIP_VDI

#include "roctracer.h"

#endif  // INC_ROCTRACER_HCC_H_
