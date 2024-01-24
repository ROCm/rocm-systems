// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

typedef union rocprofiler_hip_compiler_api_retval_u
{
    hipError_t hipError_t_retval;
    void**     voidpp_retval;
} rocprofiler_hip_compiler_api_retval_t;

typedef union rocprofiler_hip_compiler_api_args_u
{
    struct
    {
        dim3*        gridDim;
        dim3*        blockDim;
        size_t*      sharedMem;
        hipStream_t* stream;
    } __hipPopCallConfiguration;
    struct
    {
        dim3        gridDim;
        dim3        blockDim;
        size_t      sharedMem;
        hipStream_t stream;
    } __hipPushCallConfiguration;
    struct
    {
        const void* data;
    } __hipRegisterFatBinary;
    struct
    {
        void**       modules;
        const void*  hostFunction;
        char*        deviceFunction;
        const char*  deviceName;
        unsigned int threadLimit;
        uint3*       tid;
        uint3*       bid;
        dim3*        blockDim;
        dim3*        gridDim;
        int*         wSize;
    } __hipRegisterFunction;
    struct
    {
        void*       hipModule;
        void**      pointer;
        void*       init_value;
        const char* name;
        size_t      size;
        unsigned    align;
    } __hipRegisterManagedVar;
    struct
    {
        void** modules;
        void*  var;
        char*  hostVar;
        char*  deviceVar;
        int    type;
        int    ext;
    } __hipRegisterSurface;
    struct
    {
        void** modules;
        void*  var;
        char*  hostVar;
        char*  deviceVar;
        int    type;
        int    norm;
        int    ext;
    } __hipRegisterTexture;
    struct
    {
        void** modules;
        void*  var;
        char*  hostVar;
        char*  deviceVar;
        int    ext;
        size_t size;
        int    constant;
        int    global;
    } __hipRegisterVar;
    struct
    {
        void** modules;
    } __hipUnregisterFatBinary;
} rocprofiler_hip_compiler_api_args_t;
