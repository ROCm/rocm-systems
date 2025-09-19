// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ROCP_REG_TEST_WEAK) && ROCP_REG_TEST_WEAK > 0
#    pragma weak hip_init
#    pragma weak hsa_init
#    pragma weak roctxRangePush
#    pragma weak roctxRangePop
#    pragma weak ncclGetVersion
#    pragma weak rocDecCreateDecoder
#    pragma weak rocJpegStreamCreate
#endif

extern void
hip_init(void);

extern void
hsa_init(void);

extern void
roctxRangePush(const char*);

extern void
roctxRangePop(const char*);

enum ncclResult_t
{
};

extern ncclResult_t
ncclGetVersion(int* version);

enum rocDecStatus
{
};

enum rocDecDecoderHandle
{
};

enum RocDecoderCreateInfo
{
};

extern rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*  decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info);

enum RocJpegStatus
{
};

enum RocJpegStreamHandle
{
};

extern RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle);

#ifdef __cplusplus
}
#endif
