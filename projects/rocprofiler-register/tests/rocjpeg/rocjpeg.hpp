// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#define ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION 0
#define ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION  1

#include <cstddef>
#include <cstdint>

extern "C" {
// fake rccl function
enum RocJpegStatus
{
};

enum RocJpegStreamHandle
{
};

RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle)
    __attribute__((visibility("default")));
}

namespace rocjpeg
{
struct rocjpegApiFuncTable
{
    uint64_t                         size                   = 0;
    decltype(::rocJpegStreamCreate)* rocJpegStreamCreate_fn = nullptr;
};

RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle);

// populates rocjpeg api table with function pointers
inline void
initialize_rocjpeg_api_table(rocjpegApiFuncTable* dst)
{
    dst->size                   = sizeof(rocjpegApiFuncTable);
    dst->rocJpegStreamCreate_fn = &::rocjpeg::rocJpegStreamCreate;
}

// copies the api table from src to dst
inline void
copy_rocjpeg_api_table(rocjpegApiFuncTable* dst, const rocjpegApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rocjpeg
