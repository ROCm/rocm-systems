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
enum rocDecStatus
{
};

enum rocDecDecoderHandle
{
};

enum RocDecoderCreateInfo
{
};

rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*  decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info)
    __attribute__((visibility("default")));
}

namespace rocdecode
{
struct rocdecodeApiFuncTable
{
    uint64_t                         size                   = 0;
    decltype(::rocDecCreateDecoder)* rocDecCreateDecoder_fn = nullptr;
};

rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*  decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info);

// populates rocdecode api table with function pointers
inline void
initialize_rocdecode_api_table(rocdecodeApiFuncTable* dst)
{
    dst->size                   = sizeof(rocdecodeApiFuncTable);
    dst->rocDecCreateDecoder_fn = &::rocdecode::rocDecCreateDecoder;
}

// copies the api table from src to dst
inline void
copy_rocdecode_api_table(rocdecodeApiFuncTable* dst, const rocdecodeApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rocdecode
