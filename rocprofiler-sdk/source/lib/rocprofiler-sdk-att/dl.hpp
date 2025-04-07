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
#include <memory>
#include "att_decoder.h"

namespace rocprofiler
{
namespace att_wrapper
{
class DL
{
    using ParseFn  = decltype(rocprofiler_att_decoder_parse_data);
    using InfoFn   = decltype(rocprofiler_att_decoder_get_info_string);
    using StatusFn = decltype(rocprofiler_att_decoder_get_status_string);

public:
    DL(const char* libname);
    ~DL();

    ParseFn*  att_parse_data_fn = nullptr;
    InfoFn*   att_info_fn       = nullptr;
    StatusFn* att_status_fn     = nullptr;
    void*     handle            = nullptr;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
