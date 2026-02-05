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

#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"

namespace rocprofiler
{
namespace spm
{
/** @brief Wrapper to aqlprofile functions for SPM
 */
class Dlsym
{
public:
    using CreateFn  = decltype(aqlprofile_spm_create_packets);
    using DeleteFn  = decltype(aqlprofile_spm_delete_packets);
    using StartFn   = decltype(aqlprofile_spm_start);
    using StopFn    = decltype(aqlprofile_spm_stop);
    using DecodeFn  = decltype(aqlprofile_spm_decode_stream_v1);
    using QueryFn   = decltype(aqlprofile_spm_decode_query);
    using SupportFn = decltype(aqlprofile_spm_is_event_supported);

    Dlsym();
    ~Dlsym();

    bool valid() const
    {
        return create_packets_fn != nullptr && delete_packets_fn != nullptr &&
               spm_start_fn != nullptr && spm_stop_fn != nullptr && spm_decode_fn != nullptr &&
               spm_query_fn != nullptr && is_supported_fn != nullptr && handle != nullptr;
    }

    CreateFn*  create_packets_fn = nullptr;
    DeleteFn*  delete_packets_fn = nullptr;
    StartFn*   spm_start_fn      = nullptr;
    StopFn*    spm_stop_fn       = nullptr;
    DecodeFn*  spm_decode_fn     = nullptr;
    QueryFn*   spm_query_fn      = nullptr;
    SupportFn* is_supported_fn   = nullptr;
    void*      handle            = nullptr;
};

}  // namespace spm
}  // namespace rocprofiler
