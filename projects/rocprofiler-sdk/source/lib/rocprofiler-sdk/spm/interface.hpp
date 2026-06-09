// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/aqlprofile/aqlprofile.hpp"

namespace rocprofiler
{
namespace spm
{
/** @brief Wrapper to aqlprofile functions for SPM
 */
struct spm_interface
{
    using spm_create_packets_fn_t             = decltype(aqlprofile_spm_create_packets);
    using spm_start_fn_t                      = decltype(aqlprofile_spm_start);
    using spm_stop_fn_t                       = decltype(aqlprofile_spm_stop);
    using spm_decode_stream_v1_fn_t           = decltype(aqlprofile_spm_decode_stream_v1);
    using spm_decode_query_fn_t               = decltype(aqlprofile_spm_decode_query);
    using spm_is_event_supported_fn_t         = decltype(aqlprofile_spm_is_event_supported);
    using spm_query_agent_configurations_fn_t = decltype(aqlprofile_spm_query_agent_configurations);

    spm_create_packets_fn_t*             spm_create_packets             = nullptr;
    spm_start_fn_t*                      spm_start                      = nullptr;
    spm_stop_fn_t*                       spm_stop                       = nullptr;
    spm_decode_stream_v1_fn_t*           spm_decode_stream_v1           = nullptr;
    spm_decode_query_fn_t*               spm_decode_query               = nullptr;
    spm_is_event_supported_fn_t*         spm_is_event_supported         = nullptr;
    spm_query_agent_configurations_fn_t* spm_query_agent_configurations = nullptr;

    spm_interface()                     = default;
    ~spm_interface()                    = default;
    spm_interface(const spm_interface&) = delete;
    spm_interface& operator=(const spm_interface&) = delete;
};

const spm_interface*
construct_spm_interface();

}  // namespace spm
}  // namespace rocprofiler
