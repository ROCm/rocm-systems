// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "core/node_info.hpp"
#include "core/sample_cache/cache_storage_parser.hpp"
#include "core/sample_cache/metadata_storage.hpp"

namespace rocprofsys
{
namespace sample_cache
{

class rocpd_post_processing
{
public:
    rocpd_post_processing(metadata& metadata);

    void register_parser_callback(storage_parser& parser);
    void post_process_metadata();

private:
    using primary_key = size_t;

    inline void rocpd_insert_thread_id(info::thread& t_info, const node_info& n_info,
                                       const info::process& process_info) const;

    postprocessing_callback get_kernel_dispatch_callback() const;
    postprocessing_callback get_memory_copy_callback() const;
#if(ROCPROFILER_VERSION >= 600)
    postprocessing_callback get_memory_allocate_callback() const;
#endif
    postprocessing_callback get_region_callback() const;
    postprocessing_callback get_in_time_sample_callback() const;
    postprocessing_callback get_pmc_event_with_sample_callback() const;

    metadata& m_metadata;
};

}  // namespace sample_cache
}  // namespace rocprofsys
