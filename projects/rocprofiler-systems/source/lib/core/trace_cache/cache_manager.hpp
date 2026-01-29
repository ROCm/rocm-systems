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

#include "core/trace_cache/buffer_storage.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/storage_parser.hpp"

#include "library/runtime.hpp"

#include <memory>
#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{

using storage_parser_t =
    storage_parser<type_identifier_t, kernel_dispatch_sample, memory_copy_sample,
                   memory_allocate_sample, region_sample, in_time_sample,
                   pmc_event_with_sample, amd_smi_sample, cpu_freq_sample,
                   backtrace_region_sample, scratch_memory_sample>;

using buffer_storage_t = buffer_storage<flush_worker_factory_t, type_identifier_t>;

class cache_manager
{
public:
    static cache_manager& get_instance();
    buffer_storage_t&     get_buffer_storage() { return m_storage; }
    metadata_registry&    get_metadata_registry() { return *m_metadata; }
    void                  shutdown();
    void                  post_process_bulk();

private:
    cache_manager() = default;

    buffer_storage_t m_storage{ utility::get_buffered_storage_filename(
        get_root_process_id(), getpid()) };

    std::shared_ptr<metadata_registry> m_metadata{
        std::make_shared<metadata_registry>()
    };
};

inline metadata_registry&
get_metadata_registry()
{
    return cache_manager::get_instance().get_metadata_registry();
}

inline buffer_storage_t&
get_buffer_storage()
{
    return cache_manager::get_instance().get_buffer_storage();
}

}  // namespace trace_cache
}  // namespace rocprofsys
