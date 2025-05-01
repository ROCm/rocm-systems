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

#include "counter_info.hpp"
#include "generator.hpp"
#include "pc_sample_transform.hpp"
#include "statistics.hpp"
#include "stream_info.hpp"
#include "tmp_file_buffer.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <deque>

namespace rocprofiler
{
namespace tool
{
using float_type   = double;
using stats_data_t = statistics<uint64_t, float_type>;

template <typename Tp, domain_type DomainT>
struct buffered_output
{
    using type                          = Tp;
    static constexpr auto buffer_type_v = DomainT;

    explicit buffered_output(bool _enabled);
    ~buffered_output()                          = default;
    buffered_output(const buffered_output&)     = delete;
    buffered_output(buffered_output&&) noexcept = delete;
    buffered_output& operator=(const buffered_output&) = delete;
    buffered_output& operator=(buffered_output&&) noexcept = delete;

    operator bool() const { return enabled; }

    void flush();
    void read();
    void clear();
    void destroy();

    uint64_t       get_num_bytes() const;
    generator<Tp>  get_generator() const { return generator<Tp>{get_tmp_file_buffer<Tp>(DomainT)}; }
    std::deque<Tp> load_all();

    stats_entry_t stats = {};

private:
    bool enabled = false;
};

template <typename Tp, domain_type DomainT>
buffered_output<Tp, DomainT>::buffered_output(bool _enabled)
: enabled{_enabled}
{}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::flush()
{
    if(!enabled) return;

    flush_tmp_buffer<type>(buffer_type_v);
}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::read()
{
    if(!enabled) return;

    flush();

    read_tmp_file<type>(buffer_type_v);
}

template <typename Tp, domain_type DomainT>
std::deque<Tp>
buffered_output<Tp, DomainT>::load_all()
{
    auto data = std::deque<Tp>{};
    if(enabled)
    {
        auto gen = get_generator();
        for(auto ditr : gen)
        {
            for(auto itr : gen.get(ditr))
            {
                data.emplace_back(itr);
            }
        }
    }
    return data;
}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::clear()
{
    if(!enabled) return;
}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::destroy()
{
    if(!enabled) return;

    clear();
    auto*& filebuf = get_tmp_file_buffer<type>(buffer_type_v);
    if(filebuf)
    {
        file_buffer<type>* tmp = nullptr;
        std::swap(filebuf, tmp);
        tmp->buffer.destroy();
        if(tmp->file)
        {
            tmp->file.close();
            tmp->file.remove();
        }
        delete tmp;
    }
}

template <typename Tp, domain_type DomainT>
uint64_t
buffered_output<Tp, DomainT>::get_num_bytes() const
{
    if(auto*& filebuf = get_tmp_file_buffer<type>(buffer_type_v); filebuf) return filebuf->nbytes;

    return 0;
}

using hip_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_hip_api_ext_record_t, domain_type::HIP>;
using hsa_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_hsa_api_record_t, domain_type::HSA>;
using marker_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_marker_api_record_t, domain_type::MARKER>;
using rccl_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_rccl_api_record_t, domain_type::RCCL>;
using counter_collection_buffered_output_t =
    buffered_output<tool_counter_record_t, domain_type::COUNTER_COLLECTION>;
using scratch_memory_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_scratch_memory_record_t,
                    domain_type::SCRATCH_MEMORY>;
using memory_allocation_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_memory_allocation_record_t,
                    domain_type::MEMORY_ALLOCATION>;
using counter_records_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler::tool::serialized_counter_record_t,
                                         domain_type::COUNTER_VALUES>;
using pc_sampling_host_trap_buffered_output_t =
    buffered_output<rocprofiler::tool::rocprofiler_tool_pc_sampling_host_trap_record_t,
                    domain_type::PC_SAMPLING_HOST_TRAP>;
using rocdecode_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t, domain_type::ROCDECODE>;
using rocjpeg_buffered_output_t =
    buffered_output<rocprofiler_buffer_tracing_rocjpeg_api_record_t, domain_type::ROCJPEG>;
using kernel_dispatch_buffered_output_with_stream_t =
    buffered_output<tool_buffer_tracing_kernel_dispatch_ext_record_t, domain_type::KERNEL_DISPATCH>;
using memory_copy_buffered_output_with_stream_t =
    buffered_output<tool_buffer_tracing_memory_copy_ext_record_t, domain_type::MEMORY_COPY>;
using pc_sampling_stochastic_buffered_output_t =
    buffered_output<rocprofiler::tool::rocprofiler_tool_pc_sampling_stochastic_record_t,
                    domain_type::PC_SAMPLING_STOCHASTIC>;
}  // namespace tool
}  // namespace rocprofiler
