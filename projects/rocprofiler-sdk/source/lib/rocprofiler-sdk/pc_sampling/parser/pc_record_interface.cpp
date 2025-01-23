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

#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_host_trap_v0_t>(
    rocprofiler_pc_sampling_record_host_trap_v0_t** buffer,
    uint64_t                                        size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    host_trap_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_host_trap_v0_t>>(size));
    *buffer = host_trap_data.back()->samples.data();
    return size;
}

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_stochastic_v0_t>(
    rocprofiler_pc_sampling_record_stochastic_v0_t** buffer,
    uint64_t                                         size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    stochastic_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_stochastic_v0_t>>(size));
    *buffer = stochastic_data.back()->samples.data();
    return size;
}

pcsample_status_t
PCSamplingParserContext::parse(const upcoming_samples_t& upcoming,
                               const generic_sample_t*   data_,
                               int                       gfxip_major,
                               std::condition_variable&  midway_signal,
                               bool                      bRocrBufferFlip)
{
    bool bIsHostTrap = upcoming.which_sample_type == AMD_HOST_TRAP_V1;

    // Template instantiation is faster!
    auto parseSample_func =
        bIsHostTrap
            ? &PCSamplingParserContext::_parse<GFX9, rocprofiler_pc_sampling_record_host_trap_v0_t>
            : &PCSamplingParserContext::_parse<GFX9,
                                               rocprofiler_pc_sampling_record_stochastic_v0_t>;
    if(gfxip_major == 11)
        parseSample_func =
            bIsHostTrap
                ? &PCSamplingParserContext::_parse<GFX11,
                                                   rocprofiler_pc_sampling_record_host_trap_v0_t>
                : &PCSamplingParserContext::_parse<GFX11,
                                                   rocprofiler_pc_sampling_record_stochastic_v0_t>;
    else if(gfxip_major != 9)
        return PCSAMPLE_STATUS_INVALID_GFXIP;

    auto status = (this->*parseSample_func)(upcoming, data_);
    midway_signal.notify_all();

    if(!bRocrBufferFlip || status != PCSAMPLE_STATUS_SUCCESS) return status;

    return flushForgetList();
}

void
PCSamplingParserContext::newDispatch(const dispatch_pkt_id_t& pkt)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    corr_map->newDispatch(pkt);
    active_dispatches[pkt.correlation_id.internal] = pkt;
}

void
PCSamplingParserContext::completeDispatch(uint64_t correlation_id)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    forget_list.emplace(correlation_id);
}

pcsample_status_t
PCSamplingParserContext::flushForgetList()
{
    std::unique_lock<std::shared_mutex> lock(mut);
    pcsample_status_t                   status = PCSAMPLE_STATUS_SUCCESS;

    for(uint64_t id : forget_list)
    {
        if(active_dispatches.find(id) == active_dispatches.end())
        {
            status = PCSAMPLE_STATUS_PARSER_ERROR;
            continue;
        }
        const auto& pkt = active_dispatches.at(id);
        generate_id_completion_record(pkt);
        corr_map->forget(pkt);
        active_dispatches.erase(id);
    }
    forget_list.clear();
    return status;
}

bool
PCSamplingParserContext::shouldFlipRocrBuffer(const dispatch_pkt_id_t& pkt) const
{
    std::shared_lock<std::shared_mutex> lock(mut);
    return corr_map->checkDispatch(pkt);
}

template <typename PcSamplingRecordKindT>
void
PCSamplingParserContext::generate_upcoming_pc_record(
    uint64_t                              agent_id_handle,
    const PcSamplingRecordKindT*          samples,
    size_t                                num_samples,
    rocprofiler_pc_sampling_record_kind_t record_kind)
{
    auto buff_id = _agent_buffers.at(rocprofiler_agent_id_t{agent_id_handle});
    rocprofiler::buffer::instance* buff = rocprofiler::buffer::get_buffer(buff_id);

    if(!buff)
        throw std::runtime_error(fmt::format("Buffer with id: {} does not exists", buff_id.handle));

    for(size_t i = 0; i < num_samples; i++)
        buff->emplace(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING, record_kind, samples[i]);
}

template <>
void
PCSamplingParserContext::generate_upcoming_pc_record<rocprofiler_pc_sampling_record_host_trap_v0_t>(
    uint64_t                                             agent_id_handle,
    const rocprofiler_pc_sampling_record_host_trap_v0_t* samples,
    size_t                                               num_samples)
{
    this->generate_upcoming_pc_record(
        agent_id_handle, samples, num_samples, ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE);
}

template <>
void
PCSamplingParserContext::generate_upcoming_pc_record<
    rocprofiler_pc_sampling_record_stochastic_v0_t>(
    uint64_t                                              agent_id_handle,
    const rocprofiler_pc_sampling_record_stochastic_v0_t* samples,
    size_t                                                num_samples)
{
    this->generate_upcoming_pc_record(
        agent_id_handle, samples, num_samples, ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE);
}
