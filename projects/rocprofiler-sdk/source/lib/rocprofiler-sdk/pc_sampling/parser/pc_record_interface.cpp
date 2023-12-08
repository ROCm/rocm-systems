// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

uint64_t
PCSamplingParserContext::alloc(pcsample_v1_t** buffer, uint64_t size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    data.emplace_back(std::make_unique<PCSamplingData>(size));
    *buffer = data.back()->samples.data();
    return size;
}

pcsample_status_t
PCSamplingParserContext::parse(const upcoming_samples_t& upcoming,
                               const generic_sample_t*   data_,
                               int                       gfxip_major,
                               std::condition_variable&  midway_signal,
                               bool                      bRocrBufferFlip)
{
    // Template instantiation is faster!
    auto parseSample_func = &PCSamplingParserContext::_parse<GFX9>;
    if(gfxip_major == 11)
        parseSample_func = &PCSamplingParserContext::_parse<GFX11>;
    else if(gfxip_major == 0)
        parseSample_func = &PCSamplingParserContext::_parse<gfx_unknown>;
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
    active_dispatches[pkt.correlation_id] = pkt;
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
