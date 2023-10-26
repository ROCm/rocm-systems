#include "lib/rocprofiler/pc_sampling/parser/pc_record_interface.hpp"

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
