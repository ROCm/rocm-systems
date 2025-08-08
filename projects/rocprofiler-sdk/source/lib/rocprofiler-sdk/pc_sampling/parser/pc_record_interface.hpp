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

#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/correlation.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/parser_types.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>
#include <sys/types.h>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_set>

template <typename PcSamplingRecordT>
struct PCSamplingData
{
    PCSamplingData(size_t size)
    : samples(size){};
    PCSamplingData& operator=(PCSamplingData&) = delete;

    std::vector<PcSamplingRecordT> samples;
};

class PCSamplingParserContext
{
public:
    PCSamplingParserContext()
    : corr_map(std::make_unique<Parser::CorrelationMap>()){};

    /**
     * @brief Allocates some memory for samples.
     * TODO: Translate to Jonathan's buffer implementation.
     * @param[out] buffer Pointer where samples are to be written to.
     * @param[in] size Number of samples requested.
     * @returns Number of samples actually allocated on *buffer.
     */
    template <typename PcSamplingRecordT>
    uint64_t alloc(PcSamplingRecordT** buffer, uint64_t size);

    /**
     * @brief Parses a chunk of samples.
     * Call only finishes when all pc sampling records have been generated on the user buffer.
     * As an intermediate step, "midway_signal" signals when it's safe to reuse/delete "data".
     * @param[in] upcoming Metadata of upcoming samples
     * @param[in] data Pointer containing the raw hardware samples. Must match upcoming.num_samples.
     * @param[in] gfxip_major GFXIP of these samples (GFX9==9/GFX11==11).
     * @param[in] midway_signal notifies_all when the samples have been processed.
     * @param[in] bFlushCorrelationIds Set to true if this is the last batch from a ROCr buffer.
     * @returns PCSAMPLE_STATUS_SUCCESS on success.
     * @returns PCSAMPLE_STATUS_PARSER_ERROR (non-fatal) if one or more samples has invalid
     * correlation ID(s).
     * @returns PCSAMPLE_STATUS_INVALID_GFXIP (fatal) on GFXIP != 9,11,12.
     * @returns PCSAMPLE_STATUS_CALLBACK_ERROR (fatal) if memory allocation fails.
     */
    pcsample_status_t parse(const upcoming_samples_t& upcoming,
                            const generic_sample_t*   data,
                            uint32_t                  gfx_target_version,
                            std::condition_variable&  midway_signal,
                            bool                      bFlushCorrelationIds);

    /**
     * @brief Signals a dispatch completion.
     * @param[in] correlation_id Correlation ID of the completed dispatch.
     */
    void completeDispatch(uint64_t correlation_id);
    /**
     * @brief Signals a new dispatch was started.
     * Please use shouldFlipRocrBuffer() to check if the buffer must be flipped before forwarding
     * the dispatch.
     * @param[in] pkt Struct containing the dispatch packet data.
     */
    void newDispatch(const dispatch_pkt_id_t& pkt);
    /**
     * @brief Checkes if a dispatch packet will generate a collision with dorbell_id and
     * dispatch_index.
     * @param[in] pkt Struct containing the dispatch packet data.
     * @returns boolean
     */
    bool shouldFlipRocrBuffer(const dispatch_pkt_id_t& pkt) const;

    bool register_buffer_for_agent(rocprofiler_buffer_id_t buffer_id,
                                   rocprofiler_agent_id_t  agent_id)
    {
        std::unique_lock<std::shared_mutex> lock(mut);
        // Single buffer per agent is allowed
        if(_agent_buffers.count(agent_id) > 0) return false;

        _agent_buffers.emplace(agent_id, buffer_id);
        return true;
    }

    void unregister_buffer_from_agent(rocprofiler_agent_id_t agent_id)
    {
        std::unique_lock<std::shared_mutex> lock(mut);

        _agent_buffers.erase(agent_id);
    }

protected:
    /**
     * @brief Parses the given input data and generates pc sampling records.
     * Calls generate_upcoming_pc_record().
     */
    template <typename GFX, typename PcSamplingRecordT>
    pcsample_status_t _parse(const upcoming_samples_t& upcoming, const generic_sample_t* data_)
    {
        // std::shared_lock<std::shared_mutex> lock(mut);

        pcsample_status_t status      = PCSAMPLE_STATUS_SUCCESS;
        uint64_t          pkt_counter = upcoming.num_samples;
        auto              dev         = upcoming.device;
        bool              bIsHostTrap = upcoming.which_sample_type == AMD_HOST_TRAP_V1;

        while(pkt_counter > 0)
        {
            PcSamplingRecordT* samples = nullptr;
            uint64_t           memsize = alloc(&samples, pkt_counter);

            if(memsize == 0 || memsize > pkt_counter) return PCSAMPLE_STATUS_CALLBACK_ERROR;

            auto* map = corr_map.get();
            if(bIsHostTrap)
                status |= add_upcoming_samples<GFX>(dev, data_, memsize, map, samples);
            else
                status |= add_upcoming_samples<GFX>(dev, data_, memsize, map, samples);

            data_ += memsize;
            pkt_counter -= memsize;
            generate_upcoming_pc_record(dev.handle, samples, memsize);
        }

        return status;
    }

    /**
     * @brief Causes forget_corr_id records to be generated from forget_list. Clears forget_list.
     * Calls generate_id_completion_record()
     */
    pcsample_status_t flushForgetList();
    static void       generate_id_completion_record(const dispatch_pkt_id_t& pkt) { (void) pkt; };

    template <typename PcSamplingRecordT>
    void generate_upcoming_pc_record(uint64_t                 agent_id_handle,
                                     const PcSamplingRecordT* samples,
                                     size_t                   num_samples);

    template <typename PcSamplingRecordT>
    void generate_upcoming_pc_record(uint64_t                              agent_id_handle,
                                     const PcSamplingRecordT*              samples,
                                     size_t                                num_samples,
                                     rocprofiler_pc_sampling_record_kind_t record_kind);

    //! Maps doorbells and dispatch_index to correlation_id
    std::unique_ptr<Parser::CorrelationMap> corr_map;
    //! Data allocated to store host trap and stochastic samples, respectively.
    //! Temporary solution until we figured out a smooth way to copy data directly to SDK's buffers.
    std::vector<std::unique_ptr<PCSamplingData<rocprofiler_pc_sampling_record_host_trap_v0_t>>>
        host_trap_data;
    std::vector<std::unique_ptr<PCSamplingData<rocprofiler_pc_sampling_record_stochastic_v0_t>>>
        stochastic_data;
    //! Dispatches not yet completed.
    // Uses only the internal correlation_id.
    std::unordered_map<uint64_t, dispatch_pkt_id_t> active_dispatches;
    //! List of correlation ids whose dispatches have been completed and can be forgotten after the
    //! buffer flip.
    std::unordered_set<uint64_t> forget_list;

    mutable std::shared_mutex mut;

private:
    using parse_funct_ptr_t = pcsample_status_t (
        PCSamplingParserContext::*)(const upcoming_samples_t&, const generic_sample_t*);

    template <typename GFXIP>
    parse_funct_ptr_t _get_parse_func_for_method(rocprofiler_pc_sampling_method_t pcs_method);

    std::unordered_map<rocprofiler_agent_id_t, rocprofiler_buffer_id_t> _agent_buffers;
};
