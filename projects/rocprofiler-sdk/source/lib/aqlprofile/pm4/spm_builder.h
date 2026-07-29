// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_PM4_SPM_BUILDER_H_
#define SRC_PM4_SPM_BUILDER_H_

#include <stdint.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "lib/aqlprofile/pm4/cmd_config.h"
#include "lib/aqlprofile/pm4/cmd_builder.h"
#include "lib/aqlprofile/core/spm_common.hpp"
#include "lib/aqlprofile/core/logger.hpp"

namespace pm4_builder
{
class CmdBuffer;
class CmdBuilder;

// SpmBuilder config
typedef TraceConfig SpmConfig;

// Encapsulates the various Api and structures that are used to enable
// a SPM session and collect its data. Implementations of this
// interface program device specific registers to realize the functionality
class SpmBuilder
{
public:
    // Destructor of the SPM service handle
    virtual ~SpmBuilder() {}
    // Builds Pm4 command stream to program hardware registers that
    // enable a SPM session, including the issue of an event
    // to begin thread session
    virtual void Begin(CmdBuffer*             cmd_buffer,
                       const SpmConfig*       config,
                       const counters_vector& counters_vec) = 0;
    // Builds Pm4 command stream to program hardware registers that
    // disable a SPM session, including the issue of an event
    // to stop currently ongoing thread session
    virtual void End(CmdBuffer* cmd_buffer, const SpmConfig* config) = 0;
};

template <typename Builder, typename Primitives>
class GpuSpmBuilder
: public SpmBuilder
, protected Primitives
{
    typedef typename Primitives::mux_info_t mux_info_t;
    uint32_t                                wgp_per_sa_;

    void DebugTrace(uint32_t value)
    {
        CmdBuffer cmd_buffer;
        uint32_t  header[2] = {0, value};
        APPEND_COMMAND_WRAPPER((&cmd_buffer), header);
    }

    Builder builder;

public:
    explicit GpuSpmBuilder(const AgentInfo* agent_info)
    : SpmBuilder()
    , builder(acquire_ip_offset_table(agent_info))
    , wgp_per_sa_(1)
    {
        if constexpr(Primitives::GFXIP_LEVEL >= 11)
        {
            const uint32_t xcc_number = agent_info->xcc_num;
            const uint32_t se_number  = agent_info->se_num / xcc_number;
            const uint32_t sa_number  = agent_info->shader_arrays_per_se;
            if(se_number && sa_number)
            {
                wgp_per_sa_ =
                    (agent_info->cu_num / 2 + sa_number * se_number - 1) / (se_number * sa_number);
                wgp_per_sa_ /= xcc_number;
            }
        }
    }

    void Begin(CmdBuffer* cmd_buffer, const SpmConfig* config, const counters_vector& counters_vec)
    {
        // SPM parameters
        const uint32_t sampling_rate = config->sampleRate;
        const uint64_t buffer_ptr    = reinterpret_cast<uint64_t>(config->data_buffer_ptr);
        const uint32_t buffer_size   = config->data_buffer_size;
        const size_t   original_event_count = counters_vec.get_original_event_count();

        // Initialize SPM counter buffer metadata.
        // counter_map takes the index of counters_vector as input, and output an index to
        // the 16bit SPM counter buffer
        SpmBufferDesc* spm_buffer_desc = (SpmBufferDesc*) config->data_buffer_ptr;
        spm_buffer_desc->version       = 1;
        uint16_t* counter_map          = spm_buffer_desc->get_counter_map();
        memset(counter_map, 0, SPM_DESC_SIZE - sizeof(SpmBufferDesc));

        // On Vega this is needed to collect Perf Cntrs: enable clock for performance counters
        if(Primitives::GFXIP_LEVEL == 9)
            builder.BuildWriteUConfigRegPacket(
                cmd_buffer, Primitives::RLC_PERFMON_CLK_CNTL_ADDR, 1);

        // Program Grbm to broadcast messages to all shader engines
        builder.BuildWriteUConfigRegPacket(
            cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, Primitives::grbm_broadcast_value());
        // Issue a CSPartialFlush cmd including cache flush
        builder.BuildWriteWaitIdlePacket(cmd_buffer);

        // SPM counters stop
        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                           Primitives::CP_PERFMON_CNTL_ADDR,
                                           Primitives::cp_perfmon_cntl_spm_stop_value());

        // SPM counters reset
        //
        // We cannot call 'SPM counters reset' in user mode because it will reset WPTR of the
        // SPM ring buffer, RPTR must be adjusted as well but it can only be adjusted in KFD.
        // Also we don't need to reset SPM counter the same way as we do for legacy PMC,
        // because SPM counter will reset upon each new sample.
        //
        // The first reset after aqlprofile acquires SPM from KFD will be done in KFD.
        // Also each time when user mode buffer is no longer made available to KFD, KFD will
        // reset SPM counters.
        //
        // builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR,
        //                                     Primitives::cp_perfmon_cntl_reset_value());

        // Issue a CSPartialFlush cmd including cache flush
        builder.BuildWriteWaitIdlePacket(cmd_buffer);

        // Hardcode PERFMON_RING_MODE to 3 (Stall and send interrupt) to match KFD
        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                           Primitives::RLC_SPM_PERFMON_CNTL__ADDR,
                                           Primitives::rlc_spm_perfmon_cntl_value(
                                               sampling_rate, config->spm_sample_interval_type));

        // Iterate through the list of blocks to create PM4 packets to read counter values
        // Below pair.first is the block id of a counter event and pair.second is the index into
        // counters_vec of the counter event
        std::vector<std::vector<std::pair<int, int> > > counter_info_even(
            Primitives::NUMBER_OF_BLOCKS);
        std::vector<std::vector<std::pair<int, int> > > counter_info_odd(
            Primitives::NUMBER_OF_BLOCKS);

        bool has_global_32bit_spm = false;
        // distribute counter events to counter_info_even and counter_info_odd according to their
        // block id
        for(uint32_t index = 0; index < counters_vec.size(); ++index)
        {
            auto&       counter_des = counters_vec[index];
            const auto& block_des   = counter_des.block_des;

            // The builder consumes resolved per-counter SPM depth and does not re-derive policy.
            // See resolve_spm_depth() in spm_v2.cpp for the current compatibility/default rules.
            if(counter_des.spm_depth == AQLPROFILE_SPM_DEPTH_32_BITS)
            {
                counter_info_even[block_des.id].push_back({block_des.id, index});
                counter_info_odd[block_des.id].push_back({block_des.id, index});
                if(counter_des.block_info->attr & CounterBlockSpmGlobalAttr) has_global_32bit_spm = true;
            }
            else
            {
                if(counter_des.index % 2 == 0)
                    counter_info_even[block_des.id].push_back({block_des.id, index});
                else
                    counter_info_odd[block_des.id].push_back({block_des.id, index});
            }
        }

        // Sort counter_info_even and counter_info_odd by instance
        auto compare = [&counters_vec](std::pair<int, int> a, std::pair<int, int> b) {
            auto  index_a       = a.second;
            auto  index_b       = b.second;
            auto& counter_des_a = counters_vec[index_a];
            auto& counter_des_b = counters_vec[index_b];
            if(counter_des_a.spm_depth == counter_des_b.spm_depth)
            {
                if(counter_des_a.block_des.index == counter_des_b.block_des.index)
                    return counter_des_a.index < counter_des_b.index;
                else
                    return counter_des_a.block_des.index < counter_des_b.block_des.index;
            }
            else
                return counter_des_a.spm_depth > counter_des_b.spm_depth;
        };
        for(size_t i = 0; i < Primitives::NUMBER_OF_BLOCKS; ++i)
        {
            if(!counter_info_even[i].empty())
            {
                sort(counter_info_even[i].begin(), counter_info_even[i].end(), compare);
            }
            if(!counter_info_odd[i].empty())
            {
                sort(counter_info_odd[i].begin(), counter_info_odd[i].end(), compare);
            }
        }

        auto counter_map_value = [&](uint16_t index, const counter_des_t& counter_des, bool is_32bit) {
            const auto* block_info = counter_des.block_info;
            uint16_t    flags      = 0;
            if(block_info->attr & CounterBlockSaAttr) flags |= SPM_COUNTER_MAP_SA_FLAG;
            if(block_info->attr & CounterBlockWgpAttr) flags |= SPM_COUNTER_MAP_WGP_FLAG;
            if(is_32bit) flags |= SPM_COUNTER_MAP_32BIT_FLAG;
            return static_cast<uint16_t>(index | flags);
        };

        // compute segment size for global(0) and se(1)
        uint32_t ss_even[2] = {};
        uint32_t ss_odd[2]  = {};
        for(size_t i = 0; i < Primitives::NUMBER_OF_BLOCKS; ++i)
        {
            if(!counter_info_even[i].empty())
            {
                const auto& counter_des = counters_vec[counter_info_even[i][0].second];
                const auto* block_info  = counter_des.block_info;
                if(block_info->attr & CounterBlockSpmGlobalAttr)
                {
                    ss_even[0] += counter_info_even[i].size();
                }
                else
                {
                    ss_even[1] += counter_info_even[i].size();
                }
            }
            if(!counter_info_odd[i].empty())
            {
                const auto& counter_des = counters_vec[counter_info_odd[i][0].second];
                const auto* block_info  = counter_des.block_info;
                if(block_info->attr & CounterBlockSpmGlobalAttr)
                    ss_odd[0] += counter_info_odd[i].size();
                else
                    ss_odd[1] += counter_info_odd[i].size();
            }
        }

        // if SPM global is streamed we also stream time stamp.
        ss_even[0] += Primitives::RLC_SPM_TIMESTAMP_SIZE16;
        if(has_global_32bit_spm) ss_odd[0] += Primitives::RLC_SPM_TIMESTAMP_SIZE16;

        uint32_t ss[2] = {};
        for(int i = 0; i < 2; ++i)
        {
            ss_even[i] = ss_even[i] / Primitives::RLC_SPM_COUNTERS_PER_LINE +
                         uint32_t(ss_even[i] % Primitives::RLC_SPM_COUNTERS_PER_LINE > 0);
            ss_odd[i] = ss_odd[i] / Primitives::RLC_SPM_COUNTERS_PER_LINE +
                        uint32_t(ss_odd[i] % Primitives::RLC_SPM_COUNTERS_PER_LINE > 0);

            ss[i] = std::max(ss_even[i], ss_odd[i]) * 2;
        }

        // fill in mux_ram data according to even and odd arrays
        std::vector<mux_info_t> mux_ram[2];
        mux_info_t              mxinf = {0xFFFF};
        mux_info_t              mxinf_filler = {0xFFFF};

        // global mux_ram: initialize with all 0xFFFF.
        mux_ram[0].resize(ss[0] * Primitives::RLC_SPM_COUNTERS_PER_LINE + 2);
        std::fill(mux_ram[0].begin(), mux_ram[0].end(), mxinf);

        // se mux_ram: initialize with all 0xFFFF (end of muxsel).
        mux_ram[1].resize(ss[1] * Primitives::RLC_SPM_COUNTERS_PER_LINE + 2);
        std::fill(mux_ram[1].begin(), mux_ram[1].end(), mxinf);

        size_t even_idx = 0;
        size_t odd_idx  = Primitives::RLC_SPM_COUNTERS_PER_LINE;
        // follow the exact steps to fill in mux_ram as when the number of even/odd events are
        // counted Register timestamp
        for(even_idx = 0; even_idx < Primitives::RLC_SPM_TIMESTAMP_SIZE16; ++even_idx)
        {
            mxinf.data           = Primitives::spm_timestamp_muxsel(even_idx);
            mux_ram[0][even_idx] = mxinf;
            // If we have 32bit global SPM, we must align odd_idx with even_idx
            if(has_global_32bit_spm) mux_ram[0][odd_idx++] = mxinf_filler;
        }

        // Process MUX for 32bit SPM counters before 16bit SPM counters so its reserved odd
        // lane stays paired with the matching even slot before any 16-bit block consumes later
        // even positions.
        static aqlprofile_spm_depth_t depth[] = {AQLPROFILE_SPM_DEPTH_32_BITS,
                                                 AQLPROFILE_SPM_DEPTH_16_BITS};
        for(size_t n = 0; n < 2; n++)
        {
            for(size_t j = 0; j < Primitives::NUMBER_OF_BLOCKS; ++j)
            {
                if(counter_info_even[j].empty()) continue;
                const auto& counter_des_0 = counters_vec[counter_info_even[j][0].second];
                if(!(counter_des_0.block_info->attr & CounterBlockSpmGlobalAttr)) continue;
                for(size_t k = 0; k < counter_info_even[j].size(); ++k)
                {
                    const auto  index       = counter_info_even[j][k].second;
                    const auto& counter_des = counters_vec[index];
                    if(counter_des.spm_depth != depth[n]) continue;
                    if(depth[n] == AQLPROFILE_SPM_DEPTH_32_BITS)
                    {
                        const auto counter  = uint16_t(counter_des.index);
                        const auto block    = counter_des_0.block_info->spm_block_id;
                        const auto instance = uint16_t(counter_des.block_des.index);
                        mux_ram[0][even_idx] = Primitives::spm_mux_ram_value(counter, block, instance);
                        counter_map[index]   =
                            even_idx | SPM_COUNTER_MAP_GLOBAL_FLAG | SPM_COUNTER_MAP_32BIT_FLAG;
                    }
                    else
                    {
                        mux_ram[0][even_idx] = Primitives::spm_mux_ram_value(counter_des);
                        counter_map[index]   = even_idx | SPM_COUNTER_MAP_GLOBAL_FLAG;
                    }
                    even_idx = Primitives::spm_mux_ram_idx_incr(even_idx);
                }
                for(size_t k = 0; k < counter_info_odd[j].size(); ++k)
                {
                    const auto  index       = counter_info_odd[j][k].second;
                    const auto& counter_des = counters_vec[index];
                    if(counter_des.spm_depth != depth[n]) continue;
                    if(depth[n] == AQLPROFILE_SPM_DEPTH_32_BITS)
                    {
                        const auto counter  = uint16_t(counter_des.index) + 1;
                        const auto block    = counter_des_0.block_info->spm_block_id;
                        const auto instance = uint16_t(counter_des.block_des.index);
                        mux_ram[0][odd_idx] = Primitives::spm_mux_ram_value(counter, block, instance);
                    }
                    else
                    {
                        mux_ram[0][odd_idx] = Primitives::spm_mux_ram_value(counter_des);
                        counter_map[index]  = odd_idx | SPM_COUNTER_MAP_GLOBAL_FLAG;
                    }
                    odd_idx = Primitives::spm_mux_ram_idx_incr(odd_idx);
                }
            }
        }
        // fill in SE mux_ram
        even_idx = 0;
        odd_idx  = Primitives::RLC_SPM_COUNTERS_PER_LINE;

        // Process 32bit SPM counters before 16bit SPM counters so 32bit odd lanes stay paired with
        // their matching even slots before later 16bit counters consume subsequent positions.
        for(size_t n = 0; n < 2; n++)
        {
            for(size_t j = 0; j < Primitives::NUMBER_OF_BLOCKS; ++j)
            {
                if(counter_info_even[j].empty()) continue;
                const auto& counter_des_0 = counters_vec[counter_info_even[j][0].second];
                if(counter_des_0.block_info->attr & CounterBlockSpmGlobalAttr) continue;
                for(size_t k = 0; k < counter_info_even[j].size(); ++k)
                {
                    const auto  index       = counter_info_even[j][k].second;
                    const auto& counter_des = counters_vec[index];
                    if(counter_des.spm_depth != depth[n]) continue;
                    if(depth[n] == AQLPROFILE_SPM_DEPTH_32_BITS)
                    {
                        const auto counter  = uint16_t(counter_des.index);
                        const auto block    = counter_des_0.block_info->spm_block_id;
                        const auto instance = uint16_t(counter_des.block_des.index);
                        mux_ram[1][even_idx] = Primitives::spm_mux_ram_value(counter, block, instance);
                        counter_map[index]   = counter_map_value(even_idx, counter_des, true);
                    }
                    else
                    {
                        mux_ram[1][even_idx] = Primitives::spm_mux_ram_value(counter_des);
                        counter_map[index]   = counter_map_value(even_idx, counter_des, false);
                    }
                    even_idx = Primitives::spm_mux_ram_idx_incr(even_idx);
                }
                for(size_t k = 0; k < counter_info_odd[j].size(); ++k)
                {
                    const auto  index       = counter_info_odd[j][k].second;
                    const auto& counter_des = counters_vec[index];
                    if(counter_des.spm_depth != depth[n]) continue;
                    if(depth[n] == AQLPROFILE_SPM_DEPTH_32_BITS)
                    {
                        const auto counter  = uint16_t(counter_des.index) + 1;
                        const auto block    = counter_des_0.block_info->spm_block_id;
                        const auto instance = uint16_t(counter_des.block_des.index);
                        mux_ram[1][odd_idx] = Primitives::spm_mux_ram_value(counter, block, instance);
                    }
                    else
                    {
                        mux_ram[1][odd_idx] = Primitives::spm_mux_ram_value(counter_des);
                        counter_map[index]  = counter_map_value(odd_idx, counter_des, false);
                    }
                    odd_idx = Primitives::spm_mux_ram_idx_incr(odd_idx);
                }
            }
        }

        if(config->spm_sample_delay_max)
            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                               Primitives::RLC_SPM_PERFMON_SAMPLE_DELAY_MAX__ADDR,
                                               config->spm_sample_delay_max);

        if constexpr(Primitives::SPM_DELAY_PROGRAMMING_REQUIRED)
        {
            for(size_t i = 0; i < Primitives::NUMBER_OF_BLOCKS; ++i)
            {
                const bool has_even = !counter_info_even[i].empty();
                const bool has_odd  = !counter_info_odd[i].empty();
                if(!has_even && !has_odd) continue;

                const auto& counter_des =
                    counters_vec[(has_even ? counter_info_even[i][0] : counter_info_odd[i][0]).second];
                const auto* block_info = counter_des.block_info;

                if(block_info->attr & CounterBlockSpmGlobalAttr)
                {
                    // For each instance of a global block we program its delay once.
                    for(size_t j = 0; j < block_info->instance_count; ++j)
                    {
                        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                           Primitives::GRBM_GFX_INDEX_ADDR,
                                                           Primitives::grbm_inst_se_sh_index_value(j, 0, 0));
                        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                           block_info->delay_info.reg,
                                                           Primitives::get_spm_global_delay(counter_des, j));
                    }
                }
                else
                {
                    for(size_t se = 0; se < config->se_number; ++se)
                    {
                        for(size_t j = 0; j < block_info->instance_count; ++j)
                        {
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               Primitives::GRBM_GFX_INDEX_ADDR,
                                                               Primitives::grbm_inst_se_index_value(j, se));
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               block_info->delay_info.reg,
                                                               Primitives::get_spm_se_delay(counter_des, se, j));
                        }
                    }
                }
            }
            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                               Primitives::GRBM_GFX_INDEX_ADDR,
                                               Primitives::grbm_broadcast_value());
        }

        // 4. Program the Block instance streaming performance counters in order to specify which items
        //    (events) the counters should count, if any. This is done by programming the
        //    GRBM_GFX_INDEX register to specify the type of access (broadcast or instance specific)
        //    followed by the actual register value. The first step may be to clear all counters of
        //    all instances to select zero (no counting). Then program the GRBM_GFX_INDEX, followed by
        //    the [BLK]_STRMPERFMON_SELECTx register.
        uint32_t grbm_index_value_last = Primitives::grbm_broadcast_value();
        for(size_t i = 0; i < Primitives::NUMBER_OF_BLOCKS; ++i)
        {
            if(counter_info_even[i].empty()) continue;
            const auto& counter_des_0 = counters_vec[counter_info_even[i][0].second];
            const auto* block_info    = counter_des_0.block_info;
            bool        is_sqg_block  = (i == Primitives::SQ_BLOCK_ID);
            bool        is_sqc_block  = false;

            if constexpr(Primitives::GFXIP_LEVEL >= 11)
            {
                is_sqc_block = ((!is_sqg_block) && (block_info->attr & CounterBlockSqAttr)) ? true
                                                                                               : false;
            }

            if(is_sqg_block)
            {
                for(size_t k = 0; k < counter_info_even[i].size(); ++k)
                {
                    const auto&       counter_des = counters_vec[counter_info_even[i][k].second];
                    const auto&       reg_info    = block_info->counter_reg_info[counter_des.index / 2];
                    bool              is_32bit    =
                        counter_des.spm_depth == AQLPROFILE_SPM_DEPTH_32_BITS;

                    if(k == 0)
                    {
                        const uint32_t grbm_index_value = Primitives::grbm_broadcast_value();
                        if(grbm_index_value_last != grbm_index_value)
                        {
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               Primitives::GRBM_GFX_INDEX_ADDR,
                                                               grbm_index_value);
                            grbm_index_value_last = grbm_index_value;
                        }
                        if(!(Primitives::SQ_PERFCOUNTER_MASK_ADDR == Register()))
                        {
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               Primitives::SQ_PERFCOUNTER_MASK_ADDR,
                                                               Primitives::sq_mask_value(counter_des));
                        }
                        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                           reg_info.control_addr,
                                                           Primitives::sq_control_value(counter_des));
                        if(Primitives::GFXIP_LEVEL >= 11)
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               Primitives::SQG_PERFCOUNTER_CTRL2_ADDR,
                                                               Primitives::sq_control2_enable_value());
                    }

                    builder.BuildWriteUConfigRegPacket(
                        cmd_buffer,
                        reg_info.select_addr,
                        Primitives::sq_spm_select_value(counter_des, is_32bit ? 32 : 16));
                }
            }
            else if(is_sqc_block)
            {
                int                je, jo;  // je & jo store even/odd array index
                std::set<uint32_t> programmed_select_offsets;
                for(je = jo = 0; je < counter_info_even[i].size(); ++je)
                {
                    const auto& counter_des_even = counters_vec[counter_info_even[i][je].second];
                    const auto* sqc_block_info   = counter_des_even.block_info;
                    bool        is_32bit = counter_des_even.spm_depth == AQLPROFILE_SPM_DEPTH_32_BITS;

                    const auto& reg_info = sqc_block_info->counter_reg_info[counter_des_even.index / 2];

                    if(je == 0)
                    {
                        const uint32_t grbm_index_value = Primitives::grbm_broadcast_value();
                        if(grbm_index_value_last != grbm_index_value)
                        {
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               Primitives::GRBM_GFX_INDEX_ADDR,
                                                               grbm_index_value);
                            grbm_index_value_last = grbm_index_value;
                        }
                        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                           reg_info.control_addr,
                                                           Primitives::sq_control_value(counter_des_even));
                        if(Primitives::GFXIP_LEVEL >= 11)
                            builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                               Primitives::SQ_PERFCOUNTER_CTRL2_ADDR,
                                                               Primitives::sq_control2_enable_value());
                    }

                    if(!programmed_select_offsets.insert(reg_info.select_addr.offset).second)
                    {
                        if(jo < counter_info_odd[i].size()) jo++;
                        continue;
                    }

                    builder.BuildWriteConfigRegPacket(
                        cmd_buffer,
                        reg_info.select_addr,
                        Primitives::sq_spm_select_value(counter_des_even, is_32bit ? 32 : 16));

                    if(jo < counter_info_odd[i].size())
                    {
                        const auto& counter_des_odd = counters_vec[counter_info_odd[i][jo].second];
                        builder.BuildWriteConfigRegPacket(
                            cmd_buffer,
                            reg_info.select1_addr,
                            Primitives::sq_spm_select_value(counter_des_odd, is_32bit ? 32 : 16));
                        jo++;
                    }
                }
            }
            else
            {
                std::map<uint64_t, uint32_t> programmed_select_values;
                uint32_t                     grbm_index_value =
                    (block_info->attr & CounterBlockWgpAttr)
                        ? Primitives::grbm_broadcast_value()
                        : Primitives::grbm_inst_index_value(
                              Primitives::decode_spm_instance_index(block_info,
                                                                    counter_des_0.block_des.index));
                if(grbm_index_value_last != grbm_index_value)
                {
                    builder.BuildWriteUConfigRegPacket(
                        cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, grbm_index_value);
                    grbm_index_value_last = grbm_index_value;
                }
                int je, jo;  // je & jo store even/odd array index
                for(je = jo = 0; je < counter_info_even[i].size(); ++je)
                {
                    // get 16-bit SPM select value for even counters
                    const auto& counter_des = counters_vec[counter_info_even[i][je].second];
                    bool        is_32bit = counter_des.spm_depth == AQLPROFILE_SPM_DEPTH_32_BITS;
                    uint32_t    spm_select_value =
                        is_32bit ? Primitives::spm_select_value(counter_des)
                                 : Primitives::spm_even_select_value(counter_des);

                    // get 16-bit SPM select value for odd counters
                    if(jo < counter_info_odd[i].size())
                    {
                        if(!is_32bit)
                        {
                            const auto& counter_des_odd = counters_vec[counter_info_odd[i][jo].second];
                            if(counter_des_odd.block_des.index == counter_des.block_des.index)
                            {
                                spm_select_value |= Primitives::spm_odd_select_value(counter_des_odd);
                                jo++;
                            }
                        }
                        else
                            jo++;
                    }

                    int      index = (counter_des.index) >> 2;
                    int      select = ((counter_des.index) % 4) >> 1;
                    Register spm_select_addr =
                        (select == 0) ? block_info->counter_reg_info[index].select_addr
                                      : block_info->counter_reg_info[index].select1_addr;
                    // GFX12 note:
                    // - SA-scoped SPM data is still streamed through the SE line.
                    // - The non-WGP path uses grbm_inst_index_value(), which broadcasts SA.
                    // - WGP blocks share the INST field with WGP selection, so they use full GRBM broadcast.
                    // - This is only correct when the caller uses the same counter select programming for
                    //   every SA/WGP/INST participating in that SE line.
                    // - Under that constraint, broadcast and explicit topology programming converge to the
                    //   same end result while keeping the PM4 programming path simple.
#if PER_WGP_INST_SELECT
                    if((block_info->attr & CounterBlockWgpAttr) && block_info->instance_count > 1)
                    {
                        for(int wgp = 0; wgp < wgp_per_sa_; wgp++)
                        {
                            const uint32_t grbm_index_value =
                                Primitives::grbm_inst_index_value(
                                    Primitives::decode_spm_instance_index(block_info,
                                                                          counter_des.block_des.index) |
                                    (wgp << 2));
                            builder.BuildWriteUConfigRegPacket(
                                cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, grbm_index_value);
                            builder.BuildWriteConfigRegPacket(
                                cmd_buffer, spm_select_addr, spm_select_value);
                            grbm_index_value_last = grbm_index_value;
                        }
                    }
                    else
#endif
                    {
                        if(je != 0 && !(block_info->attr & CounterBlockWgpAttr))
                            grbm_index_value = Primitives::grbm_inst_index_value(
                                Primitives::decode_spm_instance_index(block_info,
                                                                      counter_des.block_des.index));
                        if(grbm_index_value_last != grbm_index_value)
                        {
                            builder.BuildWriteUConfigRegPacket(
                                cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, grbm_index_value);
                            grbm_index_value_last = grbm_index_value;
                        }

                        const uint64_t programmed_select_key =
                            (uint64_t(grbm_index_value) << 32) | spm_select_addr.offset;
                        const auto programmed_select_it =
                            programmed_select_values.find(programmed_select_key);
                        if(programmed_select_it == programmed_select_values.end())
                        {
                            programmed_select_values.insert(
                                {programmed_select_key, spm_select_value});
                            builder.BuildWriteConfigRegPacket(
                                cmd_buffer, spm_select_addr, spm_select_value);
                        }
                        else if(programmed_select_it->second != spm_select_value)
                        {
                            WARN_LOGGING(
                                "conflicting SPM select programming for grbm_index=0x{:x} "
                                "select_addr=0x{:x} old=0x{:x} new=0x{:x}",
                                grbm_index_value,
                                spm_select_addr.offset,
                                programmed_select_it->second,
                                spm_select_value);
                        }
                    }
                }
            }
        }
        if(grbm_index_value_last != Primitives::grbm_broadcast_value())
            builder.BuildWriteUConfigRegPacket(
                cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, Primitives::grbm_broadcast_value());

        // Set segment size
        uint32_t global_count = ss[0];
        uint32_t se_count     = ss[1];
        builder.BuildWriteUConfigRegPacket(
            cmd_buffer,
            Primitives::RLC_SPM_PERFMON_SEGMENT_SIZE__ADDR,
            Primitives::rlc_spm_perfmon_segment_size_value(global_count,
                                                           se_count,
                                                           config->se_number));
        if(config->spm_has_core1)
        {
            builder.BuildWriteUConfigRegPacket(
                cmd_buffer,
                Primitives::RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__ADDR,
                Primitives::rlc_spm_perfmon_segment_size_core1_value(se_count));
        }
        spm_buffer_desc->global_num_line = global_count;
        spm_buffer_desc->se_num_line     = se_count;
        spm_buffer_desc->num_se          = config->se_number;
        spm_buffer_desc->num_sa          = config->sa_number;
        spm_buffer_desc->num_wgp         = wgp_per_sa_;
        spm_buffer_desc->num_xcc         = config->xcc_number;
        spm_buffer_desc->num_events      = original_event_count;

#if defined(DEBUG_TRACE)
        static std::once_flag dump_mux_once;
        std::call_once(dump_mux_once, [&]() {
            auto dump_mux = [](const char* name, const std::vector<mux_info_t>& mux) {
                std::cout << "SPM " << name << " mux dump (16-bit words)" << std::endl;
                for(size_t line = 0; line < mux.size(); line += 16)
                {
                    std::cout << "   " << std::setw(3) << line / 16 << ":";
                    for(size_t col = 0; col < 16 && line + col < mux.size(); ++col)
                    {
                        std::cout << ' ' << std::hex << std::setw(4) << std::setfill('0')
                                  << mux[line + col].data;
                    }
                    std::cout << std::dec << std::setfill(' ') << std::endl;
                }
            };

            dump_mux("global", mux_ram[0]);
            dump_mux("se", mux_ram[1]);
        });
#endif

        // Finish MUXSEL RAM
        // 5. Program the RLC_[GLOBAL/SE]_MUXSEL_ADDR register with the starting address, likely
        // zero.
        if(!mux_ram[0].empty())
        {
            builder.BuildWriteUConfigRegPacket(
                cmd_buffer, Primitives::RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR, 0);
            builder.BuildWriteRegDataPacket(cmd_buffer,
                                            Primitives::RLC_SPM_GLOBAL_MUXSEL_DATA__ADDR,
                                            reinterpret_cast<uint32_t*>(mux_ram[0].data()),
                                            mux_ram[0].size() / 2,
                                            1);
        }
        if(!mux_ram[1].empty())
        {
            builder.BuildWriteUConfigRegPacket(
                cmd_buffer, Primitives::RLC_SPM_SE_MUXSEL_ADDR__ADDR, 0);
            builder.BuildWriteRegDataPacket(cmd_buffer,
                                            Primitives::RLC_SPM_SE_MUXSEL_DATA__ADDR,
                                            reinterpret_cast<uint32_t*>(mux_ram[1].data()),
                                            mux_ram[1].size() / 2,
                                            1);
        }
        // pm4SPM code has the following code
        builder.BuildWriteUConfigRegPacket(
            cmd_buffer, Primitives::RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR, 0);
        builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_SE_MUXSEL_ADDR__ADDR, 0);

        // Issue a CSPartialFlush cmd including cache flush
        builder.BuildWriteWaitIdlePacket(cmd_buffer);
        // Program Compute Perfcount Enable register to support perf counting
        builder.BuildWriteShRegPacket(cmd_buffer,
                                      Primitives::COMPUTE_PERFCOUNT_ENABLE_ADDR,
                                      Primitives::cp_perfcount_enable_value());
        // SPM counters start
        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                           Primitives::CP_PERFMON_CNTL_ADDR,
                                           Primitives::cp_perfmon_cntl_spm_start_value());
        // Issue a CSPartialFlush cmd including cache flush
        builder.BuildWriteWaitIdlePacket(cmd_buffer);
    }

    void End(CmdBuffer* cmd_buffer, const SpmConfig* config)
    {
        // Force one last RLC sample pulse before SPM stop so the current partial interval is
        // materialized into the stream instead of being dropped when CP_PERFMON_CNTL stops/reset SPM.
        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                           Primitives::RLC_SPM_PERFMON_CNTL__ADDR,
                                           Primitives::rlc_spm_perfmon_cntl_value(0, 0));
#if defined(DEBUG_TRACE)
        if(config->spm_sample_count_ptr && !(Primitives::RLC_SPM_SAMPLE_CNT__ADDR == Register()))
        {
            builder.BuildCopyRegDataPacket(cmd_buffer,
                                           Primitives::RLC_SPM_SAMPLE_CNT__ADDR,
                                           config->spm_sample_count_ptr,
                                           Primitives::COPY_DATA_SEL_COUNT_1DW_PRM,
                                           true);
        }
#endif
        // Program Grbm to broadcast messages to all shader engines
        builder.BuildWriteUConfigRegPacket(
            cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, Primitives::grbm_broadcast_value());
        // Issue a CSPartialFlush cmd including cache flush
        builder.BuildWriteWaitIdlePacket(cmd_buffer);
        // SPM counters stop
        builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                           Primitives::CP_PERFMON_CNTL_ADDR,
                                           Primitives::cp_perfmon_cntl_spm_stop_value());
        // SPM counters reset
        // 'SPM counters reset' must be done in KFD. See comments in Begin() for more details
        //
        // builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR,
        //                                     Primitives::cp_perfmon_cntl_reset_value());

        // On Vega this disable clock for performance counters
        if(Primitives::GFXIP_LEVEL == 9)
            builder.BuildWriteUConfigRegPacket(
                cmd_buffer, Primitives::RLC_PERFMON_CLK_CNTL_ADDR, 0);
    }
};

}  // namespace pm4_builder

#endif  // SRC_PM4_SPM_BUILDER_H_
