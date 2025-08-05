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
#include <iostream>
#include <utility>
#include <vector>

#include "pm4/cmd_config.h"

namespace pm4_builder {
class CmdBuffer;
class CmdBuilder;

// SpmBuilder config
typedef TraceConfig SpmConfig;

// Encapsulates the various Api and structures that are used to enable
// a SPM session and collect its data. Implementations of this
// interface program device specific registers to realize the functionality
class SpmBuilder {
 public:
  // Destructor of the SPM service handle
  virtual ~SpmBuilder() {}
  // Builds Pm4 command stream to program hardware registers that
  // enable a SPM session, including the issue of an event
  // to begin thread session
  virtual void Begin(CmdBuffer* cmd_buffer, const SpmConfig* config,
                     const counters_vector& counters_vec) = 0;
  // Builds Pm4 command stream to program hardware registers that
  // disable a SPM session, including the issue of an event
  // to stop currently ongoing thread session
  virtual void End(CmdBuffer* cmd_buffer, const SpmConfig* config) = 0;
};

template <typename Builder, typename Primitives>
class GpuSpmBuilder : public SpmBuilder, protected Primitives {
  typedef typename Primitives::mux_info_t mux_info_t;

  void DebugTrace(uint32_t value) {
    CmdBuffer cmd_buffer;
    uint32_t header[2] = {0, value};
    APPEND_COMMAND_WRAPPER((&cmd_buffer), header);
  }

  Builder builder;

 public:
  explicit GpuSpmBuilder(const AgentInfo* agent_info)
      : SpmBuilder(), builder(acquire_ip_offset_table(agent_info)) {}

  void Begin(CmdBuffer* cmd_buffer, const SpmConfig* config, const counters_vector& counters_vec) {
    // SPM parameters
    const uint32_t sampling_rate = config->sampleRate;
    const uint64_t buffer_ptr = reinterpret_cast<uint64_t>(config->data_buffer_ptr);
    const uint32_t buffer_size = config->data_buffer_size;

    // On Vega this is needed to collect Perf Cntrs: enable clock for performance counters
    if (Primitives::GFXIP_LEVEL == 9)
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_PERFMON_CLK_CNTL_ADDR, 1);

    // Program Grbm to broadcast messages to all shader engines
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR,
                                       Primitives::grbm_broadcast_value());
    // Issue a CSPartialFlush cmd including cache flush
    builder.BuildWriteWaitIdlePacket(cmd_buffer);
    // SPM counters reset
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR,
                                       Primitives::cp_perfmon_cntl_reset_value());

    // Initialize the [BLK]_SAMPLE_DLY_SEL registers
    // These registers are layout-dependent and allow all the blocks to receive
    // the sample signals on a specified cycle
    // global: CPC, CPF, GDS, TCC, TCA
    // SE: SX, TA, TD, TCP, SPI

    // Initialize the Performance Counter Ring Structure in memory
    // 1. Program the RLC_RING_BASE_H1/LO registers.
    // 2. Program the RLC_RING_SIZE register.
    // 3. Program the RLC_PERFMON_SEGMENT_SIZE register.
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_PERFMON_CNTL__ADDR,
                                       Primitives::rlc_spm_perfmon_cntl_value(sampling_rate));
    if (!config->spm_kfd_mode) {
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_PERFMON_RING_BASE_LO__ADDR,
                                         buffer_ptr);
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_PERFMON_RING_BASE_HI__ADDR,
                                         buffer_ptr >> 32);
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_PERFMON_RING_SIZE__ADDR,
                                         buffer_size);
    }

    // Setting VMID
    if (!config->spm_kfd_mode)
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_MC_CNTL__ADDR,
                                         Primitives::rlc_spm_mc_cntl_value());

    // Iterate through the list of blocks to create PM4 packets to read counter values
    // Below pair.first is the block id of a counter event and pair.second is the index into
    // counters_vec of the counter event
    std::vector<std::vector<std::pair<int, int> > > counter_info_even(Primitives::NUMBER_OF_BLOCKS);
    std::vector<std::vector<std::pair<int, int> > > counter_info_odd(Primitives::NUMBER_OF_BLOCKS);

    // distribute counter events to counter_info_even and counter_info_odd according to their block
    // id
    for (uint32_t index = 0; index < counters_vec.size(); ++index) {
      auto& counter_des = counters_vec[index];
      const auto& block_des = counter_des.block_des;

      if (block_des.id == Primitives::SQ_BLOCK_ID && config->spm_sq_32bit_mode) {
        counter_info_even[block_des.id].push_back({block_des.id, index});
        counter_info_odd[block_des.id].push_back({block_des.id, index});
      } else {
        if (counter_des.index % 2 == 0)
          counter_info_even[block_des.id].push_back({block_des.id, index});
        else
          counter_info_odd[block_des.id].push_back({block_des.id, index});
      }
    }

    // compute segment size for global(0) and se(1)
    uint32_t ss_even[2] = {};
    uint32_t ss_odd[2] = {};
    for (size_t i = 0; i < Primitives::NUMBER_OF_BLOCKS; ++i) {
      if (!counter_info_even[i].empty()) {
        const auto& counter_des = counters_vec[counter_info_even[i][0].second];
        const auto* block_info = counter_des.block_info;
        if (block_info->attr & CounterBlockSpmGlobalAttr) {
          ss_even[0] += counter_info_even[i].size();
        } else {
          ss_even[1] += counter_info_even[i].size();
        }
      }
      if (!counter_info_odd[i].empty()) {
        const auto& counter_des = counters_vec[counter_info_odd[i][0].second];
        const auto* block_info = counter_des.block_info;
        if (block_info->attr & CounterBlockSpmGlobalAttr)
          ss_odd[0] += counter_info_odd[i].size();
        else
          ss_odd[1] += counter_info_odd[i].size();
      }
    }

    // if SPM global is streamed we also stream time stamp.
    ss_even[0] += Primitives::RLC_SPM_TIMESTAMP_SIZE16;

    uint32_t ss[2] = {};
    for (int i = 0; i < 2; ++i) {
      ss_even[i] = ss_even[i] / Primitives::RLC_SPM_COUNTERS_PER_LINE +
                   uint32_t(ss_even[i] % Primitives::RLC_SPM_COUNTERS_PER_LINE > 0);
      ss_odd[i] = ss_odd[i] / Primitives::RLC_SPM_COUNTERS_PER_LINE +
                  uint32_t(ss_odd[i] % Primitives::RLC_SPM_COUNTERS_PER_LINE > 0);

      ss[i] = std::max(ss_even[i], ss_odd[i]) * 2;
    }

    // fill in mux_ram data according to even and odd arrays
    std::vector<mux_info_t> mux_ram[2];
    mux_info_t mxinf = {0xFFFF};

    // global mux_ram: initialize with all 0xFFFF.
    mux_ram[0].resize(ss[0] * Primitives::RLC_SPM_COUNTERS_PER_LINE + 2);
    std::fill(mux_ram[0].begin(), mux_ram[0].end(), mxinf);

    // se mux_ram: initialize with all 0xFFFF (end of muxsel).
    mux_ram[1].resize(ss[1] * Primitives::RLC_SPM_COUNTERS_PER_LINE + 2);
    std::fill(mux_ram[1].begin(), mux_ram[1].end(), mxinf);

    size_t even_idx = 0;
    size_t odd_idx = Primitives::RLC_SPM_COUNTERS_PER_LINE;
    // follow the exact steps to fill in mux_ram as when the number of even/odd events are counted
    // Register timestamp
    mxinf.data = Primitives::spm_timestamp_muxsel();
    for (even_idx = 0; even_idx < Primitives::RLC_SPM_TIMESTAMP_SIZE16; ++even_idx) {
      mux_ram[0][even_idx] = mxinf;
    }
    // fill in global mux_sram after global time stamp
    for (size_t j = 0; j < Primitives::NUMBER_OF_BLOCKS; ++j) {
      if (!counter_info_even[j].empty()) {
        const auto& counter_des = counters_vec[counter_info_even[j][0].second];
        const auto* block_info = counter_des.block_info;
        if (block_info->attr & CounterBlockSpmGlobalAttr) {
          for (size_t k = 0; k < counter_info_even[j].size(); ++k) {
            const auto& counter_des = counters_vec[counter_info_even[j][k].second];
            mux_ram[0][even_idx] = Primitives::spm_mux_ram_value(counter_des);
            even_idx = Primitives::spm_mux_ram_idx_incr(even_idx);
          }
          for (size_t k = 0; k < counter_info_odd[j].size(); ++k) {
            const auto& counter_des = counters_vec[counter_info_odd[j][k].second];
            mux_ram[0][odd_idx] = Primitives::spm_mux_ram_value(counter_des);
            odd_idx = Primitives::spm_mux_ram_idx_incr(odd_idx);
          }
        }
      }
    }
    // fill in SE mux_ram
    even_idx = 0;
    odd_idx = Primitives::RLC_SPM_COUNTERS_PER_LINE;
    for (size_t j = 0; j < Primitives::NUMBER_OF_BLOCKS; ++j) {
      // Use this code to do 32-bit SQ profiling
      if (j == Primitives::SQ_BLOCK_ID && config->spm_sq_32bit_mode) {
        for (size_t k = 0; k < counter_info_even[j].size(); ++k) {
          const auto& counter_des = counters_vec[counter_info_even[j][k].second];
          const auto counter = uint16_t(counter_des.index) * 2;
          const auto block = Primitives::SQ_BLOCK_SPM_ID;
          const auto instance = uint16_t(counter_des.block_des.index);
          mux_ram[1][even_idx] = Primitives::spm_mux_ram_value(counter, block, instance);
          even_idx = Primitives::spm_mux_ram_idx_incr(even_idx);
        }
        for (size_t k = 0; k < counter_info_odd[j].size(); ++k) {
          const auto& counter_des = counters_vec[counter_info_odd[j][k].second];
          const auto counter = uint16_t(counter_des.index) * 2 + 1;
          const auto block = Primitives::SQ_BLOCK_SPM_ID;
          const auto instance = uint16_t(counter_des.block_des.index);
          mux_ram[1][odd_idx] = Primitives::spm_mux_ram_value(counter, block, instance);
          // fix corrupted upper 16-bit by setting its mux_sel to be 0x0
          mux_ram[1][odd_idx] = mux_info_t{0x0};
          odd_idx = Primitives::spm_mux_ram_idx_incr(odd_idx);
        }
      } else {
        if (!counter_info_even[j].empty()) {
          const auto& counter_des = counters_vec[counter_info_even[j][0].second];
          const auto* block_info = counter_des.block_info;
          if (!(block_info->attr & CounterBlockSpmGlobalAttr)) {
            for (size_t k = 0; k < counter_info_even[j].size(); ++k) {
              const auto& counter_des = counters_vec[counter_info_even[j][k].second];
              mux_ram[1][even_idx] = Primitives::spm_mux_ram_value(counter_des);
              even_idx = Primitives::spm_mux_ram_idx_incr(even_idx);
            }
            for (size_t k = 0; k < counter_info_odd[j].size(); ++k) {
              const auto& counter_des = counters_vec[counter_info_odd[j][k].second];
              mux_ram[1][odd_idx] = Primitives::spm_mux_ram_value(counter_des);
              odd_idx = Primitives::spm_mux_ram_idx_incr(odd_idx);
            }
          }
        }
      }
    }

    for (const auto& counter_des : counters_vec) {
      const auto* block_info = counter_des.block_info;
      const auto& reg_info = block_info->counter_reg_info[counter_des.index];

      bool is_spm_inited = false;
      if (is_spm_inited == false) {
        is_spm_inited = true;
        if (block_info->attr & CounterBlockSpmGlobalAttr) {
          // for each instance of a global block we progam its delay
          for (size_t j = 0; j < block_info->instance_count; ++j) {
            builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR,
                                               Primitives::grbm_inst_se_sh_index_value(j, 0, 0));
            builder.BuildWriteUConfigRegPacket(cmd_buffer, block_info->delay_info[j].reg,
                                               Primitives::get_spm_global_delay(counter_des, j));
          }
        } else {
          for (size_t i = 0; i < config->spm_se_number_total; ++i) {
            for (size_t j = 0; j < block_info->instance_count; ++j) {
              int delay_index = i * block_info->instance_count + j;
              builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR,
                                                 Primitives::grbm_inst_se_index_value(j, i));
              builder.BuildWriteUConfigRegPacket(cmd_buffer,
                                                 block_info->delay_info[delay_index].reg,
                                                 Primitives::get_spm_se_delay(counter_des, i, j));
            }
          }
        }
        builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR,
                                           Primitives::grbm_broadcast_value());
      }

      // 4. Program the Block instance streaming performance counters in order to specify which
      // items
      //    (events) the counters should count, if any. This is done by programming the
      //    GRBM_GFX_INDEX register to specify the type of access (broadcast or instance specific)
      //    followed by the actual register value. The first step may be to clear all counters of
      //    all instances to select zero (no counting). Then program the GRBM_GFX_INDEX, followed by
      //    the [BLK]_STRMPERFMON_SELECTx register.
      // Setup counters
      // Configure SQ block
      if (block_info->attr & CounterBlockSqAttr) {
        builder.BuildWriteUConfigRegPacket(cmd_buffer, reg_info.select_addr,
                                           Primitives::sq_spm_select_value(counter_des));
        builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::SQ_PERFCOUNTER_MASK_ADDR,
                                           Primitives::sq_mask_value(counter_des));
        builder.BuildWriteUConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           Primitives::sq_control_value(counter_des));
      }
    }

    for (size_t i = 0; i < Primitives::NUMBER_OF_BLOCKS; ++i) {
      if (i == Primitives::SQ_BLOCK_ID) continue;

      for (size_t j = 0; j < counter_info_even[i].size(); ++j) {
        // get 16-bit SPM select value for even counters
        const auto& counter_des = counters_vec[counter_info_even[i][j].second];
        uint32_t spm_select_value = Primitives::spm_even_select_value(counter_des);

        if (j + 1 <= counter_info_odd[i].size()) {
          const auto& counter_des = counters_vec[counter_info_odd[i][j].second];
          spm_select_value |= Primitives::spm_odd_select_value(counter_des);
        }

        const auto* block_info = counter_des.block_info;
        int index = j >> 1;
        int offset = j % 2;
        uint32_t spm_select_addr =
            builder.get_addr(block_info->counter_reg_info[index].select_addr) + offset;
        builder.BuildWriteUConfigRegPacket(
            cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR,
            Primitives::grbm_inst_index_value(counter_des.block_des.index));
        builder.BuildWriteConfigRegPacket(cmd_buffer, spm_select_addr, spm_select_value);
      }
    }

    // Set segment size
    uint32_t global_count = ss[0];
    uint32_t se_count = ss[1];
    builder.BuildWriteUConfigRegPacket(
        cmd_buffer, Primitives::RLC_SPM_PERFMON_SEGMENT_SIZE__ADDR,
        Primitives::rlc_spm_perfmon_segment_size_value(global_count, se_count));
    if (config->mi100) {
      builder.BuildWriteUConfigRegPacket(
          cmd_buffer, Primitives::RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__ADDR,
          Primitives::rlc_spm_perfmon_segment_size_core1_value(se_count));
    }
    // Finish MUXSEL RAM
    // 5. Program the RLC_[GLOBAL/SE]_MUXSEL_ADDR register with the starting address, likely zero.
    if (!mux_ram[0].empty()) {
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR,
                                         0);
      builder.BuildWriteRegDataPacket(cmd_buffer, Primitives::RLC_SPM_GLOBAL_MUXSEL_DATA__ADDR,
                                      reinterpret_cast<uint32_t*>(mux_ram[0].data()),
                                      mux_ram[0].size() / 2, 1);
    }
    if (!mux_ram[1].empty()) {
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_SE_MUXSEL_ADDR__ADDR, 0);
      builder.BuildWriteRegDataPacket(cmd_buffer, Primitives::RLC_SPM_SE_MUXSEL_DATA__ADDR,
                                      reinterpret_cast<uint32_t*>(mux_ram[1].data()),
                                      mux_ram[1].size() / 2, 1);
    }
    // pm4SPM code has the following code
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR, 0);
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_SPM_SE_MUXSEL_ADDR__ADDR, 0);

    // Issue a CSPartialFlush cmd including cache flush
    builder.BuildWriteWaitIdlePacket(cmd_buffer);
    // Program Compute Perfcount Enable register to support perf counting
    builder.BuildWriteShRegPacket(cmd_buffer, Primitives::COMPUTE_PERFCOUNT_ENABLE_ADDR,
                                  Primitives::cp_perfcount_enable_value());
    // SPM counters start
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR,
                                       Primitives::cp_perfmon_cntl_spm_start_value());
    // Issue a CSPartialFlush cmd including cache flush
    builder.BuildWriteWaitIdlePacket(cmd_buffer);
  }

  void End(CmdBuffer* cmd_buffer, const SpmConfig* config) {
    // Program Grbm to broadcast messages to all shader engines
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR,
                                       Primitives::grbm_broadcast_value());
    // Issue a CSPartialFlush cmd including cache flush
    builder.BuildWriteWaitIdlePacket(cmd_buffer);
    // SPM counters stop
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR,
                                       Primitives::cp_perfmon_cntl_spm_stop_value());
    // SPM counters reset
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR,
                                       Primitives::cp_perfmon_cntl_reset_value());
    // On Vega this disable clock for performance counters
    if (Primitives::GFXIP_LEVEL == 9)
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_PERFMON_CLK_CNTL_ADDR, 0);
  }
};

}  // namespace pm4_builder

#endif  // SRC_PM4_SPM_BUILDER_H_
