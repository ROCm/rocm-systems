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

#include "pm4/gfx12/gfx12_spm_builder.hpp"

namespace pm4_builder {

void Gfx12SpmBuilder::Begin(CmdBuffer* cmd_buffer, const SpmConfig* config,
                             const counters_vector& counters_vec) {
  // SPM parameters
  const uint32_t sampling_rate = config->sampleRate;
  const uint64_t buffer_ptr = reinterpret_cast<uint64_t>(config->data_buffer_ptr);
  const uint32_t buffer_size = config->data_buffer_size;

  // Initialize SPM counter buffer metadata.
  // counter_map takes the index of counters_vector as input, and output an index to
  // the 16bit SPM counter buffer
  SpmBufferDesc* spm_buffer_desc = (SpmBufferDesc*)config->data_buffer_ptr;
  spm_buffer_desc->version = 1;
  uint16_t* counter_map = spm_buffer_desc->get_counter_map();
  memset(counter_map, 0, SPM_DESC_SIZE - sizeof(SpmBufferDesc));

  // On Vega this is needed to collect Perf Cntrs: enable clock for performance counters
  if (prim_->GetGfxipLevel() == 9)
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcPerfmonClkCntlAddr(), 1);

  // Program Grbm to broadcast messages to all shader engines
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                       prim_->GrbmBroadcastValue());
  // Issue a CSPartialFlush cmd including cache flush
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);

  // SPM counters stop
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlSpmStopValue());

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
  // builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
  //                                      prim_->CpPerfmonCntlResetValue());

  // Issue a CSPartialFlush cmd including cache flush
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);

  // Hardcode PERFMON_RING_MODE to 3 (Stall and send interrupt) to match KFD
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcSpmPerfmonCntlAddr(),
                                       prim_->RlcSpmPerfmonCntlValue(sampling_rate));

  // Iterate through the list of blocks to create PM4 packets to read counter values
  // Below pair.first is the block id of a counter event and pair.second is the index into
  // counters_vec of the counter event
  const uint32_t num_blocks = prim_->GetNumberOfBlocks();
  std::vector<std::vector<std::pair<int, int>>> counter_info_even(num_blocks);
  std::vector<std::vector<std::pair<int, int>>> counter_info_odd(num_blocks);

  // distribute counter events to counter_info_even and counter_info_odd according to their block
  // id
  for (uint32_t index = 0; index < counters_vec.size(); ++index) {
    auto& counter_des = counters_vec[index];
    const auto& block_des = counter_des.block_des;

    if (block_des.id == prim_->GetSqBlockId() && config->spm_sq_32bit_mode) {
      counter_info_even[block_des.id].push_back({block_des.id, index});
      counter_info_odd[block_des.id].push_back({block_des.id, index});
    } else {
      if (counter_des.index % 2 == 0)
        counter_info_even[block_des.id].push_back({block_des.id, index});
      else
        counter_info_odd[block_des.id].push_back({block_des.id, index});
    }
  }

  // Sort counter_info_even and counter_info_odd by instance
  auto compare = [&counters_vec](std::pair<int, int> a, std::pair<int, int> b) {
    auto index_a = a.second;
    auto index_b = b.second;
    auto& counter_des_a = counters_vec[index_a];
    auto& counter_des_b = counters_vec[index_b];
    return (counter_des_a.block_des.index < counter_des_b.block_des.index) ||
           ((counter_des_a.block_des.index == counter_des_b.block_des.index) &&
            (counter_des_a.index < counter_des_b.index));
  };
  for (size_t i = 0; i < num_blocks; ++i) {
    if (!counter_info_even[i].empty()) {
      sort(counter_info_even[i].begin(), counter_info_even[i].end(), compare);
    }
    if (!counter_info_odd[i].empty()) {
      sort(counter_info_odd[i].begin(), counter_info_odd[i].end(), compare);
    }
  }

  // compute segment size for global(0) and se(1)
  uint32_t ss_even[2] = {};
  uint32_t ss_odd[2] = {};
  for (size_t i = 0; i < num_blocks; ++i) {
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

  const uint32_t rlc_spm_timestamp_size16 = prim_->GetRlcSpmTimestampSize16();
  const uint32_t rlc_spm_counters_per_line = prim_->GetRlcSpmCountersPerLine();

  // if SPM global is streamed we also stream time stamp.
  ss_even[0] += rlc_spm_timestamp_size16;

  uint32_t ss[2] = {};
  for (int i = 0; i < 2; ++i) {
    ss_even[i] = ss_even[i] / rlc_spm_counters_per_line +
                 uint32_t(ss_even[i] % rlc_spm_counters_per_line > 0);
    ss_odd[i] = ss_odd[i] / rlc_spm_counters_per_line +
                uint32_t(ss_odd[i] % rlc_spm_counters_per_line > 0);

    ss[i] = std::max(ss_even[i], ss_odd[i]) * 2;
  }

  // fill in mux_ram data according to even and odd arrays
  std::vector<uint16_t> mux_ram[2];

  // global mux_ram: initialize with all 0xFFFF.
  mux_ram[0].resize(ss[0] * rlc_spm_counters_per_line + 2, 0xFFFF);

  // se mux_ram: initialize with all 0xFFFF (end of muxsel).
  mux_ram[1].resize(ss[1] * rlc_spm_counters_per_line + 2, 0xFFFF);

  size_t even_idx = 0;
  size_t odd_idx = rlc_spm_counters_per_line;
  // follow the exact steps to fill in mux_ram as when the number of even/odd events are counted
  // Register timestamp
  for (even_idx = 0; even_idx < rlc_spm_timestamp_size16; ++even_idx) {
    mux_ram[0][even_idx] = prim_->SpmTimestampMuxsel();
  }
  // fill in global mux_sram after global time stamp
  for (size_t j = 0; j < num_blocks; ++j) {
    if (!counter_info_even[j].empty()) {
      const auto& counter_des = counters_vec[counter_info_even[j][0].second];
      const auto* block_info = counter_des.block_info;
      if (block_info->attr & CounterBlockSpmGlobalAttr) {
        for (size_t k = 0; k < counter_info_even[j].size(); ++k) {
          const auto index = counter_info_even[j][k].second;
          const auto& counter_des = counters_vec[index];
          mux_ram[0][even_idx] = prim_->SpmMuxRamValue(counter_des);
          counter_map[index] = even_idx | 0x8000;
          even_idx = prim_->SpmMuxRamIdxIncr(even_idx);
        }
        for (size_t k = 0; k < counter_info_odd[j].size(); ++k) {
          const auto index = counter_info_odd[j][k].second;
          const auto& counter_des = counters_vec[index];
          mux_ram[0][odd_idx] = prim_->SpmMuxRamValue(counter_des);
          counter_map[index] = odd_idx | 0x8000;
          odd_idx = prim_->SpmMuxRamIdxIncr(odd_idx);
        }
      }
    }
  }
  // fill in SE mux_ram
  even_idx = 0;
  odd_idx = rlc_spm_counters_per_line;
  const uint32_t sq_block_id = prim_->GetSqBlockId();
  const uint32_t sq_block_spm_id = prim_->GetSqBlockSpmId();
  for (size_t j = 0; j < num_blocks; ++j) {
    // Use this code to do 32-bit SQ profiling
    if (j == sq_block_id && config->spm_sq_32bit_mode) {
      for (size_t k = 0; k < counter_info_even[j].size(); ++k) {
        const auto index = counter_info_even[j][k].second;
        const auto& counter_des = counters_vec[index];
        const auto counter = uint16_t(counter_des.index) * 2;
        const auto block = static_cast<uint16_t>(sq_block_spm_id);
        const auto instance = uint16_t(counter_des.block_des.index);
        mux_ram[1][even_idx] = prim_->SpmMuxRamValue(counter, block, instance);
        counter_map[index] = even_idx;
        even_idx = prim_->SpmMuxRamIdxIncr(even_idx);
      }
      for (size_t k = 0; k < counter_info_odd[j].size(); ++k) {
        const auto index = counter_info_odd[j][k].second;
        const auto& counter_des = counters_vec[index];
        const auto counter = uint16_t(counter_des.index) * 2 + 1;
        const auto block = static_cast<uint16_t>(sq_block_spm_id);
        const auto instance = uint16_t(counter_des.block_des.index);
        mux_ram[1][odd_idx] = prim_->SpmMuxRamValue(counter, block, instance);
        // fix corrupted upper 16-bit by setting its mux_sel to be 0x0
        mux_ram[1][odd_idx] = 0x0;
        odd_idx = prim_->SpmMuxRamIdxIncr(odd_idx);
      }
    } else {
      if (!counter_info_even[j].empty()) {
        const auto& counter_des = counters_vec[counter_info_even[j][0].second];
        const auto* block_info = counter_des.block_info;
        if (!(block_info->attr & CounterBlockSpmGlobalAttr)) {
          for (size_t k = 0; k < counter_info_even[j].size(); ++k) {
            const auto index = counter_info_even[j][k].second;
            const auto& counter_des = counters_vec[index];
            mux_ram[1][even_idx] = prim_->SpmMuxRamValue(counter_des);
            counter_map[index] = even_idx;
            even_idx = prim_->SpmMuxRamIdxIncr(even_idx);
          }
          for (size_t k = 0; k < counter_info_odd[j].size(); ++k) {
            const auto index = counter_info_odd[j][k].second;
            const auto& counter_des = counters_vec[index];
            mux_ram[1][odd_idx] = prim_->SpmMuxRamValue(counter_des);
            counter_map[index] = odd_idx;
            odd_idx = prim_->SpmMuxRamIdxIncr(odd_idx);
          }
        }
      }
    }
  }

  if (config->spm_sample_delay_max) {
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                         prim_->GrbmBroadcastValue());
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcSpmPerfmonSampleDelayMax(),
                                         config->spm_sample_delay_max);
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
          builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                               prim_->GrbmInstSeShIndexValue(j, 0, 0));
          builder_->BuildWriteUConfigRegPacket(cmd_buffer, block_info->delay_info.reg,
                                               prim_->GetSpmGlobalDelay(counter_des, j));
        }
      } else {
        for (size_t i = 0; i < config->se_number; ++i) {
          for (size_t j = 0; j < block_info->instance_count; ++j) {
            builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                                 prim_->GrbmInstSeIndexValue(j, i));
            builder_->BuildWriteUConfigRegPacket(cmd_buffer, block_info->delay_info.reg,
                                                 prim_->GetSpmSeDelay(counter_des, i, j));
          }
        }
      }
      builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                           prim_->GrbmBroadcastValue());
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
      builder_->BuildWriteUConfigRegPacket(cmd_buffer, reg_info.select_addr,
                                           prim_->SqSpmSelectValue(counter_des));
      builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqPerfcounterMaskAddr(),
                                           prim_->SqMaskValue(counter_des));
      builder_->BuildWriteUConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->SqControlValue(counter_des));
    }
  }

  for (size_t i = 0; i < num_blocks; ++i) {
    if (i == sq_block_id) continue;

    int instance = 0;
    int je, jo, j;  // je & jo store even/odd array index, j stores index of counter registers
    for (je = jo = j = 0; je < (int)counter_info_even[i].size(); ++je, ++j) {
      // get 16-bit SPM select value for even counters
      const auto& counter_des = counters_vec[counter_info_even[i][je].second];
      uint32_t spm_select_value = prim_->SpmEvenSelectValue(counter_des);
      if (counter_des.block_des.index != instance) {
        instance = counter_des.block_des.index;
        // Reset counter register index when instance switches
        j = 0;
      }

      // get 16-bit SPM select value for odd counters
      if (jo < (int)counter_info_odd[i].size()) {
        const auto& counter_des = counters_vec[counter_info_odd[i][jo].second];
        if (counter_des.block_des.index == instance) {
          spm_select_value |= prim_->SpmOddSelectValue(counter_des);
          jo++;
        }
      }

      const auto* block_info = counter_des.block_info;
      int index = j >> 1;
      int select = j % 2;
      Register spm_select_addr = (select == 0) ?
          block_info->counter_reg_info[index].select_addr :
          block_info->counter_reg_info[index].select1_addr;
      builder_->BuildWriteUConfigRegPacket(
          cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
          prim_->GrbmInstIndexValue(counter_des.block_des.index));
      builder_->BuildWriteConfigRegPacket(cmd_buffer, spm_select_addr, spm_select_value);
    }
  }
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                        prim_->GrbmBroadcastValue());

  // Set segment size
  uint32_t global_count = ss[0];
  uint32_t se_count = ss[1];
  builder_->BuildWriteUConfigRegPacket(
      cmd_buffer, prim_->GetRlcSpmPerfmonSegmentSize(),
      prim_->RlcSpmPerfmonSegmentSizeValue(global_count, se_count));
  if (config->spm_has_core1) {
    builder_->BuildWriteUConfigRegPacket(
        cmd_buffer, prim_->GetRlcSpmPerfmonSegmentSizeCore1(),
        prim_->RlcSpmPerfmonSegmentSizeCore1Value(se_count));
  }
  spm_buffer_desc->global_num_line = global_count;
  spm_buffer_desc->se_num_line = se_count;
  spm_buffer_desc->num_se = config->se_number;
  spm_buffer_desc->num_sa = config->sa_number;
  spm_buffer_desc->num_xcc = config->xcc_number;
  spm_buffer_desc->num_events = counters_vec.size();

  // Finish MUXSEL RAM
  // 5. Program the RLC_[GLOBAL/SE]_MUXSEL_ADDR register with the starting address, likely zero.
  if (!mux_ram[0].empty()) {
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcSpmGlobalMuxselAddr(), 0);
    builder_->BuildWriteRegDataPacket(cmd_buffer, prim_->GetRlcSpmGlobalMuxselData(),
                                      reinterpret_cast<uint32_t*>(mux_ram[0].data()),
                                      mux_ram[0].size() / 2, 1);
  }
  if (!mux_ram[1].empty()) {
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcSpmSeMuxselAddr(), 0);
    builder_->BuildWriteRegDataPacket(cmd_buffer, prim_->GetRlcSpmSeMuxselData(),
                                      reinterpret_cast<uint32_t*>(mux_ram[1].data()),
                                      mux_ram[1].size() / 2, 1);
  }
  // pm4SPM code has the following code
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcSpmGlobalMuxselAddr(), 0);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcSpmSeMuxselAddr(), 0);

  // Issue a CSPartialFlush cmd including cache flush
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
  // Program Compute Perfcount Enable register to support perf counting
  builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputePerfcountEnableAddr(),
                                  prim_->CpPerfcountEnableValue());
  // SPM counters start
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlSpmStartValue());
  // Issue a CSPartialFlush cmd including cache flush
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

void Gfx12SpmBuilder::End(CmdBuffer* cmd_buffer, const SpmConfig* config) {
  // Program Grbm to broadcast messages to all shader engines
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(),
                                       prim_->GrbmBroadcastValue());
  // Issue a CSPartialFlush cmd including cache flush
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
  // SPM counters stop
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlSpmStopValue());
  // SPM counters reset
  // 'SPM counters reset' must be done in KFD. See comments in Begin() for more details
  //
  // builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
  //                                      prim_->CpPerfmonCntlResetValue());

  // On Vega this disable clock for performance counters
  if (prim_->GetGfxipLevel() == 9)
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcPerfmonClkCntlAddr(), 0);
}

}  // namespace pm4_builder
