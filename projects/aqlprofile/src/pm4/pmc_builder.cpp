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

#include "pm4/pmc_builder.h"

#include <cstring>

#include "pm4/cmd_builder.h"

namespace pm4_builder {

// PrecExecBuilder

PrecExecBuilder::PrecExecBuilder(CmdBuilder& _builder, CmdBuffer* cmd_buffer, uint32_t target_xcc,
                                 bool is_mi300)
    : cmd_buffer_(cmd_buffer), builder_(_builder), is_mi300_(is_mi300), target_xcc_(target_xcc) {
  if (is_mi300_) {
    // PRED_EXEC aplies to MI300 only
    pos_ = cmd_buffer->DwSize();
    builder_.BuildPredExecPacket(cmd_buffer, target_xcc_, 0);
    initial_buff_size_ = cmd_buffer->DwSize();
  }
}

PrecExecBuilder::~PrecExecBuilder() {
  if (is_mi300_) {
    // PRED_EXEC aplies to MI300 only
    CmdBuffer pred_exec;
    // update first PRED_EXEC packet to its correct value
    builder_.BuildPredExecPacket(&pred_exec, target_xcc_, cmd_buffer_->DwSize() - initial_buff_size_);
    const uint32_t* data = (const uint32_t*)pred_exec.Data();

    for (size_t i = 0; i < pred_exec.DwSize(); ++i) cmd_buffer_->Assign(pos_ + i, data[i]);
  }
}

// GpuPmcBuilder

GpuPmcBuilder::GpuPmcBuilder(const aql_profile::HardwareConfig& config, CmdBuilder* builder,
                              const PrimitivesProvider* prim, bool is_concurrent)
    : PmcBuilder(),
      builder_(builder),
      prim_(prim),
      is_concurrent_(is_concurrent),
      se_number_(config.se_count / config.xcc_count),
      xcc_number_(config.xcc_count),
      xcc_per_aid_(config.xcc_per_aid),
      is_multi_xcc_(config.IsMultiXCC()),
      aid_count_(config.aid_count),
      sarrays_per_se_(config.sa_per_se_count) {
  this->wgp_per_sa_ =
      (config.cu_count / 2 + sarrays_per_se_ * se_number_ - 1) / (se_number_ * sarrays_per_se_);
  this->wgp_per_sa_ /= config.xcc_count;
  // Due to MI300 CP firmware issue we need to use mem_mapped_register mode to patch for GCEA
  // hang. Otherwise both perfcounters mode and mem_mapped_register mode should work.
  builder_->bUsePerfCounterMode = !is_multi_xcc_;
  this->asymmetric_cu_patch = config.has_asymmetric_cu_design;
}

int GpuPmcBuilder::GetNumWGPs() {
  if (prim_->GetGfxipLevel() >= 11) return wgp_per_sa_;
  return 1;
}

void GpuPmcBuilder::DebugTrace(uint32_t value) {
  CmdBuffer cmd_buffer;
  uint32_t header[2] = {0, value};
  APPEND_COMMAND_WRAPPER((&cmd_buffer), header);
}

const CounterRegInfo* GpuPmcBuilder::get_reg_table(const counter_des_t& counter_des) {
  const auto* block_info = counter_des.block_info;
  const auto& block_des = counter_des.block_des;
  auto base_index = block_des.index;
  if ((block_info->attr & CounterBlockAidAttr) && is_multi_xcc_)
    // MI300 all AID style instances fold back to per AID counter_reg_info
    base_index %= (block_info->instance_count / MAX_AID);
  base_index =
      (block_info->attr & CounterBlockExplInstAttr) ? base_index * block_info->counter_count : 0;
  return &(block_info->counter_reg_info[base_index]);
}

uint32_t GpuPmcBuilder::GetAidNumber() const {
  return aid_count_;
}

uint32_t GpuPmcBuilder::GetTargetAid(const counter_des_t& counter_des) const {
  const auto num_aid = GetAidNumber();
  const auto num_instance = counter_des.block_info->instance_count;
  const auto num_instance_per_aid = num_instance / num_aid;
  const auto instance_index = counter_des.block_des.index;
  const auto target_aid_index = instance_index / num_instance_per_aid;

  return target_aid_index;
}

uint64_t GpuPmcBuilder::get_smn_addr(uint64_t addr, uint32_t target_aid_index, bool use_aid) {
  if (is_multi_xcc_ && use_aid)
    addr |= ((uint64_t)1 << UMC_USR_BIT) | ((uint64_t)target_aid_index << UMC_AID_BIT);
  return addr;
}

uint64_t GpuPmcBuilder::get_smn_addr(const Register& reg, uint32_t target_aid_index,
                                     bool use_aid) {
  return get_smn_addr(builder_->get_addr(reg), target_aid_index, use_aid);
}

void GpuPmcBuilder::start_generic_mc_counters(CmdBuffer* cmd_buffer,
                                              const std::set<uint64_t>& instances, bool use_aid) {
  // insert master XCC PRED_EXEC packet here if it is MI300
  PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                    is_multi_xcc_ && use_aid);
  for (const auto& control_addr : instances) {
    // rpb instance clear
    builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr, prim_->McResetValue());
    // rpb instance enable
    builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr, prim_->McStartValue());
  }
}

void GpuPmcBuilder::SetGrbmGfxIndex(CmdBuffer* cmd_buffer, uint32_t value, GCMode gc_mode) {
  uint32_t xcc_id;
  if (gc_mode & GC_MODE_XCD)
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(), value);
  if (gc_mode & GC_MODE_AID) {
    for (xcc_id = 0; xcc_id < xcc_number_; xcc_id += xcc_per_aid_) {
      PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, xcc_id, is_multi_xcc_);
      builder_->BuildWritePConfigRegPacketToChiplet(cmd_buffer, prim_->GetGrbmaGfxIndexAddr(),
                                                    value, static_cast<ChipletId>(xcc_id));
    }
  }
  if (gc_mode & GC_MODE_AID_WITH_XCD_INDEX) {
    xcc_id = gc_mode & GC_MODE_XCD_ID_MASK;
    // We don't need PrecExec for this case because the caller will program PrecExec
    builder_->BuildWritePConfigRegPacketToChiplet(cmd_buffer, prim_->GetGrbmaGfxIndexAddr(), value,
                                                  static_cast<ChipletId>(xcc_id));
  }
}

void GpuPmcBuilder::SetGrbmBroadcast(CmdBuffer* cmd_buffer, GCMode gc_mode) {
  SetGrbmGfxIndex(cmd_buffer, prim_->GrbmBroadcastValue(), gc_mode);
}

void GpuPmcBuilder::SetPerfmonCntl(CmdBuffer* cmd_buffer, uint32_t value, uint32_t attr) {
  if (attr & CounterBlockCpmonAttr)
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(), value);
  if (attr & CounterBlockGrbmaAttr) {
    for (uint32_t xcc_id = 0; xcc_id < xcc_number_; xcc_id += xcc_per_aid_) {
      PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, xcc_id, is_multi_xcc_);
      builder_->BuildWritePConfigRegPacketToChiplet(cmd_buffer, prim_->GetAidPerfmonCntlAddr(),
                                                    value, static_cast<ChipletId>(xcc_id));
    }
  }
}

uint32_t GpuPmcBuilder::GetInstanceIndex(uint32_t instance_index, const GpuBlockInfo* block_info) {
  // GLARB blocks require special instance handling, so we encode instance_count into
  // instance_index. This won't impact GPUs without GLARB blocks
  return (block_info->attr & CounterBlockGlarbAttr)
             ? (instance_index | (block_info->instance_count << 16))
             : instance_index;
}

// Build PMC enable PM4 comands - enable CP counting for a specific queue
void GpuPmcBuilder::Enable(CmdBuffer* cmd_buffer) {
  // Program Compute Perfcount Enable register to support perf counting
  builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputePerfcountEnableAddr(),
                                  prim_->CpPerfcountEnableValue());
}

// Build PMC disable PM4 comands - enable CP counting for a specific queue
void GpuPmcBuilder::Disable(CmdBuffer* cmd_buffer) {
  // Program Compute Perfcount Enable register to support perf counting
  builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputePerfcountEnableAddr(),
                                  prim_->CpPerfcountDisableValue());
}

// Build PMC waite-idle PM4 comands - enable CP counting for a specific queue
void GpuPmcBuilder::WaitIdle(CmdBuffer* cmd_buffer) {
  // Program Compute Perfcount WaiteIdle register to support perf counting
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

// Build PMC start PM4 comands
void GpuPmcBuilder::Start(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) {
  // Issue barrier command
  if (!is_concurrent_) builder_->BuildWriteWaitIdlePacket(cmd_buffer);
  GCMode gc_mode_global = (counters_vec.get_attr() & CounterBlockGrbmaAttr) ? GC_MODE_ALL : GC_MODE_XCD;
  // Reset Grbm to its default state - broadcast
  SetGrbmBroadcast(cmd_buffer, gc_mode_global);
  // Disable RLC Perfmon Clock Gating
  // On Vega this is needed to collect Perf Cntrs
  if (prim_->GetGfxipLevel() == 9)
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcPerfmonClkCntlAddr(), 1);
  // Reset perf counters
  SetPerfmonCntl(cmd_buffer, prim_->CpPerfmonCntlResetValue(), counters_vec.get_attr());
  // Enable SQ Counter Control enable perfomance counter in graphics pipeline if implied
  prim_->ValidateCounters(counters_vec.get_attr());
  if (counters_vec.get_attr() & CounterBlockTcAttr) {
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqPerfcounterCtrlAddr(),
                                         prim_->SqControlEnableValue());
  }
  if (prim_->GetGfxipLevel() >= 11 &&
      (counters_vec.get_attr() & (CounterBlockTcAttr | CounterBlockSqAttr))) {
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqPerfcounterCtrl2Addr(),
                                         prim_->SqControl2EnableValue());
  }
  // Clear and enable GUS counters
  if (prim_->SupportsGusCounters()) {
    if (counters_vec.get_attr() & CounterBlockGusAttr) {
      builder_->BuildWriteConfigRegPacket(cmd_buffer, prim_->GetGusRsltCntlAddr(),
                                          prim_->GusDisableClearValue());
      builder_->BuildWriteConfigRegPacket(cmd_buffer, prim_->GetGusRsltCntlAddr(),
                                          prim_->GusStartValue());
    }
  }

  // SDMA mask
  // sdma performance monitor control value accumulator
  std::vector<std::pair<reg_addr_t, uint32_t>> sdma_select_accumulator(
      prim_->GetSdmaCounterBlockNumInstances());
  uint32_t sdma_mask = 0;
  std::map<uint32_t, uint64_t> sdmas;
  bool is_mi100 = false;
  // UMC channels and their control register (for enable/disable) per channel
  std::map<uint32_t, uint64_t> umcchs;
  // RPB/ATC are per AID block like UMC above, we save its control register (for enable/disable)
  // per AID instance
  std::set<uint64_t> rpbs;
  std::set<uint64_t> atcs;
  std::set<uint64_t> perf_cnt;
  // Programming perf counters
  for (const auto& counter_des : counters_vec) {
    const auto* block_info = counter_des.block_info;
    const auto& block_des = counter_des.block_des;
    const auto* reg_table = get_reg_table(counter_des);
    const auto& reg_info = reg_table[counter_des.index];

    if (SPISkip(block_info->attr, counter_des.id)) {
      continue;
    }

    // std:: cout << std::hex << "block id("<<block_des.id<<") index("<<block_des.index<<")
    // counter id ("<<counter_des.id
    //             <<") index("<<counter_des.index<<") sel-addr("<<reg_info.select_addr<<")" <<
    //             std::endl;

    // Set GRBM index to access proper block instance
    //
    // TODO: In order to get different event for different instance with WGP counter blocks, we
    //       need to loop through WGP instead of blindly broadcast instance. Fortunately, this
    //       is not a common practice
    const uint32_t grbm_value =
        (block_info->instance_count > 1 && !(block_info->attr & CounterBlockWgpAttr))
            ? prim_->GrbmInstIndexValue(GetInstanceIndex(block_des.index, block_info))
            : prim_->GrbmBroadcastValue();
    GCMode gc_mode = (block_info->attr & CounterBlockGrbmaAttr) ? GC_MODE_ALL : GC_MODE_XCD;
    SetGrbmGfxIndex(cmd_buffer, grbm_value, gc_mode);
    // Reset counters
    if (block_info->attr & CounterBlockMcAttr) {
      builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->McResetValue());
    }
    if (block_info->attr & CounterBlockCleanAttr) {
      for (uint32_t i = 0; i < block_info->counter_count; ++i) {
        builder_->BuildWriteConfigRegPacket(cmd_buffer,
                                            block_info->counter_reg_info[i].register_addr_lo, 0);
        builder_->BuildWriteConfigRegPacket(cmd_buffer,
                                            block_info->counter_reg_info[i].register_addr_hi, 0);
      }
    }

    // Setup counters
    if (block_info->select_value != NULL && !(block_info->attr & CounterBlockPerfCntAttr)) {
      auto select_addr = reg_info.select_addr;
      auto value = block_info->select_value(counter_des);
      if (block_info->attr & CounterBlockGrbmaAttr) {
        for (uint32_t xcc_id = 0; xcc_id < xcc_number_; xcc_id += xcc_per_aid_) {
          PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, xcc_id, is_multi_xcc_);
          builder_->BuildWritePConfigRegPacketToChiplet(cmd_buffer, select_addr, value,
                                                        static_cast<ChipletId>(xcc_id));
        }
      } else {
        builder_->BuildWriteConfigRegPacket(cmd_buffer, select_addr, value);
      }
    }
    if (block_info->attr & CounterBlockSdmaAttr) {
      const auto sdma_index = counter_des.block_des.index;
      is_mi100 = (reg_info.control_addr.offset == 0) ? true : false;
      if (is_mi100) {
        // MI100: A SDMA instance shares a common control register for PERF_SEL and ENABLE/CLEAR.
        sdma_mask |= 1u << sdma_index;
        sdma_select_accumulator[sdma_index].first = builder_->get_addr(reg_info.select_addr);
        sdma_select_accumulator[sdma_index].second |= prim_->SdmaSelectValue(counter_des);
      } else {
        // MI200 and MI300 have seperate select and control registers
        const auto sdma_index = counter_des.block_des.index;
        const auto target_aid_index = sdma_index / (block_info->instance_count / MAX_AID);
        // sdma enable/clear/stop is programmed per instance and saved in sdmas
        sdmas.insert({sdma_index, get_smn_addr(reg_info.control_addr, target_aid_index)});

        if (is_multi_xcc_) {
          PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                            is_multi_xcc_);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 0),
                                               0x04000100);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 0),
                                               0x04000100);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 0),
                                               0x04000100);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 0),
                                               0x04000100);

          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 1),
                                               0x04000100);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 1),
                                               0x04000100);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 1),
                                               0x04000100);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 1),
                                               0x04000100);

          if (xcc_number_ > 4) {
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 2),
                                                 0x04000100);
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 2),
                                                 0x04000100);
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 2),
                                                 0x04000100);
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 2),
                                                 0x04000100);

            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 3),
                                                 0x04000100);
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 3),
                                                 0x04000100);
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 3),
                                                 0x04000100);
            builder_->BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 3),
                                                 0x04000100);
          }
        }

        // insert master XCC PRED_EXEC packet here if it is MI300
        PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                          is_multi_xcc_);

        // sdma counter select is programmed per performance counter
        uint64_t select_addr = get_smn_addr(reg_info.select_addr, target_aid_index);
        builder_->BuildWritePConfigRegPacket(cmd_buffer, select_addr,
                                             prim_->SdmaSelectValue(counter_des));
      }
    }
    if (block_info->attr & (CounterBlockAidAttr | CounterBlockPerfCntAttr)) {
      bool use_aid = bool(block_info->attr & CounterBlockAidAttr);
      const auto target_aid_index = use_aid ? GetTargetAid(counter_des) : 0;
      const auto instance_index = counter_des.block_des.index;

      // insert master XCC PRED_EXEC packet here if it is MI300
      PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                        is_multi_xcc_ && use_aid);

      // umc counter select per UMC counter
      uint64_t select_addr = get_smn_addr(reg_info.select_addr, target_aid_index, use_aid);
      uint64_t control_addr = get_smn_addr(reg_info.control_addr, target_aid_index, use_aid);

      if (block_info->attr & CounterBlockUmcAttr) {
        // skip
      }
      if (block_info->attr & CounterBlockPerfCntAttr) {
        if (block_info->attr & CounterBlockRpbAttr)
          rpbs.insert(control_addr);
        else if (block_info->attr & CounterBlockAtcAttr)
          atcs.insert(control_addr);
        else
          perf_cnt.insert(control_addr);
        builder_->BuildWritePConfigRegPacket(cmd_buffer, select_addr,
                                             block_info->select_value(counter_des));
      }
    }
    // Start counters
    if (block_info->attr & CounterBlockMcAttr) {
      builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->McConfigValue(counter_des));
      builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->McStartValue());
    }
    // Configure SQ block
    if (block_info->attr & CounterBlockSqAttr) {
      if (prim_->GetGfxipLevel() == 9)
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqPerfcounterMaskAddr(),
                                             prim_->SqMaskValue(counter_des));
      builder_->BuildWriteUConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->SqControlValue(counter_des));
    }
    // Configure GUS block
    if (prim_->SupportsGusCounters()) {
      if (block_info->attr & CounterBlockGusAttr)
        builder_->BuildWriteConfigRegPacket(cmd_buffer, reg_info.select_addr,
                                            prim_->GusSelectValue(counter_des));
    }
  }
  // SDMA start for all SDMA chnnels/instances recorded earlier
  if (sdma_mask != 0) {
    // MI100
    for (uint32_t sdma_index = 0, mask = sdma_mask; mask != 0; sdma_index++, mask >>= 1) {
      if (mask & 1) {
        builder_->BuildWritePConfigRegPacket(cmd_buffer, sdma_select_accumulator[sdma_index].first,
                                             prim_->SdmaDisableClearValue());
        builder_->BuildWritePConfigRegPacket(cmd_buffer, sdma_select_accumulator[sdma_index].first,
                                             sdma_select_accumulator[sdma_index].second);
      }
    }
  }
  if (!sdmas.empty()) {
    // MI200 and MI300
    // insert master XCC PRED_EXEC packet here if it is MI300
    PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT, is_multi_xcc_);

    for (const auto& i : sdmas) {
      uint32_t sdma_index = i.first;
      uint64_t control_addr = i.second;
      // sdma per channel/instance clear
      builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr, prim_->SdmaDisableClearValue());

      // sdma per channel/instance enable
      builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr, prim_->SdmaEnableValue());
    }
  }

  // RPB start for all RPB instances
  if (!rpbs.empty()) start_generic_mc_counters(cmd_buffer, rpbs);

  // ATC start is treated the same as RPB instance
  if (!atcs.empty()) start_generic_mc_counters(cmd_buffer, atcs);

  if (!perf_cnt.empty()) start_generic_mc_counters(cmd_buffer, perf_cnt, false);

  // Reset Grbm to its default state - broadcast
  SetGrbmBroadcast(cmd_buffer, gc_mode_global);
  // Program Compute Perfcount Enable register to support perf counting
  builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputePerfcountEnableAddr(),
                                  prim_->CpPerfcountEnableValue());
  // Reset the counter list
  SetPerfmonCntl(cmd_buffer, prim_->CpPerfmonCntlResetValue(), counters_vec.get_attr());
  // Start the counter list
  SetPerfmonCntl(cmd_buffer, prim_->CpPerfmonCntlStartValue(), counters_vec.get_attr());
  // Issue barrier command to apply the commands to configure perfcounters
  if (!is_concurrent_) builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

// Build PMC read PM4 packets
uint32_t GpuPmcBuilder::ReadXccPackets(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                                       uint32_t* buf, uint32_t& read_counter, GCMode gc_mode) {
  uint32_t xcc_id = gc_mode & GC_MODE_XCD_ID_MASK;

  // Reset Grbm to its default state - broadcast
  SetGrbmBroadcast(cmd_buffer, gc_mode);

  if (prim_->GetGfxipLevel() == 10) {
    for (auto& elem : counters_vec) {
      if ((elem.block_info->attr & CounterBlockGRBMAttr) == 0) continue;
      const auto& reg_info = get_reg_table(elem)[elem.index];
      builder_->BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, buf, 3);
      break;
    }
  }

  builder_->BuildWriteWaitIdlePacket(cmd_buffer);

  // Stop GUS counters
  if (prim_->SupportsGusCounters()) {
    if (counters_vec.get_attr() & CounterBlockGusAttr)
      builder_->BuildWriteConfigRegPacket(cmd_buffer, prim_->GetGusRsltCntlAddr(),
                                          prim_->GusStopValue());
  }
  // SDMA mask
  uint32_t sdma_mask = 0;
  // UMC mask
  uint32_t umc_mask = 0;
  // Iterate through the list of blocks to create PM4 packets to read counter values
  for (const auto& counter_des : counters_vec) {
    const auto* block_info = counter_des.block_info;
    const auto& block_des = counter_des.block_des;
    const auto* reg_table = get_reg_table(counter_des);
    const auto& reg_info = reg_table[counter_des.index];

    // Skip AID mode counters
    if (block_info->attr & CounterBlockAidAttr) continue;

    // Keep PerfCnt for XCD mode, skip it for AIGC mode
    if ((block_info->attr & CounterBlockPerfCntAttr) && (gc_mode & GC_MODE_XCD)) {
      // Choose which counter to read
      builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->McConfigValue(counter_des));
      builder_->BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, buf + read_counter, 3);
      read_counter += 2;
      continue;
    }

    if (bool(block_info->attr & CounterBlockGrbmaAttr) != bool(gc_mode & GC_MODE_AID_WITH_XCD_INDEX))
      continue;

    if (SPISkip(block_info->attr, counter_des.id)) {
      read_counter += 2 * se_number_;  // Skip two 64-bit SPI counters per SE
      continue;
    }

    // Reset Grbm to its default state - broadcast
    SetGrbmBroadcast(cmd_buffer, gc_mode);

    if (block_info->attr & CounterBlockMcAttr) {
      const uint32_t grbm_value = (block_info->instance_count > 1)
                                      ? prim_->GrbmInstIndexValue(block_des.index)
                                      : prim_->GrbmBroadcastValue();
      SetGrbmGfxIndex(cmd_buffer, grbm_value);
      builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           prim_->McConfigValue(counter_des));
      builder_->BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, buf + read_counter, 3);
      read_counter += 2;
    } else if (block_info->attr & CounterBlockSdmaAttr) {
      // Stop SDMA: this code path appplies only to non-MI300
      if (reg_info.control_addr.offset == 0) {
        // MI100: stopped per instance
        const uint32_t mask = 1u << counter_des.block_des.index;
        if ((sdma_mask & mask) == 0) {
          sdma_mask |= mask;
          auto control_addr =
              (reg_info.control_addr.offset == 0) ? reg_info.select_addr : reg_info.control_addr;
          builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                               prim_->SdmaStopValue(counter_des));
        }
      } else {
        // MI200: stopped per counter to choose which counter to read
        builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                             prim_->SdmaStopValue(counter_des));
      }
      // Read SDMA
      uint32_t dw_mask = 0x1;
      if (reg_info.register_addr_hi.offset != 0) dw_mask = 0x3;
      if (buf != nullptr) {
        buf[read_counter]     = 0;
        buf[read_counter + 1] = 0;
      }
      builder_->BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, buf + read_counter, dw_mask);
      read_counter += 2;
    } else if (block_info->attr & CounterBlockUmcAttr) {
      // skip
    } else {
      const uint32_t se_end_index = (block_info->attr & CounterBlockSeAttr) ? se_number_ : 1;
      const uint32_t sa_end_index = (block_info->attr & CounterBlockSaAttr) ? sarrays_per_se_ : 1;
      for (uint32_t se_index = 0; se_index < se_end_index; ++se_index)
        for (uint32_t sarray = 0; sarray < sa_end_index; ++sarray) {
          uint32_t grbm_value = prim_->GrbmBroadcastValue();
          if ((block_info->instance_count > 1) && (block_info->attr & CounterBlockSaAttr)) {
            grbm_value =
                prim_->GrbmInstSeShIndexValue(block_des.index, se_index, sarray);
          } else if ((block_info->instance_count > 1) &&
                     (block_info->attr & CounterBlockSeAttr)) {
            grbm_value = prim_->GrbmInstSeIndexValue(block_des.index, se_index);
          } else if (block_info->instance_count > 1) {
            grbm_value =
                prim_->GrbmInstIndexValue(GetInstanceIndex(block_des.index, block_info));
          } else if (block_info->attr & CounterBlockSeAttr) {
            grbm_value = prim_->GrbmSeIndexValue(se_index);
          }

          bool bIsWGPcounter11 =
              prim_->GetGfxipLevel() == 11 && (block_info->attr & CounterBlockSqAttr);
          bool bIsWGPcounter12 =
              prim_->GetGfxipLevel() >= 12 && (block_info->attr & CounterBlockWgpAttr);

          if (bIsWGPcounter11) {
            for (int wgp = 0; wgp < wgp_per_sa_; wgp++) {
              grbm_value = prim_->GrbmSeShWgpIndexValue(se_index, sarray, wgp);
              SetGrbmGfxIndex(cmd_buffer, grbm_value);
              builder_->BuildCopyCounterDataPacket(
                  cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                  buf + read_counter, 1);
              read_counter += 2;
            }
          } else if (bIsWGPcounter12) {
            for (int wgp = 0; wgp < wgp_per_sa_; wgp++) {
              // TODO: This patch is needed to avoid soft-hang for some WGP
              //       blocks, will remove after CU mask support is added to
              //       agent_info
              if (asymmetric_cu_patch && sarray == 1 && wgp == 8) {
                if (buf != nullptr) {
                  buf[read_counter] = 0;
                  buf[read_counter + 1] = 0;
                }
                read_counter += 2;
                continue;
              }
              if (block_info->instance_count > 1)
                grbm_value = prim_->GrbmInstSeShWgpIndexValue(block_des.index,
                                                               se_index, sarray, wgp);
              else
                grbm_value = prim_->GrbmSeShWgpIndexValue(se_index, sarray, wgp);
              SetGrbmGfxIndex(cmd_buffer, grbm_value);
              uint32_t dw_mask = reg_info.register_addr_hi.offset ? 3 : 1;
              builder_->BuildCopyCounterDataPacket(
                  cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                  buf + read_counter, dw_mask);
              if (buf && (dw_mask == 1))
                buf[read_counter + 1] = 0;
              read_counter += 2;
            }
          } else {
            SetGrbmGfxIndex(cmd_buffer, grbm_value, gc_mode);
            if (block_info->attr & CounterBlockGrbmaAttr)
              builder_->BuildCopyCounterDataPacketFromChiplet(
                  cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                  buf + read_counter, 3, static_cast<ChipletId>(xcc_id));
            else
              builder_->BuildCopyCounterDataPacket(
                  cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                  buf + read_counter, 3);
            read_counter += 2;
          }
        }
    }
  }
  // Reset Grbm to its default state - broadcast
  SetGrbmBroadcast(cmd_buffer, gc_mode);
  // Return amount of data to read
  return read_counter * sizeof(uint32_t);
}

// Build PMC stop PM4 comands
void GpuPmcBuilder::Stop(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) {
  GCMode gc_mode = (counters_vec.get_attr() & CounterBlockGrbmaAttr) ? GC_MODE_ALL : GC_MODE_XCD;
  // Reset Grbm to its default state - broadcast
  SetGrbmBroadcast(cmd_buffer, gc_mode);

  uint32_t sdma_mask = 0;
  if (counters_vec.get_attr() & (CounterBlockAidAttr | CounterBlockPerfCntAttr)) {
    for (const auto& counter_des : counters_vec) {
      const auto* block_info = counter_des.block_info;
      const auto& block_des = counter_des.block_des;
      const auto* reg_table = get_reg_table(counter_des);
      const auto& reg_info = reg_table[counter_des.index];

      if (block_info->attr & CounterBlockAidAttr) {
        // MI300 AID blocks: UMC/RPB/ATC/SDMA event insert master XCC PRED_EXEC packet here
        PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                          is_multi_xcc_);

        const auto target_aid_index = GetTargetAid(counter_des);
        uint64_t smn_control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);

        if (block_info->attr & CounterBlockUmcAttr) {
          // Stop UMC
        } else if (block_info->attr & (CounterBlockRpbAttr | CounterBlockAtcAttr)) {
          // Stop RPB/ATC
          builder_->BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr, 0);
        } else if (block_info->attr & CounterBlockSdmaAttr) {
          // Stop SDMA
          if (reg_info.control_addr.offset == 0) {
            // MI100: stopped per instance
            const uint32_t mask = 1u << counter_des.block_des.index;
            if ((sdma_mask & mask) == 0) {
              sdma_mask |= mask;
              auto control_addr = (reg_info.control_addr.offset == 0) ? reg_info.select_addr
                                                                      : reg_info.control_addr;
              builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                                   prim_->SdmaStopValue(counter_des));
            }
          } else if (is_multi_xcc_) {
            // MI300 SDMA event: insert master XCC PRED_EXEC packet here
            builder_->BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr,
                                                 prim_->SdmaStopValue(counter_des));
          } else {
            // MI200: stopped per counter to choose which counter to read
            builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                                 prim_->SdmaStopValue(counter_des));
          }
        }
      } else if (block_info->attr & CounterBlockPerfCntAttr) {
        // Stop Per-XCD PerfCnt
        builder_->BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr, 0);
      }
    }
  }

  // Issue barrier command to wait commands to complete
  SetPerfmonCntl(cmd_buffer, prim_->CpPerfmonCntlStopValue(), counters_vec.get_attr());

  // Enable RLC Perfmon Clock Gating. On Vega this
  // was disabled during Perf Cntrs collection session
  if (prim_->GetGfxipLevel() == 9)
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcPerfmonClkCntlAddr(), 0);

  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

// Build PMC read PM4 comands
uint32_t GpuPmcBuilder::Read(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                             void* data_buffer) {
  uint32_t* buf = reinterpret_cast<uint32_t*>(data_buffer);
  uint32_t read_counter = 0;
  auto counters_attr = counters_vec.get_attr();

  SetPerfmonCntl(cmd_buffer, prim_->CpPerfmonCntlReadValue(), counters_vec.get_attr());

  // counters have UMC events: MI300 Loop over MI300 XCCs for each counter_des
  if (counters_attr & CounterBlockAidAttr) {
    for (const auto& counter_des : counters_vec) {
      const auto* block_info = counter_des.block_info;
      const auto& block_des = counter_des.block_des;
      const auto* reg_table = get_reg_table(counter_des);
      const auto& reg_info = reg_table[counter_des.index];

      if (block_info->attr & CounterBlockUmcAttr) {
        // skip
      } else if (block_info->attr & CounterBlockSdmaAttr) {
        // insert master XCC PRED_EXEC packet accordingly
        PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                          is_multi_xcc_);

        const auto sdma_index = counter_des.block_des.index;
        const auto target_aid_index = sdma_index >> 2;

        // Read SDMA
        uint32_t dw_mask = 0x1;
        if (reg_info.register_addr_hi.offset != 0) {
          // MI200 and MI300 both have register_addr_hi
          // Select which SDMA perf counter to read
          auto smn_control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr,
                                               prim_->SdmaStopValue(counter_des));
          dw_mask = 0x3;
        }
        auto smn_register_addr_lo = get_smn_addr(reg_info.register_addr_lo, target_aid_index);
        auto smn_register_addr_hi = get_smn_addr(reg_info.register_addr_hi, target_aid_index);
        builder_->BuildCopyCounterDataPacket(cmd_buffer, smn_register_addr_lo,
                                             smn_register_addr_hi, buf + read_counter, dw_mask);
        read_counter += 2;
      } else if ((block_info->attr & CounterBlockAidAttr)) {
        // Read UMC/ATC/RPB
        // insert master XCC PRED_EXEC packet accordingly
        PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, VIRTUALXCCID_SELECT,
                                          is_multi_xcc_);

        const auto target_aid_index = GetTargetAid(counter_des);
        if (counters_attr & (CounterBlockRpbAttr | CounterBlockAtcAttr)) {
          // For RPB program to choose which counter to read
          uint64_t control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);
          builder_->BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                               prim_->McConfigValue(counter_des));
        }
        auto smn_register_addr_lo = get_smn_addr(reg_info.register_addr_lo, target_aid_index);
        auto smn_register_addr_hi = get_smn_addr(reg_info.register_addr_hi, target_aid_index);
        builder_->BuildCopyCounterDataPacket(cmd_buffer, smn_register_addr_lo,
                                             smn_register_addr_hi, buf + read_counter, 3);
        read_counter += 2;
      }
    }
  }
  for (uint32_t xcc_selected = 0; xcc_selected < xcc_number_; ++xcc_selected) {
    PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, xcc_selected, is_multi_xcc_);
    ReadXccPackets(cmd_buffer, counters_vec, buf, read_counter);
  }
  // AIGC blocks
  if (counters_vec.get_attr() & CounterBlockGrbmaAttr) {
    for (uint32_t xcc_selected = 0; xcc_selected < xcc_number_; xcc_selected += xcc_per_aid_) {
      PrecExecBuilder prec_exec_builder(*builder_, cmd_buffer, xcc_selected, is_multi_xcc_);
      GCMode gc_mode = (GCMode)(GC_MODE_AID_WITH_XCD_INDEX | xcc_selected);
      ReadXccPackets(cmd_buffer, counters_vec, buf, read_counter, gc_mode);
    }
  }

  builder_->BuildCacheFlushPacket(cmd_buffer, reinterpret_cast<size_t>(buf),
                                  read_counter * sizeof(uint32_t));

  // Return amount of data to read
  return read_counter * sizeof(uint32_t);
}

};  // namespace pm4_builder
