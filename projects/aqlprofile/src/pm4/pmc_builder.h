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

#ifndef SRC_PM4_PMC_BUILDER_H_
#define SRC_PM4_PMC_BUILDER_H_

#include <stdint.h>

#include <set>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "def/gpu_block_info.h"
#include "pm4/cmd_config.h"
#include "util/hsa_rsrc_factory.h"

namespace pm4_builder {
// MI300 UMC constants
constexpr uint32_t VIRTUALXCCID_SELECT = 0;
constexpr uint32_t UMC_MASTER_XCC = 2;
constexpr uint32_t MAX_AID = 4;
constexpr uint32_t UMC_USR_BIT = 34 - 2;
constexpr uint32_t UMC_AID_BIT = 32 - 2;
constexpr uint32_t UMC_SAMPLE_BYTE_SIZE = 8;

constexpr size_t SPI_SPECIAL_CNT = 0x1000000;
inline bool SPISkip(size_t block, size_t id) {
  return (block & CounterBlockSPIAttr) != 0 && id >= SPI_SPECIAL_CNT;
}

class CmdBuffer;
class CmdBuilder;

// helper class for building PrecExec packet
template <typename Builder>
class PrecExecBuilder {
 public:
  PrecExecBuilder(Builder& _builder, CmdBuffer* cmd_buffer, uint32_t target_xcc, bool is_mi300)
      : cmd_buffer_(cmd_buffer), builder{_builder}, is_mi300_(is_mi300), target_xcc_(target_xcc) {
    if (is_mi300_) {
      // PRED_EXEC aplies to MI300 only
      pos_ = cmd_buffer->DwSize();
      builder.BuildPredExecPacket(cmd_buffer, target_xcc_, 0);
      initial_buff_size_ = cmd_buffer->DwSize();
    }
  }

  ~PrecExecBuilder() {
    if (is_mi300_) {
      // PRED_EXEC aplies to MI300 only
      CmdBuffer pred_exec;
      // update first PRED_EXEC packet to its correct value
      builder.BuildPredExecPacket(&pred_exec, target_xcc_,
                                  cmd_buffer_->DwSize() - initial_buff_size_);
      const uint32_t* data = (const uint32_t*)pred_exec.Data();

      for (size_t i = 0; i < pred_exec.DwSize(); ++i) cmd_buffer_->Assign(pos_ + i, data[i]);
    }
  }

 private:
  CmdBuffer* cmd_buffer_{nullptr};
  Builder& builder;
  bool is_mi300_{false};
  uint32_t target_xcc_{0};
  int pos_{0};
  int initial_buff_size_{0};
};

// PMC PM4 commands builder virtual interface
class PmcBuilder {
 public:
  PmcBuilder() {}
  virtual ~PmcBuilder() {}
  // Generate enable profiling commands
  virtual void Enable(CmdBuffer* cmd_buffer) = 0;
  // Generate disable profiling commands
  virtual void Disable(CmdBuffer* cmd_buffer) = 0;
  // Generate wait for GPU idle commands
  virtual void WaitIdle(CmdBuffer* cmd_buffer) = 0;
  // Generate start profiling commands.
  virtual void Start(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) = 0;
  // Generate stop profiling commands.
  // Return actual required data buffer size.
  virtual void Stop(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) = 0;
  // Generate read profiling commands.
  // Return actual required data buffer size.
  virtual uint32_t Read(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                        void* data_buffer) = 0;
  virtual int GetNumWGPs() = 0;
};

// PMC PM4 commands builder template
template <typename Builder, typename Primitives, bool concurrent>
class GpuPmcBuilder : public PmcBuilder, protected Primitives {
 private:
  typedef uint32_t reg_addr_t;
  // Shader Engines number on the GPU
  uint32_t se_number_;
  uint32_t wgp_per_sa;
  uint32_t sarrays_per_se;
  // XCC number on the GPU
  uint32_t xcc_number_;
  Builder builder;

  void DebugTrace(uint32_t value) {
    CmdBuffer cmd_buffer;
    uint32_t header[2] = {0, value};
    APPEND_COMMAND_WRAPPER((&cmd_buffer), header);
  }

  // Reg-info table getting helper
  const CounterRegInfo* get_reg_table(const counter_des_t& counter_des) {
    const auto* block_info = counter_des.block_info;
    const auto& block_des = counter_des.block_des;
    auto base_index = block_des.index;
    if ((block_info->attr & CounterBlockAidAttr) && (xcc_number_ > 1))
      // MI300 all AID style instances fold back to per AID counter_reg_info
      base_index %= (block_info->instance_count / MAX_AID);
    base_index =
        (block_info->attr & CounterBlockExplInstAttr) ? base_index * block_info->counter_count : 0;
    return &(block_info->counter_reg_info[base_index]);
  }

  uint32_t GetAidNumber() const { return (xcc_number_ > 1) ? 4 : 1; }

  uint32_t GetTargetAid(const counter_des_t& counter_des) const {
    const auto num_aid = GetAidNumber();
    const auto num_instance = counter_des.block_info->instance_count;
    const auto num_instance_per_aid = num_instance / num_aid;
    const auto instance_index = counter_des.block_des.index;
    const auto target_aid_index = instance_index / num_instance_per_aid;

    return target_aid_index;
  }

  // helper function to convert a 32-bit address to a 64-bit SMN address.
  // Returns the address seen by UMC_MASTER_XCC of register at reg_addr on target_aid_index.
  uint64_t get_smn_addr(uint64_t addr, uint32_t target_aid_index) {
    if (xcc_number_ > 1)
      addr |= ((uint64_t)1 << UMC_USR_BIT) | ((uint64_t)target_aid_index << UMC_AID_BIT);
    return addr;
  }

  uint64_t get_smn_addr(const Register& reg, uint32_t target_aid_index) {
    return get_smn_addr(builder.get_addr(reg), target_aid_index);
  }

  // start counters for rpb-block like instances
  void start_generic_mc_counters(CmdBuffer* cmd_buffer,
                                 const std::map<uint32_t, uint64_t>& instances) {
    // insert master XCC PRED_EXEC packet here if it is MI300
    PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                               xcc_number_ > 1);
    for (const auto& i : instances) {
      uint64_t control_addr = i.second;
      // rpb instance clear
      builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr, Primitives::mc_reset_value());
      // rpb instance enable
      builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr, Primitives::mc_start_value());
    }
  }

  // 'attr' is reserved for future expansion
  void SetGrbmGfxIndex(CmdBuffer* cmd_buffer, uint32_t value, uint32_t attr = 0) {
    builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::GRBM_GFX_INDEX_ADDR, value);
  }

  // 'attr' is reserved for future expansion
  void SetGrbmBroadcast(CmdBuffer* cmd_buffer, uint32_t attr = 0) {
    SetGrbmGfxIndex(cmd_buffer, Primitives::grbm_broadcast_value());
  }

  void SetPerfmonCntl(CmdBuffer* cmd_buffer, uint32_t value, uint32_t attr) {
    if (attr & CounterBlockCpmonAttr)
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::CP_PERFMON_CNTL_ADDR, value);
  }

 public:
  explicit GpuPmcBuilder(const AgentInfo* agent_info)
      : PmcBuilder(),
        builder(acquire_ip_offset_table(agent_info)),
        se_number_(agent_info->se_num / agent_info->xcc_num),
        xcc_number_(agent_info->xcc_num),
        sarrays_per_se(agent_info->shader_arrays_per_se) {
    this->wgp_per_sa =
        (agent_info->cu_num / 2 + sarrays_per_se * se_number_ - 1) / (se_number_ * sarrays_per_se);
    // Due to MI300 CP firmware issue we need to use mem_mapped_register mode to patch for GCEA
    // hang. Otherwise both perfcounters mode and mem_mapped_register mode should work.
    builder.bUsePerfCounterMode = (xcc_number_ > 1) ? false : true;
  }

  int GetNumWGPs() override {
    if (Primitives::GFXIP_LEVEL >= 11) return wgp_per_sa;
    return 1;
  };

  // Build PMC enable PM4 comands - enable CP counting for a specific queue
  void Enable(CmdBuffer* cmd_buffer) {
    // Program Compute Perfcount Enable register to support perf counting
    builder.BuildWriteShRegPacket(cmd_buffer, Primitives::COMPUTE_PERFCOUNT_ENABLE_ADDR,
                                  Primitives::cp_perfcount_enable_value());
  }
  // Build PMC disable PM4 comands - enable CP counting for a specific queue
  void Disable(CmdBuffer* cmd_buffer) {
    // Program Compute Perfcount Enable register to support perf counting
    builder.BuildWriteShRegPacket(cmd_buffer, Primitives::COMPUTE_PERFCOUNT_ENABLE_ADDR,
                                  Primitives::cp_perfcount_disable_value());
  }
  // Build PMC waite-idle PM4 comands - enable CP counting for a specific queue
  void WaitIdle(CmdBuffer* cmd_buffer) {
    // Program Compute Perfcount WaiteIdle register to support perf counting
    builder.BuildWriteWaitIdlePacket(cmd_buffer);
  }

  // Build PMC start PM4 comands
  void Start(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) override {
    // Issue barrier command
    if (!concurrent) builder.BuildWriteWaitIdlePacket(cmd_buffer);
    // Reset Grbm to its default state - broadcast
    SetGrbmBroadcast(cmd_buffer, counters_vec.get_attr());
    // Disable RLC Perfmon Clock Gating
    // On Vega this is needed to collect Perf Cntrs
    if (Primitives::GFXIP_LEVEL == 9)
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_PERFMON_CLK_CNTL_ADDR, 1);
    // Reset perf counters
    SetPerfmonCntl(cmd_buffer, Primitives::cp_perfmon_cntl_reset_value(), counters_vec.get_attr());
    // Enable SQ Counter Control enable perfomance counter in graphics pipeline if implied
    Primitives::validate_counters(counters_vec.get_attr());
    if (counters_vec.get_attr() & CounterBlockTcAttr) {
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::SQ_PERFCOUNTER_CTRL_ADDR,
                                         Primitives::sq_control_enable_value());
    }
    if (Primitives::GFXIP_LEVEL >= 11 &&
        (counters_vec.get_attr() & (CounterBlockTcAttr | CounterBlockSqAttr))) {
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::SQ_PERFCOUNTER_CTRL2_ADDR,
                                         Primitives::sq_control2_enable_value());
    }
#if defined(_GFX10_PRIMITIVES_H_) || defined(_GFX11_PRIMITIVES_H_)
    // Clear and enable GUS counters
    if (counters_vec.get_attr() & CounterBlockGusAttr) {
      builder.BuildWriteConfigRegPacket(cmd_buffer, Primitives::GUS_PERFCOUNTER_RSLT_CNTL_ADDR,
                                        Primitives::gus_disable_clear_value());
      builder.BuildWriteConfigRegPacket(cmd_buffer, Primitives::GUS_PERFCOUNTER_RSLT_CNTL_ADDR,
                                        Primitives::gus_start_value());
    }
#endif

    // SDMA mask
    // sdma performance monitor control value accumulator
    std::pair<reg_addr_t, uint32_t>
        sdma_select_accumulator[Primitives::SDMA_COUNTER_BLOCK_NUM_INSTANCES];
    uint32_t sdma_mask = 0;
    std::map<uint32_t, uint64_t> sdmas;
    bool is_mi100 = false;
    // UMC channels and their control register (for enable/disable) per channel
    std::map<uint32_t, uint64_t> umcchs;
    // RPB/ATC are per AID block like UMC above, we save its control register (for enable/disable)
    // per AID instance
    std::map<uint32_t, uint64_t> rpbs;
    std::map<uint32_t, uint64_t> atcs;
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
      const uint32_t grbm_value = (block_info->instance_count > 1 && !(block_info->attr & CounterBlockWgpAttr))
                                      ? Primitives::grbm_inst_index_value(block_des.index)
                                      : Primitives::grbm_broadcast_value();
      SetGrbmGfxIndex(cmd_buffer, grbm_value, block_info->attr);
      // Reset counters
      if (block_info->attr & CounterBlockMcAttr) {
        builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           Primitives::mc_reset_value());
      }
      if (block_info->attr & CounterBlockCleanAttr) {
        for (uint32_t i = 0; i < block_info->counter_count; ++i) {
          builder.BuildWriteConfigRegPacket(cmd_buffer,
                                            block_info->counter_reg_info[i].register_addr_lo, 0);
          builder.BuildWriteConfigRegPacket(cmd_buffer,
                                            block_info->counter_reg_info[i].register_addr_hi, 0);
        }
      }

      // Setup counters
      if (block_info->select_value != NULL && !(block_info->attr & CounterBlockRpbAttr) &&
          !(block_info->attr & CounterBlockAtcAttr)) {
        builder.BuildWriteConfigRegPacket(cmd_buffer, reg_info.select_addr,
                                          block_info->select_value(counter_des));
      }
      if (block_info->attr & CounterBlockSdmaAttr) {
        const auto sdma_index = counter_des.block_des.index;
        is_mi100 = (reg_info.control_addr.offset == 0) ? true : false;
        if (is_mi100) {
          // MI100: A SDMA instance shares a common control register for PERF_SEL and ENABLE/CLEAR.
          sdma_mask |= 1u << sdma_index;
          sdma_select_accumulator[sdma_index].first = builder.get_addr(reg_info.select_addr);
          sdma_select_accumulator[sdma_index].second |= Primitives::sdma_select_value(counter_des);
        } else {
          // MI200 and MI300 have seperate select and control registers
          const auto sdma_index = counter_des.block_des.index;
          const auto target_aid_index = sdma_index / (block_info->instance_count / MAX_AID);
          // sdma enable/clear/stop is programmed per instance and saved in sdmas
          sdmas.insert({sdma_index, get_smn_addr(reg_info.control_addr, target_aid_index)});

          if (xcc_number_ > 1) {
            PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                       xcc_number_ > 1);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 0),
                                               0x04000100);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 0),
                                               0x04000100);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 0),
                                               0x04000100);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 0),
                                               0x04000100);

            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 1),
                                               0x04000100);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 1),
                                               0x04000100);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 1),
                                               0x04000100);
            builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 1),
                                               0x04000100);

            if (xcc_number_ > 4) {
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 2),
                                                 0x04000100);
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 2),
                                                 0x04000100);
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 2),
                                                 0x04000100);
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 2),
                                                 0x04000100);

              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x4B30 >> 2, 3),
                                                 0x04000100);
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x6330 >> 2, 3),
                                                 0x04000100);
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x651B0 >> 2, 3),
                                                 0x04000100);
              builder.BuildWritePConfigRegPacket(cmd_buffer, get_smn_addr(0x661B0 >> 2, 3),
                                                 0x04000100);
            }
          }

          // insert master XCC PRED_EXEC packet here if it is MI300
          PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                     xcc_number_ > 1);

          // sdma counter select is programmed per performance counter
          uint64_t select_addr = get_smn_addr(reg_info.select_addr, target_aid_index);
          builder.BuildWritePConfigRegPacket(cmd_buffer, select_addr,
                                             Primitives::sdma_select_value(counter_des));
        }
      }
      if (block_info->attr & CounterBlockAidAttr) {
        const auto target_aid_index = GetTargetAid(counter_des);
        const auto instance_index = counter_des.block_des.index;

        // insert master XCC PRED_EXEC packet here if it is MI300
        PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                   xcc_number_ > 1);

        // umc counter select per UMC counter
        uint64_t select_addr = get_smn_addr(reg_info.select_addr, target_aid_index);
        uint64_t control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);

        if (block_info->attr & CounterBlockUmcAttr) {
          // skip
        }
        if (block_info->attr & CounterBlockRpbAttr || block_info->attr & CounterBlockAtcAttr) {
          if (block_info->attr & CounterBlockRpbAttr)
            rpbs.insert({instance_index, control_addr});
          else
            atcs.insert({instance_index, control_addr});
          builder.BuildWritePConfigRegPacket(cmd_buffer, select_addr,
                                             block_info->select_value(counter_des));
        }
      }
      // Start counters
      if (block_info->attr & CounterBlockMcAttr) {
        builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           Primitives::mc_config_value(counter_des));
        builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           Primitives::mc_start_value());
      }
      // Configure SQ block
      if (block_info->attr & CounterBlockSqAttr) {
        if (Primitives::GFXIP_LEVEL == 9)
          builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::SQ_PERFCOUNTER_MASK_ADDR,
                                             Primitives::sq_mask_value(counter_des));
        builder.BuildWriteUConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           Primitives::sq_control_value(counter_des));
      }
#if defined(_GFX10_PRIMITIVES_H_) || defined(_GFX11_PRIMITIVES_H_)
      // Configure GUS block
      if (block_info->attr & CounterBlockGusAttr)
        builder.BuildWriteConfigRegPacket(cmd_buffer, reg_info.select_addr,
                                          Primitives::gus_select_value(counter_des));
#endif
    }
    // SDMA start for all SDMA chnnels/instances recorded earlier
    if (sdma_mask != 0) {
      // MI100
      for (uint32_t sdma_index = 0, mask = sdma_mask; mask != 0; sdma_index++, mask >>= 1) {
        if (mask & 1) {
          builder.BuildWritePConfigRegPacket(cmd_buffer, sdma_select_accumulator[sdma_index].first,
                                             Primitives::sdma_disable_clear_value());
          builder.BuildWritePConfigRegPacket(cmd_buffer, sdma_select_accumulator[sdma_index].first,
                                             sdma_select_accumulator[sdma_index].second);
        }
      }
    }
    if (!sdmas.empty()) {
      // MI200 and MI300
      // insert master XCC PRED_EXEC packet here if it is MI300
      PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                 xcc_number_ > 1);

      for (const auto& i : sdmas) {
        uint32_t sdma_index = i.first;
        uint64_t control_addr = i.second;
        // sdma per channel/instance clear
        builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                           Primitives::sdma_disable_clear_value());

        // sdma per channel/instance enable
        builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                           Primitives::sdma_enable_value());
      }
    }

    // RPB start for all RPB instances
    if (!rpbs.empty()) start_generic_mc_counters(cmd_buffer, rpbs);

    // ATC start is treated the same as RPB instance
    if (!atcs.empty()) start_generic_mc_counters(cmd_buffer, atcs);

    // Reset Grbm to its default state - broadcast
    SetGrbmBroadcast(cmd_buffer, counters_vec.get_attr());
    // Program Compute Perfcount Enable register to support perf counting
    builder.BuildWriteShRegPacket(cmd_buffer, Primitives::COMPUTE_PERFCOUNT_ENABLE_ADDR,
                                  Primitives::cp_perfcount_enable_value());
    // Reset the counter list
    SetPerfmonCntl(cmd_buffer, Primitives::cp_perfmon_cntl_reset_value(), counters_vec.get_attr());
    // Start the counter list
    SetPerfmonCntl(cmd_buffer, Primitives::cp_perfmon_cntl_start_value(), counters_vec.get_attr());
    // Issue barrier command to apply the commands to configure perfcounters
    if (!concurrent) builder.BuildWriteWaitIdlePacket(cmd_buffer);
  }

  // Build PMC read PM4 packets
  uint32_t ReadXccPackets(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                          void* data_buffer, uint32_t& read_counter) {
    // Reset Grbm to its default state - broadcast
    SetGrbmBroadcast(cmd_buffer, counters_vec.get_attr());

    if (Primitives::GFXIP_LEVEL == 10) {
      for (auto& elem : counters_vec) {
        if ((elem.block_info->attr & CounterBlockGRBMAttr) == 0) continue;
        const auto& reg_info = get_reg_table(elem)[elem.index];
        builder.BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, data_buffer, 3);
        break;
      }
    }

    builder.BuildWriteWaitIdlePacket(cmd_buffer);

#if defined(_GFX10_PRIMITIVES_H_) || defined(_GFX11_PRIMITIVES_H_)
    // Stop GUS counters
    if (counters_vec.get_attr() & CounterBlockGusAttr)
      builder.BuildWriteConfigRegPacket(cmd_buffer, Primitives::GUS_PERFCOUNTER_RSLT_CNTL_ADDR,
                                        Primitives::gus_stop_value());
#endif
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

      // Skip UMC/SDMA/ATC/RPB counters
      if (block_info->attr & CounterBlockAidAttr) continue;

      if (SPISkip(block_info->attr, counter_des.id)) {
        read_counter += 2 * se_number_;  // Skip two 64-bit SPI counters per SE
        continue;
      }

      // Reset Grbm to its default state - broadcast
      SetGrbmBroadcast(cmd_buffer, counters_vec.get_attr());

      if (block_info->attr & CounterBlockMcAttr) {
        const uint32_t grbm_value = (block_info->instance_count > 1)
                                        ? Primitives::grbm_inst_index_value(block_des.index)
                                        : Primitives::grbm_broadcast_value();
        SetGrbmGfxIndex(cmd_buffer, grbm_value);
        builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                           Primitives::mc_config_value(counter_des));
        uint32_t* data = reinterpret_cast<uint32_t*>(data_buffer) + read_counter;
        builder.BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, data, 3);
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
            builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                               Primitives::sdma_stop_value(counter_des));
          }
        } else {
          // MI200: stopped per counter to choose which counter to read
          builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                             Primitives::sdma_stop_value(counter_des));
        }
        // Read SDMA
        uint32_t dw_mask = 0x1;
        if (reg_info.register_addr_hi.offset != 0) dw_mask = 0x3;
        uint32_t* data = reinterpret_cast<uint32_t*>(data_buffer) + read_counter;
        if (data_buffer != 0) {
          *reinterpret_cast<uint64_t*>(data) = 0;
        }
        builder.BuildCopyCounterDataPacket(cmd_buffer, reg_info.register_addr_lo,
                                           reg_info.register_addr_hi, data, dw_mask);
        read_counter += 2;
      } else if (block_info->attr & CounterBlockUmcAttr) {
        // skip
      } else {
        const uint32_t se_end_index = (block_info->attr & CounterBlockSeAttr) ? se_number_ : 1;
        const uint32_t sa_end_index = (block_info->attr & CounterBlockSaAttr) ? 2 : 1;
        for (uint32_t se_index = 0; se_index < se_end_index; ++se_index)
          for (uint32_t sarray = 0; sarray < sa_end_index; ++sarray) {
            uint32_t grbm_value = Primitives::grbm_broadcast_value();
            if ((block_info->instance_count > 1) && (block_info->attr & CounterBlockSaAttr)) {
              grbm_value =
                  Primitives::grbm_inst_se_sh_index_value(block_des.index, se_index, sarray);
            } else if ((block_info->instance_count > 1) &&
                       (block_info->attr & CounterBlockSeAttr)) {
              grbm_value = Primitives::grbm_inst_se_index_value(block_des.index, se_index);
            } else if (block_info->instance_count > 1) {
              grbm_value = Primitives::grbm_inst_index_value(block_des.index);
            } else if (block_info->attr & CounterBlockSeAttr) {
              grbm_value = Primitives::grbm_se_index_value(se_index);
            }

          bool bIsWGPcounter11 = Primitives::GFXIP_LEVEL == 11 && (block_info->attr & CounterBlockSqAttr);
          bool bIsWGPcounter12 = Primitives::GFXIP_LEVEL >= 12 && (block_info->attr & CounterBlockWgpAttr);

          if (bIsWGPcounter11) {
            for (int wgp=0; wgp<wgp_per_sa; wgp++) {
              grbm_value = Primitives::grbm_se_sh_wgp_index_value(se_index, sarray, wgp);
              SetGrbmGfxIndex(cmd_buffer, grbm_value);
              builder.BuildCopyCounterDataPacket(
                  cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                  reinterpret_cast<uint32_t*>(data_buffer) + read_counter, 1);
              read_counter += 2;
            }
          } else if (bIsWGPcounter12) {
            for (int wgp=0; wgp<wgp_per_sa; wgp++) {
              if (block_info->instance_count > 1)
                grbm_value = Primitives::grbm_inst_se_sh_wgp_index_value(block_des.index, se_index, sarray, wgp);
              else
                grbm_value = Primitives::grbm_se_sh_wgp_index_value(se_index, sarray, wgp);
              SetGrbmGfxIndex(cmd_buffer, grbm_value);
              uint32_t dw_mask = reg_info.register_addr_hi.offset ? 3 : 1;
              builder.BuildCopyCounterDataPacket(
                  cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                  reinterpret_cast<uint32_t*>(data_buffer) + read_counter, dw_mask);
              if (data_buffer && (dw_mask == 1))
                  *(reinterpret_cast<uint32_t*>(data_buffer) + read_counter + 1) = 0;
              read_counter += 2;
            }
          } else {
            SetGrbmGfxIndex(cmd_buffer, grbm_value, block_info->attr);
            builder.BuildCopyCounterDataPacket(
                cmd_buffer, reg_info.register_addr_lo, reg_info.register_addr_hi,
                reinterpret_cast<uint32_t*>(data_buffer) + read_counter, 3);
            read_counter += 2;
          }
        }
      }
    }
    // Reset Grbm to its default state - broadcast
    SetGrbmBroadcast(cmd_buffer, counters_vec.get_attr());
    // Return amount of data to read
    return read_counter * sizeof(uint32_t);
  }

  // Build PMC stop PM4 comands
  void Stop(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) override {
    // Reset Grbm to its default state - broadcast
    SetGrbmBroadcast(cmd_buffer, counters_vec.get_attr());

    uint32_t sdma_mask = 0;
    if (counters_vec.get_attr() & CounterBlockAidAttr)
      for (const auto& counter_des : counters_vec) {
        const auto* block_info = counter_des.block_info;
        const auto& block_des = counter_des.block_des;
        const auto* reg_table = get_reg_table(counter_des);
        const auto& reg_info = reg_table[counter_des.index];

        if (!(block_info->attr & CounterBlockAidAttr))
          // skip all non-AID blocks
          continue;

        // MI300 AID blocks: UMC/RPB/ATC/SDMA event insert master XCC PRED_EXEC packet here
        PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                   xcc_number_ > 1);

        const auto target_aid_index = GetTargetAid(counter_des);
        uint64_t smn_control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);

        if (block_info->attr & CounterBlockUmcAttr) {
          // Stop UMC
        } else if (block_info->attr & (CounterBlockRpbAttr | CounterBlockAtcAttr)) {
          // Stop RPB/ATC
          builder.BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr, 0);
        } else if (block_info->attr & CounterBlockSdmaAttr) {
          // Stop SDMA
          if (reg_info.control_addr.offset == 0) {
            // MI100: stopped per instance
            const uint32_t mask = 1u << counter_des.block_des.index;
            if ((sdma_mask & mask) == 0) {
              sdma_mask |= mask;
              auto control_addr = (reg_info.control_addr.offset == 0) ? reg_info.select_addr
                                                                      : reg_info.control_addr;
              builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                                 Primitives::sdma_stop_value(counter_des));
            }
          } else if (xcc_number_ > 1) {
            // MI300 SDMA event: insert master XCC PRED_EXEC packet here
            builder.BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr,
                                               Primitives::sdma_stop_value(counter_des));
          } else {
            // MI200: stopped per counter to choose which counter to read
            builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                               Primitives::sdma_stop_value(counter_des));
          }
        }
      }
    sdma_mask = 0;
    if (counters_vec.get_attr() & CounterBlockAidAttr)
      for (const auto& counter_des : counters_vec) {
        const auto* block_info = counter_des.block_info;
        const auto& block_des = counter_des.block_des;
        const auto* reg_table = get_reg_table(counter_des);
        const auto& reg_info = reg_table[counter_des.index];

        if (!(block_info->attr & CounterBlockAidAttr))
          // skip all non-AID blocks
          continue;

        // MI300 AID blocks: UMC/RPB/ATC/SDMA event insert master XCC PRED_EXEC packet here
        PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                   xcc_number_ > 1);

        const auto target_aid_index = GetTargetAid(counter_des);
        uint64_t smn_control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);

        if (block_info->attr & CounterBlockUmcAttr) {
          // Stop UMC
        } else if (block_info->attr & (CounterBlockRpbAttr | CounterBlockAtcAttr)) {
          // Stop RPB/ATC
          builder.BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr, 0);
        } else if (block_info->attr & CounterBlockSdmaAttr) {
          // Stop SDMA
          if (reg_info.control_addr.offset == 0) {
            // MI100: stopped per instance
            const uint32_t mask = 1u << counter_des.block_des.index;
            if ((sdma_mask & mask) == 0) {
              sdma_mask |= mask;
              auto control_addr = (reg_info.control_addr.offset == 0) ? reg_info.select_addr
                                                                      : reg_info.control_addr;
              builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                                 Primitives::sdma_stop_value(counter_des));
            }
          } else if (xcc_number_ > 1) {
            // MI300 SDMA event: insert master XCC PRED_EXEC packet here
            builder.BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr,
                                               Primitives::sdma_stop_value(counter_des));
          } else {
            // MI200: stopped per counter to choose which counter to read
            builder.BuildWritePConfigRegPacket(cmd_buffer, reg_info.control_addr,
                                               Primitives::sdma_stop_value(counter_des));
          }
        }
      }

    // Issue barrier command to wait commands to complete
    SetPerfmonCntl(cmd_buffer, Primitives::cp_perfmon_cntl_stop_value(), counters_vec.get_attr());

    // Enable RLC Perfmon Clock Gating. On Vega this
    // was disabled during Perf Cntrs collection session
    if (Primitives::GFXIP_LEVEL == 9)
      builder.BuildWriteUConfigRegPacket(cmd_buffer, Primitives::RLC_PERFMON_CLK_CNTL_ADDR, 0);

    builder.BuildWriteWaitIdlePacket(cmd_buffer);
  }

  // Build PMC read PM4 comands
  uint32_t Read(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                void* data_buffer) override {
    uint32_t read_counter = 0;
    auto counters_attr = counters_vec.get_attr();

    SetPerfmonCntl(cmd_buffer, Primitives::cp_perfmon_cntl_read_value(), counters_vec.get_attr());

    // counters have UMC events: MI300 Loop over MI300 XCCs for each counter_des
    if (counters_attr & CounterBlockAidAttr)
      for (const auto& counter_des : counters_vec) {
        const auto* block_info = counter_des.block_info;
        const auto& block_des = counter_des.block_des;
        const auto* reg_table = get_reg_table(counter_des);
        const auto& reg_info = reg_table[counter_des.index];

        if (block_info->attr & CounterBlockUmcAttr) {
          // skip
        } else if (block_info->attr & CounterBlockSdmaAttr) {
          // insert master XCC PRED_EXEC packet accordingly
          PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                     xcc_number_ > 1);

          const auto sdma_index = counter_des.block_des.index;
          const auto target_aid_index = sdma_index >> 2;

          // Read SDMA
          uint32_t dw_mask = 0x1;
          if (reg_info.register_addr_hi.offset != 0) {
            // MI200 and MI300 both have register_addr_hi
            // Select which SDMA perf counter to read
            auto smn_control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);
            builder.BuildWritePConfigRegPacket(cmd_buffer, smn_control_addr,
                                               Primitives::sdma_stop_value(counter_des));
            dw_mask = 0x3;
          }
          uint32_t* smn_data_buffer = reinterpret_cast<uint32_t*>(data_buffer) + read_counter;
          auto smn_register_addr_lo = get_smn_addr(reg_info.register_addr_lo, target_aid_index);
          auto smn_register_addr_hi = get_smn_addr(reg_info.register_addr_hi, target_aid_index);
          builder.BuildCopyCounterDataPacket(cmd_buffer, smn_register_addr_lo, smn_register_addr_hi,
                                             smn_data_buffer, dw_mask);
          read_counter += 2;
        } else if ((block_info->attr & CounterBlockAidAttr)) {
          // Read UMC/ATC/RPB
          // insert master XCC PRED_EXEC packet accordingly
          PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, VIRTUALXCCID_SELECT,
                                                     xcc_number_ > 1);

          const auto target_aid_index = GetTargetAid(counter_des);
          if (counters_attr & (CounterBlockRpbAttr | CounterBlockAtcAttr)) {
            // For RPB program to choose which counter to read
            uint64_t control_addr = get_smn_addr(reg_info.control_addr, target_aid_index);
            builder.BuildWritePConfigRegPacket(cmd_buffer, control_addr,
                                               Primitives::mc_config_value(counter_des));
          }
          uint32_t* smn_data_buffer = reinterpret_cast<uint32_t*>(data_buffer) + read_counter;
          auto smn_register_addr_lo = get_smn_addr(reg_info.register_addr_lo, target_aid_index);
          auto smn_register_addr_hi = get_smn_addr(reg_info.register_addr_hi, target_aid_index);
          builder.BuildCopyCounterDataPacket(cmd_buffer, smn_register_addr_lo, smn_register_addr_hi,
                                             smn_data_buffer, 3);
          read_counter += 2;
        }
      }
    for (size_t xcc_selected = 0; xcc_selected < xcc_number_; ++xcc_selected) {
      PrecExecBuilder<Builder> prec_exec_builder(builder, cmd_buffer, xcc_selected,
                                                 xcc_number_ > 1);
      ReadXccPackets(cmd_buffer, counters_vec, data_buffer, read_counter);
    }

    builder.BuildCacheFlushPacket(cmd_buffer, size_t(data_buffer), read_counter * sizeof(uint32_t));

    // Return amount of data to read
    return read_counter * sizeof(uint32_t);
  }
};

};  // namespace pm4_builder

#endif  // SRC_PM4_PMC_BUILDER_H_
