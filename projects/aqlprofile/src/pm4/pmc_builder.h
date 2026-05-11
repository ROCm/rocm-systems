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

#include "core/hardware_config.hpp"
#include "def/gpu_block_info.h"
#include "pm4/cmd_config.h"
#include "pm4/primitives_provider.hpp"

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

enum GCMode {
  GC_MODE_XCD_ID_MASK = 0xFF,
  GC_MODE_XCD = 0x100,
  GC_MODE_AID = 0x200,
  GC_MODE_AID_WITH_XCD_INDEX = 0x400,
  GC_MODE_ALL = GC_MODE_XCD | GC_MODE_AID,
};

class CmdBuffer;
class CmdBuilder;

// helper class for building PrecExec packet
class PrecExecBuilder {
 public:
  PrecExecBuilder(CmdBuilder& _builder, CmdBuffer* cmd_buffer, uint32_t target_xcc, bool is_mi300);
  ~PrecExecBuilder();

 private:
  CmdBuffer* cmd_buffer_{nullptr};
  CmdBuilder& builder_;
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

// PMC PM4 commands builder (non-template, runtime-polymorphic)
class GpuPmcBuilder : public PmcBuilder {
 private:
  typedef uint32_t reg_addr_t;
  // Shader Engines number on the GPU
  uint32_t se_number_;
  uint32_t wgp_per_sa_;
  uint32_t sarrays_per_se_;
  // XCC topology
  uint32_t xcc_number_;
  uint32_t xcc_per_aid_;
  bool is_multi_xcc_;   // xcc_number_ > 1: enables PRED_EXEC packets and AID-aware addressing
  uint32_t aid_count_;  // Total AID count: xcc_number_ / xcc_per_aid_

  CmdBuilder* builder_;
  const PrimitivesProvider* prim_;
  bool is_concurrent_;

  bool asymmetric_cu_patch;  // Set from HardwareConfig::has_asymmetric_cu_design

  void DebugTrace(uint32_t value);

  // Reg-info table getting helper
  const CounterRegInfo* get_reg_table(const counter_des_t& counter_des);

  uint32_t GetAidNumber() const;
  uint32_t GetTargetAid(const counter_des_t& counter_des) const;

  // helper function to convert a 32-bit address to a 64-bit SMN address.
  // Returns the address seen by UMC_MASTER_XCC of register at reg_addr on target_aid_index.
  uint64_t get_smn_addr(uint64_t addr, uint32_t target_aid_index, bool use_aid = true);
  uint64_t get_smn_addr(const Register& reg, uint32_t target_aid_index, bool use_aid = true);

  // start counters for rpb-block like instances
  void start_generic_mc_counters(CmdBuffer* cmd_buffer, const std::set<uint64_t>& instances,
                                 bool use_aid = true);

  void SetGrbmGfxIndex(CmdBuffer* cmd_buffer, uint32_t value, GCMode gc_mode = GC_MODE_XCD);
  void SetGrbmBroadcast(CmdBuffer* cmd_buffer, GCMode gc_mode);
  void SetPerfmonCntl(CmdBuffer* cmd_buffer, uint32_t value, uint32_t attr);
  uint32_t GetInstanceIndex(uint32_t instance_index, const GpuBlockInfo* block_info);

 public:
  explicit GpuPmcBuilder(const aql_profile::HardwareConfig& config, CmdBuilder* builder,
                         const PrimitivesProvider* prim, bool is_concurrent);

  int GetNumWGPs() override;

  // Build PMC enable PM4 comands - enable CP counting for a specific queue
  void Enable(CmdBuffer* cmd_buffer);
  // Build PMC disable PM4 comands - enable CP counting for a specific queue
  void Disable(CmdBuffer* cmd_buffer);
  // Build PMC waite-idle PM4 comands - enable CP counting for a specific queue
  void WaitIdle(CmdBuffer* cmd_buffer);

  // Build PMC start PM4 comands
  void Start(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) override;

  // Build PMC read PM4 packets
  uint32_t ReadXccPackets(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                          uint32_t* buf, uint32_t& read_counter, GCMode gc_mode = GC_MODE_XCD);

  // Build PMC stop PM4 comands
  void Stop(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) override;

  // Build PMC read PM4 comands
  uint32_t Read(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                void* data_buffer) override;
};

};  // namespace pm4_builder

#endif  // SRC_PM4_PMC_BUILDER_H_
