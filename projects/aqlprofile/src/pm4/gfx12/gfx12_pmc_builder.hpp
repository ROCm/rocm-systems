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

#ifndef SRC_PM4_GFX12_PMC_BUILDER_HPP_
#define SRC_PM4_GFX12_PMC_BUILDER_HPP_

#include <stdint.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/hw/hardware_config.hpp"
#include "def/gpu_block_info.h"
#include "pm4/cmd_builder.h"
#include "pm4/cmd_config.h"
#include "pm4/pmc_builder.h"
#include "pm4/primitives_provider.hpp"

namespace pm4_builder {
// MI300 UMC constants (also defined in phase13 pmc_builder.h, redeclared here for isolation)
// These are constexpr so duplicate declarations are fine as long as values match.

enum Gfx12GCMode {
  GFX12_GC_MODE_XCD_ID_MASK = 0xFF,
  GFX12_GC_MODE_XCD = 0x100,
  GFX12_GC_MODE_AID = 0x200,
  GFX12_GC_MODE_AID_WITH_XCD_INDEX = 0x400,
  GFX12_GC_MODE_ALL = GFX12_GC_MODE_XCD | GFX12_GC_MODE_AID,
};

class CmdBuffer;
class CmdBuilder;

/// Helper class for building PredExec packets (non-template variant for GFX12).
class Gfx12PrecExecBuilder {
 public:
  Gfx12PrecExecBuilder(CmdBuilder& _builder, CmdBuffer* cmd_buffer, uint32_t target_xcc,
                       bool is_mi300);
  ~Gfx12PrecExecBuilder();

 private:
  CmdBuffer* cmd_buffer_{nullptr};
  CmdBuilder& builder_;
  bool is_mi300_{false};
  uint32_t target_xcc_{0};
  int pos_{0};
  int initial_buff_size_{0};
};

/// Non-template PMC PM4 commands builder for GFX12, using PrimitivesProvider for runtime dispatch.
class Gfx12PmcBuilder : public PmcBuilder {
 private:
  typedef uint32_t reg_addr_t;
  uint32_t se_number_;
  uint32_t wgp_per_sa_;
  uint32_t sarrays_per_se_;
  uint32_t xcc_number_;
  uint32_t xcc_per_aid_;
  bool is_multi_xcc_;
  uint32_t aid_count_;

  CmdBuilder* builder_;
  const PrimitivesProvider* prim_;
  bool is_concurrent_;

  bool asymmetric_cu_patch;

  const CounterRegInfo* get_reg_table(const counter_des_t& counter_des);

  uint32_t GetAidNumber() const;
  uint32_t GetTargetAid(const counter_des_t& counter_des) const;

  uint64_t get_smn_addr(uint64_t addr, uint32_t target_aid_index, bool use_aid = true);
  uint64_t get_smn_addr(const Register& reg, uint32_t target_aid_index, bool use_aid = true);

  void start_generic_mc_counters(CmdBuffer* cmd_buffer, const std::set<uint64_t>& instances,
                                 bool use_aid = true);

  void SetGrbmGfxIndex(CmdBuffer* cmd_buffer, uint32_t value,
                       Gfx12GCMode gc_mode = GFX12_GC_MODE_XCD);
  void SetGrbmBroadcast(CmdBuffer* cmd_buffer, Gfx12GCMode gc_mode);
  void SetPerfmonCntl(CmdBuffer* cmd_buffer, uint32_t value, uint32_t attr);
  uint32_t GetInstanceIndex(uint32_t instance_index, const GpuBlockInfo* block_info);

 public:
  explicit Gfx12PmcBuilder(const aql_profile::HardwareConfig& config, CmdBuilder* builder,
                           const PrimitivesProvider* prim, bool is_concurrent);

  int GetNumWGPs() override;

  void Enable(CmdBuffer* cmd_buffer) override;
  void Disable(CmdBuffer* cmd_buffer) override;
  void WaitIdle(CmdBuffer* cmd_buffer) override;
  void Start(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) override;

  uint32_t ReadXccPackets(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                          uint32_t* buf, uint32_t& read_counter,
                          Gfx12GCMode gc_mode = GFX12_GC_MODE_XCD);

  void Stop(CmdBuffer* cmd_buffer, const counters_vector& counters_vec) override;
  uint32_t Read(CmdBuffer* cmd_buffer, const counters_vector& counters_vec,
                void* data_buffer) override;
};

}  // namespace pm4_builder

#endif  // SRC_PM4_GFX12_PMC_BUILDER_HPP_
