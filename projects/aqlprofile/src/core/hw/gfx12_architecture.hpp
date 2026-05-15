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

#ifndef SRC_CORE_ARCHITECTURES_GFX12_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_GFX12_ARCHITECTURE_HPP_

#include "core/hw/hardware_architecture.hpp"
#include "util/hsa_rsrc_factory.h"

namespace aql_profile {

/// GFX12 architecture implementation (RDNA4 — gfx1200 / gfx1201).
/// Owns the block table and HardwareConfig for this GPU family.
/// PM4 builders (GpuPmcBuilder, GpuSpmBuilder, GpuSqttBuilder) remain
/// instantiated in Gfx1200Factory using the existing Primitives template.
class Gfx12Architecture : public HardwareArchitecture {
 public:
  /// Tag type: pass to the protected constructor to skip InitializeBlockTable().
  /// Used by derived classes that call InitializeBlockTable() themselves.
  struct DeferBlockTableInit {};

  explicit Gfx12Architecture(const AgentInfo* agent_info);
  ~Gfx12Architecture() override = default;

  const HardwareConfig& GetConfig() const override { return config_; }
  bool IsGFX12() const override { return true; }

  const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const override;
  uint32_t FindBlockByName(const char* name) const override;
  uint32_t GetBlockCount() const override { return block_count_; }
  const GpuBlockInfo** GetBlockTable() const override { return block_table_; }

  int GetAccumLowID() const override { return 1; }
  int GetAccumHiID()  const override { return 1; }

 protected:
  /// Constructor for derived classes: runs InitializeConfig but defers
  /// InitializeBlockTable() so the derived class can call it once itself.
  Gfx12Architecture(const AgentInfo* agent_info, DeferBlockTableInit);

  void InitializeConfig(const AgentInfo* agent_info);
  virtual void InitializeBlockTable();

  HardwareConfig config_;
  const GpuBlockInfo** block_table_;
  uint32_t block_count_;
};

/// GFX12 gfx1201 variant — same as gfx1200 with 4 block-table overrides
/// (CHC, GCEA, GCEA_SE, GL2C have higher instance counts on gfx1201).
class Gfx1201Architecture : public Gfx12Architecture {
 public:
  explicit Gfx1201Architecture(const AgentInfo* agent_info);

 protected:
  void InitializeBlockTable() override;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_GFX12_ARCHITECTURE_HPP_
