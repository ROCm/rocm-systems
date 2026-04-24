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

#ifndef SRC_CORE_ARCHITECTURES_GFX9_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_GFX9_ARCHITECTURE_HPP_

#include "core/hardware_architecture.hpp"
#include "util/hsa_rsrc_factory.h"

namespace aql_profile {

/// GFX9 architecture implementation (Vega series)
/// Base class for gfx900, gfx902, gfx906, and specialized MI100/MI200/MI300
class Gfx9Architecture : public HardwareArchitecture {
 public:
  explicit Gfx9Architecture(const AgentInfo* agent_info);
  virtual ~Gfx9Architecture() = default;

  // HardwareArchitecture interface
  const HardwareConfig& GetConfig() const override { return config_; }
  const RegisterSchema& GetRegisterSchema() const override { return register_schema_; }
  const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const override;
  uint32_t FindBlockByName(const char* name) const override;
  uint32_t GetBlockCount() const override;
  pm4_builder::CmdBuilder* CreateCmdBuilder() const override;

  // Architecture version
  bool IsGFX9() const override { return true; }

 protected:
  // Protected constructor for derived classes (MI100, MI200, MI300)
  Gfx9Architecture() = default;

  // Initialize hardware config - can be overridden by derived classes
  virtual void InitializeConfig(const AgentInfo* agent_info);

  // Initialize register schema - can be extended by derived classes
  virtual void InitializeRegisterSchema();

  // Initialize block table - can be overridden by derived classes
  virtual void InitializeBlockTable();

  HardwareConfig config_;
  RegisterSchema register_schema_;
  const GpuBlockInfo** block_table_;
  uint32_t block_count_;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_GFX9_ARCHITECTURE_HPP_
