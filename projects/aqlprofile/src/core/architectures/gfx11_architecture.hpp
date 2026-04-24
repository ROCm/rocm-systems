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

#ifndef SRC_CORE_ARCHITECTURES_GFX11_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_GFX11_ARCHITECTURE_HPP_

#include "core/hardware_architecture.hpp"
#include "util/hsa_rsrc_factory.h"

namespace aql_profile {

/// GFX11 architecture implementation (RDNA 3)
/// Covers Navi 31, Navi 32, Navi 33 (gfx1100, gfx1101, gfx1102, gfx1103)
class Gfx11Architecture : public HardwareArchitecture {
 public:
  explicit Gfx11Architecture(const AgentInfo* agent_info);
  virtual ~Gfx11Architecture() = default;

  // HardwareArchitecture interface
  const HardwareConfig& GetConfig() const override { return config_; }
  const RegisterSchema& GetRegisterSchema() const override { return register_schema_; }
  const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const override;
  uint32_t FindBlockByName(const char* name) const override;
  uint32_t GetBlockCount() const override;
  pm4_builder::CmdBuilder* CreateCmdBuilder() const override;

  // Architecture version
  bool IsGFX11() const override { return true; }

  // GFX11 specific
  int GetNumWGPs() const override;

 protected:
  void InitializeConfig(const AgentInfo* agent_info);
  void InitializeRegisterSchema();
  void InitializeBlockTable();

  HardwareConfig config_;
  RegisterSchema register_schema_;
  const GpuBlockInfo** block_table_;
  uint32_t block_count_;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_GFX11_ARCHITECTURE_HPP_
