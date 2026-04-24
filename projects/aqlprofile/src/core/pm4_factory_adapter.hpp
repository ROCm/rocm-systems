// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions.
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

#ifndef SRC_CORE_PM4_FACTORY_ADAPTER_HPP_
#define SRC_CORE_PM4_FACTORY_ADAPTER_HPP_

#include "core/pm4_factory.h"
#include "core/hardware_architecture.hpp"

namespace aql_profile {

/// Adapter that allows Pm4Factory to use the new HardwareArchitecture system
/// This provides backward compatibility during migration
class Pm4FactoryAdapter : public Pm4Factory {
 public:
  /// Create factory using new architecture system
  explicit Pm4FactoryAdapter(HardwareArchitecture* architecture);
  virtual ~Pm4FactoryAdapter();

  // Pm4Factory interface
  gpu_id_t GetGpuId() const override;
  bool IsConcurrent() const override { return concurrent_mode_; }
  bool SpmKfdMode() const override { return spm_kfd_mode_; }

  pm4_builder::CmdBuilder* GetCmdBuilder() const override { return cmd_builder_; }
  pm4_builder::PmcBuilder* GetPmcBuilder() const override { return pmc_builder_; }
  pm4_builder::SpmBuilder* GetSpmBuilder() const override { return spm_builder_; }
  pm4_builder::SqttBuilder* GetSqttBuilder() const override { return sqtt_builder_; }

  uint32_t GetShaderEnginesNumber() const override;
  uint32_t GetShaderArraysNumber() const override;
  uint32_t GetComputeUnitNumber() const override;
  uint32_t GetSQTTBufferAlignment() const override { return 0x1000; }
  const char* GetGFX() const override;

  bool IsGFX9() const override { return architecture_->IsGFX9(); }
  bool IsGFX10() const override { return architecture_->IsGFX10(); }
  bool IsGFX11() const override { return architecture_->IsGFX11(); }
  bool IsGFX12() const override { return architecture_->IsGFX12(); }

  uint32_t GetXccNumber() const override;
  uint32_t GetSpmSampleDelayMax() override;

  const GpuBlockInfo* GetBlockInfo(const aqlprofile_pmc_event_t* event) const override;
  const GpuBlockInfo* GetBlockInfo(const event_t* event) const override;
  const GpuBlockInfo* GetBlockInfo(const uint32_t& block_id) const override;

  size_t GetNumEvents(uint32_t block_name) const override;
  size_t GetBytesNeeded(uint32_t block_name) const override;

  uint32_t FindBlock(const char* name) const override;

  int GetNumWGPs() const override;
  int GetAccumLowID() const override;
  int GetAccumHiID() const override;

  // Access to architecture
  const HardwareArchitecture* GetArchitecture() const { return architecture_; }

 private:
  void InitializeBuilders();
  gpu_id_t MapToLegacyGpuId() const;

  HardwareArchitecture* architecture_;
  bool concurrent_mode_;
  bool spm_kfd_mode_;
  mutable std::string gfx_name_;
};

}  // namespace aql_profile

#endif  // SRC_CORE_PM4_FACTORY_ADAPTER_HPP_
