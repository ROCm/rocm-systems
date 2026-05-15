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

#ifndef SRC_PM4_GFX12_SQTT_BUILDER_HPP_
#define SRC_PM4_GFX12_SQTT_BUILDER_HPP_

#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "core/hw/hardware_config.hpp"
#include "pm4/cmd_builder.h"
#include "pm4/primitives_provider.hpp"
#include "pm4/sqtt_builder.h"

namespace pm4_builder {

/* Class responsible for locking PM4 packets to a specific XCC (mask).
Non-template variant using CmdBuilder* for use by Gfx12SqttBuilder.
Starts locking future packets on constructor.
Stops locking when the destructor is called.
The builder and cmdbuffer must be valid for the entire lifetime of this class. */
class Gfx12XccPacketLock {
 public:
  Gfx12XccPacketLock(CmdBuilder& _builder, CmdBuffer* cmd_buffer, uint32_t xcc_number,
                     uint32_t xcc_mask);
  virtual ~Gfx12XccPacketLock();

 private:
  CmdBuilder& builder_;
  CmdBuffer* cmd_buffer;
  uint32_t xcc_initial_cmd_size;
  uint32_t xcc_mask;
  uint32_t xcc_number;
};

/// Non-template SQTT builder for GFX12, using PrimitivesProvider for runtime dispatch.
class Gfx12SqttBuilder : public SqttBuilder {
  CmdBuilder* builder_;
  const PrimitivesProvider* prim_;

 public:
  explicit Gfx12SqttBuilder(const aql_profile::HardwareConfig& config, CmdBuilder* builder,
                             const PrimitivesProvider* prim, uint32_t timestamp_freq_hz)
      : builder_(builder),
        prim_(prim),
        xcc_number_(config.xcc_count),
        is_multi_xcc_(config.IsMultiXCC()),
        se_number_total(config.se_count),
        timestamp_freq(timestamp_freq_hz),
        cu_per_se(config.cu_count / config.se_count) {}

  virtual size_t GetUTCErrorMask() const override { return prim_->GetTtControlUtcErrMask(); }
  virtual size_t GetBufferFullMask() const override { return prim_->GetTtControlFullMask(); }
  virtual size_t GetLockDownFailMask() const override { return prim_->GetTtLockdownFailMask(); }
  virtual size_t GetWritePtrMask() const override { return prim_->GetTtWritePtrMask(); }
  virtual size_t GetWritePtrBlk() const override { return 32; }
  virtual size_t BufferAlignment() const override { return prim_->GetTtBuffAlignShift(); }

  void SetGRBMToBroadcast(CmdBuffer* cmd_buffer);
  void Select_GRBM_SE_SH0(CmdBuffer* cmd_buffer, int se_index);
  void StartPerfMon(CmdBuffer* cmd_buffer, TraceConfig* config);
  void StopPerfMon(CmdBuffer* cmd_buffer);

  void Begin(CmdBuffer* cmd_buffer, TraceConfig* config) override;
  void End(CmdBuffer* cmd_buffer, TraceConfig* config) override;
  void ReadValues(CmdBuffer* cmd_buffer, const TraceConfig* config, size_t se_index);

  uint32_t GetXCCNumber() const { return xcc_number_; }
  uint64_t PopCount(uint64_t se_mask) const;
  bool isXccEnabled(int xcc, uint64_t se_number_xcc, TraceConfig* config);
  uint64_t GetBaseStep(TraceConfig* config) const;

  virtual hsa_status_t InsertCodeobjMarker(CmdBuffer* cmd_buffer, uint32_t data,
                                           unsigned channel) override;
  virtual void InsertTimestampMarker(CmdBuffer* cmd_buffer, uint64_t* addr) override;
  void WriteConfigPacket(CmdBuffer* cmdbuf, const Register& reg, uint32_t value);
  void GetStatusPacket(CmdBuffer* cmd_buffer, TraceConfig* config, TraceControl& control,
                       int se_id) override;
  void Swapbuffer(CmdBuffer* cmd_buffer, TraceConfig* config, void* addr, void* prev, int se_id,
                  bool buf1) override;

  size_t se_number_total{};
  size_t xcc_number_{};
  bool is_multi_xcc_{};
  uint32_t timestamp_freq{};
  uint32_t cu_per_se{};
};

}  // namespace pm4_builder

#endif  // SRC_PM4_GFX12_SQTT_BUILDER_HPP_
