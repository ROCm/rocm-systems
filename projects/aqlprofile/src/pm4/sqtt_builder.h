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

#ifndef SRC_PM4_SQTT_BUILDER_H_
#define SRC_PM4_SQTT_BUILDER_H_

#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "pm4/cmd_config.h"
#include "pm4/primitives_provider.hpp"
#include "trace_decoder_instrument.h"

#define SQTT_PERFCOUNTER_TOKEN (1u << 14)
#define SQTT_PERFCOUNTER_SIMD_MASK 24

namespace pm4_builder {
class CmdBuffer;
class CmdBuilder;

/* Class responsible for locking PM4 packets to a specific XCC (mask).
Starts locking future packets on constructor.
Stops locking when the destructor is called.
The builder and cmdbuffer must be valid for the entire lifetime os this class. */
class XCC_Packet_Lock {
 public:
  XCC_Packet_Lock(CmdBuilder& _builder, CmdBuffer* cmd_buffer, uint32_t xcc_number,
                  uint32_t xcc_mask);
  virtual ~XCC_Packet_Lock();

 private:
  CmdBuilder& builder_;
  CmdBuffer* cmd_buffer;
  uint32_t xcc_initial_cmd_size;
  uint32_t xcc_mask;
  uint32_t xcc_number;
};

// Thread traces status register indices to determine
// status of thread trace run

struct TraceControl
{
  uint32_t status{0};
  uint32_t cntr{0};
  uint32_t wptr{0};
  uint32_t status2{0};
  uint64_t gpu_clock_cnt_start{0};
  uint64_t gpu_clock_cnt_end{0};
  uint32_t status_double_buffer{0};
};

// Encapsulates the various Api and structures that are used to enable
// a thread trace session and collect its data. Implementations of this
// interface program device specific registers to realize the functionality
class SqttBuilder {
 public:
  // Destructor of the thread trace service handle
  virtual ~SqttBuilder() {}
  // Builds Pm4 command stream to program hardware registers that
  // enable a thread trace session, including the issue of an event
  // to begin thread session
  virtual void Begin(CmdBuffer* cmd_buffer, TraceConfig* config) = 0;
  // Builds Pm4 command stream to program hardware registers that
  // disable a thread trace session, including the issue of an event
  // to stop currently ongoing thread session
  virtual void End(CmdBuffer* cmd_buffer, TraceConfig* config) = 0;
  // Builds Pm4 command stream to program hardware registers that
  // inserts "data" into the SQTT buffer as USERDATA_2 (data_lo) and USERDATA_3 (data_hi)
  virtual hsa_status_t InsertCodeobjMarker(CmdBuffer* cmd_buffer, uint32_t data, unsigned channel) = 0;
  // Builds PM4 command stream to swap SQTT buffer to the next
  virtual void Swapbuffer(CmdBuffer* cmd_buffer, TraceConfig* config, void* addr, void* prev, int se_id, bool buf1) = 0;
  // Builds PM4 command stream to query status bit
  virtual void GetStatusPacket(CmdBuffer* cmd_buffer, TraceConfig* config, TraceControl& control, int se_id) = 0;

  virtual void InsertTimestampMarker(CmdBuffer* cmd_buffer, uint64_t* addr) {};

  // Returns TT_CONTROL_UTC_ERR_MASK
  virtual size_t GetUTCErrorMask() const = 0;
  // Returns TT_CONTROL_FULL_MASK
  virtual size_t GetBufferFullMask() const = 0;
  // Returns TT_LOCKDOWN_FAIL for double buffering
  virtual size_t GetLockDownFailMask() const = 0;
  // Returns TT_WRITE_PTR_MASK
  virtual size_t GetWritePtrMask() const = 0;
  // Returns size of block in bytes per increment in WPTR
  virtual size_t GetWritePtrBlk() const = 0;
  // Returns number of bits used for TTrace buffer alignement (e.g. 12 for 4KB alignment)
  virtual size_t BufferAlignment() const = 0;
};

class GpuSqttBuilder : public SqttBuilder {
  CmdBuilder* builder_;
  const PrimitivesProvider* prim_;

  void DebugTrace(uint32_t value);

 public:
  explicit GpuSqttBuilder(const AgentInfo* agent_info, CmdBuilder* builder,
                          const PrimitivesProvider* prim)
      : builder_(builder),
        prim_(prim),
        xcc_number_(agent_info->xcc_num),
        se_number_total(agent_info->se_num),
        timestamp_freq(agent_info->timestamp_freq),
        cu_per_se(agent_info->cu_num / agent_info->se_num) {}

  // Returns TT_CONTROL_UTC_ERR_MASK
  virtual size_t GetUTCErrorMask() const override { return prim_->GetTtControlUtcErrMask(); }
  // Returns TT_CONTROL_FULL_MASK
  virtual size_t GetBufferFullMask() const override { return prim_->GetTtControlFullMask(); }
  virtual size_t GetLockDownFailMask() const override { return prim_->GetTtLockdownFailMask(); }
  // Returns TT_WRITE_PTR_MASK
  virtual size_t GetWritePtrMask() const override { return prim_->GetTtWritePtrMask(); }
  // Returns size of block in bytes per increment in WPTR
  virtual size_t GetWritePtrBlk() const override { return 32; }
  // Returns number of bits used for TTrace buffer alignement (e.g. 12 for 4KB alignment)
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
  void GetStatusPacket(CmdBuffer* cmd_buffer, TraceConfig* config, TraceControl& control, int se_id) override;
  void Swapbuffer(CmdBuffer* cmd_buffer, TraceConfig* config, void* addr, void* prev, int se_id, bool buf1) override;

  size_t se_number_total{};
  size_t xcc_number_{};
  uint32_t timestamp_freq{};
  uint32_t cu_per_se{};
};

}  // namespace pm4_builder

#endif  // SRC_PM4_SQTT_BUILDER_H_
