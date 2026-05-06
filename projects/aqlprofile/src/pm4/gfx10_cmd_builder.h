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

#ifndef SRC_PM4_GFX10_CMD_BUILDER_H_
#define SRC_PM4_GFX10_CMD_BUILDER_H_

#include <assert.h>
#include "pm4/cmd_builder.h"

namespace pm4_builder {

/// @brief class Gfx10CmdBuilder implements the virtual class CmdBuilder
/// for GFX10 gpus
class Gfx10CmdBuilder : public CmdBuilder {
 private:
  static uint32_t MakePacket3Header(uint32_t opcode, size_t packet_size);

 public:
  Gfx10CmdBuilder(const reg_base_offset_table* _table) : CmdBuilder(_table){};

  static constexpr bool IsPrivilegedConfigReg(uint32_t addr) {
    return ((addr >= 0x00002000u) && (addr <= 0x00009fffu));
  }

  void BuildBarrierCommand(CmdBuffer* cmdBuf);
  void BuildWriteWaitIdlePacket(CmdBuffer* cmdbuf) override { BuildBarrierCommand(cmdbuf); }
  void BuildCacheFlushPacket(CmdBuffer* cmdbuf, size_t addr, size_t size);
  void BuildWaitRegMemCommand(CmdBuffer* cmdbuf, bool mem_space, uint64_t wait_addr, bool func_eq,
                              uint32_t mask_val, uint32_t wait_val) override;
  void BuildWriteShRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) override;
  void BuildCopyRegDataPacket(CmdBuffer* cmdbuf, uint32_t src_reg_addr, const void* dst_addr,
                              uint32_t size, bool wait) override;
  void BuildMutexAcquirePacket(CmdBuffer* cmdbuf, size_t addr) override;
  void BuildMutexReleasePacket(CmdBuffer* cmdbuf, size_t addr) override;
  void BuildPrimeL2(CmdBuffer* cmdBuf, uint64_t addr) override {}
  void BuildPredExecPacket(CmdBuffer* cmdbuf, uint32_t xcc_id = 0, uint32_t exec_count = 0) {}
  void BuildWriteUConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) override;
  void BuildWritePConfigRegPacket(CmdBuffer* cmdbuf, uint64_t addr, uint32_t value);
  void BuildWriteConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value);
  uint32_t BuildCopyCounterDataPacket(CmdBuffer* cmdbuf, uint64_t src_reg_addr_lo,
                                      uint64_t src_reg_addr_hi, const uint32_t* dst_addr,
                                      uint32_t dw_mask);
  void BuildWriteRegDataPacket(CmdBuffer* cmdbuf, uint32_t dst_reg_addr, const uint32_t* data,
                               uint32_t count, bool wait);
  void BuildNopPacket(CmdBuffer* cmdbuf, uint32_t num_dwords);
  void BuildThreadTraceEventFinish(CmdBuffer* cmdBuf) {}
  void BuildIndirectBufferCmd(CmdBuffer* cmdbuf, const void* cmd_addr, std::size_t cmd_size);

  void BuildWriteRegDataPacket(CmdBuffer* cmd, const Register& reg, const uint32_t* data,
                               uint32_t count, bool wait);
};

}  // namespace pm4_builder

#endif  //  SRC_PM4_GFX10_CMD_BUILDER_H_
