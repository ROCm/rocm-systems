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
#include "def/gfx10_def.h"
#include "pm4/cmd_builder.h"

namespace pm4_builder {
/// @brief class Gfx10CmdBuilder implements the virtual class CmdBuilder
/// for GFX10 gpus
class Gfx10CmdBuilder : public CmdBuilder {
 private:
  // @brief Return a Type 3 PM4 packet header
  static uint32_t MakePacket3Header(uint32_t opcode, size_t packet_size) {
    uint32_t count = packet_size / sizeof(uint32_t) - 2;
    uint32_t header = PACKET3(opcode, count);
    return header;
  }

 public:
  Gfx10CmdBuilder(const reg_base_offset_table* _table) : CmdBuilder(_table){};
  static constexpr bool IsPrivilegedConfigReg(uint32_t addr) {
    return ((addr >= CONFIG_SPACE_START) && (addr <= CONFIG_SPACE_END));
  }

  void BuildBarrierCommand(CmdBuffer* cmdBuf) {
    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_EVENT_WRITE, 2 * sizeof(uint32_t));

    uint32_t dword2 =
        PACKET3_EVENT_WRITE__EVENT_TYPE(CS_PARTIAL_FLUSH) |
        PACKET3_EVENT_WRITE__EVENT_INDEX(PACKET3_EVENT_WRITE__EVENT_INDEX__CS_PARTIAL_FLUSH);

    // build the pm4mec_event_write command which has 2 Dwords
    uint32_t pm4mec_event_write_cmd[2] = {header, dword2};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdBuf, pm4mec_event_write_cmd);
  }

  void BuildWriteWaitIdlePacket(CmdBuffer* cmdbuf) override { BuildBarrierCommand(cmdbuf); }

  void BuildCacheFlushPacket(CmdBuffer* cmdbuf, size_t addr, size_t size) {
    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_ACQUIRE_MEM, 8 * sizeof(uint32_t));

    // Specify the base address of memory to invalidate.
    // The address must be 256 byte aligned.
    uint32_t dword5 = PACKET3_ACQUIRE_MEM__COHER_BASE_LO((uint32_t(addr >> 8)));
    uint32_t dword6 = PACKET3_ACQUIRE_MEM__COHER_BASE_HI((uint8_t(addr >> 40)));

    // Specify the size of memory to invalidate.
    // Size is specified in terms of 256 byte chunks. A coher_size
    // of 0xFFFFFFFF actually specified 0xFFFFFFFF00 (40 bits)
    // of memory. The field coher_size_hi specifies memory from
    // bits 40-64 for a total of 256 TB.
    size = ((addr % 256 + size) >> 8) + ((size + 0xFF) >> 8) - (size >> 8);
    uint32_t dword3 = PACKET3_ACQUIRE_MEM__COHER_SIZE((uint32_t(size)));
    uint32_t dword4 = PACKET3_ACQUIRE_MEM__COHER_SIZE_HI((uint32_t(size >> 32)));

    // Specify the poll interval for determing if operation is complete
    uint32_t dword7 = PACKET3_ACQUIRE_MEM__POLL_INTERVAL(0x10);

    // Program GCR Control Register. Initialize L2 Cache flush
    // for Non-Coherent memory blocks
    uint32_t dword8 = PACKET3_ACQUIRE_MEM__GCR_CNTL(
        ((GCR_CNTL__SEQ_FORWARD & GCR_CNTL__SEQ_MASK) | GCR_CNTL__GL2_WB_MASK));

    // build the pm4mec_acquire_mem command which has 8 Dwords
    uint32_t pm4mec_acquire_mem_cmd[8] = {header, 0,      dword3, dword4,
                                          dword5, dword6, dword7, dword8};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_acquire_mem_cmd);
  }

  void BuildWaitRegMemCommand(CmdBuffer* cmdbuf, bool mem_space, uint64_t wait_addr, bool func_eq,
                              uint32_t mask_val, uint32_t wait_val) override {
    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_WAIT_REG_MEM, 7 * sizeof(uint32_t));

    uint32_t dword7 = PACKET3_WAIT_REG_MEM__POLL_INTERVAL(0x04);
    uint32_t dword2_operation =
        PACKET3_WAIT_REG_MEM__OPERATION(PACKET3_WAIT_REG_MEM__OPERATION__WAIT_REG_MEM);

    // Apply the space to which addr belongs
    uint32_t dword2_mem_space =
        mem_space
            ? PACKET3_WAIT_REG_MEM__MEM_SPACE(PACKET3_WAIT_REG_MEM__MEM_SPACE__MEMORY_SPACE)
            : PACKET3_WAIT_REG_MEM__MEM_SPACE(PACKET3_WAIT_REG_MEM__MEM_SPACE__REGISTER_SPACE);

    // Apply the function - equal / not equal desired by user
    uint32_t dword2_function =
        func_eq ? PACKET3_WAIT_REG_MEM__FUNCTION(
                      PACKET3_WAIT_REG_MEM__FUNCTION__EQUAL_TO_THE_REFERENCE_VALUE)
                : PACKET3_WAIT_REG_MEM__FUNCTION(
                      PACKET3_WAIT_REG_MEM__FUNCTION__NOT_EQUAL_REFERENCE_VALUE);

    uint32_t dword2 = dword2_operation | dword2_mem_space | dword2_function;

    // Apply the mask on value at address/register
    uint32_t dword6 = PACKET3_WAIT_REG_MEM__MASK(mask_val);

    // Value to use in applying equal / not equal function
    uint32_t dword5 = PACKET3_WAIT_REG_MEM__REFERENCE(wait_val);

    // The address to poll should be DWord (4 byte) aligned
    // Update upper 32 bit address if addr is not a register
    uint32_t dword3 = 0;
    uint32_t dword4 = 0;
    if (mem_space) {
      assert(!(wait_addr & 0x3) && "WaitRegMem address must be 4 byte aligned");
      dword3 = PACKET3_WAIT_REG_MEM__MEM_POLL_ADDR_LO((Low32(wait_addr) >> 2));
      dword4 = PACKET3_WAIT_REG_MEM__MEM_POLL_ADDR_HI(High32(wait_addr));
    } else
      dword3 = PACKET3_WAIT_REG_MEM__REG_POLL_ADDR(wait_addr);

    // build the pm4mec_wait_reg_mem command which has 7 Dwords
    uint32_t pm4mec_wait_reg_mem_cmd[7] = {header, dword2, dword3, dword4, dword5, dword6, dword7};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_wait_reg_mem_cmd);
  }

  void BuildWriteShRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) override {
    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_SET_SH_REG, 2 * sizeof(uint32_t) + sizeof(value));
    uint32_t dword2 = PACKET3_SET_SH_REG__REG_OFFSET((addr - PERSISTENT_SPACE_START)) |
                      PACKET3_SET_SH_REG__INDEX(PACKET3_SET_SH_REG__INDEX__DEFAULT);

    // build the pm4mec_set_sh_reg command which has 3 Dwords
    uint32_t pm4mec_set_sh_reg_cmd[3] = {header, dword2, value};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_set_sh_reg_cmd);
  }

  void BuildCopyRegDataPacket(CmdBuffer* cmdbuf, uint32_t src_reg_addr, const void* dst_addr,
                              uint32_t size, bool wait) override {
    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_COPY_DATA, 6 * sizeof(uint32_t));

    uint32_t dword2 =
        (IsPrivilegedConfigReg(src_reg_addr)
             ? PACKET3_COPY_DATA__SRC_SEL(PACKET3_COPY_DATA__SRC_SEL__PERFCOUNTERS)
             : PACKET3_COPY_DATA__SRC_SEL(PACKET3_COPY_DATA__SRC_SEL__MEM_MAPPED_REGISTER)) |
        PACKET3_COPY_DATA__SRC_CACHE_POLICY(PACKET3_COPY_DATA__SRC_CACHE_POLICY__LRU) |
        PACKET3_COPY_DATA__DST_SEL(PACKET3_COPY_DATA__DST_SEL__TC_L2) |
        PACKET3_COPY_DATA__DST_CACHE_POLICY(PACKET3_COPY_DATA__DST_CACHE_POLICY__LRU) |
        (wait ? PACKET3_COPY_DATA__WR_CONFIRM(PACKET3_COPY_DATA__WR_CONFIRM__WAIT_FOR_CONFIRMATION)
              : PACKET3_COPY_DATA__WR_CONFIRM(
                    PACKET3_COPY_DATA__WR_CONFIRM__DO_NOT_WAIT_FOR_CONFIRMATION)) |
        ((size == 0) ? PACKET3_COPY_DATA__COUNT_SEL(PACKET3_COPY_DATA__COUNT_SEL__32_BITS_OF_DATA)
                     : PACKET3_COPY_DATA__COUNT_SEL(PACKET3_COPY_DATA__COUNT_SEL__64_BITS_OF_DATA));

    // Specify the source register offset
    uint32_t dword3 = PACKET3_COPY_DATA__SRC_REG_OFFSET(src_reg_addr);

    // Specify the destination memory address
    uint32_t dword6 = PACKET3_COPY_DATA__DST_ADDR_HI(PtrHigh32(dst_addr));
    uint32_t dword5 = 0;
    if (size == 0) {
      dword5 |= PACKET3_COPY_DATA__DST_32B_ADDR_LO((PtrLow32(dst_addr) >> 2));
    } else {
      dword5 |= PACKET3_COPY_DATA__DST_64B_ADDR_LO((PtrLow32(dst_addr) >> 3));
    }

    // build the pm4mec_copy_data command which has 6 Dwords
    uint32_t pm4mec_copy_data_cmd[6] = {header, dword2, dword3, 0, dword5, dword6};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_copy_data_cmd);
  }

  void BuildMutexAcquirePacket(CmdBuffer* cmdbuf, size_t addr) override {
    constexpr uint32_t GL2_OP_ATOMIC_CMPSWAP_RTN_32 = 8;
    uint32_t header = MakePacket3Header(PACKET3_ATOMIC_MEM, 9 * sizeof(uint32_t));

    uint32_t dword2 =
        PACKET3_ATOMIC_MEM__COMMAND(PACKET3_ATOMIC_MEM__COMMAND__LOOP_UNTIL_COMPARE_SATISFIED) |
        PACKET3_ATOMIC_MEM__ATOMIC(GL2_OP_ATOMIC_CMPSWAP_RTN_32);
    uint32_t dword9 = PACKET3_ATOMIC_MEM__LOOP_INTERVAL(4);

    uint32_t dword3 = PACKET3_ATOMIC_MEM__ADDR_LO(uint32_t(addr));
    uint32_t dword4 = PACKET3_ATOMIC_MEM__ADDR_HI((addr >> 32));
    uint32_t dword5 = PACKET3_ATOMIC_MEM__SRC_DATA_LO(MakeMutexSlot());
    uint32_t dword6 = PACKET3_ATOMIC_MEM__CMP_DATA_LO(0);

    // build the pm4mec_atomic_mem command which has 9 Dwords
    uint32_t pm4_mec_atomic_mem_cmd[9] = {header, dword2, dword3, dword4, dword5,
                                          dword6, 0,      0,      dword9};
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4_mec_atomic_mem_cmd);
  }

  void BuildMutexReleasePacket(CmdBuffer* cmdbuf, size_t addr) override {
    constexpr uint32_t GL2_OP_ATOMIC_SWAP_RTN_32 = 7;
    uint32_t header = MakePacket3Header(PACKET3_ATOMIC_MEM, 9 * sizeof(uint32_t));

    uint32_t dword2 = PACKET3_ATOMIC_MEM__COMMAND(PACKET3_ATOMIC_MEM__COMMAND__SINGLE_PASS_ATOMIC) |
                      PACKET3_ATOMIC_MEM__ATOMIC(GL2_OP_ATOMIC_SWAP_RTN_32);

    uint32_t dword3 = PACKET3_ATOMIC_MEM__ADDR_LO(uint32_t(addr));
    uint32_t dword4 = PACKET3_ATOMIC_MEM__ADDR_HI((addr >> 32));
    uint32_t dword5 = PACKET3_ATOMIC_MEM__SRC_DATA_LO(0);

    // build the pm4mec_atomic_mem command which has 9 Dwords
    uint32_t pm4_mec_atomic_mem_cmd[9] = {header, dword2, dword3, dword4, dword5, 0, 0, 0, 0};
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4_mec_atomic_mem_cmd);
  }

  void BuildPrimeL2(CmdBuffer* cmdBuf, uint64_t addr) override {};

  void BuildPredExecPacket(CmdBuffer* cmdbuf, uint32_t xcc_id = 0, uint32_t exec_count = 0) {}

  void BuildWriteUConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) override {
    // Initialize the command header
    uint32_t header =
        MakePacket3Header(PACKET3_SET_UCONFIG_REG, 2 * sizeof(uint32_t) + sizeof(value));
    uint32_t dword2 = PACKET3_SET_UCONFIG_REG__REG_OFFSET((addr - UCONFIG_SPACE_START));

    // build the pm4mec_set_uconfig_reg command which has 3 Dwords
    uint32_t pm4mec_set_uconfig_reg_cmd[3] = {header, dword2, value};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_set_uconfig_reg_cmd);
  }

  void BuildWritePConfigRegPacket(CmdBuffer* cmdbuf, uint64_t addr, uint32_t value) {
    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_COPY_DATA, 6 * sizeof(uint32_t));

    uint32_t dword2 =
        PACKET3_COPY_DATA__SRC_SEL(PACKET3_COPY_DATA__SRC_SEL__IMMEDIATE_DATA) |
        PACKET3_COPY_DATA__SRC_CACHE_POLICY(PACKET3_COPY_DATA__SRC_CACHE_POLICY__LRU) |
        (IsPrivilegedConfigReg(addr) && bUsePerfCounterMode
             ? PACKET3_COPY_DATA__DST_SEL(PACKET3_COPY_DATA__DST_SEL__PERFCOUNTERS)
             : PACKET3_COPY_DATA__DST_SEL(PACKET3_COPY_DATA__DST_SEL__MEM_MAPPED_REGISTER)) |
        PACKET3_COPY_DATA__DST_CACHE_POLICY(PACKET3_COPY_DATA__DST_CACHE_POLICY__LRU) |
        PACKET3_COPY_DATA__WR_CONFIRM(PACKET3_COPY_DATA__WR_CONFIRM__DO_NOT_WAIT_FOR_CONFIRMATION) |
        PACKET3_COPY_DATA__COUNT_SEL(PACKET3_COPY_DATA__COUNT_SEL__32_BITS_OF_DATA);

    uint32_t dword3 = PACKET3_COPY_DATA__IMM_DATA(value);

    // extend register address to all 64-bit
    uint32_t dword5 = Low32(addr);
    uint32_t dword6 = PACKET3_COPY_DATA__DST_ADDR_HI(High32(addr));

    // build the pm4mec_copy_data command which has 6 Dwords
    uint32_t pm4mec_copy_data_cmd[6] = {header, dword2, dword3, 0, dword5, dword6};

    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_copy_data_cmd);
  }

  void BuildWriteConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) {
    return IsPrivilegedConfigReg(addr) ? BuildWritePConfigRegPacket(cmdbuf, addr, value)
                                       : BuildWriteUConfigRegPacket(cmdbuf, addr, value);
  }

  uint32_t BuildCopyCounterDataPacket(CmdBuffer* cmdbuf, uint64_t src_reg_addr_lo,
                                      uint64_t src_reg_addr_hi, const void* dst_addr,
                                      uint32_t dw_mask) {
    uint32_t read_counter = 0;
    if (dw_mask & 0x1) {
      BuildCopyRegDataPacket(cmdbuf, src_reg_addr_lo, (uint32_t*)dst_addr + read_counter,
                             PACKET3_COPY_DATA__COUNT_SEL__32_BITS_OF_DATA, false);
      ++read_counter;
    }
    if (dw_mask & 0x2) {
      BuildCopyRegDataPacket(cmdbuf, src_reg_addr_hi, (uint32_t*)dst_addr + read_counter,
                             PACKET3_COPY_DATA__COUNT_SEL__32_BITS_OF_DATA, false);
      ++read_counter;
    }
    return read_counter;
  }

  void BuildWriteRegDataPacket(CmdBuffer* cmdbuf, uint32_t dst_reg_addr, const uint32_t* data,
                               uint32_t count, bool wait) {
    // Initialize the command header
    uint32_t header =
        MakePacket3Header(PACKET3_WRITE_DATA, 4 * sizeof(uint32_t) + count * sizeof(data[0]));

    // ordinal2
    uint32_t dword2 =
        PACKET3_WRITE_DATA__DST_SEL(
            PACKET3_WRITE_DATA__DST_SEL__MEM_MAPPED_REGISTER) |  // mem-mapped reg
        PACKET3_WRITE_DATA__ADDR_INCR(
            PACKET3_WRITE_DATA__ADDR_INCR__DO_NOT_INCREMENT_ADDRESS) |  // not increment address
        (wait ? PACKET3_WRITE_DATA__WR_CONFIRM(
                    PACKET3_WRITE_DATA__WR_CONFIRM__WAIT_FOR_WRITE_CONFIRMATION)
              : PACKET3_WRITE_DATA__WR_CONFIRM(
                    PACKET3_WRITE_DATA__WR_CONFIRM__DO_NOT_WAIT_FOR_WRITE_CONFIRMATION));

    // ordinal3
    uint32_t dword3 = PACKET3_WRITE_DATA__DST_MMREG_ADDR(dst_reg_addr);  // mem-mapped reg

    // ordinal4
    uint32_t dword4 = PACKET3_WRITE_DATA__DST_MEM_ADDR_HI(0);  // mem-mapped reg

    // build the pm4mec_write_data command which has 4 Dwords
    uint32_t pm4mec_write_data_cmd[4] = {header, dword2, dword3, dword4};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_write_data_cmd);

    // appending data dwords
    for (uint32_t i = 0; i < count; ++i) {
      APPEND_COMMAND_WRAPPER(cmdbuf, data[i]);
    }

    if (count & 1) {
      // Insert a NOP spacer
      BuildNopPacket(cmdbuf, 1);
    }
  }

  void BuildNopPacket(CmdBuffer* cmdbuf, uint32_t num_dwords) {
    uint32_t header = MakePacket3Header(PACKET3_NOP, num_dwords * sizeof(uint32_t));
    APPEND_COMMAND_WRAPPER(cmdbuf, header);
    if (num_dwords > 1) {
      std::vector<uint32_t> data_block((num_dwords - 1), 0);
      APPEND_COMMAND_WRAPPER(cmdbuf, data_block.data(), (num_dwords - 1));
    }
  }

  void BuildThreadTraceEventFinish(CmdBuffer* cmdBuf) {}

  void BuildIndirectBufferCmd(CmdBuffer* cmdbuf, const void* cmd_addr, std::size_t cmd_size) {
    // Verify the address is 4-byte aligned
    assert(!(uintptr_t(cmd_addr) & 0x3) && "IndirectBuffer address must be 4 byte aligned");

    // Initialize the command header
    uint32_t header = MakePacket3Header(PACKET3_INDIRECT_BUFFER, 4 * sizeof(uint32_t));

    // Specify the address of indirect buffer encoding cmd stream
    uint32_t dword2 = PACKET3_INDIRECT_BUFFER__IB_BASE_LO((PtrLow32(cmd_addr) >> 2));
    uint32_t dword3 = PACKET3_INDIRECT_BUFFER__IB_BASE_HI(PtrHigh32(cmd_addr));

    // Specify the size of indirect buffer and cache policy to set
    // upon executing the cmds of indirect buffer
    uint32_t dword4 =
        PACKET3_INDIRECT_BUFFER__VALID(1) |
        PACKET3_INDIRECT_BUFFER__IB_SIZE((cmd_size / sizeof(uint32_t))) |
        PACKET3_INDIRECT_BUFFER__CACHE_POLICY(PACKET3_INDIRECT_BUFFER__CACHE_POLICY__STREAM);

    // build the pm4mec_indirect_buffer command which has 4 Dwords
    uint32_t pm4mec_indirect_buffer_cmd[4] = {header, dword2, dword3, dword4};
    // Append the built command into output Command Buffer
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4mec_indirect_buffer_cmd);
  }

  void BuildWriteShRegPacket(CmdBuffer* cmd, const Register& reg, uint32_t value) {
    BuildWriteShRegPacket(cmd, get_addr(reg), value);
  }

  void BuildWriteUConfigRegPacket(CmdBuffer* cmd, const Register& reg, uint32_t value) {
    BuildWriteUConfigRegPacket(cmd, get_addr(reg), value);
  }

  void BuildWritePConfigRegPacket(CmdBuffer* cmd, const Register& reg, uint32_t value) {
    BuildWritePConfigRegPacket(cmd, get_addr(reg), value);
  }

  void BuildCopyCounterDataPacket(CmdBuffer* cmd, const Register& reg_lo, const Register& reg_hi,
                                  const void* dst_addr, uint32_t mask) {
    BuildCopyCounterDataPacket(cmd, (mask & 1 << 0) ? get_addr(reg_lo) : 0,
                               (mask & 1 << 1) ? get_addr(reg_hi) : 0, dst_addr, mask);
  }

  void BuildWriteConfigRegPacket(CmdBuffer* cmd, const Register& reg, uint32_t value) {
    BuildWriteConfigRegPacket(cmd, get_addr(reg), value);
  }

  void BuildWaitRegMemCommand(CmdBuffer* cmd, bool mem_space, const Register& wait_addr,
                              bool func_eq, uint32_t mask_val, uint32_t wait_val) {
    BuildWaitRegMemCommand(cmd, mem_space, get_addr(wait_addr), func_eq, mask_val, wait_val);
  }

  void BuildWriteRegDataPacket(CmdBuffer* cmd, const Register& reg, const uint32_t* data,
                               uint32_t count, bool wait) {
    BuildWriteRegDataPacket(cmd, get_addr(reg), data, count, wait);
  }

  void BuildCopyRegDataPacket(CmdBuffer* cmd, const Register& reg, const void* dst_addr,
                              uint32_t size, bool wait) {
    BuildCopyRegDataPacket(cmd, get_addr(reg), dst_addr, size, wait);
  }
};
}  // namespace pm4_builder

#endif  //  SRC_PM4_GFX10_CMD_BUILDER_H_
