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

// Header file for CmdBuilder and CmdBuffer interfaces

#ifndef SRC_PM4_CMD_BUILDER_H_
#define SRC_PM4_CMD_BUILDER_H_

#include <stdint.h>
#include <string.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>
#include <atomic>
#include "util/reg_offsets.h"

#define APPEND_COMMAND_WRAPPER(cmdbuf, ...)  \
  do {                                       \
    PrintCommand(__FUNCTION__, __VA_ARGS__); \
    cmdbuf->Append(__VA_ARGS__);             \
  } while (false);

namespace pm4_builder {

namespace {

#if defined(DEBUG_TRACE)
template <typename T, typename std::enable_if<sizeof(T) % sizeof(uint32_t) == 0, int>::type = 0>
constexpr size_t PacketsSize(T&& packet) {
  return sizeof(packet) / sizeof(uint32_t);
}
template <typename T, typename... Ts>
constexpr size_t PacketsSize(T&& packet, Ts&&... rest) {
  return PacketsSize(std::forward<T>(packet)) + PacketsSize(std::forward<Ts>(rest)...);
}

template <typename T, typename std::enable_if<sizeof(T) % sizeof(uint32_t) == 0, int>::type = 0>
void PrintPackets(std::ostream& stream, T&& packet) {
  for (size_t i = 0; i < sizeof(packet) / sizeof(uint32_t); ++i)
    stream << " " << std::hex << std::right << std::setw(8) << std::setfill('0')
           << reinterpret_cast<const uint32_t*>(&packet)[i];
}
template <typename T, typename... Ts>
void PrintPackets(std::ostream& stream, T&& packet, Ts&&... rest) {
  PrintPackets(stream, std::forward<T>(packet));
  PrintPackets(stream, std::forward<Ts>(rest)...);
}
#endif

template <typename... Ts>
void PrintCommand(const char* function_name, Ts&&... packets) {
#if defined(DEBUG_TRACE)
  std::ostringstream oss;
  oss << "'" << function_name << "' size(" << std::dec << PacketsSize(std::forward<Ts>(packets)...)
      << ")";
  std::clog << std::setw(40) << std::left << oss.str() << ":";
  PrintPackets(std::clog, std::forward<Ts>(packets)...);
  std::clog << std::setfill(' ') << std::endl;
#endif
}

}  // namespace

/// @brief Implements the interface CmdBuffer and thus can be used to
/// translate various Gpu commands as byte stream.
/// @note: The Api does not require implementations to be thread safe.
/// Users are therefore required to be access in a serialized manner.
class CmdBuffer {
 public:
  /// @brief Append the command into the underlying buffer
  /// @param packet Buffer containing one or more instances of Gpu commands
  template <typename T, typename std::enable_if<sizeof(T) % sizeof(uint32_t) == 0, int>::type = 0>
  void Append(T&& packet) {
    size_t pos = data_.size();
    data_.resize(pos + sizeof(packet) / sizeof(uint32_t));
    memcpy(&data_[pos], &packet, sizeof(T));
  }
  template <typename... Ts>
  void Append(Ts&&... packets) {
    using expander = int[];
    (void)expander{0, (Append(std::forward<Ts>(packets)), 0)...};
  }

  /// @brief Append num_dwords of data to the buffer
  /// @param data_pointer points to the data to append
  /// @param num_dwords number of dwords
  void Append(uint32_t* data_pointer, uint32_t num_dwords) {
    size_t pos = data_.size();
    data_.resize(pos + num_dwords);
    memcpy(&data_[pos], data_pointer, num_dwords);
  }

  /// @brief Return size of Gpu commands in bytes in the underlying buffer
  size_t Size() const { return data_.size() * sizeof(data_[0]); }

  /// @brief Return size of Gpu commands in DWord in the underlying buffer
  size_t DwSize() const { return data_.size(); }

  void Assign(uint32_t index, uint32_t value) { data_[index] = value; }

  /// @brief Return address of the start of accumulated commands.
  const void* Data() const { return &data_[0]; }

  /// @brief Clear buffer.
  void Clear() { return data_.clear(); }

 private:
  /// @brief Defines Gpu command buffer as a vector of uint32_t
  std::vector<uint32_t> data_;
};

/// @brief Specifies the public interface of CmdBuilder for use by
/// clients to build Gpu command streams.
class CmdBuilder {
 public:
  CmdBuilder(const reg_base_offset_table* _table) : ip_offset_table(_table) {}

  /// @brief Build and copy a WaitIdle Gpu command into command buffer
  /// @param cmdbuf Pointer to command buffer to be appended
  virtual void BuildWriteWaitIdlePacket(CmdBuffer* cmdbuf) = 0;

  /// @brief Builds a Gpu command to wait until condition is realized
  /// @param cmdbuf command buffer to be appended with launch command
  /// @param mem_space if the address is in memory or is a register offset
  /// @param wait_addr address to wait on
  /// @param func_eq true means equal, false means not-equal
  /// @param mask_val Mask to apply on value from addr in comparison
  /// @param wait_val value to apply for the func given above
  virtual void BuildWaitRegMemCommand(CmdBuffer* cmdbuf, bool mem_space, uint64_t wait_addr,
                                      bool func_eq, uint32_t mask_val, uint32_t wait_val) = 0;

  /// @brief Build CP command to program a Gpu register
  /// @param cmdbuf Pointer to command buffer to be appended
  /// @param addr Register to be programmed
  /// @param value Value to write into register
  virtual void BuildWriteUConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) = 0;

  /// @brief Build and copy WriteShReg command
  /// @param cmdbuf Pointer to command buffer to be appended
  /// @param addr Offset of the register
  /// @param value Value to write into register
  virtual void BuildWriteShRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) = 0;

  /// @brief Build a Gpu command that copies data from a specified
  /// source register to destination
  /// @param cmdbuf Pointer to command buffer to be appended
  /// @param src_sel Source register selector
  /// @param src_reg_add 32-bit Source register address of the data to read from
  /// @param dst_addr Destination address for the data to be written to
  /// @param size Size of the data to be written
  /// @param  wait True if Gpu command should confirm the write operation
  /// operation has completed successfully
  virtual void BuildCopyRegDataPacket(CmdBuffer* cmdbuf, uint32_t src_reg_addr,
                                      const void* dst_addr, uint32_t size, bool wait) = 0;

  /// @brief Same as the above function but with 64-bit src_reg_addr.
  virtual void BuildCopyRegDataPacket(CmdBuffer* cmdbuf, uint64_t src_reg_addr,
                                      const void* dst_addr, uint32_t size, bool wait) {
    BuildCopyRegDataPacket(cmdbuf, uint32_t(src_reg_addr), dst_addr, size, wait);
  }

  /// @brief Builds the Gpu command to reference indirectly a stream
  /// of other Gpu commands. The launch command is then copied into
  /// the command buffer parameter.
  /// @param cmdBuf command buffer to be appended with launch command
  /// @param cmd_addr Address of command buffer carrying command stream
  /// @param cmd_size Size of dispatch command stream in bytes
  virtual void BuildIndirectBufferCmd(CmdBuffer* cmdbuf, const void* cmd_addr,
                                      std::size_t cmd_size) = 0;

  virtual void BuildMutexAcquirePacket(CmdBuffer* cmdbuf, size_t addr) = 0;
  virtual void BuildMutexReleasePacket(CmdBuffer* cmdbuf, size_t addr) = 0;

  virtual uint32_t MakeMutexSlot() {
    static std::atomic<size_t> slot{1};
    return uint32_t(slot.fetch_add(1)) | (1u << 31);
  }

  /// @brief Builds a NOP packet
  /// @param cmdBuf command buffer to be appended with launch command
  /// @param num_dwords number of dwords of the whole NOP packet, including the Header
  virtual void BuildNopPacket(CmdBuffer* cmdbuf, uint32_t num_dwords) = 0;

  /// @brief Builds a TT Finish Event
  /// @param cmdBuf  command buffer to be appended with launch command
  virtual void BuildThreadTraceEventFinish(CmdBuffer* cmdBuf) = 0;

  /// @brief Prime pages of ATCL2
  /// @param cmdBuf  command buffer to be appended with launch command
  virtual void BuildPrimeL2(CmdBuffer* cmdBuf, uint64_t addr) = 0;

  /// @brief Release resources used by CmdBuilder
  virtual ~CmdBuilder(){};

  bool bUsePerfCounterMode{true};
  // is this an MI300 command builder?

  uint32_t get_addr(const Register& reg) {
    if (!ip_offset_table) {
      std::cerr << "CmdBuilder::get_addr(): IP offset table is NULL!" << std::endl;
      std::abort();
    }
    return (*ip_offset_table)[reg];
  }

  virtual void BuildWriteConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) = 0;

 private:
  const reg_base_offset_table* const ip_offset_table;
};

/// @brief Returns the lower 32-bits of a value
constexpr uint32_t Low32(uint64_t u) { return static_cast<uint32_t>(u); }

/// @brief Returns the upper 32-bits of a value
constexpr uint32_t High32(uint64_t u) { return static_cast<uint32_t>(u >> 32); }

/// @brief Returns the lower 32-bits of an address
inline uint32_t PtrLow32(const void* p) { return Low32(reinterpret_cast<uintptr_t>(p)); }

/// @brief Returns the upper 32-bits of an address
inline uint32_t PtrHigh32(const void* p) { return High32(reinterpret_cast<uintptr_t>(p)); }

}  // namespace pm4_builder

#endif  // SRC_PM4_CMD_BUILDER_H_
