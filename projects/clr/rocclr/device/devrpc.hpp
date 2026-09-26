/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "top.hpp"
#include "device/device.hpp"
#include <cstddef>
#include <cstdint>

/// ROCm-specific RPC opcodes that replace the legacy hostcall buffer services.
enum : uint32_t {
  /// Calls a host-side function pointer with up to seven uint64_t arguments and
  /// returns up to two uint64_t results.
  ///   [in]  data[0]    = function pointer  void (*)(uint64_t *out, const uint64_t *in)
  ///   [in]  data[1..7] = input arguments
  ///   [out] data[0..1] = output values
  ROCM_HOSTCALL_FUNCTION = ('r' << 24) | 0,

  /// Device memory allocation and deallocation.  Replaces SERVICE_DEVMEM.
  ///   [in]  data[0] = address (non-zero to free this VA)
  ///   [in]  data[1] = size    (when data[0] == 0, allocate this many bytes)
  ///   [out] data[0] = virtual address of the new allocation, or 0
  ROCM_HOSTCALL_DEVMEM = ('r' << 24) | 1,

  /// Address-sanitizer violation report.
  ///   [in]  data[0] = faulting address
  ///   [in]  data[1] = program counter
  ///   [in]  data[2] = workgroup index X
  ///   [in]  data[3] = workgroup index Y
  ///   [in]  data[4] = workgroup index Z
  ///   [in]  data[5] = wave ID
  ///   [in]  data[6] = access info (bit 0 = is_write; upper 32 bits nonzero = non-fatal)
  ///   [in]  data[7] = access size in bytes
  ROCM_HOSTCALL_SANITIZER = ('r' << 24) | 2,

  /// Hostcall-style printf/fprintf using the message descriptor protocol.
  ///   [in]  data[0]    = message descriptor (BEGIN/END flags, length, 56-bit ID)
  ///   [in]  data[1..7] = payload elements (count given by descriptor length field)
  ///   [out] data[0]    = updated descriptor (with assigned ID) or printf return value
  ///   [out] data[1]    = second return value (on END)
  ROCM_HOSTCALL_PRINTF = ('r' << 24) | 3,
};

namespace amd {

class MessageHandler;

constexpr uint64_t kMaxRpcPortCount = 16384;

struct RpcBufferInfo {
  void* buffer;
  uint32_t num_lanes;
  uint32_t port_count;
  const amd::Device* device;
  MessageHandler* messages = nullptr;
};

/// Returns true if the RPC headers were found at compile time, i.e. the
/// compiler was built with LLVM libc support.
bool rpcAvailable();

size_t getRpcBufferSize(uint32_t num_lanes, uint32_t num_ports);

/// Service pending RPC requests from the given buffer. Returns true if at least
/// one port was serviced.
bool processRpcBuffer(RpcBufferInfo* info);

/// Drain all pending RPC requests from the given buffer.
void flushRpcBuffer(RpcBufferInfo* info);

/// Returns the expected size in bytes of the RPC client symbol.
size_t getRpcClientSize();

/// Construct an RPC client pointing at \p buffer and write the raw bytes to
/// \p staging.
void initRpcClient(void* staging, void* buffer, uint32_t num_ports);

/// Extract the interrupt doorbell fields from \p signal and write them into the
/// RPC buffer so the device can fire interrupts to wake the server thread.
void initRpcDoorbell(void* buffer, void* signal_handle);

}  // namespace amd
