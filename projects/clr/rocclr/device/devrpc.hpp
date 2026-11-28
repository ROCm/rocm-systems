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

namespace amd {

constexpr uint64_t kMaxRpcPortCount = 16384;

struct RpcBufferInfo {
  void* buffer;
  uint32_t num_lanes;
  uint32_t port_count;
  amd::Device* device;
  device::Signal* signal = nullptr;
  device::Signal* interrupt = nullptr;
  bool owns_signal = false;
};

/// Returns true if the RPC headers were found at compile time, i.e. the
/// compiler was built with LLVM libc support.
bool rpcAvailable();

size_t getRpcBufferSize(uint32_t num_lanes, uint32_t num_ports);

/// Service pending RPC requests from the given buffer. Returns true if at least
/// one port was serviced.
bool processRpcBuffer(RpcBufferInfo* info);

/// Service one bounded pass of pending RPC requests from the given buffer.
void flushRpcBuffer(RpcBufferInfo* info);

/// Returns the expected size in bytes of the RPC client symbol.
size_t getRpcClientSize();

/// Construct an RPC client pointing at \p buffer and write the raw bytes to
/// \p staging.
void initRpcClient(void* staging, void* buffer, uint32_t num_ports);

/// Use \p signal for the RPC counter and \p interrupt_signal for the interrupt,
/// or \p signal for both when no interrupt signal is supplied.
void initRpcDoorbell(void* buffer, device::Signal& signal,
                     device::Signal* interrupt_signal = nullptr);

}  // namespace amd
