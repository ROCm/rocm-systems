// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_PM4_H_
#define ROCJITSU_VM_AMDGPU_PM4_H_

/// @file
/// @brief PM4 packets, shader registers and submission lifetime state.

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace rocjitsu::amdgpu {

class GraphicsDraw;

/// @brief Compute register offsets relative to the SH register aperture.
inline constexpr uint32_t kPm4ComputeStartX = 0x204;
inline constexpr uint32_t kPm4ComputeNumThreadX = 0x207;
inline constexpr uint32_t kPm4ComputeNumThreadY = 0x208;
inline constexpr uint32_t kPm4ComputeNumThreadZ = 0x209;
inline constexpr uint32_t kPm4ComputePgmLo = 0x20c;
inline constexpr uint32_t kPm4ComputePgmHi = 0x20d;
inline constexpr uint32_t kPm4ComputeScratchLo = 0x210;
inline constexpr uint32_t kPm4ComputeScratchHi = 0x211;
inline constexpr uint32_t kPm4ComputePgmRsrc1 = 0x212;
inline constexpr uint32_t kPm4ComputePgmRsrc2 = 0x213;
inline constexpr uint32_t kPm4ComputeTmpringSize = 0x218;
inline constexpr uint32_t kPm4ComputeUserData0 = 0x240;

/// @brief Type-3 packet opcodes accepted by the command processor.
enum class Pm4Opcode : uint32_t {
  Nop = 0x10,
  SetBase = 0x11,
  ClearState = 0x12,
  DrawIndex2 = 0x27,
  ContextControl = 0x28,
  DrawIndexAuto = 0x2d,
  NumInstances = 0x2f,
  PfpSyncMe = 0x42,
  SetContextReg = 0x69,
  SetContextRegPairs = 0xb8,
  SetContextRegPairsPacked = 0xb9,
  DispatchDirect = 0x15,
  DispatchIndirect = 0x16,
  WriteData = 0x37,
  WaitRegMem = 0x3c,
  IndirectBuffer = 0x3f,
  CopyData = 0x40,
  EventWrite = 0x46,
  ReleaseMem = 0x49,
  DmaData = 0x50,
  AcquireMem = 0x58,
  LoadShRegIndex = 0x63,
  SetShReg = 0x76,
  SetShRegIndex = 0x9b,
  SetUconfigReg = 0x79,
  SetUconfigRegIndex = 0x7a,
  SetShRegPairs = 0xba,
  SetUconfigRegPairs = 0xbe,
};

/// @brief Resident waves lease scratch slots independently of physical CU wave IDs.
/// @details A driver may size its scratch BO for a shader's occupancy, below the maximum
/// number of hardware wave slots. Retiring a wave makes its slot reusable.
class Pm4ScratchPool : public std::enable_shared_from_this<Pm4ScratchPool> {
public:
  explicit Pm4ScratchPool(uint32_t slots) {
    for (uint32_t i = slots; i; --i)
      free_.push_back(i - 1);
  }
  bool available(uint32_t count) {
    std::lock_guard lock(mutex_);
    return free_.size() >= count;
  }
  std::shared_ptr<uint32_t> acquire() {
    auto slot = std::make_unique<uint32_t>();
    {
      std::lock_guard lock(mutex_);
      if (free_.empty())
        throw std::runtime_error("PM4 scratch pool exhausted");
      *slot = free_.back();
      free_.pop_back();
    }
    return {slot.release(), [pool = shared_from_this()](uint32_t *slot) {
              std::lock_guard lock(pool->mutex_);
              pool->free_.push_back(*slot);
              delete slot;
            }};
  }

private:
  std::mutex mutex_;
  std::vector<uint32_t> free_;
};

/// @brief GPU address and remaining dword count of one indirect buffer.
struct Pm4IndirectBuffer {
  uint64_t address = 0;
  uint32_t dwords = 0;
};

/// @brief Shared launch/wave failure status; wake the CP to cancel the owning queue.
struct Pm4FailureState {
  std::atomic<bool> failed{false};
  std::function<void()> wake;
  void fail() {
    failed.store(true, std::memory_order_release);
    if (wake)
      wake();
  }
};

/// @brief One DRM command submission and its asynchronous fence callbacks.
/// @details The CP evaluates dependencies without blocking its engine thread and
/// publishes completion only after all commands retire.
struct Pm4Submission {
  bool graphics_engine = false;
  std::deque<Pm4IndirectBuffer> buffers;
  std::shared_ptr<Pm4FailureState> failure = std::make_shared<Pm4FailureState>();
  std::function<bool()> ready;
  std::function<void(bool)> complete;
};

/// @brief Shader and graphics register files and ordered submissions for one CP queue.
struct Pm4QueueState {
  uint64_t indirect_base = 0;
  uint32_t num_instances = 1;
  std::shared_ptr<GraphicsDraw> draw;
  std::array<uint32_t, 0x400> sh_registers{};
  std::array<uint32_t, 0x2000> context_registers{};
  std::array<uint32_t, 0x4000> uconfig_registers{};
  std::deque<Pm4Submission> submissions;
};

} // namespace rocjitsu::amdgpu

#endif
