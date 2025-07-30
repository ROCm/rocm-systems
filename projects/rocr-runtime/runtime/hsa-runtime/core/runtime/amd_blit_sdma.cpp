////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_blit_sdma.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/inc/sdma_registers.h"
#include "core/inc/signal.h"
#include "core/inc/interrupt_signal.h"
#include "core/inc/default_signal.h"

namespace rocr {
namespace AMD {

inline uint32_t ptrlow32(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

inline uint32_t ptrhigh32(const void* p) {
#if defined(HSA_LARGE_MODEL)
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) >> 32);
#else
  return 0;
#endif
}

const size_t BlitSdmaBase::kQueueSize = 1024 * 1024;
const size_t BlitSdmaBase::kCopyPacketSize = sizeof(SDMA_PKT_COPY_LINEAR);
const size_t BlitSdmaBase::kMaxSingleCopySize = SDMA_PKT_COPY_LINEAR::kMaxSize_;
const size_t BlitSdmaBase::kMaxSingleFillSize = SDMA_PKT_CONSTANT_FILL::kMaxSize_;

// Initialize size of various sDMA commands use by this module
template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::linear_copy_command_size_ = sizeof(SDMA_PKT_COPY_LINEAR);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::fill_command_size_ = sizeof(SDMA_PKT_CONSTANT_FILL);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::fence_command_size_ = sizeof(SDMA_PKT_FENCE);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::poll_command_size_ = sizeof(SDMA_PKT_POLL_REGMEM);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::flush_command_size_ = sizeof(SDMA_PKT_POLL_REGMEM);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::atomic_command_size_ = sizeof(SDMA_PKT_ATOMIC);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::timestamp_command_size_ = sizeof(SDMA_PKT_TIMESTAMP);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::trap_command_size_ = sizeof(SDMA_PKT_TRAP);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
const uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                        useGCR>::gcr_command_size_ = sizeof(SDMA_PKT_GCR);

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BlitSdma()
    : agent_(NULL),
      queue_start_addr_(NULL),
      bytes_queued_(0),
      parity_(false),
      cached_reserve_index_(0),
      cached_commit_index_(0),
      platform_atomic_support_(true),
      hdp_flush_support_(false),
      gang_leader_(false),
      is_ganged_(false),
      min_submission_size_(0) {
  std::memset(&queue_resource_, 0, sizeof(queue_resource_));
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::~BlitSdma() {}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::Initialize(
    const core::Agent& agent, bool use_xgmi, size_t linear_copy_size_override, int rec_eng) {
  if (queue_start_addr_ != NULL) {
    // Already initialized.
    return HSA_STATUS_SUCCESS;
  }

  if (agent.device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR;
  }

  agent_ = reinterpret_cast<AMD::GpuAgent*>(&const_cast<core::Agent&>(agent));

  if (HSA_PROFILE_FULL == agent_->profile()) {
    assert(false && "Only support SDMA for dgpu currently");
    return HSA_STATUS_ERROR;
  }

  // Some GFX9 devices require a minimum of 64 DWORDS per ring buffer submission.
  if (agent_->supported_isas()[0]->GetVersion() >= core::Isa::Version(9, 0, 0) &&
     (agent_->supported_isas()[0]->GetVersion() <= core::Isa::Version(9, 0, 4) ||
     agent_->supported_isas()[0]->GetVersion() == core::Isa::Version(9, 0, 12))) {
    min_submission_size_ = 256;
  }

  const core::Runtime::LinkInfo& link =
            core::Runtime::runtime_singleton_->GetLinkInfo( agent_->node_id(),
                core::Runtime::runtime_singleton_->cpu_agents()[0]->node_id());
  if (agent_->supported_isas()[0]->GetVersion() == core::Isa::Version(7, 0, 1)) {
    platform_atomic_support_ = false;
  } else {
    platform_atomic_support_ = link.info.atomic_support_64bit;
  }

  // HDP flush supported on gfx900 and forward.
  // gfx90a can support xGMI host to device connections so bypass HDP flush
  // in this case.
  // gfx101x seems to have issues with HDP flushes
  if (agent_->supported_isas()[0]->GetMajorVersion() >= 9 &&
      !(agent_->supported_isas()[0]->GetMajorVersion() == 10 && agent_->supported_isas()[0]->GetMinorVersion() == 1)) {
    hdp_flush_support_ = link.info.link_type != HSA_AMD_LINK_INFO_TYPE_XGMI;
  }

  // Allocate queue buffer.
  queue_start_addr_ =
      (char*)agent_->system_allocator()(kQueueSize, 0x1000, core::MemoryRegion::AllocateExecutable);

  if (queue_start_addr_ == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  MAKE_NAMED_SCOPE_GUARD(cleanupOnException, [&]() { Destroy(agent); };);
  std::memset(queue_start_addr_, 0, kQueueSize);

  bytes_written_.resize(kQueueSize);

  // Access kernel driver to initialize the queue control block
  // This call binds user mode queue object to underlying compute
  // device. ROCr creates queues that are of two kinds: PCIe optimized
  // and xGMI optimized. Which queue to create is indicated via input
  // boolean flag
  const HSA_QUEUE_TYPE kQueueType_ = rec_eng >= 0 ? HSA_QUEUE_SDMA_BY_ENG_ID :
                                     (use_xgmi ? HSA_QUEUE_SDMA_XGMI : HSA_QUEUE_SDMA);
  if (agent_->driver().CreateQueue(agent_->node_id(), kQueueType_, 100, HSA_QUEUE_PRIORITY_MAXIMUM,
                                   rec_eng, queue_start_addr_, kQueueSize, nullptr,
                                   queue_resource_) != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  cached_reserve_index_ = *reinterpret_cast<RingIndexTy*>(queue_resource_.Queue_write_ptr);
  cached_commit_index_ = cached_reserve_index_;

  if (core::g_use_interrupt_wait) {
    signals_[0].reset(new core::InterruptSignal(0));
    signals_[1].reset(new core::InterruptSignal(0));
  } else {
    signals_[0].reset(new core::DefaultSignal(0));
    signals_[1].reset(new core::DefaultSignal(0));
  }

  max_single_linear_copy_size_ = linear_copy_size_override;

  cleanupOnException.Dismiss();
  return HSA_STATUS_SUCCESS;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::Destroy(
    const core::Agent& agent) {
  // Release all allocated resources and reset them to zero.

  if (queue_resource_.QueueId != 0) {
    // Release queue resources from the kernel
    auto err = agent_->driver().DestroyQueue(queue_resource_.QueueId);
    assert(err == HSA_STATUS_SUCCESS);
    memset(&queue_resource_, 0, sizeof(queue_resource_));
  }

  if (queue_start_addr_ != NULL) {
    // Release queue buffer.
    agent_->system_deallocator()(queue_start_addr_);
  }

  queue_start_addr_ = NULL;
  cached_reserve_index_ = 0;
  cached_commit_index_ = 0;

  signals_[0].reset();
  signals_[1].reset();

  return HSA_STATUS_SUCCESS;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                      useGCR>::SubmitBlockingCommand(const void* cmd, size_t cmd_size,
                                                     uint64_t size) {
  ScopedAcquire<KernelMutex> lock(&lock_);

  // Alternate between completion signals
  // Using two allows overlapping command writing and copies
  core::Signal* completionSignal;
  if (parity_)
    completionSignal = signals_[0].get();
  else
    completionSignal = signals_[1].get();
  parity_ ^= true;

  // Wait for prior operation with this signal to complete
  completionSignal->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 0, -1, HSA_WAIT_STATE_BLOCKED);

  // Mark signal as in use, guard against exception leaving the signal in an unusable state.
  completionSignal->StoreRelaxed(2);
  MAKE_SCOPE_GUARD([&]() { completionSignal->StoreRelaxed(0); });
  lock.Release();

  std::vector<core::Signal*> gang_signals(0);

  // Submit command and wait for completion
  hsa_status_t ret =
      SubmitCommand(cmd, cmd_size, size, std::vector<core::Signal*>(), *completionSignal,
                    gang_signals);
  completionSignal->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 1, -1, HSA_WAIT_STATE_BLOCKED);
  return ret;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::SubmitCommand(
    const void* cmd, size_t cmd_size, uint64_t size, const std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal, std::vector<core::Signal*>& gang_signals) {

  uint32_t num_poll_command = 0;

  // Cached copy of dep_signals[i]->LoadRelaxed
  uint64_t dep_signals_value[HSA_MAX_DEP_SIGNALS];

  for (size_t i = 0; i < dep_signals.size(); ++i) {
    // The signal is 64 bit value, and poll checks for 32 bit value.
    // If the signal is already 0, then we do not need to poll.
    // If the upper 32-bits of the signal is 0, then we only need to poll the
    // lower 32-bits
    dep_signals_value[i] = dep_signals[i]->LoadRelaxed();
    if (dep_signals_value[i]) {
      num_poll_command++;
      if (dep_signals_value[i] >> 32)
        num_poll_command++;
    }
  }

  // Workaround for rare-issue on gfx908 where SDMA_OP_POLL_REGMEM returns before
  // polled memory is cleared
  static bool doublePoll = agent_->supported_isas()[0]->GetMajorVersion() == 9 &&
                           agent_->supported_isas()[0]->GetMinorVersion() == 0 &&
                           agent_->supported_isas()[0]->GetStepping() != 10;
  if (doublePoll)
    num_poll_command *= 2;

  const uint32_t total_poll_command_size =
      (num_poll_command * poll_command_size_);

  // Load the profiling state early in case the user disable or enable the
  // profiling in the middle of the call.
  const bool profiling_enabled = agent_->profiling_enabled();

  uint64_t* start_ts_addr = nullptr;
  uint64_t* end_ts_addr = nullptr;
  uint32_t total_timestamp_command_size = 0;

  // Gang leader polls gang item completions and does final decrement or
  // completion of gang signal to prevent race between poll and signal
  // destruction.
  uint32_t total_gang_complete_command_size = poll_command_size_ +
         (platform_atomic_support_ ? atomic_command_size_ : fence_command_size_);
  uint32_t total_gang_command_size = gang_leader_ ?
          static_cast<uint32_t>(gang_signals.size()) * total_gang_complete_command_size : 0;

  if (profiling_enabled && (gang_leader_ || gang_signals.empty())) {
    out_signal.GetSdmaTsAddresses(start_ts_addr, end_ts_addr);
    total_timestamp_command_size = 2 * timestamp_command_size_;
  }

  // On agent that does not support platform atomic, we replace it with
  // one or two fence packet(s) to update the signal value. The reason fence
  // is used and not write packet is because the SDMA engine may overlap a
  // serial copy/write packets.
  const uint64_t completion_signal_value =
      static_cast<uint64_t>(out_signal.LoadRelaxed() - 1);
  const size_t sync_command_size = (platform_atomic_support_)
                                       ? atomic_command_size_
                                       : (completion_signal_value > UINT32_MAX)
                                             ? 2 * fence_command_size_
                                             : fence_command_size_;

  // If the signal is an interrupt signal, we also need to make SDMA engine to
  // send interrupt packet to IH.
  const size_t interrupt_command_size =
      (out_signal.signal_.event_mailbox_ptr != 0)
          ? (fence_command_size_ + trap_command_size_)
          : 0;

  // Add space for acquire or release Hdp flush command
  uint32_t flush_cmd_size = 0;
  if (core::Runtime::runtime_singleton_->flag().enable_sdma_hdp_flush()) {
    if ((HwIndexMonotonic) && (hdp_flush_support_)) {
      flush_cmd_size = flush_command_size_;
    }
  }

  // Add space for cache flush.
  if (useGCR) flush_cmd_size += gcr_command_size_ * 2;

  const uint32_t total_command_size = total_poll_command_size + cmd_size + sync_command_size +
      total_timestamp_command_size + interrupt_command_size + flush_cmd_size + total_gang_command_size;
  const uint32_t pad_size = total_command_size < min_submission_size_ ?
                            min_submission_size_ - total_command_size : 0;

  RingIndexTy curr_index;
  char* command_addr;
  uint64_t prior_bytes, post_bytes;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
    if (command_addr == nullptr) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    prior_bytes = bytes_queued_;
    bytes_queued_ += size;
    post_bytes = bytes_queued_;
  }
  uint32_t wrapped_index = WrapIntoRing(curr_index);

  for (size_t i = 0; i < dep_signals.size(); ++i) {
    if (dep_signals_value[i]) {
      uint32_t* signal_addr =
          reinterpret_cast<uint32_t*>(dep_signals[i]->ValueLocation());

      if (dep_signals_value[i] >> 32) {
        // Wait for the higher 32 bits to 0.
        BuildPollCommand(command_addr, &signal_addr[1], 0);
        command_addr += poll_command_size_;
        bytes_written_[wrapped_index] = prior_bytes;
        wrapped_index += poll_command_size_;

        if (doublePoll) {
          BuildPollCommand(command_addr, &signal_addr[1], 0);
          command_addr += poll_command_size_;
          bytes_written_[wrapped_index] = prior_bytes;
          wrapped_index += poll_command_size_;
        }
      }
      // Then wait for the lower 32 bits to 0.
      BuildPollCommand(command_addr, &signal_addr[0], 0);
      command_addr += poll_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += poll_command_size_;

      if (doublePoll) {
        BuildPollCommand(command_addr, &signal_addr[0], 0);
        command_addr += poll_command_size_;
        bytes_written_[wrapped_index] = prior_bytes;
        wrapped_index += poll_command_size_;
      }
    }
  }

  if (profiling_enabled && (gang_leader_ || gang_signals.empty())) {
    BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(start_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += timestamp_command_size_;
  }

  // Issue a Hdp flush cmd
  if (core::Runtime::runtime_singleton_->flag().enable_sdma_hdp_flush()) {
    if ((HwIndexMonotonic) && (hdp_flush_support_)) {
      BuildHdpFlushCommand(command_addr);
      command_addr += flush_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += flush_command_size_;
    }
  }

  // Issue cache invalidate
  if (useGCR) {
    BuildGCRCommand(command_addr, true);
    command_addr += gcr_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += gcr_command_size_;
  }

  // Do the command after all polls are satisfied.
  memcpy(command_addr, cmd, cmd_size);
  command_addr += cmd_size;
  bytes_written_.fill(wrapped_index, wrapped_index + cmd_size, prior_bytes);
  wrapped_index += cmd_size;

  // Issue cache writeback
  if (useGCR) {
    BuildGCRCommand(command_addr, false);
    command_addr += gcr_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += gcr_command_size_;
  }

  if (profiling_enabled && (gang_leader_ || gang_signals.empty())) {
    assert(IsMultipleOf(end_ts_addr, 32));
    BuildGetGlobalTimestampCommand(command_addr,
                                   reinterpret_cast<void*>(end_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += timestamp_command_size_;
  }

  // Wait for non-leaders gang items to complete
  if (gang_leader_) {
    for (int i = 0; i < gang_signals.size(); i++) {
      uint32_t* gang_signal_addr =
          reinterpret_cast<uint32_t*>(gang_signals[i]->ValueLocation());
      BuildPollCommand(command_addr, gang_signal_addr, 1);
      command_addr += poll_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += poll_command_size_;

      // After non-leader gang-items have completed, decrement the gang signal value.
      if (platform_atomic_support_) {
        BuildAtomicDecrementCommand(command_addr, gang_signal_addr);
        command_addr += atomic_command_size_;
        bytes_written_[wrapped_index] = post_bytes;
        wrapped_index += atomic_command_size_;
      } else {
        BuildFenceCommand(command_addr, gang_signal_addr, 0);
        command_addr += fence_command_size_;
        bytes_written_[wrapped_index] = post_bytes;
        wrapped_index += fence_command_size_;
      }
    }
  }

  // After transfer is completed, decrement the signal value.
  if (platform_atomic_support_) {
    BuildAtomicDecrementCommand(command_addr, out_signal.ValueLocation());
    command_addr += atomic_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += atomic_command_size_;
  } else {
    uint32_t* signal_value_location = reinterpret_cast<uint32_t*>(out_signal.ValueLocation());
    if (completion_signal_value > UINT32_MAX) {
      BuildFenceCommand(command_addr, signal_value_location + 1,
                        static_cast<uint32_t>(completion_signal_value >> 32));
      command_addr += fence_command_size_;
      bytes_written_[wrapped_index] = post_bytes;
      wrapped_index += fence_command_size_;
    }

    BuildFenceCommand(command_addr, signal_value_location,
                      static_cast<uint32_t>(completion_signal_value));
    command_addr += fence_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += fence_command_size_;
  }

  // Update mailbox event and send interrupt to IH.
  if (out_signal.signal_.event_mailbox_ptr != 0) {
    BuildFenceCommand(command_addr,
                      reinterpret_cast<uint32_t*>(out_signal.signal_.event_mailbox_ptr),
                      static_cast<uint32_t>(out_signal.signal_.event_id));
    command_addr += fence_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += fence_command_size_;

    BuildTrapCommand(command_addr, out_signal.signal_.event_id);
    command_addr += trap_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += trap_command_size_;
  }

  // Pad size is DWORD aligned since all commands are dword aligned.
  // Insert NOP header DWORD with value of the number of null DWORDs shifted
  // by 16 bits to pad total submission.
  if (pad_size) {
    memset(command_addr, 0, pad_size);
    uint32_t *dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
    dword_command_addr[0] = (pad_size/4 - 1) << 16;
  }

  ReleaseWriteAddress(curr_index, total_command_size + pad_size);

  return HSA_STATUS_SUCCESS;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                      useGCR>::SubmitLinearCopyCommand(void* dst, const void* src, size_t size) {
  // Break the copy into multiple copy operation incase the copy size exceeds
  // the SDMA linear copy limit.
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                               kMaxSingleCopySize;
  const uint32_t num_copy_command = (size + max_copy_size - 1) / max_copy_size;

  std::vector<SDMA_PKT_COPY_LINEAR> buff(num_copy_command);
  BuildCopyCommand(reinterpret_cast<char*>(&buff[0]), num_copy_command, dst, src, size);

  return SubmitBlockingCommand(&buff[0], buff.size() * sizeof(SDMA_PKT_COPY_LINEAR), size);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                      useGCR>::SubmitLinearCopyCommand(void* dst, const void* src, size_t size,
                                                       std::vector<core::Signal*>& dep_signals,
                                                       core::Signal& out_signal,
                                                       std::vector<core::Signal*>& gang_signals) {
  // Break the copy into multiple copy operations when the copy size exceeds
  // the SDMA linear copy limit.
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                               kMaxSingleCopySize;
  const uint32_t num_copy_command = (size + max_copy_size - 1) / max_copy_size;

  // Assemble copy packets.
  std::vector<SDMA_PKT_COPY_LINEAR> buff(num_copy_command);
  BuildCopyCommand(reinterpret_cast<char*>(&buff[0]), num_copy_command, dst, src, size);

  return SubmitCommand(&buff[0], buff.size() * sizeof(SDMA_PKT_COPY_LINEAR), size, dep_signals,
                       out_signal, gang_signals);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t
BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::SubmitCopyRectCommand(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal) {
  // Hardware requires DWORD alignment for base address, pitches
  // Also confirm that we have a geometric rect (copied block does not wrap an edge).
  if (((uintptr_t)dst->base) % 4 != 0 || ((uintptr_t)src->base) % 4 != 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Copy rect base address not aligned.");
  if (((uintptr_t)dst->pitch) % 4 != 0 || ((uintptr_t)src->pitch) % 4 != 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect pitch not aligned.");
  if (((uintptr_t)dst->slice) % 4 != 0 || ((uintptr_t)src->slice) % 4 != 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect slice not aligned.");
  if (uint64_t(src_offset->x) + range->x > src->pitch ||
      uint64_t(dst_offset->x) + range->x > dst->pitch)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect width out of range.");
  if ((src->slice != 0) && (uint64_t(src_offset->y) + range->y) > src->slice / src->pitch)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect height out of range.");
  if ((dst->slice != 0) && (uint64_t(dst_offset->y) + range->y) > dst->slice / dst->pitch)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect height out of range.");
  if (range->z > 1 && (src->slice == 0 || dst->slice == 0))
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect slice needed.");

  // GFX12 or later use a different packet format that is incompatible (fields changed in size and location).
  const bool isGFX12Plus =
                        (agent_->supported_isas()[0]->GetMajorVersion() >= 12);

  // Common and GFX12 packet must match in size to use same code for vector/append.
  static_assert(sizeof(SDMA_PKT_COPY_LINEAR_RECT) == sizeof(SDMA_PKT_COPY_LINEAR_RECT_GFX12), "");

  const uint max_pitch = 1 << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::pitch_bits : SDMA_PKT_COPY_LINEAR_RECT::pitch_bits);

  std::vector<SDMA_PKT_COPY_LINEAR_RECT> pkts;
  std::vector<uint64_t> bytes_moved;
  auto append = [&](size_t size) {
    assert(size == sizeof(SDMA_PKT_COPY_LINEAR_RECT) && "SDMA packet size missmatch");
    pkts.emplace_back(SDMA_PKT_COPY_LINEAR_RECT());
    return &pkts.back();
  };

  // Do wide pitch 2D copies along X-Z
  if (range->z == 1 && (src->pitch > max_pitch || dst->pitch > max_pitch)) {
    hsa_pitched_ptr_t Src = *src;
    hsa_pitched_ptr_t Dst = *dst;
    hsa_dim3_t Soff = *src_offset;
    hsa_dim3_t Doff = *dst_offset;
    hsa_dim3_t Range = *range;

    Src.base = static_cast<char*>(Src.base) + Soff.z * Src.slice + Soff.y * Src.pitch;
    Dst.base = static_cast<char*>(Dst.base) + Doff.z * Dst.slice + Doff.y * Dst.pitch;
    Soff.y = Soff.z = 0;
    Doff.y = Doff.z = 0;

    Src.slice = Src.pitch;
    Src.pitch = 0;
    Dst.slice = Dst.pitch;
    Dst.pitch = 0;

    Range.z = Range.y;
    Range.y = 1;

    BuildCopyRectCommand(append, &Dst, &Doff, &Src, &Soff, &Range);
  } else {
    BuildCopyRectCommand(append, dst, dst_offset, src, src_offset, range);
  }

  uint64_t size = static_cast<uint64_t>(range->x) * static_cast<uint64_t>(range->y) * range->z;

  std::vector<core::Signal*> gang_signals(0);

  return SubmitCommand(&pkts[0], pkts.size() * sizeof(SDMA_PKT_COPY_LINEAR_RECT), size, dep_signals,
                       out_signal, gang_signals);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
                      useGCR>::SubmitLinearFillCommand(void* ptr, uint32_t value, size_t count) {
  const size_t size = count * sizeof(uint32_t);

  const uint32_t num_fill_command = (size + kMaxSingleFillSize - 1) / kMaxSingleFillSize;

  std::vector<SDMA_PKT_CONSTANT_FILL> buff(num_fill_command);
  BuildFillCommand(reinterpret_cast<char*>(&buff[0]), num_fill_command, ptr, value, count);

  return SubmitBlockingCommand(&buff[0], buff.size() * sizeof(SDMA_PKT_CONSTANT_FILL), size);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
hsa_status_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::EnableProfiling(
    bool enable) {
  return HSA_STATUS_SUCCESS;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
char* BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::AcquireWriteAddress(
    uint32_t cmd_size, RingIndexTy& curr_index) {
  // Ring is full when all but one byte is written.
  if (cmd_size >= kQueueSize) {
    return nullptr;
  }

  while (true) {
    curr_index = atomic::Load(&cached_reserve_index_, std::memory_order_acquire);

    // Check whether a linear region of the requested size is available.
    // If == cmd_size: region is at beginning of ring.
    // If < cmd_size: region intersects end of ring, pad with no-ops and retry.
    if (WrapIntoRing(curr_index + cmd_size) < cmd_size) {
      PadRingToEnd(curr_index);
      continue;
    }

    // Check whether the engine has finished using this region.
    const RingIndexTy new_index = curr_index + cmd_size;

    if (CanWriteUpto(new_index) == false) {
      // Wait for read index to move and try again.
      os::YieldThread();
      continue;
    }

    // Try to reserve this part of the ring.
    if (atomic::Cas(&cached_reserve_index_, new_index, curr_index, std::memory_order_release) ==
        curr_index) {
      return queue_start_addr_ + WrapIntoRing(curr_index);
    }

    // Another thread reserved curr_index, try again.
    os::YieldThread();
  }

  return nullptr;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
              useGCR>::UpdateWriteAndDoorbellRegister(RingIndexTy curr_index,
                                                      RingIndexTy new_index) {
  while (true) {
    // Make sure that the address before ::curr_index is already released.
    // Otherwise the CP may read invalid packets.
    if (atomic::Load(&cached_commit_index_, std::memory_order_acquire) == curr_index) {
      if (core::Runtime::runtime_singleton_->flag().sdma_wait_idle()) {
        // TODO: remove when sdma wpointer issue is resolved.
        // Wait until the SDMA engine finish processing all packets before
        // updating the wptr and doorbell.
        while (WrapIntoRing(*reinterpret_cast<RingIndexTy*>(queue_resource_.Queue_read_ptr)) !=
               WrapIntoRing(curr_index)) {
          os::YieldThread();
        }
      }

      // Update write pointer and doorbell register.
      *reinterpret_cast<RingIndexTy*>(queue_resource_.Queue_write_ptr) =
          (HwIndexMonotonic ? new_index : WrapIntoRing(new_index));

      // Ensure write pointer is visible to GPU before doorbell.
      std::atomic_thread_fence(std::memory_order_release);

      *reinterpret_cast<RingIndexTy*>(queue_resource_.Queue_DoorBell) =
          (HwIndexMonotonic ? new_index : WrapIntoRing(new_index));

      atomic::Store(&cached_commit_index_, new_index, std::memory_order_release);
      break;
    }

    // Waiting for another thread to submit preceding commands first.
    os::YieldThread();
  }
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::ReleaseWriteAddress(
    RingIndexTy curr_index, uint32_t cmd_size) {
  if (cmd_size > kQueueSize) {
    assert(false && "cmd_addr is outside the queue buffer range");
    return;
  }

  UpdateWriteAndDoorbellRegister(curr_index, curr_index + cmd_size);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::PadRingToEnd(
    RingIndexTy curr_index) {
  // Reserve region from here to the end of the ring.
  RingIndexTy new_index = curr_index + (kQueueSize - WrapIntoRing(curr_index));

  // Check whether the engine has finished using this region.
  if (CanWriteUpto(new_index) == false) {
    // Wait for read index to move and try again.
    return;
  }

  if (atomic::Cas(&cached_reserve_index_, new_index, curr_index, std::memory_order_release) ==
      curr_index) {
    // Write and submit NOP commands in reserved region.
    char* nop_address = queue_start_addr_ + WrapIntoRing(curr_index);
    memset(nop_address, 0, new_index - curr_index);

    // Pad pending bytes tracking
    bytes_written_.fill(WrapIntoRing(curr_index), WrapIntoRing(new_index), bytes_queued_);

    UpdateWriteAndDoorbellRegister(curr_index, new_index);
  }
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
uint32_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::WrapIntoRing(
    RingIndexTy index) {
  return index & (kQueueSize - 1);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
bool BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::CanWriteUpto(
    RingIndexTy upto_index) {
  // Get/calculate the monotonic read index.
  RingIndexTy hw_read_index = *reinterpret_cast<RingIndexTy*>(queue_resource_.Queue_read_ptr);
  RingIndexTy read_index;

  if (HwIndexMonotonic) {
    read_index = hw_read_index;
  } else {
    // Calculate distance from commit index to HW read index.
    // Commit index is always < kQueueSize away from HW read index.
    RingIndexTy commit_index = atomic::Load(&cached_commit_index_, std::memory_order_relaxed);
    RingIndexTy dist_to_read_index = WrapIntoRing(commit_index - hw_read_index);
    read_index = commit_index - dist_to_read_index;
  }

  // Check whether the read pointer has passed the given index.
  // At most we can submit (kQueueSize - 1) bytes at a time.
  return (upto_index - read_index) < kQueueSize;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildFenceCommand(
    char* fence_command_addr, uint32_t* fence, uint32_t fence_value) {
  assert(fence_command_addr != NULL);
  SDMA_PKT_FENCE* packet_addr =
      reinterpret_cast<SDMA_PKT_FENCE*>(fence_command_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_FENCE));

  packet_addr->HEADER_UNION.op = SDMA_OP_FENCE;

  if (agent_->supported_isas()[0]->GetMajorVersion() >= 10) {
    packet_addr->HEADER_UNION.mtype = 3;
  }

  packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(fence);

  packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence);

  packet_addr->DATA_UNION.data = fence_value;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildCopyCommand(
    char* cmd_addr, uint32_t num_copy_command, void* dst, const void* src, size_t size) {
  size_t cur_size = 0;
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                                                              kMaxSingleCopySize;
  for (uint32_t i = 0; i < num_copy_command; ++i) {
    const uint32_t copy_size =
        static_cast<uint32_t>(std::min((size - cur_size), max_copy_size));

    void* cur_dst = static_cast<char*>(dst) + cur_size;
    const void* cur_src = static_cast<const char*>(src) + cur_size;

    SDMA_PKT_COPY_LINEAR* packet_addr =
        reinterpret_cast<SDMA_PKT_COPY_LINEAR*>(cmd_addr);

    memset(packet_addr, 0, sizeof(SDMA_PKT_COPY_LINEAR));

    packet_addr->HEADER_UNION.op = SDMA_OP_COPY;
    packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;

    if (max_copy_size == (1 << 30) -1)
      packet_addr->COUNT_UNION.count_ext.count = copy_size + SizeToCountOffset;
    else
      packet_addr->COUNT_UNION.count.count = copy_size + SizeToCountOffset;

    packet_addr->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
    packet_addr->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);

    packet_addr->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst);
    packet_addr->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst);

    cmd_addr += linear_copy_command_size_;
    cur_size += copy_size;
  }

  assert(cur_size == size);
}

/*
Copies are done in terms of elements (1, 2, 4, 8, or 16 bytes) and have alignment restrictions.
Elements are coded by the log2 of the element size in bytes (ie. element 0=1 byte, 4=16 byte).
This routine breaks a large rect into tiles that can be handled by hardware.  Pitches and offsets
must be representable in terms of elements in all tiles of the copy.
*/
template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildCopyRectCommand(
    const std::function<void*(size_t)>& append, const hsa_pitched_ptr_t* dst,
    const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset,
    const hsa_dim3_t* range) {
  // Returns the index of the first set bit (ie log2 of the largest power of 2 that evenly divides
  // width), the largest element that perfectly covers width.
  // width | 16 ensures that we don't return a higher element than is supported and avoids
  // issues with 0.
  auto maxAlignedElement = [](size_t width) {
    return __builtin_ctz(width | 16);
  };

  // GFX12 or later use a different packet format that is incompatible (fields changed in size and location).
  const bool isGFX12Plus =
                      (agent_->supported_isas()[0]->GetMajorVersion() >= 12);

  // Limits in terms of element count
  const uint32_t max_pitch = 1    << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::pitch_bits   : SDMA_PKT_COPY_LINEAR_RECT::pitch_bits);
  const uint64_t max_slice = 1ULL << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::slice_bits   : SDMA_PKT_COPY_LINEAR_RECT::slice_bits);
  const uint32_t max_x     = 1    << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::rect_xy_bits : SDMA_PKT_COPY_LINEAR_RECT::rect_xy_bits);
  const uint32_t max_y     = 1    << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::rect_xy_bits : SDMA_PKT_COPY_LINEAR_RECT::rect_xy_bits);
  const uint32_t max_z     = 1    << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::rect_z_bits  : SDMA_PKT_COPY_LINEAR_RECT::rect_z_bits);

  // Find maximum element that describes the pitch and slice.
  // Pitch and slice must both be represented in units of elements.  No element larger than this
  // may be used in any tile as the pitches would not be exactly represented.
  int max_ele = Min(maxAlignedElement(src->pitch), maxAlignedElement(dst->pitch));
  if (range->z != 1)  // Only need to consider slice if HW will copy along Z.
    max_ele = Min(max_ele, maxAlignedElement(src->slice), maxAlignedElement(dst->slice));

  /*
  Find the minimum element size that will be needed for any tile.

  No subdivision of a range admits a larger element size for the smallest element in any subdivision
  than the element size that covers the whole range, though some can be worse (this is easily model
  checked).  Subdividing with any element larger than the covering element won't change the covering
  element of the remainder
  ( Range%Element = (Range-N*LargerElement)%Element since LargerElement%Element=0 ).
    Ex. range->x=71, assume max range is 16 elements:  We can break at 64 giving tiles:
    [0,63], [64-70] (width 64 & 7).  64 is covered by element 4 (16B) and 7 is covered by element 0
    (1B).  Exactly covering 71 requires using element 0.

  Base addresses in each tile must be DWORD aligned, if not then the offset from an aligned address
  must be represented in elements.  This may reduce the size of the element, but since elements are
  integer multiples of each other this is harmless.

  src and dst base has already been checked for DWORD alignment so we only need to consider the
  offset here.
  */
  int min_ele = Min(max_ele, maxAlignedElement(range->x), maxAlignedElement(src_offset->x % 4),
                    maxAlignedElement(dst_offset->x % 4));

  // Check that pitch and slice can be represented in the tile with the smallest element
  if ((src->pitch >> min_ele) > max_pitch || (dst->pitch >> min_ele) > max_pitch)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Copy rect pitch out of limits.\n");
  if (range->z != 1) {  // Only need to consider slice if HW will copy along Z.
    if ((src->slice >> min_ele) > max_slice || (dst->slice >> min_ele) > max_slice)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Copy rect slice out of limits.\n");
  }

  // Break copy into tiles
  for (uint32_t z = 0; z < range->z; z += max_z) {
    for (uint32_t y = 0; y < range->y; y += max_y) {
      uint32_t x = 0;
      while (x < range->x) {
        uint32_t width = range->x - x;

        // Get largest element which describes the start of this tile after its base address has
        // been aligned.  Base addresses must be DWORD (4 byte) aligned.
        int aligned_ele = Min(maxAlignedElement((src_offset->x + x) % 4),
                              maxAlignedElement((dst_offset->x + x) % 4), max_ele);

        // Get largest permissible element which exactly covers width
        int element = Min(maxAlignedElement(width), aligned_ele);
        int xcount = width >> element;

        // If width is too large then width is at least max_x bytes (bigger than any element) so
        // drop the width restriction and clip element count to max_x.
        if (xcount > max_x) {
          element = aligned_ele;
          xcount = Min(width >> element, max_x);
        }

        // Get base addresses and offsets for this tile.
        uintptr_t sbase = (uintptr_t)src->base + src_offset->x + x +
            (src_offset->y + y) * src->pitch + (src_offset->z + z) * src->slice;
        uintptr_t dbase = (uintptr_t)dst->base + dst_offset->x + x +
            (dst_offset->y + y) * dst->pitch + (dst_offset->z + z) * dst->slice;
        uint soff = (sbase % 4) >> element;
        uint doff = (dbase % 4) >> element;
        sbase &= ~3ull;
        dbase &= ~3ull;

        x += xcount << element;

        // GFX12 has a different packet format that is incompatible with pre-GFX12.
        if (isGFX12Plus) {
          SDMA_PKT_COPY_LINEAR_RECT_GFX12* pkt =
            (SDMA_PKT_COPY_LINEAR_RECT_GFX12*)append(sizeof(SDMA_PKT_COPY_LINEAR_RECT));
          *pkt = {};
          pkt->HEADER_UNION.op = SDMA_OP_COPY;
          pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_RECT;
          pkt->HEADER_UNION.element = element;
          pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = sbase;
          pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = sbase >> 32;
          pkt->SRC_PARAMETER_1_UNION.src_offset_x = soff;
          pkt->SRC_PARAMETER_2_UNION.src_pitch = (src->pitch >> element) - 1;
          pkt->SRC_PARAMETER_3_UNION.src_slice_pitch =
            (range->z == 1) ? 0 : (src->slice >> element) - 1;
          pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = dbase;
          pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = dbase >> 32;
          pkt->DST_PARAMETER_1_UNION.dst_offset_x = doff;
          pkt->DST_PARAMETER_2_UNION.dst_pitch = (dst->pitch >> element) - 1;
          pkt->DST_PARAMETER_3_UNION.dst_slice_pitch =
            (range->z == 1) ? 0 : (dst->slice >> element) - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_x = xcount - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_y = Min(range->y - y, max_y) - 1;
          pkt->RECT_PARAMETER_2_UNION.rect_z = Min(range->z - z, max_z) - 1;
        } else {  // Pre-GFX12, common packet used
          SDMA_PKT_COPY_LINEAR_RECT* pkt =
            (SDMA_PKT_COPY_LINEAR_RECT*)append(sizeof(SDMA_PKT_COPY_LINEAR_RECT));
          *pkt = {};
          pkt->HEADER_UNION.op = SDMA_OP_COPY;
          pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_RECT;
          pkt->HEADER_UNION.element = element;
          pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = sbase;
          pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = sbase >> 32;
          pkt->SRC_PARAMETER_1_UNION.src_offset_x = soff;
          pkt->SRC_PARAMETER_2_UNION.src_pitch = (src->pitch >> element) - 1;
          pkt->SRC_PARAMETER_3_UNION.src_slice_pitch =
            (range->z == 1) ? 0 : (src->slice >> element) - 1;
          pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = dbase;
          pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = dbase >> 32;
          pkt->DST_PARAMETER_1_UNION.dst_offset_x = doff;
          pkt->DST_PARAMETER_2_UNION.dst_pitch = (dst->pitch >> element) - 1;
          pkt->DST_PARAMETER_3_UNION.dst_slice_pitch =
            (range->z == 1) ? 0 : (dst->slice >> element) - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_x = xcount - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_y = Min(range->y - y, max_y) - 1;
          pkt->RECT_PARAMETER_2_UNION.rect_z = Min(range->z - z, max_z) - 1;
	}
      }
    }
  }
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildFillCommand(
    char* cmd_addr, uint32_t num_fill_command, void* ptr, uint32_t value, size_t count) {
  char* cur_ptr = reinterpret_cast<char*>(ptr);
  const uint32_t maxDwordCount = kMaxSingleFillSize / sizeof(uint32_t);
  SDMA_PKT_CONSTANT_FILL* packet_addr = reinterpret_cast<SDMA_PKT_CONSTANT_FILL*>(cmd_addr);

  for (uint32_t i = 0; i < num_fill_command; i++) {
    assert(count != 0 && "SDMA fill command count error.");
    const uint32_t fill_count = Min(count, size_t(maxDwordCount));

    memset(packet_addr, 0, sizeof(SDMA_PKT_CONSTANT_FILL));

    packet_addr->HEADER_UNION.op = SDMA_OP_CONST_FILL;
    packet_addr->HEADER_UNION.fillsize = 2;  // DW fill

    packet_addr->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_ptr);
    packet_addr->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_ptr);

    packet_addr->DATA_UNION.src_data_31_0 = value;

    packet_addr->COUNT_UNION.count = (fill_count + SizeToCountOffset) * sizeof(uint32_t);

    packet_addr++;
    cur_ptr += fill_count * sizeof(uint32_t);
    count -= fill_count;
  }
  assert(count == 0 && "SDMA fill command count error.");
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildPollCommand(
    char* cmd_addr, void* addr, uint32_t reference) {
  SDMA_PKT_POLL_REGMEM* packet_addr =
      reinterpret_cast<SDMA_PKT_POLL_REGMEM*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_POLL_REGMEM));

  packet_addr->HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
  packet_addr->HEADER_UNION.mem_poll = 1;
  packet_addr->HEADER_UNION.func = 0x3;  // IsEqual.
  packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(addr);
  packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);

  packet_addr->VALUE_UNION.value = reference;

  packet_addr->MASK_UNION.mask = 0xffffffff;  // Compare the whole content.

  packet_addr->DW5_UNION.interval = 0x04;
  packet_addr->DW5_UNION.retry_count = 0xfff;  // Retry forever.
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
              useGCR>::BuildAtomicDecrementCommand(char* cmd_addr, void* addr) {
  SDMA_PKT_ATOMIC* packet_addr = reinterpret_cast<SDMA_PKT_ATOMIC*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_ATOMIC));

  packet_addr->HEADER_UNION.op = SDMA_OP_ATOMIC;
  packet_addr->HEADER_UNION.operation = SDMA_ATOMIC_ADD64;

  packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(addr);
  packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);

  packet_addr->SRC_DATA_LO_UNION.src_data_31_0 = 0xffffffff;
  packet_addr->SRC_DATA_HI_UNION.src_data_63_32 = 0xffffffff;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset,
              useGCR>::BuildGetGlobalTimestampCommand(char* cmd_addr, void* write_address) {
  SDMA_PKT_TIMESTAMP* packet_addr =
      reinterpret_cast<SDMA_PKT_TIMESTAMP*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_TIMESTAMP));

  packet_addr->HEADER_UNION.op = SDMA_OP_TIMESTAMP;
  packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_TIMESTAMP_GET_GLOBAL;

  packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(write_address);
  packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(write_address);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildTrapCommand(
    char* cmd_addr, uint32_t event_id) {
  SDMA_PKT_TRAP* packet_addr =
      reinterpret_cast<SDMA_PKT_TRAP*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_TRAP));

  packet_addr->HEADER_UNION.op = SDMA_OP_TRAP;
  packet_addr->INT_CONTEXT_UNION.int_ctx = event_id;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildHdpFlushCommand(
    char* cmd_addr) {
  assert(cmd_addr != NULL);
  SDMA_PKT_POLL_REGMEM* addr = reinterpret_cast<SDMA_PKT_POLL_REGMEM*>(cmd_addr);
  memcpy(addr, &hdp_flush_cmd, flush_command_size_);
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
void BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::BuildGCRCommand(
    char* cmd_addr, bool invalidate) {
  assert(cmd_addr != NULL);
  assert(useGCR && "Unsupported SDMA command - GCR.");
  SDMA_PKT_GCR* addr = reinterpret_cast<SDMA_PKT_GCR*>(cmd_addr);
  memset(addr, 0, sizeof(SDMA_PKT_GCR));
  addr->HEADER_UNION.op = SDMA_OP_GCR;
  addr->HEADER_UNION.sub_op = SDMA_SUBOP_USER_GCR;
  addr->WORD2_UNION.GCR_CONTROL_GL2_WB = 1;
  addr->WORD2_UNION.GCR_CONTROL_GLK_WB = 1;
  if (invalidate) {
    addr->WORD2_UNION.GCR_CONTROL_GL2_INV = 1;
    addr->WORD2_UNION.GCR_CONTROL_GL1_INV = 1;
    addr->WORD2_UNION.GCR_CONTROL_GLV_INV = 1;
    addr->WORD2_UNION.GCR_CONTROL_GLK_INV = 1;
  }
  // Discarding all lines for now.
  addr->WORD2_UNION.GCR_CONTROL_GL2_RANGE = 0;
}

template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
uint64_t BlitSdma<RingIndexTy, HwIndexMonotonic, SizeToCountOffset, useGCR>::PendingBytes() {
  RingIndexTy commit = atomic::Load(&cached_commit_index_, std::memory_order_acquire);
  RingIndexTy hw_read_index = *reinterpret_cast<RingIndexTy*>(queue_resource_.Queue_read_ptr);
  RingIndexTy read;
  if (HwIndexMonotonic) {
    read = hw_read_index;
  } else {
    RingIndexTy dist_to_read_index = WrapIntoRing(commit - hw_read_index);
    read = commit - dist_to_read_index;
  }

  if (commit == read) return 0;
  return bytes_queued_ - bytes_written_[WrapIntoRing(read)];
}

template class BlitSdma<uint32_t, false, 0, false>;
template class BlitSdma<uint64_t, true, -1, false>;
template class BlitSdma<uint64_t, true, -1, true>;

}  // namespace amd
}  // namespace rocr
