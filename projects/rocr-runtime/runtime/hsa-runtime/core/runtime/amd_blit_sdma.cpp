////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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
#include <core/util/utils.h>

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

const size_t BlitSdmaBase::kQueueSize = 1024 * 1024 * 8;
const size_t BlitSdmaBase::kCopyPacketSize = sizeof(SDMA_PKT_COPY_LINEAR);
const size_t BlitSdmaBase::kMaxSingleCopySize = SDMA_PKT_COPY_LINEAR::kMaxSize_;
const size_t BlitSdmaBase::kMaxSingleFillSize = SDMA_PKT_CONSTANT_FILL::kMaxSize_;

// Initialize size of various sDMA commands use by this module
template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::linear_copy_command_size_ = sizeof(SDMA_PKT_COPY_LINEAR);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::broadcast_copy_command_size_ = sizeof(SDMA_PKT_COPY_LINEAR_BROADCAST);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::swap_copy_command_size_ = sizeof(SDMA_PKT_COPY_LINEAR_SWAP);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::fill_command_size_ = sizeof(SDMA_PKT_CONSTANT_FILL);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::fence_command_size_ = sizeof(SDMA_PKT_FENCE);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::poll_command_size_ = sizeof(SDMA_PKT_POLL_REGMEM);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::flush_command_size_ = sizeof(SDMA_PKT_POLL_REGMEM);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::atomic_command_size_ = sizeof(SDMA_PKT_ATOMIC);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::timestamp_command_size_ = sizeof(SDMA_PKT_TIMESTAMP);

template <bool useGCR, bool scopeFields> const uint32_t BlitSdma<useGCR, scopeFields>::trap_command_size_ = sizeof(SDMA_PKT_TRAP);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::fence_64b_command_size_ = sizeof(SDMA_PKT_FENCE_64B_GFX1250);

template <bool useGCR, bool scopeFields>
const uint32_t BlitSdma<useGCR, scopeFields>::poll_64b_command_size_ = sizeof(SDMA_PKT_POLL_MEM_64B_GFX1250);

template <bool useGCR,bool scopeFields>
uint32_t BlitSdma<useGCR, scopeFields>::gcr_command_size() {
  if (is_gfx125plus_) {
    return sizeof(SDMA_PKT_GCR_GFX1250);
  }
  return sizeof(SDMA_PKT_GCR);
}

template <bool useGCR, bool scopeFields>
BlitSdma<useGCR, scopeFields>::BlitSdma()
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
      min_submission_size_(0),
      needs_kmt_doorbell_(false),
      sdma_wait_idle_(false),
      is_dxg_(false),
      enable_sdma_hdp_flush_(false),
      sw_poll_workaround_(false),
      queue_wptr_(nullptr),
      queue_rptr_(nullptr),
      queue_doorbell_(nullptr),
      broadcast_supported_(false),
      multicast_supported_(false),
      is_gfx125plus_(false),
      swap_supported_(false),
      indirect_copy_supported_(false) {
  std::memset(&queue_resource_, 0, sizeof(queue_resource_));
}

template <bool useGCR, bool scopeFields> BlitSdma<useGCR, scopeFields>::~BlitSdma() {}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::Initialize(const core::Agent& agent, bool use_xgmi,
                                          size_t linear_copy_size_override, int rec_eng) {
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

  // Cache ISA version for capability detection below.
  const auto isa_version = agent_->supported_isas()[0]->GetVersion();
  const auto major = agent_->supported_isas()[0]->GetMajorVersion();
  const auto minor = agent_->supported_isas()[0]->GetMinorVersion();
  const auto stepping = agent_->supported_isas()[0]->GetStepping();

  // Some GFX9 devices require a minimum of 64 DWORDS per ring buffer submission.
  if (isa_version >= core::Isa::Version(9, 0, 0) &&
     (isa_version <= core::Isa::Version(9, 0, 4) ||
      isa_version == core::Isa::Version(9, 0, 12))) {
    min_submission_size_ = 256;
  }

  const core::Runtime::LinkInfo& link =
            core::Runtime::runtime_singleton_->GetLinkInfo( agent_->node_id(),
                core::Runtime::runtime_singleton_->cpu_agents()[0]->node_id());
  if (isa_version == core::Isa::Version(7, 0, 1)) {
    platform_atomic_support_ = false;
  } else {
    platform_atomic_support_ = link.info.atomic_support_64bit;
  }

  // HDP flush supported on gfx900 and forward.
  // gfx90a can support xGMI host to device connections so bypass HDP flush
  // in this case.
  // gfx101x seems to have issues with HDP flushes
  if (major >= 9 && !(major == 10 && minor == 1)) {
    hdp_flush_support_ = link.info.link_type != HSA_AMD_LINK_INFO_TYPE_XGMI;
  }

  is_gfx125plus_ = (major == 12 && minor >= 5);

  // The linear wait/signal-indirect copy packet is available on gfx125+.
  indirect_copy_supported_ = is_gfx125plus_;

  // Multicast (one src -> N dst) linear copy packet is gfx125+.
  multicast_supported_ = is_gfx125plus_;

  // Broadcast linear copy supported on MI200+ and all SDMA 5.x/6.x+.
  if (major >= 10) {
    broadcast_supported_ = true;
  } else if (major == 9) {
    broadcast_supported_ = (minor >= 4) || (minor == 0 && stepping >= 10);
    swap_supported_ = (minor >= 4);
  }

  // Allocate queue buffer.
  queue_start_addr_ =
      (char*)agent_->system_allocator()(kQueueSize, 0x1000, core::MemoryRegion::AllocateExecutable);

  if (queue_start_addr_ == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  MAKE_NAMED_SCOPE_GUARD(cleanupOnException, [&]() { Destroy(); };);
  std::memset(queue_start_addr_, 0, kQueueSize);

  bytes_written_.resize(kQueueSize);

  // Access kernel driver to initialize the queue control block
  // This call binds user mode queue object to underlying compute
  // device. ROCr creates queues that are of two kinds: PCIe optimized
  // and xGMI optimized. Which queue to create is indicated via input
  // boolean flag
  const HSA_QUEUE_TYPE kQueueType_ = rec_eng >= 0 ? HSA_QUEUE_SDMA_BY_ENG_ID :
                                     (use_xgmi ? HSA_QUEUE_SDMA_XGMI : HSA_QUEUE_SDMA);
  if (agent_->driver().CreateQueue(agent_->node_id(), kQueueType_, 100, HSA::HSA_AMD_QUEUE_PRIORITY_MAXIMUM,
                                   rec_eng, queue_start_addr_, kQueueSize, 0, nullptr,
                                   queue_resource_) != HSA_STATUS_SUCCESS) {
    LogPrint(HSA_AMD_LOG_FLAG_INFO, "Failed to create queue, size=%d, type=%d,"
       " priority=%d, engine_id=%d", kQueueSize, kQueueType_, HSA_QUEUE_PRIORITY_MAXIMUM, rec_eng);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  // Cache MMIO pointers to avoid repeated struct access + reinterpret_cast in hot paths.
  queue_wptr_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_write_ptr);
  queue_rptr_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_read_ptr);
  queue_doorbell_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_DoorBell);

  cached_reserve_index_ = *queue_wptr_;
  cached_commit_index_ = cached_reserve_index_;

  // Cache platform/flag checks to avoid pointer chasing in the hot path.
  is_dxg_ = core::Runtime::runtime_singleton_->thunkLoader()->IsDXG();
  needs_kmt_doorbell_ = is_dxg_ ||
                        core::Runtime::runtime_singleton_->thunkLoader()->IsDTIF();
  sdma_wait_idle_ = core::Runtime::runtime_singleton_->flag().sdma_wait_idle();
  enable_sdma_hdp_flush_ = core::Runtime::runtime_singleton_->flag().enable_sdma_hdp_flush();

  // Cache gfx90x SW poll workaround flag to avoid static-local guard overhead
  // on every SubmitCommand call.
  sw_poll_workaround_ = agent_->supported_isas()[0]->GetMajorVersion() == 9 &&
                        agent_->supported_isas()[0]->GetMinorVersion() == 0 &&
                        agent_->supported_isas()[0]->GetStepping() != 10;

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

template <bool useGCR, bool scopeFields> hsa_status_t BlitSdma<useGCR, scopeFields>::Destroy() {
  // Release all allocated resources and reset them to zero.

  if (queue_resource_.QueueId != 0) {
    // Release queue resources from the kernel
    auto err = agent_->driver().DestroyQueue(queue_resource_.QueueId);
    assert(err == HSA_STATUS_SUCCESS);
    (void)err;
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

class CommandCallBackData {
  public:
    CommandCallBackData(const void* cmd,
                        size_t cmd_size,
                        uint64_t size,
                        size_t num_dep_signals,
                        core::Signal& out_signal,
                        std::vector<core::Signal*>& gang_signals,
                        BlitSdmaBase* owner):
                        cmd_size_(cmd_size),
                        size_(size),
                        num_dep_signals_(num_dep_signals),
                        out_signal_(&out_signal),
                        owner_(owner) {
      cmd_ = malloc(cmd_size);
      if (cmd == nullptr)
        throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                              "Failed to allocate data for copy callback.");

      memcpy(cmd_, cmd, cmd_size);

      for (auto gang_sig: gang_signals)
        gang_signals_.push_back(gang_sig);
    }

    ~CommandCallBackData() {
      free(cmd_);
    }

    void *cmd_;
    size_t cmd_size_;
    uint64_t size_;
    size_t num_dep_signals_;
    core::Signal* out_signal_;
    std::vector<core::Signal*> gang_signals_;
    BlitSdmaBase* owner_;
};

static bool DepSignalCompleteHandler(hsa_signal_value_t signal_value, void *arg ) {
  CommandCallBackData* callbackData = reinterpret_cast<CommandCallBackData*>(arg);

  if (--callbackData->num_dep_signals_ == 0) {
    /* Callback SubmitCommand with no dependent signals */
    const std::vector<core::Signal*> dep_signals(0);

    callbackData->owner_->SubmitCommand(callbackData->cmd_,
                                        callbackData->cmd_size_,
                                        callbackData->size_,
                                        dep_signals,
                                        *(callbackData->out_signal_),
                                        callbackData->gang_signals_);
    delete callbackData;
  }
  return false;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitBlockingCommand(const void* cmd, size_t cmd_size,
                                                     uint64_t size) {
  std::unique_lock<std::mutex> lock(lock_);

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
  lock.unlock();

  std::vector<core::Signal*> gang_signals(0);

  // Submit command and wait for completion
  hsa_status_t ret =
      SubmitCommand(cmd, cmd_size, size, std::vector<core::Signal*>(), *completionSignal,
                    gang_signals);
  completionSignal->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 1, -1, HSA_WAIT_STATE_BLOCKED);
  return ret;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitCommand(const void* cmd, size_t cmd_size, uint64_t size,
                                             const std::vector<core::Signal*>& dep_signals,
                                             core::Signal& out_signal,
                                             std::vector<core::Signal*>& gang_signals) {
  uint32_t num_poll_command = 0;
  uint32_t num_poll_signals = 0;

  // Cached copy of dep_signals[i]->LoadRelaxed
  uint64_t dep_signals_value[HSA_MAX_DEP_SIGNALS];

  for (size_t i = 0; i < dep_signals.size(); ++i) {
    // The signal is 64 bit value, and poll checks for 32 bit value.
    // If the signal is already 0, then we do not need to poll.
    // If the upper 32-bits of the signal is 0, then we only need to poll the
    // lower 32-bits
    dep_signals_value[i] = dep_signals[i]->LoadRelaxed();
    if (dep_signals_value[i]) {
      num_poll_signals++;
      num_poll_command++;
      if (dep_signals_value[i] >> 32)
        num_poll_command++;
    }
  }

  // Workaround for rare issue on gfx90x asics where SDMA_OP_POLL_REGMEM returns before
  // polled memory is cleared. Use SetAsyncSignalHandler to poll the signal signal
  // value on host-side. Once all the dependent signals are cleared, DepSignalCompleteHandler
  // will call SubmitCommand(..) again without any dependent-signals.
  if (sw_poll_workaround_ && num_poll_signals) {
    CommandCallBackData* callbackArgs =
      new CommandCallBackData(cmd, cmd_size, size, num_poll_signals, out_signal, gang_signals, this);

    if (callbackArgs == nullptr) {
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                             "Failed to allocate data for copy callback.");
    }

    for (size_t i = 0; i < dep_signals.size(); ++i) {
      if (dep_signals_value[i]) {
        core::Runtime::runtime_singleton_->SetAsyncSignalHandler(
                                         core::Signal::Convert(dep_signals[i]),
                                         HSA_SIGNAL_CONDITION_EQ, 0, DepSignalCompleteHandler,
                                         reinterpret_cast<void*>(callbackArgs));
      }
    }
    return HSA_STATUS_SUCCESS;
  }

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
  if (enable_sdma_hdp_flush_) {
    if (hdp_flush_support_) {
      flush_cmd_size = flush_command_size_;
    }
  }

  // Add space for cache flush.
  if (useGCR) flush_cmd_size += gcr_command_size() * 2;

  const uint32_t total_command_size = total_poll_command_size + cmd_size + sync_command_size +
      total_timestamp_command_size + interrupt_command_size + flush_cmd_size + total_gang_command_size;
  const uint32_t pad_size = total_command_size < min_submission_size_ ?
                            min_submission_size_ - total_command_size :
                            is_dxg_ ?
                              AlignUp(total_command_size, 64) - total_command_size : 0;

  uint64_t curr_index;
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
      }
      // Then wait for the lower 32 bits to 0.
      BuildPollCommand(command_addr, &signal_addr[0], 0);
      command_addr += poll_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += poll_command_size_;
    }
  }

  if (profiling_enabled && (gang_leader_ || gang_signals.empty())) {
    BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(start_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += timestamp_command_size_;
  }

  // Issue a Hdp flush cmd
  if (enable_sdma_hdp_flush_) {
    if (hdp_flush_support_) {
      BuildHdpFlushCommand(command_addr);
      command_addr += flush_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += flush_command_size_;
    }
  }

  // Issue cache invalidate
  if (useGCR) {
    BuildGCRCommand(command_addr, true);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += gcr_command_size();
  }

  // Do the command after all polls are satisfied.
  memcpy(command_addr, cmd, cmd_size);
  command_addr += cmd_size;
  bytes_written_.fill(wrapped_index, wrapped_index + cmd_size, prior_bytes);
  wrapped_index += cmd_size;

  // Issue cache writeback
  if (useGCR) {
    BuildGCRCommand(command_addr, false);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += gcr_command_size();
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

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitLinearCopyBodyWaitSignal(
    void* dst, const void* src, size_t size,
    const std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal,
    bool fused_notify) {

  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_
                                                            : kMaxSingleCopySize;
  const uint32_t num_copy_command =
      static_cast<uint32_t>((size + max_copy_size - 1) / max_copy_size);

  const bool profiling_enabled = agent_->profiling_enabled();

  uint64_t* start_ts_addr = nullptr;
  uint64_t* end_ts_addr = nullptr;
  uint32_t total_timestamp_command_size = 0;

  if (profiling_enabled) {
    out_signal.GetSdmaTsAddresses(start_ts_addr, end_ts_addr);
    total_timestamp_command_size = 2 * timestamp_command_size_;
  }

  const uint32_t wait_dws = dep_signals.empty() ? 0 : 7;
  const uint32_t per_pkt_dws = 1 + wait_dws + 6 + 5;
  const uint32_t total_copy_dws = num_copy_command * per_pkt_dws;

  const uint32_t extra_polls = (dep_signals.size() > 1)
      ? static_cast<uint32_t>(dep_signals.size() - 1) : 0;
  const uint32_t extra_poll_bytes = extra_polls * poll_64b_command_size_;

  const uint32_t copy_bytes = total_copy_dws * sizeof(uint32_t);

  const bool has_mailbox = (out_signal.signal_.event_mailbox_ptr != 0);
  const bool emit_prologue = fused_notify && useGCR;
  const bool emit_epilogue = fused_notify && (useGCR || has_mailbox);

  const uint32_t prologue_bytes = emit_prologue ? gcr_command_size() : 0;
  uint32_t epilogue_bytes = 0;
  if (emit_epilogue) {
    if (useGCR) epilogue_bytes += gcr_command_size();
    if (has_mailbox) epilogue_bytes += fence_command_size_ + trap_command_size_;
  }

  const uint32_t total_command_size =
      prologue_bytes + extra_poll_bytes + total_timestamp_command_size +
      copy_bytes + epilogue_bytes;

  const uint32_t pad_size = total_command_size < min_submission_size_
      ? min_submission_size_ - total_command_size
      : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

  uint64_t curr_index;
  char* command_addr;
  uint64_t prior_bytes, post_bytes;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
    if (command_addr == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    prior_bytes = bytes_queued_;
    bytes_queued_ += size;
    post_bytes = bytes_queued_;
  }
  uint32_t wrapped_index = WrapIntoRing(curr_index);

  if (num_copy_command > 1)
    out_signal.AddRelaxed(num_copy_command - 1);

  if (emit_prologue) {
    BuildGCRCommand(command_addr, true);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += gcr_command_size();
  }

  for (uint32_t i = 1; i < dep_signals.size(); ++i) {
    BuildPoll64bCommand(command_addr, dep_signals[i]->ValueLocation(), 0);
    command_addr += poll_64b_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += poll_64b_command_size_;
  }

  if (profiling_enabled) {
    BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(start_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += timestamp_command_size_;
  }

  const core::Signal* wait_sig = dep_signals.empty() ? nullptr : dep_signals[0];
  BuildWaitSignalCopyCommand(command_addr, num_copy_command,
                             dst, src, size,
                             wait_sig, &out_signal);
  bytes_written_.fill(wrapped_index, wrapped_index + copy_bytes, prior_bytes);
  bytes_written_[wrapped_index + copy_bytes - sizeof(uint32_t)] = post_bytes;
  command_addr += copy_bytes;
  wrapped_index += copy_bytes;

  if (profiling_enabled) {
    assert(IsMultipleOf(end_ts_addr, 32));
    BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(end_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += timestamp_command_size_;
  }

  if (emit_epilogue) {
    if (useGCR) {
      BuildGCRCommand(command_addr, false);
      command_addr += gcr_command_size();
      bytes_written_[wrapped_index] = post_bytes;
      wrapped_index += gcr_command_size();
    }

    if (has_mailbox) {
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
  }

  if (pad_size) {
    memset(command_addr, 0, pad_size);
    uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
    dword_command_addr[0] = (pad_size / 4 - 1) << 16;
  }

  ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitPrologue(
    const std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal,
    core::Signal& prologue_signal,
    bool fused_bodies) {

  const bool profiling_enabled = agent_->profiling_enabled();
  const bool emit_deps = !fused_bodies || profiling_enabled;

  uint32_t num_poll_command = 0;
  uint64_t dep_signals_value[HSA_MAX_DEP_SIGNALS];

  if (emit_deps) {
    for (size_t i = 0; i < dep_signals.size(); ++i) {
      dep_signals_value[i] = dep_signals[i]->LoadRelaxed();
      if (dep_signals_value[i]) {
        if (is_gfx125plus_) {
          num_poll_command++;
        } else {
          num_poll_command++;
          if (dep_signals_value[i] >> 32)
            num_poll_command++;
        }
      }
    }
  }

  const uint32_t per_poll_size = (is_gfx125plus_) ? poll_64b_command_size_ : poll_command_size_;
  const uint32_t total_poll_command_size = num_poll_command * per_poll_size;

  uint64_t* start_ts_addr = nullptr;
  uint64_t* end_ts_addr = nullptr;
  uint32_t total_timestamp_command_size = 0;

  if (profiling_enabled) {
    out_signal.GetSdmaTsAddresses(start_ts_addr, end_ts_addr);
    total_timestamp_command_size = timestamp_command_size_;
  }

  uint32_t flush_cmd_size = 0;
  if (emit_deps && enable_sdma_hdp_flush_ && hdp_flush_support_)
    flush_cmd_size = flush_command_size_;
  if (useGCR) flush_cmd_size += gcr_command_size();

  const size_t prologue_signal_cmd_size =
      is_gfx125plus_ ? fence_64b_command_size_ : fence_command_size_;

  const uint32_t total_command_size = total_poll_command_size +
      total_timestamp_command_size + flush_cmd_size + prologue_signal_cmd_size;
  const uint32_t pad_size = total_command_size < min_submission_size_
      ? min_submission_size_ - total_command_size
      : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

  uint64_t curr_index;
  char* command_addr;
  uint64_t prior_bytes;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
    if (command_addr == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    prior_bytes = bytes_queued_;
  }
  uint32_t wrapped_index = WrapIntoRing(curr_index);

  if (emit_deps) {
    for (size_t i = 0; i < dep_signals.size(); ++i) {
      if (dep_signals_value[i]) {
        if (is_gfx125plus_) {
          BuildPoll64bCommand(command_addr, dep_signals[i]->ValueLocation(), 0);
          command_addr += poll_64b_command_size_;
          bytes_written_[wrapped_index] = prior_bytes;
          wrapped_index += poll_64b_command_size_;
        } else {
          uint32_t* signal_addr =
              reinterpret_cast<uint32_t*>(dep_signals[i]->ValueLocation());
          if (dep_signals_value[i] >> 32) {
            BuildPollCommand(command_addr, &signal_addr[1], 0);
            command_addr += poll_command_size_;
            bytes_written_[wrapped_index] = prior_bytes;
            wrapped_index += poll_command_size_;
          }
          BuildPollCommand(command_addr, &signal_addr[0], 0);
          command_addr += poll_command_size_;
          bytes_written_[wrapped_index] = prior_bytes;
          wrapped_index += poll_command_size_;
        }
      }
    }
  }

  if (profiling_enabled) {
    BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(start_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += timestamp_command_size_;
  }

  if (emit_deps && enable_sdma_hdp_flush_ && hdp_flush_support_) {
    BuildHdpFlushCommand(command_addr);
    command_addr += flush_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += flush_command_size_;
  }

  if (useGCR) {
    BuildGCRCommand(command_addr, true);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += gcr_command_size();
  }

  if (is_gfx125plus_) {
    BuildFence64bCommand(command_addr, prologue_signal.ValueLocation(), 0);
    command_addr += fence_64b_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += fence_64b_command_size_;
  } else {
    uint32_t* sig_loc = reinterpret_cast<uint32_t*>(prologue_signal.ValueLocation());
    BuildFenceCommand(command_addr, sig_loc, 0);
    command_addr += fence_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += fence_command_size_;
  }

  if (pad_size) {
    memset(command_addr, 0, pad_size);
    uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
    dword_command_addr[0] = (pad_size / 4 - 1) << 16;
  }

  ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitEpilogue(
    core::Signal& out_signal,
    const std::vector<core::Signal*>& body_signals) {

  const bool profiling_enabled = agent_->profiling_enabled();

  uint64_t* start_ts_addr = nullptr;
  uint64_t* end_ts_addr = nullptr;
  uint32_t total_timestamp_command_size = 0;

  if (profiling_enabled) {
    out_signal.GetSdmaTsAddresses(start_ts_addr, end_ts_addr);
    total_timestamp_command_size = timestamp_command_size_;
  }

  // Drain poll size: with platform atomics poll out_signal for 1 (one poll).
  // Without platform atomics poll each body_signal for 0 (N polls).
  const uint32_t single_poll_size = is_gfx125plus_
      ? poll_64b_command_size_
      : poll_command_size_;
  const uint32_t body_poll_size = body_signals.empty()
      ? single_poll_size
      : static_cast<uint32_t>(body_signals.size()) * single_poll_size;

  uint32_t gcr_cmd_size = 0;
  if (useGCR) gcr_cmd_size = gcr_command_size();

  // Final signal write: epilogue drives out_signal to 0.
  const size_t sync_command_size = is_gfx125plus_
      ? fence_64b_command_size_
      : (platform_atomic_support_ ? atomic_command_size_ : fence_command_size_);

  const size_t interrupt_command_size =
      (out_signal.signal_.event_mailbox_ptr != 0)
          ? (fence_command_size_ + trap_command_size_)
          : 0;

  const uint32_t total_command_size = body_poll_size + gcr_cmd_size +
      total_timestamp_command_size + sync_command_size + interrupt_command_size;
  const uint32_t pad_size = total_command_size < min_submission_size_
      ? min_submission_size_ - total_command_size
      : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

  uint64_t curr_index;
  char* command_addr;
  uint64_t prior_bytes;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
    if (command_addr == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    prior_bytes = bytes_queued_;
  }
  uint32_t wrapped_index = WrapIntoRing(curr_index);

  // Drain: wait for bodies to finish.
  if (body_signals.empty()) {
    // Platform atomics: bodies decremented out_signal; poll for 1.
    if (is_gfx125plus_) {
      BuildPoll64bCommand(command_addr, out_signal.ValueLocation(), 1);
      command_addr += poll_64b_command_size_;
    } else {
      uint32_t* sig_lo = reinterpret_cast<uint32_t*>(out_signal.ValueLocation());
      BuildPollCommand(command_addr, &sig_lo[0], 1);
      command_addr += poll_command_size_;
    }
  } else {
    // No platform atomics (never gfx125+): bodies fenced body_signals to 0.
    for (core::Signal* bs : body_signals) {
      uint32_t* sig_lo = reinterpret_cast<uint32_t*>(bs->ValueLocation());
      BuildPollCommand(command_addr, &sig_lo[0], 0);
      command_addr += poll_command_size_;
    }
  }
  bytes_written_[wrapped_index] = prior_bytes;
  wrapped_index += body_poll_size;

  if (useGCR) {
    BuildGCRCommand(command_addr, false);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += gcr_command_size();
  }

  if (profiling_enabled) {
    assert(IsMultipleOf(end_ts_addr, 32));
    BuildGetGlobalTimestampCommand(command_addr,
                                   reinterpret_cast<void*>(end_ts_addr));
    command_addr += timestamp_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += timestamp_command_size_;
  }

  // Write out_signal to 0 (final completion).
  if (is_gfx125plus_) {
    BuildFence64bCommand(command_addr, out_signal.ValueLocation(), 0);
    command_addr += fence_64b_command_size_;
  } else if (platform_atomic_support_) {
    BuildAtomicDecrementCommand(command_addr, out_signal.ValueLocation());
    command_addr += atomic_command_size_;
  } else {
    uint32_t* sig_lo = reinterpret_cast<uint32_t*>(out_signal.ValueLocation());
    BuildFenceCommand(command_addr, sig_lo, 0);
    command_addr += fence_command_size_;
  }
  bytes_written_[wrapped_index] = prior_bytes;
  wrapped_index += sync_command_size;

  if (out_signal.signal_.event_mailbox_ptr != 0) {
    BuildFenceCommand(command_addr,
                      reinterpret_cast<uint32_t*>(out_signal.signal_.event_mailbox_ptr),
                      static_cast<uint32_t>(out_signal.signal_.event_id));
    command_addr += fence_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += fence_command_size_;

    BuildTrapCommand(command_addr, out_signal.signal_.event_id);
    command_addr += trap_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += trap_command_size_;
  }

  if (pad_size) {
    memset(command_addr, 0, pad_size);
    uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
    dword_command_addr[0] = (pad_size / 4 - 1) << 16;
  }

  ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitFusedCoordinator(
    const std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal,
    core::Signal* prologue_signal,
    bool fused_bodies,
    hsa_amd_memory_copy_op_type_t op,
    const std::vector<void*>& dsts,
    const std::vector<const void*>& srcs,
    const std::vector<size_t>& sizes_a,
    const std::vector<size_t>& sizes_b,
    bool indirect_src, bool indirect_dst,
    const std::vector<core::Signal*>& body_deps) {

  const size_t num_entries = dsts.size();
  const bool profiling_enabled = agent_->profiling_enabled();
  const bool need_prologue = prologue_signal != nullptr;
  const bool is_swap = (op == HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP);
  const bool is_indirect = (indirect_src || indirect_dst);
  const bool has_mailbox = (out_signal.signal_.event_mailbox_ptr != 0);

  // Epilogue fast path: the epilogue exists to (a) raise the completion
  // interrupt (fence+trap) once all engines finish, (b) order a GCR cache
  // writeback before signalling, and (c) emit the profiling end-timestamp.
  // When none of those apply, the poll+fence are redundant: every body packet's
  // SIGNAL stage already atomically subtracts 1 from out_signal *after* its copy
  // (SDMA executes in-order per engine), so the last chunk's sub is the correct
  // completion edge on its own.  We drop the whole epilogue and shift the signal
  // target down by one (below) so that final sub lands on the done value.
  const bool skip_epilogue = !useGCR && !profiling_enabled && !has_mailbox;

  // --- Prologue size computation ---
  uint32_t prologue_size = 0;
  uint32_t num_poll_command = 0;
  uint64_t dep_signals_value[HSA_MAX_DEP_SIGNALS];

  if (need_prologue) {
    const bool emit_deps = !fused_bodies || profiling_enabled;
    if (emit_deps) {
      for (size_t i = 0; i < dep_signals.size(); ++i) {
        dep_signals_value[i] = dep_signals[i]->LoadRelaxed();
        if (dep_signals_value[i]) num_poll_command++;
      }
    }

    prologue_size += num_poll_command * poll_64b_command_size_;

    if (profiling_enabled) prologue_size += timestamp_command_size_;

    if (emit_deps && enable_sdma_hdp_flush_ && hdp_flush_support_)
      prologue_size += flush_command_size_;
    if (useGCR) prologue_size += gcr_command_size();

    prologue_size += fence_64b_command_size_;
  }

  // --- Body size computation (gfx125+ fused WaitSignal path) ---
  const size_t max_copy_size = is_swap
      ? SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250::kMaxSize_
      : (max_single_linear_copy_size_ ? max_single_linear_copy_size_ : kMaxSingleCopySize);

  const uint32_t wait_dws = body_deps.empty() ? 0 : 7;
  const uint32_t per_pkt_dws = 1 + wait_dws + 6 + 5;

  std::vector<uint32_t> chunks(num_entries);
  uint32_t total_chunks = 0;
  uint64_t total_bytes = 0;
  for (size_t i = 0; i < num_entries; ++i) {
    const size_t entry_size = is_swap
        ? std::max(sizes_a[i], sizes_b[i]) : sizes_a[i];
    chunks[i] = is_indirect ? 1
        : static_cast<uint32_t>((entry_size + max_copy_size - 1) / max_copy_size);
    total_chunks += chunks[i];
    total_bytes += entry_size;
  }
  const uint32_t body_copy_bytes = total_chunks * per_pkt_dws * sizeof(uint32_t);

  const uint32_t body_extra_polls = (body_deps.size() > 1)
      ? static_cast<uint32_t>(body_deps.size() - 1) : 0;
  const uint32_t body_extra_poll_bytes = body_extra_polls * poll_64b_command_size_;

  const uint32_t body_size = body_extra_poll_bytes + body_copy_bytes;

  // --- Epilogue size computation ---
  uint32_t epilogue_size = 0;

  if (!skip_epilogue) {
    epilogue_size += poll_64b_command_size_;

    if (useGCR) epilogue_size += gcr_command_size();
    if (profiling_enabled) epilogue_size += timestamp_command_size_;

    epilogue_size += fence_64b_command_size_;

    if (has_mailbox)
      epilogue_size += fence_command_size_ + trap_command_size_;
  }

  // --- Total reservation ---
  const uint32_t total_command_size = prologue_size + body_size + epilogue_size;
  const uint32_t pad_size = total_command_size < min_submission_size_
      ? min_submission_size_ - total_command_size
      : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

  uint64_t curr_index;
  char* command_addr;
  uint64_t prior_bytes, post_bytes;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
    if (command_addr == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    prior_bytes = bytes_queued_;
    bytes_queued_ += total_bytes;
    post_bytes = bytes_queued_;
  }
  uint32_t wrapped_index = WrapIntoRing(curr_index);

  // Arm out_signal for multi-chunk body entries (same as SubmitBodies).
  if (total_chunks > num_entries)
    out_signal.AddRelaxed(total_chunks - num_entries);

  // With no epilogue fence to perform the final completion decrement, the body
  // SIGNAL subs must reach the done value themselves: drop the target by one.
  if (skip_epilogue)
    out_signal.SubRelaxed(1);

  // === PROLOGUE SECTION ===
  if (need_prologue) {
    const bool emit_deps = !fused_bodies || profiling_enabled;

    if (emit_deps) {
      for (size_t i = 0; i < dep_signals.size(); ++i) {
        if (dep_signals_value[i]) {
          BuildPoll64bCommand(command_addr, dep_signals[i]->ValueLocation(), 0);
          command_addr += poll_64b_command_size_;
          bytes_written_[wrapped_index] = prior_bytes;
          wrapped_index += poll_64b_command_size_;
        }
      }
    }

    if (profiling_enabled) {
      uint64_t* start_ts_addr = nullptr;
      uint64_t* end_ts_addr = nullptr;
      out_signal.GetSdmaTsAddresses(start_ts_addr, end_ts_addr);
      BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(start_ts_addr));
      command_addr += timestamp_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += timestamp_command_size_;
    }

    if (emit_deps && enable_sdma_hdp_flush_ && hdp_flush_support_) {
      BuildHdpFlushCommand(command_addr);
      command_addr += flush_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += flush_command_size_;
    }

    if (useGCR) {
      BuildGCRCommand(command_addr, true);
      command_addr += gcr_command_size();
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += gcr_command_size();
    }

    BuildFence64bCommand(command_addr, prologue_signal->ValueLocation(), 0);
    command_addr += fence_64b_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += fence_64b_command_size_;
  }

  // === BODY SECTION (fused WaitSignal path) ===
  for (uint32_t i = 1; i < body_deps.size(); ++i) {
    BuildPoll64bCommand(command_addr, body_deps[i]->ValueLocation(), 0);
    command_addr += poll_64b_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += poll_64b_command_size_;
  }

  const core::Signal* wait_sig = body_deps.empty() ? nullptr : body_deps[0];
  const uint32_t copy_start = wrapped_index;

  for (size_t i = 0; i < num_entries; ++i) {
    if (is_indirect) {
      BuildWaitSignalIndirectCopyCommand(command_addr,
          dsts[i], srcs[i], sizes_a[i],
          indirect_src, indirect_dst,
          wait_sig, &out_signal);
    } else if (is_swap) {
      BuildWaitSignalSwapCommand(command_addr, chunks[i],
          dsts[i], const_cast<void*>(srcs[i]),
          sizes_a[i], sizes_b[i],
          wait_sig, &out_signal);
    } else {
      BuildWaitSignalCopyCommand(command_addr, chunks[i],
          dsts[i], srcs[i], sizes_a[i],
          wait_sig, &out_signal);
    }
    const uint32_t entry_bytes = chunks[i] * per_pkt_dws * sizeof(uint32_t);
    command_addr += entry_bytes;
    wrapped_index += entry_bytes;
  }
  bytes_written_.fill(copy_start, copy_start + body_copy_bytes, prior_bytes);
  bytes_written_[copy_start + body_copy_bytes - sizeof(uint32_t)] = post_bytes;

  // === EPILOGUE SECTION ===
  // Skipped entirely on the fast path: the body SIGNAL subs are the completion
  // edge (see skip_epilogue above).
  if (!skip_epilogue) {
    BuildPoll64bCommand(command_addr, out_signal.ValueLocation(), 1);
    command_addr += poll_64b_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += poll_64b_command_size_;

    if (useGCR) {
      BuildGCRCommand(command_addr, false);
      command_addr += gcr_command_size();
      bytes_written_[wrapped_index] = post_bytes;
      wrapped_index += gcr_command_size();
    }

    if (profiling_enabled) {
      uint64_t* start_ts_addr = nullptr;
      uint64_t* end_ts_addr = nullptr;
      out_signal.GetSdmaTsAddresses(start_ts_addr, end_ts_addr);
      assert(IsMultipleOf(end_ts_addr, 32));
      BuildGetGlobalTimestampCommand(command_addr, reinterpret_cast<void*>(end_ts_addr));
      command_addr += timestamp_command_size_;
      bytes_written_[wrapped_index] = post_bytes;
      wrapped_index += timestamp_command_size_;
    }

    BuildFence64bCommand(command_addr, out_signal.ValueLocation(), 0);
    command_addr += fence_64b_command_size_;
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += fence_64b_command_size_;

    if (has_mailbox) {
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
  }

  if (pad_size) {
    memset(command_addr, 0, pad_size);
    uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
    dword_command_addr[0] = (pad_size / 4 - 1) << 16;
  }

  ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitBodies(
    hsa_amd_memory_copy_op_type_t op,
    void* const* dst_list,
    const void* const* src_list,
    const size_t* size_list,
    const std::vector<uint32_t>& indices,
    bool indirect_src, bool indirect_dst,
    const std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal,
    core::Signal* body_signal) {

  const size_t num_entries = indices.size();
  if (num_entries == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const bool is_swap = (op == HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP);
  const bool is_indirect = (indirect_src || indirect_dst);

  if (is_indirect && !indirect_copy_supported_)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (is_swap && !swap_supported_ && !is_gfx125plus_)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (is_swap) {
    constexpr size_t kAlign = SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250::kAlignment_;
    for (size_t i = 0; i < num_entries; ++i) {
      const uint32_t d = indices[i];
      if ((reinterpret_cast<uintptr_t>(dst_list[d]) & (kAlign - 1)) != 0 ||
          (reinterpret_cast<uintptr_t>(src_list[d]) & (kAlign - 1)) != 0)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  const size_t max_copy_size = is_swap
      ? SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250::kMaxSize_
      : (max_single_linear_copy_size_ ? max_single_linear_copy_size_ : kMaxSingleCopySize);

  if (is_gfx125plus_) {
    // --- Fused WaitSignal path: every chunk carries wait+copy+signal in one packet ---
    const uint32_t wait_dws = dep_signals.empty() ? 0 : 7;
    const uint32_t per_pkt_dws = 1 + wait_dws + 6 + 5;

    std::vector<uint32_t> chunks(num_entries);
    uint32_t total_chunks = 0;
    uint64_t total_bytes = 0;
    for (size_t i = 0; i < num_entries; ++i) {
      const size_t entry_size = size_list[indices[i]];
      if (is_indirect && entry_size > max_copy_size)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      chunks[i] = is_indirect ? 1
          : static_cast<uint32_t>((entry_size + max_copy_size - 1) / max_copy_size);
      total_chunks += chunks[i];
      total_bytes += entry_size;
    }
    const uint32_t total_copy_dws = total_chunks * per_pkt_dws;
    const uint32_t copy_bytes = total_copy_dws * sizeof(uint32_t);

    const uint32_t extra_polls = (dep_signals.size() > 1)
        ? static_cast<uint32_t>(dep_signals.size() - 1) : 0;
    const uint32_t extra_poll_bytes = extra_polls * poll_64b_command_size_;

    const uint32_t total_command_size = extra_poll_bytes + copy_bytes;
    const uint32_t pad_size = total_command_size < min_submission_size_
        ? min_submission_size_ - total_command_size
        : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

    uint64_t curr_index;
    char* command_addr;
    uint64_t prior_bytes, post_bytes;
    {
      std::lock_guard<std::mutex> lock(reservation_lock_);
      command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
      if (command_addr == nullptr)
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      prior_bytes = bytes_queued_;
      bytes_queued_ += total_bytes;
      post_bytes = bytes_queued_;
    }
    uint32_t wrapped_index = WrapIntoRing(curr_index);

    if (total_chunks > num_entries)
      out_signal.AddRelaxed(total_chunks - num_entries);

    for (uint32_t i = 1; i < dep_signals.size(); ++i) {
      BuildPoll64bCommand(command_addr, dep_signals[i]->ValueLocation(), 0);
      command_addr += poll_64b_command_size_;
      bytes_written_[wrapped_index] = prior_bytes;
      wrapped_index += poll_64b_command_size_;
    }

    const core::Signal* wait_sig = dep_signals.empty() ? nullptr : dep_signals[0];
    const uint32_t copy_start = wrapped_index;

    for (size_t i = 0; i < num_entries; ++i) {
      const uint32_t d = indices[i];
      if (is_indirect) {
        BuildWaitSignalIndirectCopyCommand(command_addr,
            dst_list[d], src_list[d], size_list[d],
            indirect_src, indirect_dst,
            wait_sig, &out_signal);
      } else if (is_swap) {
        BuildWaitSignalSwapCommand(command_addr, chunks[i],
            dst_list[d], const_cast<void*>(src_list[d]),
            size_list[d], size_list[d],
            wait_sig, &out_signal);
      } else {
        BuildWaitSignalCopyCommand(command_addr, chunks[i],
            dst_list[d], src_list[d], size_list[d],
            wait_sig, &out_signal);
      }
      const uint32_t entry_bytes = chunks[i] * per_pkt_dws * sizeof(uint32_t);
      command_addr += entry_bytes;
      wrapped_index += entry_bytes;
    }
    bytes_written_.fill(copy_start, copy_start + copy_bytes, prior_bytes);
    bytes_written_[copy_start + copy_bytes - sizeof(uint32_t)] = post_bytes;

    if (pad_size) {
      memset(command_addr, 0, pad_size);
      uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
      dword_command_addr[0] = (pad_size / 4 - 1) << 16;
    }

    ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  } else {
    // --- Classic path: single reservation, poll + copies + signal completion ---
    // Entries are serialized on one engine.  With platform atomics: single
    // atomic dec of out_signal.  Without: fence body_signal to 0 (epilogue
    // polls body_signals then fences out_signal).
    //
    // The classic path always leads with a poll on the prologue signal
    // (dep_signals[0]); every internal caller supplies one. Guard against
    // external callers passing empty dep_signals which would null-deref below.
    if (dep_signals.empty())
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    const size_t classic_max = is_swap
        ? SDMA_PKT_COPY_LINEAR_SWAP::kMaxSize_
        : (max_single_linear_copy_size_ ? max_single_linear_copy_size_ : kMaxSingleCopySize);

    core::Signal* prologue_sig = dep_signals.empty() ? nullptr : dep_signals[0];
    const uint32_t poll_size = poll_command_size_;

    const uint32_t per_pkt_dws = is_swap
        ? static_cast<uint32_t>(sizeof(SDMA_PKT_COPY_LINEAR_SWAP) / sizeof(uint32_t))
        : static_cast<uint32_t>(sizeof(SDMA_PKT_COPY_LINEAR) / sizeof(uint32_t));

    std::vector<uint32_t> chunks(num_entries);
    uint32_t total_copy_bytes = 0;
    uint64_t total_bytes = 0;
    for (size_t i = 0; i < num_entries; ++i) {
      const size_t entry_size = size_list[indices[i]];
      chunks[i] = static_cast<uint32_t>((entry_size + classic_max - 1) / classic_max);
      total_copy_bytes += chunks[i] * per_pkt_dws * sizeof(uint32_t);
      total_bytes += entry_size;
    }

    const uint32_t signal_cmd_size = platform_atomic_support_
        ? atomic_command_size_ : fence_command_size_;
    const uint32_t total_command_size = poll_size + total_copy_bytes + signal_cmd_size;
    const uint32_t pad_size = total_command_size < min_submission_size_
        ? min_submission_size_ - total_command_size
        : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

    uint64_t curr_index;
    char* command_addr;
    uint64_t prior_bytes, post_bytes;
    {
      std::lock_guard<std::mutex> lock(reservation_lock_);
      command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
      if (command_addr == nullptr)
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      prior_bytes = bytes_queued_;
      bytes_queued_ += total_bytes;
      post_bytes = bytes_queued_;
    }
    uint32_t wrapped_index = WrapIntoRing(curr_index);

    // Poll prologue signal.
    uint32_t* poll_addr = reinterpret_cast<uint32_t*>(prologue_sig->ValueLocation());
    BuildPollCommand(command_addr, &poll_addr[0], 0);
    command_addr += poll_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += poll_command_size_;

    // Copy packets (all entries, serialized).
    const uint32_t copy_start = wrapped_index;
    for (size_t i = 0; i < num_entries; ++i) {
      const uint32_t d = indices[i];
      const uint32_t entry_copy_bytes = chunks[i] * per_pkt_dws * sizeof(uint32_t);

      if (is_swap) {
        BuildSwapCopyCommand(command_addr, chunks[i],
                             dst_list[d], const_cast<void*>(src_list[d]), size_list[d]);
      } else {
        BuildCopyCommand(command_addr, chunks[i], dst_list[d], src_list[d], size_list[d]);
      }
      command_addr += entry_copy_bytes;
      wrapped_index += entry_copy_bytes;
    }

    // Signal body completion.
    if (platform_atomic_support_) {
      BuildAtomicDecrementCommand(command_addr, out_signal.ValueLocation());
      command_addr += atomic_command_size_;
    } else {
      uint32_t* sig_lo = reinterpret_cast<uint32_t*>(body_signal->ValueLocation());
      BuildFenceCommand(command_addr, sig_lo, 0);
      command_addr += fence_command_size_;
    }
    wrapped_index += signal_cmd_size;

    bytes_written_.fill(copy_start, copy_start + total_copy_bytes +
        signal_cmd_size, prior_bytes);
    bytes_written_[wrapped_index - sizeof(uint32_t)] = post_bytes;

    if (pad_size) {
      memset(command_addr, 0, pad_size);
      uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
      dword_command_addr[0] = (pad_size / 4 - 1) << 16;
    }

    ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  }

  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitLinearCopyCommand(void* dst, const void* src, size_t size) {
  // Break the copy into multiple copy operation incase the copy size exceeds
  // the SDMA linear copy limit.
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                               kMaxSingleCopySize;
  const uint32_t num_copy_command = (size + max_copy_size - 1) / max_copy_size;

  // Avoid heap allocation for common single-packet case.
  SDMA_PKT_COPY_LINEAR stack_buff;
  std::vector<SDMA_PKT_COPY_LINEAR> heap_buff(num_copy_command > 1 ? num_copy_command : 0);
  auto* buff = num_copy_command <= 1 ? &stack_buff : heap_buff.data();

  BuildCopyCommand(reinterpret_cast<char*>(buff), num_copy_command, dst, src, size);

  return SubmitBlockingCommand(buff, num_copy_command * sizeof(SDMA_PKT_COPY_LINEAR), size);
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitLinearCopyCommand(void* dst, const void* src, size_t size,
                                                       std::vector<core::Signal*>& dep_signals,
                                                       core::Signal& out_signal,
                                                       std::vector<core::Signal*>& gang_signals) {
  // Break the copy into multiple copy operations when the copy size exceeds
  // the SDMA linear copy limit.
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                               kMaxSingleCopySize;
  const uint32_t num_copy_command = (size + max_copy_size - 1) / max_copy_size;

  // Avoid heap allocation for common single-packet case.
  SDMA_PKT_COPY_LINEAR stack_buff;
  std::vector<SDMA_PKT_COPY_LINEAR> heap_buff(num_copy_command > 1 ? num_copy_command : 0);
  auto* buff = num_copy_command <= 1 ? &stack_buff : heap_buff.data();

  BuildCopyCommand(reinterpret_cast<char*>(buff), num_copy_command, dst, src, size);

  return SubmitCommand(buff, num_copy_command * sizeof(SDMA_PKT_COPY_LINEAR), size, dep_signals,
                       out_signal, gang_signals);
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitLinearCopyBroadcastCommand(
    const std::vector<void*>& dsts, const void* src, size_t size,
    std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal) {

  if (!broadcast_supported_) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (dsts.empty() || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                                                              kMaxSingleCopySize;
  const uint32_t num_chunks = static_cast<uint32_t>((size + max_copy_size - 1) / max_copy_size);
  const uint64_t total_bytes_moved = static_cast<uint64_t>(size) * dsts.size();
  std::vector<core::Signal*> no_gang;

  constexpr size_t kMask = SDMA_PKT_COPY_LINEAR_BROADCAST::kDstAlignMask_;
  for (size_t i = 0; i + 1 < dsts.size(); i += 2) {
    if ((reinterpret_cast<uintptr_t>(dsts[i]) & kMask) !=
        (reinterpret_cast<uintptr_t>(dsts[i + 1]) & kMask))
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Each broadcast packet copies from one src to two dsts.
  // An odd trailing destination falls back to a regular linear copy.
  const uint32_t num_pairs = static_cast<uint32_t>(dsts.size() / 2);
  const bool has_remainder = (dsts.size() % 2) != 0;

  // Total command buffer: broadcast packets for each pair, plus linear packets
  // for the remainder destination, all multiplied by the number of size chunks.
  const size_t broadcast_bytes = num_pairs * num_chunks *
                                 static_cast<size_t>(broadcast_copy_command_size_);
  const size_t linear_bytes = has_remainder ?
                              (num_chunks * static_cast<size_t>(linear_copy_command_size_)) : 0;
  const size_t total_cmd_size = broadcast_bytes + linear_bytes;

  std::vector<char> cmd_buf(total_cmd_size, 0);
  char* cmd_ptr = cmd_buf.data();

  // Build broadcast packets for each destination pair.
  for (uint32_t p = 0; p < num_pairs; ++p) {
    BuildBroadcastCopyCommand(cmd_ptr, num_chunks,
                              dsts[p * 2], dsts[p * 2 + 1], src, size);
    cmd_ptr += num_chunks * broadcast_copy_command_size_;
  }

  // Handle the remaining odd destination with a regular linear copy.
  if (has_remainder) {
    BuildCopyCommand(cmd_ptr, num_chunks, dsts.back(),
                     src, size);
  }

  return SubmitCommand(cmd_buf.data(), total_cmd_size, total_bytes_moved,
                       dep_signals, out_signal, no_gang);
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitLinearCopyMulticastCommand(
    const std::vector<void*>& dsts, const void* src, size_t size,
    std::vector<core::Signal*>& dep_signals,
    core::Signal& out_signal,
    bool profiling_enabled) {

  if (!multicast_supported_)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (dsts.empty() || size == 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  // The multicast packet's num_of_destination is a 10-bit field encoded as
  // (count - 1), so at most 1024 destinations fit; reject more to avoid
  // truncating the field and corrupting the packet.
  constexpr size_t kMaxMulticastDsts = 1u << 10;  // 1024
  if (dsts.size() > kMaxMulticastDsts)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const uint32_t num_dsts = static_cast<uint32_t>(dsts.size());
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_
                                                            : kMaxSingleCopySize;
  const uint32_t num_chunks =
      static_cast<uint32_t>((size + max_copy_size - 1) / max_copy_size);
  const uint64_t total_bytes_moved = static_cast<uint64_t>(size) * num_dsts;

  // Profiling needs the classic prologue/copy/epilogue path to collect start/end
  // timestamps, so it uses the plain multicast packet via SubmitCommand.  When
  // profiling is off we fuse the GCR prologue, poll+copy+signal and the GCR
  // writeback + mailbox epilogue into a single reservation: dep_signals[0] folds
  // into the packet WAIT and out_signal is decremented by the packet SIGNAL on
  // the final chunk.
  if (profiling_enabled) {
    std::vector<core::Signal*> no_gang;
    // 5 fixed DWs (header, count, parameter, src_addr lo/hi) + 2 DWs per dst.
    const size_t pkt_bytes = (5 + 2 * static_cast<size_t>(num_dsts)) * sizeof(uint32_t);
    const size_t total_cmd_size = num_chunks * pkt_bytes;

    std::vector<char> cmd_buf(total_cmd_size, 0);
    BuildMulticastCopyCommand(cmd_buf.data(), num_chunks, dsts, src, size);

    return SubmitCommand(cmd_buf.data(), total_cmd_size, total_bytes_moved,
                         dep_signals, out_signal, no_gang);
  }

  const uint32_t num_copy_command = num_chunks;

  // Per-chunk copy core: count (1) + parameter (1) + src lo/hi (2) + N dst
  // lo/hi pairs (2N).  The 7-DW WAIT prefix is on the first chunk only and the
  // 5-DW SIGNAL suffix on the last (variable-length packet, see
  // BuildMulticastWaitSignalCopyCommand).
  // Option (b): every chunk carries wait (if there is a dep) + copy + signal.
  // The WAIT block is reserved only when there is a dep to fold in, matching the
  // builder's compaction; the SIGNAL block is always present.
  const uint32_t wait_dws = dep_signals.empty() ? 0 : 7;
  const uint32_t core_dws = 4 + 2 * num_dsts;
  const uint32_t per_pkt_dws = 1 + wait_dws + core_dws + 5;
  const uint32_t total_copy_dws = num_copy_command * per_pkt_dws;

  const uint32_t extra_polls = (dep_signals.size() > 1)
      ? static_cast<uint32_t>(dep_signals.size() - 1) : 0;
  const uint32_t extra_poll_bytes = extra_polls * poll_64b_command_size_;

  const uint32_t copy_bytes = total_copy_dws * sizeof(uint32_t);

  // Multicast is always issued standalone on a single engine, so the notify
  // prologue (GCR invalidate) and epilogue (poll out_signal, GCR writeback,
  // mailbox + trap) are folded into this same reservation.  Mirrors the linear
  // SubmitBodies fused path.
  const bool has_mailbox = (out_signal.signal_.event_mailbox_ptr != 0);
  const uint32_t prologue_bytes = useGCR ? gcr_command_size() : 0;
  uint32_t epilogue_bytes = 0;
  if (useGCR) epilogue_bytes += gcr_command_size();
  if (has_mailbox) epilogue_bytes += fence_command_size_ + trap_command_size_;

  const uint32_t total_command_size =
      prologue_bytes + extra_poll_bytes + copy_bytes + epilogue_bytes;

  const uint32_t pad_size = total_command_size < min_submission_size_
      ? min_submission_size_ - total_command_size
      : is_dxg_ ? AlignUp(total_command_size, 64) - total_command_size : 0;

  uint64_t curr_index;
  char* command_addr;
  uint64_t prior_bytes, post_bytes;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    command_addr = AcquireWriteAddress(total_command_size + pad_size, curr_index);
    if (command_addr == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    prior_bytes = bytes_queued_;
    bytes_queued_ += total_bytes_moved;
    post_bytes = bytes_queued_;
  }
  uint32_t wrapped_index = WrapIntoRing(curr_index);

  // Each of the N chunks decrements out_signal once (one signal per
  // multicast packet, independent of dst count), so pre-load it by N-1 before
  // the doorbell.
  if (num_copy_command > 1)
    out_signal.AddRelaxed(num_copy_command - 1);

  // Prologue: GCR invalidate before the copy.
  if (useGCR) {
    BuildGCRCommand(command_addr, true);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += gcr_command_size();
  }

  for (uint32_t i = 1; i < dep_signals.size(); ++i) {
    BuildPoll64bCommand(command_addr, dep_signals[i]->ValueLocation(), 0);
    command_addr += poll_64b_command_size_;
    bytes_written_[wrapped_index] = prior_bytes;
    wrapped_index += poll_64b_command_size_;
  }

  const core::Signal* wait_sig = dep_signals.empty() ? nullptr : dep_signals[0];
  BuildMulticastWaitSignalCopyCommand(command_addr, num_copy_command,
                                      dsts, src, size, wait_sig, &out_signal);
  bytes_written_.fill(wrapped_index, wrapped_index + copy_bytes, prior_bytes);
  bytes_written_[wrapped_index + copy_bytes - sizeof(uint32_t)] = post_bytes;
  command_addr += copy_bytes;
  wrapped_index += copy_bytes;

  // Epilogue: GCR writeback (if any), then notify KFD.  No self-poll on
  // out_signal: the ring is in-order and the multicast packet's inline SIGNAL
  // already executes after its copy, so the copy is drained by the time we get
  // here.  Polling out_signal — the very address the packet itself signals —
  // deadlocks the gfx1250 SDMA engine (the inline-signal write is not observed
  // by the immediately-following POLL_REGMEM), wedging every later packet.
  if (useGCR) {
    BuildGCRCommand(command_addr, false);
    command_addr += gcr_command_size();
    bytes_written_[wrapped_index] = post_bytes;
    wrapped_index += gcr_command_size();
  }

  if (has_mailbox) {
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

  if (pad_size) {
    memset(command_addr, 0, pad_size);
    uint32_t* dword_command_addr = reinterpret_cast<uint32_t*>(command_addr);
    dword_command_addr[0] = (pad_size / 4 - 1) << 16;
  }

  ReleaseWriteAddress(curr_index, total_command_size + pad_size);
  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitCopyRectCommand(
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

template <bool useGCR, bool scopeFields>
hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitLinearFillCommand(void* ptr, uint32_t value, size_t count) {
  const size_t size = count * sizeof(uint32_t);

  const uint32_t num_fill_command = (size + kMaxSingleFillSize - 1) / kMaxSingleFillSize;

  // Avoid heap allocation for common single-packet case.
  SDMA_PKT_CONSTANT_FILL stack_buff;
  std::vector<SDMA_PKT_CONSTANT_FILL> heap_buff(num_fill_command > 1 ? num_fill_command : 0);
  auto* buff = num_fill_command <= 1 ? &stack_buff : heap_buff.data();

  BuildFillCommand(reinterpret_cast<char*>(buff), num_fill_command, ptr, value, count);

  return SubmitBlockingCommand(buff, num_fill_command * sizeof(SDMA_PKT_CONSTANT_FILL), size);
}

template <bool useGCR, bool scopeFields> hsa_status_t BlitSdma<useGCR, scopeFields>::EnableProfiling(bool enable) {
  return HSA_STATUS_SUCCESS;
}

template <bool useGCR, bool scopeFields>
char* BlitSdma<useGCR, scopeFields>::AcquireWriteAddress(uint32_t cmd_size, uint64_t& curr_index) {
  // Ring is full when all but one byte is written.
  if (cmd_size >= kQueueSize) {
    return nullptr;
  }

  curr_index = atomic::Load(&cached_reserve_index_, std::memory_order_acquire);

  while (true) {
    // Check whether a linear region of the requested size is available.
    // If == cmd_size: region is at beginning of ring.
    // If < cmd_size: region intersects end of ring, pad with no-ops and retry.
    if (WrapIntoRing(curr_index + cmd_size) < cmd_size) {
      PadRingToEnd(curr_index);
      curr_index = atomic::Load(&cached_reserve_index_, std::memory_order_acquire);
      continue;
    }

    // Check whether the engine has finished using this region.
    const uint64_t new_index = curr_index + cmd_size;

    if (CanWriteUpto(new_index) == false) {
      // Wait for read index to move and try again.
      os::YieldThread();
      curr_index = atomic::Load(&cached_reserve_index_, std::memory_order_acquire);
      continue;
    }

    // Try to reserve this part of the ring.
    uint64_t observed = atomic::Cas(&cached_reserve_index_, new_index, curr_index,
                                    std::memory_order_release);
    if (observed == curr_index) {
      return queue_start_addr_ + WrapIntoRing(curr_index);
    }

    // CAS failed -- reuse the observed value directly, skip redundant atomic Load.
    curr_index = observed;
    _mm_pause();
  }

  return nullptr;
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::UpdateWriteAndDoorbellRegister(uint64_t curr_index, uint64_t new_index) {
  while (true) {
    // Make sure that the address before ::curr_index is already released.
    // Otherwise the CP may read invalid packets.
    uint64_t commit_index = atomic::Load(&cached_commit_index_, std::memory_order_acquire);
    if (commit_index == curr_index) {
      if (sdma_wait_idle_) {
        // TODO: remove when sdma wpointer issue is resolved.
        // Wait until the SDMA engine finish processing all packets before
        // updating the wptr and doorbell.
        while (WrapIntoRing(*queue_rptr_) != WrapIntoRing(curr_index)) {
          os::YieldThread();
        }
      }

      // Update write pointer and doorbell register.
      *queue_wptr_ = new_index;

      // Keep compiler ordering between wptr and doorbell writes. On x86 with
      // WB/coherent queue state, hardware ordering ensures the device observes
      // the wptr update before processing the doorbell.
      // this is ensured by release semantics
      // Atomic write to prevent TSAN race when multiple threads ring doorbell
      atomic::Store(queue_doorbell_, new_index, std::memory_order_release);
      if (needs_kmt_doorbell_) {
        HSAKMT_CALL(hsaKmtQueueRingDoorbell(queue_resource_.QueueId, new_index));
      }

      LogPrint(HSA_AMD_LOG_FLAG_SDMA,
               "SDMA ReleaseWrite: QueueId=%u, cmd_size=%lu, wptr=0x%lx, rptr=0x%lx",
               queue_resource_.QueueId, new_index - curr_index, new_index,
               *queue_rptr_);

      atomic::Store(&cached_commit_index_, new_index, std::memory_order_release);
      break;
    }

    // Waiting for another thread to submit preceding commands first.
    // Use mwaitx to efficiently monitor cached_commit_index_ instead of
    // burning CPU cycles.
    if (core::g_use_mwaitx) {
      timer::DoMwaitx(static_cast<int64_t*>(static_cast<void*>(&cached_commit_index_)),
                      static_cast<int64_t>(commit_index),
                      10000, true);
    } else {
      os::YieldThread();
    }
  }
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::ReleaseWriteAddress(uint64_t curr_index, uint32_t cmd_size) {
  if (cmd_size > kQueueSize) {
    assert(false && "cmd_addr is outside the queue buffer range");
    return;
  }

  UpdateWriteAndDoorbellRegister(curr_index, curr_index + cmd_size);
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::PadRingToEnd(uint64_t curr_index) {
  // Reserve region from here to the end of the ring.
  uint64_t new_index = curr_index + (kQueueSize - WrapIntoRing(curr_index));

  // Check whether the engine has finished using this region.
  if (CanWriteUpto(new_index) == false) {
    // Engine hasn't freed this region yet.  Pause briefly.
    _mm_pause();
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

template <bool useGCR, bool scopeFields> uint32_t BlitSdma<useGCR, scopeFields>::WrapIntoRing(uint64_t index) {
  return index & (kQueueSize - 1);
}

template <bool useGCR, bool scopeFields> bool BlitSdma<useGCR, scopeFields>::CanWriteUpto(uint64_t upto_index) {
  // Get/calculate the monotonic read index.
  uint64_t hw_read_index = *queue_rptr_;

  // Check whether the read pointer has passed the given index.
  // At most we can submit (kQueueSize - 1) bytes at a time.
  return (upto_index - hw_read_index) < kQueueSize;
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildFenceCommand(char* fence_command_addr, uint32_t* fence,
                                         uint32_t fence_value) {
  assert(fence_command_addr != NULL);

  // GFX12 or later use a different packet format that is incompatible (fields changed in size and location).
  if (agent_->supported_isas()[0]->GetMajorVersion() >= 12) {
    SDMA_PKT_FENCE_GFX12* packet_addr =
      reinterpret_cast<SDMA_PKT_FENCE_GFX12*>(fence_command_addr);

    memset(packet_addr, 0, sizeof(SDMA_PKT_FENCE_GFX12));

    packet_addr->HEADER_UNION.op = SDMA_OP_FENCE;
    packet_addr->HEADER_UNION.mtype = 3;

    /* We only use fence on signals and they are in system memory */
    packet_addr->HEADER_UNION.sys = 1;

    if (scopeFields)
      packet_addr->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

    packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(fence);
    packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence);

    packet_addr->DATA_UNION.data = fence_value;
  } else {
    SDMA_PKT_FENCE* packet_addr =
      reinterpret_cast<SDMA_PKT_FENCE*>(fence_command_addr);

    memset(packet_addr, 0, sizeof(SDMA_PKT_FENCE));

    packet_addr->HEADER_UNION.op = SDMA_OP_FENCE;

    if (agent_->supported_isas()[0]->GetMajorVersion() >= 10) {
      packet_addr->HEADER_UNION.mtype = 3;
    }

    packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(fence);
    packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence);

    packet_addr->DATA_UNION.data = fence_value;\
  }
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildCopyCommand(char* cmd_addr, uint32_t num_copy_command, void* dst,
                                        const void* src, size_t size) {
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

    if (scopeFields) packet_addr->HEADER_UNION.npd = 1;

    if (max_copy_size == max_single_linear_copy_size_)
      packet_addr->COUNT_UNION.count_ext.count = copy_size - 1; /* count is 1-based */
    else
      packet_addr->COUNT_UNION.count.count = copy_size - 1; /* count is 1-based */

    if (scopeFields) {
      packet_addr->PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
      packet_addr->PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
    }

    packet_addr->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
    packet_addr->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);

    packet_addr->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst);
    packet_addr->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst);

    cmd_addr += linear_copy_command_size_;
    cur_size += copy_size;
  }

  assert(cur_size == size);
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildBroadcastCopyCommand(char* cmd_addr, uint32_t num_copy_command,
                                                  void* dst1, void* dst2,
                                                  const void* src, size_t size) {
  [[maybe_unused]] constexpr size_t kMask = SDMA_PKT_COPY_LINEAR_BROADCAST::kDstAlignMask_;
  assert((reinterpret_cast<uintptr_t>(dst1) & kMask) ==
         (reinterpret_cast<uintptr_t>(dst2) & kMask));
  size_t cur_size = 0;
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                                                              kMaxSingleCopySize;
  for (uint32_t i = 0; i < num_copy_command; ++i) {
    const uint32_t copy_size =
        static_cast<uint32_t>(std::min((size - cur_size), max_copy_size));

    void* cur_dst1 = static_cast<char*>(dst1) + cur_size;
    void* cur_dst2 = static_cast<char*>(dst2) + cur_size;
    const void* cur_src = static_cast<const char*>(src) + cur_size;

    SDMA_PKT_COPY_LINEAR_BROADCAST* packet_addr =
        reinterpret_cast<SDMA_PKT_COPY_LINEAR_BROADCAST*>(cmd_addr);

    memset(packet_addr, 0, sizeof(SDMA_PKT_COPY_LINEAR_BROADCAST));

    packet_addr->HEADER_UNION.op = SDMA_OP_COPY;
    packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_BROADCAST;
    packet_addr->HEADER_UNION.broadcast = 1;

    if (max_copy_size == max_single_linear_copy_size_)
      packet_addr->COUNT_UNION.count_ext.count = copy_size - 1;
    else
      packet_addr->COUNT_UNION.count.count = copy_size - 1;

    packet_addr->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
    packet_addr->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);

    packet_addr->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst1);
    packet_addr->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst1);

    packet_addr->DST2_ADDR_LO_UNION.dst2_addr_31_0 = ptrlow32(cur_dst2);
    packet_addr->DST2_ADDR_HI_UNION.dst2_addr_63_32 = ptrhigh32(cur_dst2);

    cmd_addr += broadcast_copy_command_size_;
    cur_size += copy_size;
  }

  assert(cur_size == size);
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildMulticastCopyCommand(
    char* cmd_addr, uint32_t num_copy_command,
    const std::vector<void*>& dsts, const void* src, size_t size) {

  const uint32_t num_dsts = static_cast<uint32_t>(dsts.size());
  const size_t pkt_bytes = (5 + 2 * static_cast<size_t>(num_dsts)) * sizeof(uint32_t);
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_ :
                                                              kMaxSingleCopySize;
  size_t cur_size = 0;
  for (uint32_t i = 0; i < num_copy_command; ++i) {
    const uint32_t copy_size =
        static_cast<uint32_t>(std::min(size - cur_size, max_copy_size));

    memset(cmd_addr, 0, pkt_bytes);

    SDMA_PKT_COPY_LINEAR_MULTICAST_GFX1250* pkt =
        reinterpret_cast<SDMA_PKT_COPY_LINEAR_MULTICAST_GFX1250*>(cmd_addr);

    pkt->HEADER_UNION.op = SDMA_OP_COPY;
    pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;

    pkt->COUNT_UNION.count = copy_size - 1;

    pkt->PARAMETER_UNION.num_of_destination = num_dsts - 1;
    pkt->PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    pkt->PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;

    const void* cur_src = static_cast<const char*>(src) + cur_size;
    pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
    pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);

    uint32_t* dst_dw = reinterpret_cast<uint32_t*>(cmd_addr) + 5;
    for (uint32_t d = 0; d < num_dsts; ++d) {
      const void* cur_dst = static_cast<const char*>(dsts[d]) + cur_size;
      dst_dw[d * 2]     = ptrlow32(cur_dst);
      dst_dw[d * 2 + 1] = ptrhigh32(cur_dst);
    }

    cmd_addr += pkt_bytes;
    cur_size += copy_size;
  }

  assert(cur_size == size);
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildMulticastWaitSignalCopyCommand(
    char* cmd_addr, uint32_t num_copy_command,
    const std::vector<void*>& dsts, const void* src, size_t size,
    const core::Signal* wait_signal, core::Signal* signal_signal) {

  const uint32_t num_dsts = static_cast<uint32_t>(dsts.size());
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_
                                                            : kMaxSingleCopySize;
  size_t cur_size = 0;

  for (uint32_t i = 0; i < num_copy_command; ++i) {
    const uint32_t copy_size =
        static_cast<uint32_t>(std::min(size - cur_size, max_copy_size));

    // Option (b) chunk synchronization: every chunk waits on the same input
    // signal and decrements the same output signal, so completion is correct
    // even if the HW overlaps chunk copies (caller pre-loads out_signal by N-1).
    const bool do_wait = (wait_signal != nullptr);
    const bool do_signal = (signal_signal != nullptr);

    // Encode the fixed fields (header, optional wait, count, parameter, src and
    // optional signal) into a full-layout scratch struct.  The N destination
    // address pairs are written manually because the struct models only a
    // single destination: per the MAS extra destinations cascade as 2-DW pairs
    // and push the signal block out by 2*(N-1) DWs.  As with the other
    // wait/signal packets the buffer is variable length, so a chunk without
    // wait and/or signal omits those DW blocks entirely.
    SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX1250 pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.HEADER_UNION.op = SDMA_OP_COPY;
    pkt.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;
    pkt.HEADER_UNION.wait = do_wait ? 1 : 0;
    pkt.HEADER_UNION.signal = do_signal ? 1 : 0;

    if (do_wait) {
      pkt.WAIT_FUNCTION_UNION.wait_function = 0x3;  // Equal
      void* wait_addr = const_cast<core::Signal*>(wait_signal)->ValueLocation();
      pkt.WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
      pkt.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
      pkt.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 = 0;
      pkt.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 = 0;
      pkt.WAIT_MASK_LO_UNION.wait_mask_31_0 = 0xffffffff;
      pkt.WAIT_MASK_HI_UNION.wait_mask_63_32 = 0xffffffff;
    }

    pkt.COUNT_UNION.count = copy_size - 1;
    pkt.COPY_PARAMETER_UNION.num_of_destination = num_dsts - 1;
    pkt.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    pkt.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;

    const char* cur_src = reinterpret_cast<const char*>(src) + cur_size;
    pkt.SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
    pkt.SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);

    if (do_signal) {
      pkt.SIGNAL_OPERATION_UNION.signal_operation = 0x70;  // 64b sub
      pkt.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
      void* sig_addr = signal_signal->ValueLocation();
      pkt.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(sig_addr) >> 3;
      pkt.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(sig_addr);
      pkt.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
      pkt.SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
    }

    // Emit compacted: header (DW0), optional wait (scratch DW1-7), then
    // count+parameter+src (scratch DW8-11), then N dst pairs, then optional
    // signal (scratch DW14-18).  cmd_addr advances by exactly the DWs emitted.
    const uint32_t* s = reinterpret_cast<const uint32_t*>(&pkt);
    uint32_t* out_dw = reinterpret_cast<uint32_t*>(cmd_addr);
    uint32_t n = 0;
    out_dw[n++] = s[0];
    if (do_wait)
      for (uint32_t d = 1; d <= 7; ++d) out_dw[n++] = s[d];
    out_dw[n++] = s[8];   // count
    out_dw[n++] = s[9];   // parameter (num_of_destination + scopes)
    out_dw[n++] = s[10];  // src lo
    out_dw[n++] = s[11];  // src hi
    for (uint32_t j = 0; j < num_dsts; ++j) {
      const char* cur_dst = reinterpret_cast<const char*>(dsts[j]) + cur_size;
      out_dw[n++] = ptrlow32(cur_dst);
      out_dw[n++] = ptrhigh32(cur_dst);
    }
    if (do_signal)
      for (uint32_t d = 14; d <= 18; ++d) out_dw[n++] = s[d];

    cmd_addr += n * sizeof(uint32_t);
    cur_size += copy_size;
  }

  assert(cur_size == size);
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildSwapCopyCommand(char* cmd_addr, uint32_t num_copy_command,
                                            void* addr_a, void* addr_b, size_t size) {
  // gfx1250 uses a dedicated swap packet that carries per-operand cache scope
  // (the mechanism that replaces GCR cache invalidate/writeback); other archs
  // use the legacy swap packet. The two are the same size, so they share the
  // swap_copy_command_size_ stride and the classic body swap buffer.
  static_assert(sizeof(SDMA_PKT_COPY_LINEAR_SWAP_GFX1250) == sizeof(SDMA_PKT_COPY_LINEAR_SWAP),
                "gfx1250 swap packet must match legacy swap packet size for shared stride");
  const bool use_gfx1250 = scopeFields && is_gfx125plus_;

  const size_t kAlign = use_gfx1250 ? SDMA_PKT_COPY_LINEAR_SWAP_GFX1250::kAlignment_
                                    : SDMA_PKT_COPY_LINEAR_SWAP::kAlignment_;
  assert((reinterpret_cast<uintptr_t>(addr_a) & (kAlign - 1)) == 0);
  assert((reinterpret_cast<uintptr_t>(addr_b) & (kAlign - 1)) == 0);

  size_t cur_size = 0;
  const size_t max_copy_size = use_gfx1250 ? SDMA_PKT_COPY_LINEAR_SWAP_GFX1250::kMaxSize_
                                           : SDMA_PKT_COPY_LINEAR_SWAP::kMaxSize_;
  for (uint32_t i = 0; i < num_copy_command; ++i) {
    const uint32_t copy_size =
        static_cast<uint32_t>(std::min((size - cur_size), max_copy_size));

    void* cur_addr_a = static_cast<char*>(addr_a) + cur_size;
    void* cur_addr_b = static_cast<char*>(addr_b) + cur_size;

    if (use_gfx1250) {
      SDMA_PKT_COPY_LINEAR_SWAP_GFX1250* pkt =
          reinterpret_cast<SDMA_PKT_COPY_LINEAR_SWAP_GFX1250*>(cmd_addr);

      memset(pkt, 0, sizeof(SDMA_PKT_COPY_LINEAR_SWAP_GFX1250));

      pkt->HEADER_UNION.op = SDMA_OP_COPY;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;

      pkt->COUNT_UNION.count = copy_size - 1;

      pkt->PARAMETER_UNION.scope_a = SDMA_MEMORY_SCOPE_SYS;
      pkt->PARAMETER_UNION.scope_b = SDMA_MEMORY_SCOPE_SYS;

      pkt->ADDR_A_LO_UNION.DW_3_DATA = ptrlow32(cur_addr_a);
      pkt->ADDR_A_HI_UNION.addr_a_63_32 = ptrhigh32(cur_addr_a);

      pkt->ADDR_B_LO_UNION.DW_5_DATA = ptrlow32(cur_addr_b);
      pkt->ADDR_B_HI_UNION.addr_b_63_32 = ptrhigh32(cur_addr_b);
    } else {
      SDMA_PKT_COPY_LINEAR_SWAP* packet_addr =
          reinterpret_cast<SDMA_PKT_COPY_LINEAR_SWAP*>(cmd_addr);

      memset(packet_addr, 0, sizeof(SDMA_PKT_COPY_LINEAR_SWAP));

      packet_addr->HEADER_UNION.op = SDMA_OP_COPY;
      packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;

      packet_addr->COUNT_UNION.count = copy_size - 1;

      packet_addr->ADDR_A_LO_UNION.DW_3_DATA = ptrlow32(cur_addr_a);
      packet_addr->ADDR_A_HI_UNION.addr_a_63_32 = ptrhigh32(cur_addr_a);

      packet_addr->ADDR_B_LO_UNION.DW_5_DATA = ptrlow32(cur_addr_b);
      packet_addr->ADDR_B_HI_UNION.addr_b_63_32 = ptrhigh32(cur_addr_b);
    }

    cmd_addr += swap_copy_command_size_;
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
template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildCopyRectCommand(const std::function<void*(size_t)>& append,
                                            const hsa_pitched_ptr_t* dst,
                                            const hsa_dim3_t* dst_offset,
                                            const hsa_pitched_ptr_t* src,
                                            const hsa_dim3_t* src_offset, const hsa_dim3_t* range) {
  // Returns the index of the first set bit (ie log2 of the largest power of 2 that evenly divides
  // width), the largest element that perfectly covers width.
  // width | 16 ensures that we don't return a higher element than is supported and avoids
  // issues with 0.
  auto maxAlignedElement = [](size_t width) {
    return rocr::os::Ctz(width | 16);
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
  auto max_ele = Min(maxAlignedElement(src->pitch), maxAlignedElement(dst->pitch));
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
  auto min_ele = Min(max_ele, maxAlignedElement(range->x), maxAlignedElement(src_offset->x % 4),
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
        auto aligned_ele = Min(maxAlignedElement((src_offset->x + x) % 4),
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
          if (scopeFields) pkt->HEADER_UNION.npd = 1;
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
          pkt->RECT_PARAMETER_2_UNION.gfx12.rect_z = Min(range->z - z, max_z) - 1;
          if (scopeFields) {
            pkt->RECT_PARAMETER_2_UNION.gfx1250.dst_scope = SDMA_MEMORY_SCOPE_SYS;
            pkt->RECT_PARAMETER_2_UNION.gfx1250.src_scope = SDMA_MEMORY_SCOPE_SYS;
          }
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

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildFillCommand(char* cmd_addr, uint32_t num_fill_command,
                                                     void* ptr, uint32_t value, size_t count) {
  char* cur_ptr = reinterpret_cast<char*>(ptr);
  const uint32_t maxDwordCount = kMaxSingleFillSize / sizeof(uint32_t);
  SDMA_PKT_CONSTANT_FILL* packet_addr = reinterpret_cast<SDMA_PKT_CONSTANT_FILL*>(cmd_addr);

  for (uint32_t i = 0; i < num_fill_command; i++) {
    assert(count != 0 && "SDMA fill command count error.");
    const uint32_t fill_count = Min(count, size_t(maxDwordCount));

    memset(packet_addr, 0, sizeof(SDMA_PKT_CONSTANT_FILL));

    packet_addr->HEADER_UNION.op = SDMA_OP_CONST_FILL;
    if (scopeFields) {
      packet_addr->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
      packet_addr->HEADER_UNION.npd = 1;
    }
    packet_addr->HEADER_UNION.fillsize = 2;  // DW fill

    packet_addr->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_ptr);
    packet_addr->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_ptr);

    packet_addr->DATA_UNION.src_data_31_0 = value;

    /* count is 1-based */
    packet_addr->COUNT_UNION.count = (fill_count - 1) * sizeof(uint32_t);

    packet_addr++;
    cur_ptr += fill_count * sizeof(uint32_t);
    count -= fill_count;
  }
  assert(count == 0 && "SDMA fill command count error.");
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildPollCommand(char* cmd_addr, void* addr, uint32_t reference) {
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
  if (scopeFields) packet_addr->DW5_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildPoll64bCommand(char* cmd_addr, void* addr, uint64_t reference) {
  SDMA_PKT_POLL_MEM_64B_GFX1250* pkt =
      reinterpret_cast<SDMA_PKT_POLL_MEM_64B_GFX1250*>(cmd_addr);

  memset(pkt, 0, sizeof(SDMA_PKT_POLL_MEM_64B_GFX1250));

  pkt->HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_POLL_MEM_64B;
  pkt->HEADER_UNION.func = 0x3;  // Equal

  pkt->ADDR_LO_UNION.addr_31_3 = ptrlow32(addr) >> 3;
  pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);

  pkt->REFERENCE_LO_UNION.reference_31_0 = static_cast<uint32_t>(reference);
  pkt->REFERENCE_HI_UNION.reference_63_32 = static_cast<uint32_t>(reference >> 32);

  pkt->MASK_LO_UNION.mask_31_0 = 0xffffffff;
  pkt->MASK_HI_UNION.mask_63_32 = 0xffffffff;

  pkt->HEADER_UNION.sys = 1;  // Address is in system memory.
  pkt->DW7_UNION.retry_count = 0;  // Infinite retry

  if (scopeFields)
    pkt->DW7_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildFence64bCommand(char* cmd_addr, void* fence_addr,
                                                         uint64_t fence_value) {
  SDMA_PKT_FENCE_64B_GFX1250* pkt =
      reinterpret_cast<SDMA_PKT_FENCE_64B_GFX1250*>(cmd_addr);

  memset(pkt, 0, sizeof(SDMA_PKT_FENCE_64B_GFX1250));

  pkt->HEADER_UNION.op = SDMA_OP_FENCE;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_FENCE_64B;
  pkt->HEADER_UNION.mtype = 3;
  // Signal memory is in system memory.
  pkt->HEADER_UNION.sys = 1;

  if (scopeFields)
    pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  pkt->ADDR_LO_UNION.addr_31_3 = ptrlow32(fence_addr) >> 3;
  pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence_addr);

  pkt->DATA_LO_UNION.data_31_0 = static_cast<uint32_t>(fence_value);
  pkt->DATA_HI_UNION.data_63_32 = static_cast<uint32_t>(fence_value >> 32);
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildWaitSignalCopyCommand(
    char* cmd_addr, uint32_t num_copy_command,
    void* dst, const void* src, size_t size,
    const core::Signal* wait_signal,
    core::Signal* signal_signal) {

  size_t cur_size = 0;
  const size_t max_copy_size = max_single_linear_copy_size_ ? max_single_linear_copy_size_
                                                            : kMaxSingleCopySize;

  for (uint32_t i = 0; i < num_copy_command; ++i) {
    const uint32_t copy_size =
        static_cast<uint32_t>(std::min(size - cur_size, max_copy_size));

    // Option (b) chunk synchronization: every chunk waits on the same input
    // signal and decrements the same output signal.  This is correct even when
    // the HW overlaps chunk copies, since out_signal only reaches 0 once all
    // chunks have signalled (the caller pre-loads it by N-1).  A scheme that
    // signals only on the last chunk could complete early under overlap.
    const bool do_wait = (wait_signal != nullptr);
    const bool do_signal = (signal_signal != nullptr);

    // The WaitSignal packet is variable length: per the MAS, the 7 WAIT DWs
    // (DW1-7) and 5 SIGNAL DWs (DW14-18) are present in the buffer only when
    // their header bit is set ("should not appear" otherwise).  We encode the
    // packet into a full-layout scratch struct, then emit only the DW blocks
    // that are present so a compacted chunk (no wait) places the copy block
    // immediately after the header where the engine expects it.
    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX1250 pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.HEADER_UNION.op = SDMA_OP_COPY;
    pkt.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
    pkt.HEADER_UNION.wait = do_wait ? 1 : 0;
    pkt.HEADER_UNION.signal = do_signal ? 1 : 0;

    if (do_wait) {
      pkt.WAIT_FUNCTION_UNION.wait_function = 0x3;  // Equal
      void* wait_addr = const_cast<core::Signal*>(wait_signal)->ValueLocation();
      pkt.WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
      pkt.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
      pkt.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 = 0;
      pkt.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 = 0;
      pkt.WAIT_MASK_LO_UNION.wait_mask_31_0 = 0xffffffff;
      pkt.WAIT_MASK_HI_UNION.wait_mask_63_32 = 0xffffffff;
    }

    pkt.COPY_COUNT_UNION.copy_count = copy_size - 1;
    pkt.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    pkt.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;

    const char* cur_src = reinterpret_cast<const char*>(src) + cur_size;
    char* cur_dst = reinterpret_cast<char*>(dst) + cur_size;
    pkt.SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
    pkt.SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);
    pkt.DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst);
    pkt.DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst);

    if (do_signal) {
      pkt.SIGNAL_OPERATION_UNION.signal_operation = 0x70;  // 64b sub
      pkt.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
      void* sig_addr = signal_signal->ValueLocation();
      pkt.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(sig_addr) >> 3;
      pkt.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(sig_addr);
      pkt.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
      pkt.SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
    }

    // Emit compacted: header (DW0), optional wait (DW1-7), copy (DW8-13),
    // optional signal (DW14-18).  cmd_addr advances by exactly the DWs emitted.
    const uint32_t* pkt_dw = reinterpret_cast<const uint32_t*>(&pkt);
    uint32_t* out_dw = reinterpret_cast<uint32_t*>(cmd_addr);
    uint32_t n = 0;
    out_dw[n++] = pkt_dw[0];
    if (do_wait)
      for (uint32_t d = 1; d <= 7; ++d) out_dw[n++] = pkt_dw[d];
    for (uint32_t d = 8; d <= 13; ++d) out_dw[n++] = pkt_dw[d];
    if (do_signal)
      for (uint32_t d = 14; d <= 18; ++d) out_dw[n++] = pkt_dw[d];

    cmd_addr += n * sizeof(uint32_t);
    cur_size += copy_size;
  }
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildWaitSignalIndirectCopyCommand(
    char* cmd_addr,
    void* dst, const void* src, size_t size,
    bool indirect_src, bool indirect_dst,
    const core::Signal* wait_signal,
    core::Signal* signal_signal) {

  const bool do_wait = (wait_signal != nullptr);
  const bool do_signal = (signal_signal != nullptr);

  SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX1250* pkt =
      reinterpret_cast<SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX1250*>(cmd_addr);
  memset(pkt, 0, sizeof(SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX1250));

  pkt->HEADER_UNION.op = SDMA_OP_COPY;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_INDIRECT;
  pkt->HEADER_UNION.indirect_src = indirect_src ? 1 : 0;
  pkt->HEADER_UNION.indirect_dst = indirect_dst ? 1 : 0;
  pkt->HEADER_UNION.wait = do_wait ? 1 : 0;
  pkt->HEADER_UNION.signal = do_signal ? 1 : 0;

  if (do_wait) {
    pkt->WAIT_FUNCTION_UNION.wait_function = 0x3;  // Equal
    void* wait_addr = const_cast<core::Signal*>(wait_signal)->ValueLocation();
    pkt->WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
    pkt->WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
    pkt->WAIT_REFERENCE_LO_UNION.wait_reference_31_0 = 0;
    pkt->WAIT_REFERENCE_HI_UNION.wait_reference_63_32 = 0;
    pkt->WAIT_MASK_LO_UNION.wait_mask_31_0 = 0xffffffff;
    pkt->WAIT_MASK_HI_UNION.wait_mask_63_32 = 0xffffffff;
  }

  pkt->COPY_COUNT_UNION.copy_count = static_cast<uint32_t>(size) - 1;
  pkt->COPY_PARAMETER_UNION.copy_dst_scope = SDMA_MEMORY_SCOPE_SYS;
  pkt->COPY_PARAMETER_UNION.copy_src_scope = SDMA_MEMORY_SCOPE_SYS;
  pkt->COPY_PARAMETER_UNION.indirect_addr_scope = SDMA_MEMORY_SCOPE_SYS;

  pkt->SRC_ADDR_LO_UNION.copy_src_addr_31_0 = ptrlow32(src);
  pkt->SRC_ADDR_HI_UNION.copy_src_addr_63_32 = ptrhigh32(src);
  pkt->DST_ADDR_LO_UNION.copy_dst_addr_31_0 = ptrlow32(dst);
  pkt->DST_ADDR_HI_UNION.copy_dst_addr_63_32 = ptrhigh32(dst);

  if (do_signal) {
    pkt->SIGNAL_OPERATION_UNION.signal_operation = 0x70;  // 64b sub
    pkt->SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
    void* sig_addr = signal_signal->ValueLocation();
    pkt->SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(sig_addr) >> 3;
    pkt->SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(sig_addr);
    pkt->SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
    pkt->SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
  }
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildWaitSignalSwapCommand(
    char* cmd_addr, uint32_t num_copy_command,
    void* addr_a, void* addr_b, size_t size_a, size_t size_b,
    const core::Signal* wait_signal,
    core::Signal* signal_signal) {

  size_t cur_a = 0, cur_b = 0;
  const size_t max_copy_size = SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250::kMaxSize_;

  for (uint32_t i = 0; i < num_copy_command; ++i) {
    // Option (b) chunk synchronization: every chunk waits on the same input
    // signal and decrements the same output signal, so completion is correct
    // even if the HW overlaps chunk copies (caller pre-loads out_signal by N-1).
    const bool do_wait = (wait_signal != nullptr);
    const bool do_signal = (signal_signal != nullptr);

    const uint32_t chunk_a = static_cast<uint32_t>(std::min(size_a - cur_a, max_copy_size));
    const uint32_t chunk_b = static_cast<uint32_t>(std::min(size_b - cur_b, max_copy_size));

    const char* p_a = reinterpret_cast<const char*>(addr_a) + cur_a;
    const char* p_b = reinterpret_cast<const char*>(addr_b) + cur_b;

    // Variable-length packet (see BuildWaitSignalCopyCommand): encode into a
    // full-layout scratch struct, then emit only the DW blocks present so a
    // compacted chunk places the copy block right after the header.
    SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250 pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.HEADER_UNION.op     = SDMA_OP_COPY;
    pkt.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
    pkt.HEADER_UNION.wait   = do_wait   ? 1 : 0;
    pkt.HEADER_UNION.signal = do_signal ? 1 : 0;

    if (do_wait) {
      pkt.WAIT_FUNCTION_UNION.wait_function  = 0x3;  // Equal
      void* wa = const_cast<core::Signal*>(wait_signal)->ValueLocation();
      pkt.WAIT_ADDR_LO_UNION.wait_addr_31_3  = ptrlow32(wa) >> 3;
      pkt.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wa);
      pkt.WAIT_MASK_LO_UNION.wait_mask_31_0  = 0xffffffff;
      pkt.WAIT_MASK_HI_UNION.wait_mask_63_32 = 0xffffffff;
    }

    pkt.COUNT_UNION.count                    = chunk_a - 1;
    pkt.COPY_PARAMETER_UNION.scope_a         = SDMA_MEMORY_SCOPE_SYS;
    pkt.COPY_PARAMETER_UNION.scope_b         = SDMA_MEMORY_SCOPE_SYS;
    pkt.ADDR_A_LO_UNION.addr_a_31_0          = ptrlow32(p_a);
    pkt.ADDR_A_HI_UNION.addr_a_63_32         = ptrhigh32(p_a);
    pkt.ADDR_B_LO_UNION.addr_b_31_0          = ptrlow32(p_b);
    pkt.ADDR_B_HI_UNION.addr_b_63_32         = ptrhigh32(p_b);

    if (do_signal) {
      pkt.SIGNAL_OPERATION_UNION.signal_operation = 0x70;  // 64b sub
      pkt.SIGNAL_OPERATION_UNION.signal_scope     = SDMA_MEMORY_SCOPE_SYS;
      void* sa = signal_signal->ValueLocation();
      pkt.SIGNAL_ADDR_LO_UNION.signal_addr_31_3   = ptrlow32(sa) >> 3;
      pkt.SIGNAL_ADDR_HI_UNION.signal_addr_63_32  = ptrhigh32(sa);
      pkt.SIGNAL_DATA_LO_UNION.signal_data_31_0   = 1;
    }

    // Emit compacted: header (DW0), optional wait (DW1-7), copy (DW8-13),
    // optional signal (DW14-18).
    const uint32_t* pkt_dw = reinterpret_cast<const uint32_t*>(&pkt);
    uint32_t* out_dw = reinterpret_cast<uint32_t*>(cmd_addr);
    uint32_t n = 0;
    out_dw[n++] = pkt_dw[0];
    if (do_wait)
      for (uint32_t d = 1; d <= 7; ++d) out_dw[n++] = pkt_dw[d];
    for (uint32_t d = 8; d <= 13; ++d) out_dw[n++] = pkt_dw[d];
    if (do_signal)
      for (uint32_t d = 14; d <= 18; ++d) out_dw[n++] = pkt_dw[d];

    cmd_addr += n * sizeof(uint32_t);
    cur_a += chunk_a;
    cur_b += chunk_b;
  }
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildAtomicDecrementCommand(char* cmd_addr, void* addr) {
  SDMA_PKT_ATOMIC* packet_addr = reinterpret_cast<SDMA_PKT_ATOMIC*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_ATOMIC));

  packet_addr->HEADER_UNION.op = SDMA_OP_ATOMIC;
  packet_addr->HEADER_UNION.operation = SDMA_ATOMIC_ADD64;
  if (scopeFields) packet_addr->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(addr);
  packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);

  packet_addr->SRC_DATA_LO_UNION.src_data_31_0 = 0xffffffff;
  packet_addr->SRC_DATA_HI_UNION.src_data_63_32 = 0xffffffff;
}

template <bool useGCR, bool scopeFields>
void BlitSdma<useGCR, scopeFields>::BuildGetGlobalTimestampCommand(char* cmd_addr, void* write_address) {
  SDMA_PKT_TIMESTAMP* packet_addr =
      reinterpret_cast<SDMA_PKT_TIMESTAMP*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_TIMESTAMP));

  packet_addr->HEADER_UNION.op = SDMA_OP_TIMESTAMP;
  packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_TIMESTAMP_GET_GLOBAL;

  if (scopeFields) packet_addr->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(write_address);
  packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(write_address);
}

template <bool useGCR, bool scopeFields> void BlitSdma<useGCR, scopeFields>::BuildTrapCommand(char* cmd_addr, uint32_t event_id) {
  SDMA_PKT_TRAP* packet_addr =
      reinterpret_cast<SDMA_PKT_TRAP*>(cmd_addr);

  memset(packet_addr, 0, sizeof(SDMA_PKT_TRAP));

  packet_addr->HEADER_UNION.op = SDMA_OP_TRAP;
  packet_addr->INT_CONTEXT_UNION.int_ctx = event_id;
}

template <bool useGCR, bool scopeFields> void BlitSdma<useGCR, scopeFields>::BuildHdpFlushCommand(char* cmd_addr) {
  assert(cmd_addr != NULL);
  SDMA_PKT_POLL_REGMEM* addr = reinterpret_cast<SDMA_PKT_POLL_REGMEM*>(cmd_addr);
  memcpy(addr, &hdp_flush_cmd, flush_command_size_);
}

template <bool useGCR, bool scopeFields> void BlitSdma<useGCR, scopeFields>::BuildGCRCommand(char* cmd_addr, bool invalidate) {
  assert(cmd_addr != NULL);
  assert(useGCR && "Unsupported SDMA command - GCR.");

  if (is_gfx125plus_) {
    SDMA_PKT_GCR_GFX1250* addr = reinterpret_cast<SDMA_PKT_GCR_GFX1250*>(cmd_addr);
    memset(addr, 0, sizeof(SDMA_PKT_GCR_TAG_GFX1250));
    addr->HEADER_UNION.op = SDMA_OP_GCR;
    addr->HEADER_UNION.sub_op = SDMA_SUBOP_USER_GCR;
    if (invalidate) {
      addr->WORD3_UNION.GCR_CONTROL_GL2_SCOPE = 1; // system-scope
      addr->WORD3_UNION.GCR_CONTROL_GL2_INV = 1;
    } else {
      addr->WORD3_UNION.GCR_CONTROL_GL2_SCOPE = 1; // system-scope
      addr->WORD3_UNION.GCR_CONTROL_GL2_WB = 1;
    }
    // Discarding all lines for now.
    addr->WORD3_UNION.GCR_CONTROL_GL2_RANGE = 0;
  } else {
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
}

template <bool useGCR, bool scopeFields> uint64_t BlitSdma<useGCR, scopeFields>::PendingBytes() {
  uint64_t commit = atomic::Load(&cached_commit_index_, std::memory_order_acquire);
  uint64_t hw_read_index = *queue_rptr_;

  if (commit == hw_read_index) return 0;
  return bytes_queued_ - bytes_written_[WrapIntoRing(hw_read_index)];
}

template class BlitSdma<false, false>;  // BlitSdmaV4
template class BlitSdma<true, false>;   // BlitSdmaV5
template class BlitSdma<false, true>;   // BlitSdmaV6

}  // namespace amd
}  // namespace rocr
