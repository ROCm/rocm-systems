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

#include "core/inc/intercept_queue.h"
#include "core/inc/intercept_queue_logic.h"
#include "core/inc/amd_aql_queue.h"
#include "core/inc/default_signal.h"
#include "core/util/timer.h"
#include "core/util/utils.h"
#include "inc/hsa_api_trace.h"

namespace rocr {
namespace core {

namespace {

// Determine if a packet is the AMD_AQL_FORMAT_INTERCEPT_MARKER packet. Loads
// the packet header non-atomically. That is permissable if the calling thread
// has previously loaded the header atomically to determine if it is not an
// INVALID packet. Once a packet is no longer INVALID its ownership belongs to
// the packer processor.
bool inline IsInterceptMarkerPacket(const AqlPacket* packet) {
  return (AqlPacket::type(packet->packet.header) == HSA_PACKET_TYPE_VENDOR_SPECIFIC) &&
      (packet->amd_vendor.format == AMD_AQL_FORMAT_INTERCEPT_MARKER);
}

// Number of consecutive slots starting at `write` (at most `limit`) that the packet processor
// has retired, i.e. whose header reads INVALID. The HSA System Architecture specification
// (Queues) lets a producer write a slot only once read_index has passed it AND its format field
// reads INVALID. The processor is required to store INVALID before moving read_index and
// normally does, but on MI250 the order is rarely (about 1e-7 per packet) inverted: the INVALID
// store lands after read_index moved, and a header written in between is lost to it (see
// PlanSubmit). The processor retires slots in order, so a prefix count is enough; a new ring is
// filled with INVALID packets, so the first lap passes too.
uint64_t RetiredSlots(AqlPacket* ring, uint64_t mask, uint64_t write, uint64_t limit) {
  uint64_t n = 0;
  for (; n < limit; ++n) {
    const uint16_t header =
        atomic::Load(&ring[(write + n) & mask].packet.header, std::memory_order_relaxed);
    if (AqlPacket::type(header) != HSA_PACKET_TYPE_INVALID) break;
  }
  return n;
}

}  // namespace

struct InterceptFrame {
  InterceptQueue* queue;
  uint64_t pkt_index;
  size_t interceptor_index;
};

static thread_local InterceptFrame Cursor = {nullptr, 0, 0};

static const uint16_t kInvalidHeader = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierHeader = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

bool InterceptQueue::IsPendingRetryPoint() const {
  // Whether a retry barrier has been inserted but its completion has not yet been observed,
  // used to avoid putting multiple retry packets on the wrapped queue. The retry barrier
  // completes against a dedicated signal (retry_doorbell_) whose async handler clears
  // retry_outstanding_, so that flag is the authoritative source. The wrapped queue read
  // index is intentionally NOT consulted: it can advance past the retry packet before the
  // completion handler runs, which would let a second retry be inserted while the first is
  // still outstanding and let the delayed first handler then clear the newer retry's state.
  return retry_outstanding_.load(std::memory_order_acquire);
}

InterceptQueue::InterceptQueue(std::unique_ptr<Queue> queue)
    : QueueProxy(std::move(queue)),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      next_packet_(0),
      quit_(false),
      active_(true) {
  // retry_outstanding_ defaults to false, so IsPendingRetryPoint() is false before the
  // first retry barrier is inserted.
  assert(!IsPendingRetryPoint() &&
         "Packet intercept error: initial retry state is incompatible with IsPendingRetryPoint.\n");
  buffer_ = SharedArray<AqlPacket, 4096>(wrapped->amd_queue_.hsa_queue.size);
  amd_queue_.hsa_queue.base_address = reinterpret_cast<void*>(&buffer_[0]);

  // Pre-allocate staging buffer with queue size
  staging_buffer_.resize(wrapped->amd_queue_.hsa_queue.size);

  // Fill the ring buffer with invalid packet headers.
  // Leave packet content uninitialized to help trigger application errors.
  for (uint32_t pkt_id = 0; pkt_id < wrapped->amd_queue_.hsa_queue.size; ++pkt_id) {
    buffer_[pkt_id].packet.header = HSA_PACKET_TYPE_INVALID;
  }

  // Match the queue's signal ABI block to async_doorbell_'s
  // This allows us to use the queue's signal ABI block from devices to trigger async_doorbell while
  // host side use jumps directly to the queue's signal implementation.
  if (!core::g_use_interrupt_wait)
    async_doorbell_ = new DefaultSignal(DOORBELL_MAX);
  else
    async_doorbell_ = new InterruptSignal(DOORBELL_MAX);
  MAKE_NAMED_SCOPE_GUARD(sigGuard, [&]() { async_doorbell_->DestroySignal(); });
  this->signal_ = async_doorbell_->signal_;
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);

  // Install an async handler for device side dispatches.
  auto err = Runtime::runtime_singleton_->SetAsyncSignalHandler(
      core::Signal::Convert(async_doorbell_), HSA_SIGNAL_CONDITION_NE,
      async_doorbell_->LoadRelaxed(), HandleAsyncDoorbell, this);
  if (err != HSA_STATUS_SUCCESS)
    throw AMD::hsa_exception(err, "Doorbell handler registration failed.\n");

  // Dedicated retry-barrier completion signal (see header). DOORBELL_MAX init; a completing
  // retry barrier decrements it, tripping the NE async handler.
  if (!core::g_use_interrupt_wait)
    retry_doorbell_ = new DefaultSignal(DOORBELL_MAX);
  else
    retry_doorbell_ = new InterruptSignal(DOORBELL_MAX);
  MAKE_NAMED_SCOPE_GUARD(retrySigGuard, [&]() { retry_doorbell_->DestroySignal(); });

  err = Runtime::runtime_singleton_->SetAsyncSignalHandler(
      core::Signal::Convert(retry_doorbell_), HSA_SIGNAL_CONDITION_NE,
      retry_doorbell_->LoadRelaxed(), HandleRetryDoorbell, this);
  if (err != HSA_STATUS_SUCCESS)
    throw AMD::hsa_exception(err, "Retry doorbell handler registration failed.\n");

  // Install copy submission interceptor.
  AddInterceptor(Submit, this);

  sigGuard.Dismiss();
  retrySigGuard.Dismiss();
}

InterceptQueue::~InterceptQueue() {
  active_ = false;

  // Kill the async doorbell handler
  // Doorbell may not be used during or after queue destroy, however an interrupt may be in flight.
  // Ensure doorbell value is not 0, mark for exit, wake handler and wait for termination value.
  async_doorbell_->StoreRelaxed(DOORBELL_MAX);
  quit_ = true;
  hsa_signal_value_t val = async_doorbell_->ExchRelaxed(1);
  if (val != 0)
    async_doorbell_->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 0, -1, HSA_WAIT_STATE_BLOCKED);
  async_doorbell_->DestroySignal();

  // Kill the retry doorbell handler (quit_ already set), same teardown protocol
  // as async_doorbell_ above.
  retry_doorbell_->StoreRelaxed(DOORBELL_MAX);
  val = retry_doorbell_->ExchRelaxed(1);
  if (val != 0)
    retry_doorbell_->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 0, -1, HSA_WAIT_STATE_BLOCKED);
  retry_doorbell_->DestroySignal();
}

bool InterceptQueue::HandleAsyncDoorbell(hsa_signal_value_t value, void* arg) {
  InterceptQueue* queue = reinterpret_cast<InterceptQueue*>(arg);
  if (queue->quit_) {
    queue->async_doorbell_->StoreRelaxed(0);
    return false;
  }
  queue->async_doorbell_->StoreRelaxed(DOORBELL_MAX);
  queue->StoreRelease(value);
  return true;
}

bool InterceptQueue::HandleRetryDoorbell(hsa_signal_value_t value, void* arg) {
  InterceptQueue* queue = reinterpret_cast<InterceptQueue*>(arg);
  if (queue->quit_) {
    queue->retry_doorbell_->StoreRelaxed(0);
    return false;
  }
  queue->retry_doorbell_->StoreRelaxed(DOORBELL_MAX);
  // Tracked retry barrier completed. Clearing here (the only clear site) lets Submit()
  // insert a fresh barrier if overflow_ remains, keeping at most one outstanding.
  queue->retry_outstanding_.store(false, std::memory_order_release);
  queue->StoreRelease(value);
  return true;
}

void InterceptQueue::PacketWriter(const void* pkts, uint64_t pkt_count) {
  assert(Cursor.interceptor_index > 0 &&
         "Packet intercept error: final submit handler must not call PacketWritter.\n");
  --Cursor.interceptor_index;
  auto& handler = Cursor.queue->interceptors[Cursor.interceptor_index];
  handler.first(pkts, pkt_count, Cursor.pkt_index, handler.second, PacketWriter);
  // Restore index as the same rewrite handler may call the PacketWriter more than once.
  ++Cursor.interceptor_index;
}

void InterceptQueue::Submit(const void* pkts, uint64_t pkt_count, uint64_t user_pkt_index,
                            void* data, hsa_amd_queue_intercept_packet_writer writer) {
  InterceptQueue* queue = reinterpret_cast<InterceptQueue*>(data);
  const AqlPacket* packets = (const AqlPacket*)pkts;

  // Submit final packet transform to hardware.
  uint64_t submitted_count = queue->Submit(packets, pkt_count);
  if (submitted_count == pkt_count) return;

  // Could not submit all the final packets, stash unsubmitted ones for later.
  assert(queue->overflow_.empty() && "Packet intercept error: overflow buffer not empty.\n");
  for (uint64_t i = submitted_count; i < pkt_count; i++)
    queue->overflow_.push_back(packets[i]);
}

uint64_t InterceptQueue::Submit(const AqlPacket* packets, uint64_t count) {
  if (count == 0) return 0;

  uint64_t marker_count = 0;
  for (uint64_t i = 0; i < count; i++) {
    if (IsInterceptMarkerPacket(&packets[i])) ++marker_count;
  }

  AqlPacket* ring = reinterpret_cast<AqlPacket*>(wrapped->amd_queue_.hsa_queue.base_address);
  const uint64_t qsize = wrapped->amd_queue_.hsa_queue.size;
  uint64_t mask = qsize - 1;

  // Only a hardware packet processor retires slots by writing INVALID headers; a software queue
  // need not, and must not be gated on it.
  const bool processor_retires_slots = AMD::AqlQueue::IsType(wrapped.get());

  // When nothing fits and no retry barrier is outstanding, the packet processor is still
  // releasing slots (a retry barrier has just completed, or the ring was filled to its last
  // slot), so a short wait is almost always enough: on MI250 the read index trails the
  // processor's INVALID store by up to ~210 us. The wait is time-bounded because it holds lock_
  // on the doorbell or async events thread; when it runs out, StoreRelaxed() re-arms the drain
  // and the next round waits again, until the processor releases a slot.
  const timer::fast_clock::duration kMaxSlotWait = std::chrono::microseconds(500);
  timer::fast_clock::time_point wait_start;
  bool waiting = false;

  while (true) {
    uint64_t write = wrapped->LoadWriteIndexRelaxed();
    uint64_t read = wrapped->LoadReadIndexRelaxed();
    bool pending_retry_point = IsPendingRetryPoint();

    // At most one slot per non-marker packet plus one for a retry barrier can be used this
    // iteration, and the ring has only qsize of them.
    uint64_t limit = count - marker_count + 1;
    if (limit > qsize) limit = qsize;
    const uint64_t retired_slots =
        processor_retires_slots ? RetiredSlots(ring, mask, write, limit) : UINT64_MAX;

    // Pure slot-accounting decision (overflow/underflow hardening, unit tested). The
    // non-atomic write/read snapshot (transient read > write) is handled inside PlanSubmit.
    const auto plan = intercept_queue_logic::PlanSubmit(write, read, qsize, count, marker_count,
                                                        pending_retry_point, !overflow_.empty(),
                                                        retired_slots);
    if (plan.needs_wakeup) {
      const timer::fast_clock::time_point now = timer::fast_clock::now();
      if (!waiting) {
        waiting = true;
        wait_start = now;
      }
      if (now - wait_start < kMaxSlotWait) {
        os::YieldThread();
        continue;
      }
    }
    uint64_t submitted_count = plan.submitted_count;

    // If packets remain unsubmitted, insert a retry barrier to drain them later
    // (PlanSubmit already required a free slot for it).
    if (plan.insert_retry_barrier) {
      uint64_t barrier = wrapped->AddWriteIndexRelaxed(1);
      assert(barrier == write &&
             "Packet intercept error: wrapped queue has been updated by another thread.\n");
      ++write;

      // Mark the retry outstanding BEFORE publishing the packet and ringing the doorbell.
      // The GPU may complete the barrier immediately; HandleRetryDoorbell() must then observe
      // retry_outstanding_ == true so it clears it (otherwise the flag is left stuck true and
      // overflow_ is stranded).
      retry_outstanding_.store(true, std::memory_order_release);

      // Barrier wakes async queue processing; completion signal is the dedicated
      // retry_doorbell_ so HandleRetryDoorbell() tracks it unambiguously.
      ring[barrier & mask].packet.body = {};
      ring[barrier & mask].barrier_and.completion_signal = Signal::Convert(retry_doorbell_);
      if (wrapped->IsDeviceMemRingBuf() && needsPcieOrdering()) {
        // Ensure the packet body is written as header may get reordered when writing over PCIE
        store_fence();
      }
      // Release-publish the header, then ring the doorbell.
      atomic::Store(&ring[barrier & mask].barrier_and.header, kBarrierHeader,
                    std::memory_order_release);
      HSA::hsa_signal_store_screlease(wrapped->amd_queue_.hsa_queue.doorbell_signal, barrier);
    }

    // Attempt to reserve useable queue space if some packets need to be
    // submitted.
    uint64_t new_write = submitted_count == 0
        ? write
        : wrapped->CasWriteIndexRelaxed(write, write + submitted_count);
    if (new_write == write) {
      uint64_t packets_index = 0;
      uint64_t write_index = 0;
      uint64_t first_written_packet_index;
      while (submitted_count > 0 || (packets_index < count && IsInterceptMarkerPacket(&packets[packets_index]))) {
        // Ensure the marker packet callback is invoked before following
        // packets are made available for the packet processor.
        if (IsInterceptMarkerPacket(&packets[packets_index])) {
          const amd_aql_intercept_marker_t* marker_packet =
              reinterpret_cast<const amd_aql_intercept_marker_t*>(&packets[packets_index]);
          marker_packet->callback(marker_packet, &wrapped->amd_queue_.hsa_queue,
                                  write + write_index);
        } else {
          if (write_index == 0) {
            // Leave the header of the first packet as INVALID so packet
            // processor will not start processing any packets until all have
            // been written and the first packet header atomically store
            // released.
            ring[(write + write_index) & mask].packet.body = packets[packets_index].packet.body;
            first_written_packet_index = packets_index;
          } else {
            ring[(write + write_index) & mask] = packets[packets_index];
          }
          ++write_index;
          --submitted_count;
        }
        ++packets_index;
      }
      if (write_index != 0) {
        if (wrapped->IsDeviceMemRingBuf() && needsPcieOrdering()) {
          // Ensure the packet body is written as header may get reordered when writing over PCIE
          store_fence();
        }
        atomic::Store(&ring[write & mask].packet.header, packets[first_written_packet_index].packet.header,
                      std::memory_order_release);
        HSA::hsa_signal_store_screlease(wrapped->amd_queue_.hsa_queue.doorbell_signal,
                                        write + write_index - 1);
      }
      return packets_index;
    }
  }
}

void InterceptQueue::StoreRelaxed(hsa_signal_value_t value) {
  if (!active_) return;

  // If called recursively defer to async doorbell thread.
  if (Cursor.queue != nullptr) {
    debug_print("Likely incorrect queue use observed in an interceptor.\n");
    async_doorbell_->StoreRelaxed(value);
    return;
  }

  std::lock_guard<std::mutex> lock(lock_);

  // Submit overflow packets.
  if (!overflow_.empty()) {
    uint64_t submitted_count = Submit(&overflow_[0], overflow_.size());

    if (submitted_count < overflow_.size()) {
      overflow_.erase(overflow_.begin(), overflow_.begin() + submitted_count);
      // Since there was no space to submit all the overflow packets, there is
      // no space for other packets either.
      EnsureDrainWakeup();
      return;
    }

    // All overflow packets have been submitted.
    overflow_.clear();
  }

  Cursor.queue = this;

  AqlPacket* ring = reinterpret_cast<AqlPacket*>(amd_queue_.hsa_queue.base_address);
  uint64_t mask = wrapped->amd_queue_.hsa_queue.size - 1;

  // Loop over valid packets and process.
  uint64_t end = LoadWriteIndexAcquire();

  // Can only process packets that are occupying slots in the queue buffer. No
  // need to add a barrier packet to ensure the extra packets are processed as
  // the producer must ring the doorbell once the extra packets are made valid.
  if (end > next_packet_ + amd_queue_.hsa_queue.size)
    end = next_packet_ + amd_queue_.hsa_queue.size;

  uint64_t i = next_packet_;
  uint64_t invalid_header_i = end;

  while (i < end) {
    // Load the packet header as atomic acquire as it may have been written by
    // another thread as atomic release. This ensures the rest of the packet
    // fields are visible. Once loaded and proven not to be INVALID, further
    // loads by this thread can be non-atomic.
    uint16_t header = atomic::Load(&ring[i & mask].packet.header, std::memory_order_acquire);
    if (!AqlPacket::IsValid(header)) {
      invalid_header_i = i;
      break;
    }
    ++i;

    // Only allow the rewrite of one packet to be on the overflow queue. When
    // packets are put on the overflow queue a barrier packet will also be
    // added which has an async handler that will ring the doorbell, That
    // doorbell ring will ensure this function is re-invoked to put the
    // overflow packets on the hardware queue and continue rewriting packets on
    // the intercept queue.
    if (!overflow_.empty()) break;
  }

  // Process callbacks.
  uint64_t packet_count = i - next_packet_;
  if (packet_count) {
    Cursor.interceptor_index = interceptors.size() - 1;
    Cursor.pkt_index = next_packet_;
    auto& handler = interceptors[Cursor.interceptor_index];

    // Check if packets wrap around the ring buffer boundary using unmasked indices.
    // The interceptor callback expects packets to be contiguous in memory.
    if ((next_packet_ + packet_count) > ((next_packet_ & ~mask) + amd_queue_.hsa_queue.size)) {
      // Packets wrap around - use pre-allocated staging buffer
      for (uint64_t j = 0; j < packet_count; ++j) {
        staging_buffer_[j] = ring[(next_packet_ + j) & mask];
      }
      handler.first(staging_buffer_.data(), packet_count, next_packet_,
                    handler.second, PacketWriter);
    } else {
      // Packets are contiguous in the ring buffer
      handler.first(&ring[next_packet_ & mask], packet_count, next_packet_,
                                                 handler.second, PacketWriter);
    }

    if (IsDeviceMemRingBuf() && needsPcieOrdering()) {
      // Ensure the packet body is written as header may get reordered when writing over PCIE
      store_fence();
    }
  }
  i = next_packet_;
  while (i < std::min(end, invalid_header_i)) {
    // Invalidate consumed packets.
    atomic::Store(&ring[i & mask].packet.header, kInvalidHeader, std::memory_order_release);
    // Packet has now been processed so advance the read index.
    ++i;
  }

  next_packet_ = i;
  Cursor.queue = nullptr;
  atomic::Store(&amd_queue_.read_dispatch_id, next_packet_, std::memory_order_release);

  // The final rewriter may have left packets in overflow_.
  if (!overflow_.empty()) EnsureDrainWakeup();
}

void InterceptQueue::EnsureDrainWakeup() {
  // Called with lock_ held and overflow_ non-empty. A pending retry barrier wakes the drain
  // through HandleRetryDoorbell() when the processor reaches it. Without one nothing on the
  // device refers to this queue any more, and the producer may already be blocked on the full
  // proxy ring, so the drain would never run again: re-run it from the async doorbell thread.
  // HandleAsyncDoorbell() resets the signal, so any other value trips its NE condition; the
  // teardown handshake is keyed on quit_, not on the value, so next_packet_ (>= 1 here) cannot be
  // mistaken for it.
  if (IsPendingRetryPoint() || quit_) return;
  async_doorbell_->StoreRelaxed(next_packet_);
}

hsa_status_t InterceptQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID: 
    case HSA_QUEUE_INFO_USE_COUNT:
    case HSA_QUEUE_INFO_HW_ID:
    case HSA_AMD_QUEUE_INFO_ENGINE_TYPE:
    case HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID: {
      if (!AMD::AqlQueue::IsType(wrapped.get())) return HSA_STATUS_ERROR_INVALID_QUEUE;

      AMD::AqlQueue* aqlQueue = static_cast<AMD::AqlQueue*>(wrapped.get());
      return aqlQueue->GetInfo(attribute, value);
    }
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

}  // namespace core
}  // namespace rocr
