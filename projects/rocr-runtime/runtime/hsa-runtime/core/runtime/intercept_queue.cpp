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
#include "core/inc/amd_aql_queue.h"
#include "core/inc/default_signal.h"
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

bool InterceptQueue::IsPendingRetryPoint(uint64_t wrapped_current_read_index) const {
  // This function is intended to determine if the last retry barrier packet
  // has definitely not been processed in order to avoid putting multiple retry
  // packets on the wrapped queue.
  //
  // The AQL protocol allows the packet processor to advance the read index any
  // time after the producer advances the write index. It does not specify the
  // latest that the read index must be advanced. This makes it impossible to
  // use the read index to determine if a packet has definitely not been
  // processed.
  //
  // This code assumes that the read index will be advanced no later than the
  // start of processing the next packet. So at worst, if the read index equals
  // the retry index the packet may have already been processed, and its
  // completion signal updated (perhaps that was the cause of entering
  // InterceptQueue::StoreRelaxed that is now invoking this function). But if
  // the read index is less than the retry index, then the packet has not yet
  // been processed, This implies that the minimum queue size is 3 (enforced in
  // hsa_amd_queue_intercept_create): a non-retry packet, a retry packet that
  // is being processed, and space for a new retry packet.
  //
  // FIXME: The above assumption can be removed by using a distinct interrupt
  // signal for the retry packet completion signal, and tracking when that
  // signal is updated and invokes its async handler. Currently the wrapped
  // queue doorbell signal is also being used as the retry completion signal.
  // If that is done then the minimum queue size needs to be changed from 3 to
  // 2 (enforced in hsa_amd_queue_intercept_create).
  return retry_index_ > wrapped_current_read_index;
}

InterceptQueue::InterceptQueue(std::unique_ptr<Queue> queue)
    : QueueProxy(std::move(queue)),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      next_packet_(0),
      retry_index_(0),
      quit_(false),
      active_(true) {
  // Initial retry_index_ value must ensure that
  // InterceptQueue::IsPendingRetryPoint will return false before the first
  // retry barrier packet is inserted.
  assert(!IsPendingRetryPoint(next_packet_) &&
         "Packet intercept error: initial retry index is incompatible with IsPendingRetryPoint.\n");
  buffer_ = SharedArray<AqlPacket, 4096>(wrapped->amd_queue_.hsa_queue.size);
  amd_queue_.hsa_queue.base_address = reinterpret_cast<void*>(&buffer_[0]);

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

  // Install copy submission interceptor.
  AddInterceptor(Submit, this);

  sigGuard.Dismiss();
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
  uint64_t mask = wrapped->amd_queue_.hsa_queue.size - 1;

  while (true) {
    uint64_t write = wrapped->LoadWriteIndexRelaxed();
    uint64_t read = wrapped->LoadReadIndexRelaxed();
    uint64_t free_slots = wrapped->amd_queue_.hsa_queue.size - (write - read);
    bool pending_retry_point = IsPendingRetryPoint(read);

    uint64_t submitted_count = count - marker_count;

    // If the number of packets is greater than the wrapped queue size, then we
    // can never submit them all at once. So submit what will fit, leaving one
    // slot free for the retry barrier packet if it is not already on the
    // queue.
    if (submitted_count >= wrapped->amd_queue_.hsa_queue.size) {
      submitted_count = free_slots - (pending_retry_point ? 0 : 1);
    }

    // Prefer to either submit all the packets, or none of the packets. This
    // ensures that all the packets of a rewrite will be on the queue at the
    // same time. This may be desirable for some rewrites. So if out of space
    // defer packet insertion. Always make sure there is a free slot available
    // for the retry barrier packet if there is not already one present.
    else if (free_slots < submitted_count + (pending_retry_point ? 0 : 1)) {
      submitted_count = 0;
    }

    // If we are not submitting all the packets, we need to ensure there is a
    // retry packet to cause the remaining packets to be submitted. If there is
    // not already a pending retry point add one.
    if (submitted_count < (count - marker_count) && !pending_retry_point) {
      // Reserve one slot for the barrier packet. There will always be at least
      // one free slot.
      assert(free_slots >= 1 &&
             "Packet intercept error: there is no free slot for a retry barrier packet.\n");
      // Reserve a slot for the barrier packet.
      uint64_t barrier = wrapped->AddWriteIndexRelaxed(1);
      assert(barrier == write &&
             "Packet intercept error: wrapped queue has been updated by another thread.\n");
      ++write;

      // Submit barrier which will wake async queue processing.
      ring[barrier & mask].packet.body = {};
      ring[barrier & mask].barrier_and.completion_signal = Signal::Convert(async_doorbell_);
      if (wrapped->IsDeviceMemRingBuf() && needsPcieOrdering()) {
        // Ensure the packet body is written as header may get reordered when writing over PCIE
        _mm_sfence();
      }
      atomic::Store(&ring[barrier & mask].barrier_and.header, kBarrierHeader,
                    std::memory_order_release);
      // Update the wrapped queue's doorbell so it knows there is a new packet in the queue.
      HSA::hsa_signal_store_screlease(wrapped->amd_queue_.hsa_queue.doorbell_signal, barrier);

      // Record the retry point
      retry_index_ = barrier;
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
          _mm_sfence();
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

  ScopedAcquire<KernelMutex> lock(&lock_);

  // Submit overflow packets.
  if (!overflow_.empty()) {
    uint64_t submitted_count = Submit(&overflow_[0], overflow_.size());

    if (submitted_count < overflow_.size()) {
      overflow_.erase(overflow_.begin(), overflow_.begin() + submitted_count);
      // Since there was no space to submit all the overflow packets, there is
      // no space for other packets either.
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
  while (i < end) {
    // Load the packet header as atomic acquire as it may have been written by
    // another thread as atomic release. This ensures the rest of the packet
    // fields are visible. Once loaded and proven not to be INVALID, further
    // loads by this thread can be non-atomic.
    uint16_t header = atomic::Load(&ring[i & mask].packet.header, std::memory_order_acquire);
    if (!AqlPacket::IsValid(header)) break;

    // Process callbacks.
    Cursor.interceptor_index = interceptors.size() - 1;
    Cursor.pkt_index = i;
    auto& handler = interceptors[Cursor.interceptor_index];
    handler.first(&ring[i & mask], 1, i, handler.second, PacketWriter);
    if (IsDeviceMemRingBuf() && needsPcieOrdering()) {
      // Ensure the packet body is written as header may get reordered when writing over PCIE
      _mm_sfence();
    }
    // Invalidate consumed packet.
    atomic::Store(&ring[i & mask].packet.header, kInvalidHeader, std::memory_order_release);

    // Packet has now been processed so advance the read index.
    ++i;

    // Only allow the rewrite of one packet to be on the overflow queue. When
    // packets are put on the overflow queue a barrier packet will also be
    // added which has an async handler that will ring the doorbell, That
    // doorbell ring will ensure this function is re-invoked to put the
    // overflow packets on the hardware queue and continue rewriting packets on
    // the intercept queue.
    if (!overflow_.empty()) break;
  }

  next_packet_ = i;
  Cursor.queue = nullptr;
  atomic::Store(&amd_queue_.read_dispatch_id, next_packet_, std::memory_order_release);
}

hsa_status_t InterceptQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID: {
      if (!AMD::AqlQueue::IsType(wrapped.get())) return HSA_STATUS_ERROR_INVALID_QUEUE;

      AMD::AqlQueue* aqlQueue = static_cast<AMD::AqlQueue*>(wrapped.get());
      return aqlQueue->GetInfo(attribute, value);
    }
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

}  // namespace core
}  // namespace rocr
