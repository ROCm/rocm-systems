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
#include "inc/amd_hsa_signal.h"
#include "inc/hsa_api_trace.h"

namespace rocr {
namespace core {

namespace {

using MetadataPacket = AqlMetadataPrefetchPacket;

constexpr uint16_t kMutableAqlHeaderFlags = ((1u << HSA_PACKET_HEADER_WIDTH_BARRIER) - 1)
        << HSA_PACKET_HEADER_BARRIER |
    ((1u << HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE) - 1)
        << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
    ((1u << HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE) - 1)
        << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

// Determine if a packet is the AMD_AQL_FORMAT_INTERCEPT_MARKER packet. Loads
// the packet header non-atomically. That is permissable if the calling thread
// has previously loaded the header atomically to determine if it is not an
// INVALID packet. Once a packet is no longer INVALID its ownership belongs to
// the packer processor.
bool inline IsInterceptMarkerPacket(const AqlPacket* packet) {
  return (AqlPacket::type(packet->packet.header) == HSA_PACKET_TYPE_VENDOR_SPECIFIC) &&
      (packet->amd_vendor.format == AMD_AQL_FORMAT_INTERCEPT_MARKER);
}

void FillNoopMetadata(MetadataPacket& metadata) {
  metadata = {};
  metadata.packet.header0.type = HSA_PACKET_TYPE_INVALID;
  metadata.packet.header1.type = HSA_PACKET_TYPE_INVALID;
  metadata.packet.header2.type = HSA_PACKET_TYPE_INVALID;
  metadata.packet.header3.type = HSA_PACKET_TYPE_INVALID;
}

void WriteMetadataSlot(void* metadata_ring, uint64_t slot_index, uint64_t mask,
                       const MetadataPacket& metadata) {
  reinterpret_cast<MetadataPacket*>(metadata_ring)[slot_index & mask] = metadata;
}

bool MaskedPacketMatch(const AqlPacket& a, const AqlPacket& b) {
  return AqlPacket::type(a.packet.header) == AqlPacket::type(b.packet.header) &&
      (a.packet.header & ~kMutableAqlHeaderFlags) == (b.packet.header & ~kMutableAqlHeaderFlags) &&
      memcmp(reinterpret_cast<const uint8_t*>(&a) + 2, reinterpret_cast<const uint8_t*>(&b) + 2,
             54) == 0;
}

bool CompletionSignalMatch(const AqlPacket& a, const AqlPacket& b) {
  constexpr size_t kCompletionSignalOffset = sizeof(AqlPacket) - sizeof(hsa_signal_t);
  return memcmp(reinterpret_cast<const uint8_t*>(&a) + kCompletionSignalOffset,
                reinterpret_cast<const uint8_t*>(&b) + kCompletionSignalOffset,
                sizeof(hsa_signal_t)) == 0;
}

uint32_t GetSignalEventId(hsa_signal_t signal) {
  if (signal.handle == 0) return 0;
  return reinterpret_cast<const amd_signal_t*>(signal.handle)->event_id;
}

void UpdateMetadataEventId(const AqlPacket& packet, MetadataPacket& metadata) {
  switch (AqlPacket::type(packet.packet.header)) {
    case HSA_PACKET_TYPE_KERNEL_DISPATCH:
      metadata.packet.event_id = GetSignalEventId(packet.dispatch.completion_signal);
      break;
    case HSA_PACKET_TYPE_BARRIER_AND:
      metadata.packet.event_id = GetSignalEventId(packet.barrier_and.completion_signal);
      break;
    case HSA_PACKET_TYPE_BARRIER_OR:
      metadata.packet.event_id = GetSignalEventId(packet.barrier_or.completion_signal);
      break;
    case HSA_PACKET_TYPE_VENDOR_SPECIFIC:
      if (packet.amd_vendor.format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
        metadata.packet.event_id = GetSignalEventId(packet.ext_dispatch.completion_signal);
      } else if (packet.amd_vendor.format == HSA_AMD_PACKET_TYPE_BARRIER_VALUE) {
        const auto* barrier = reinterpret_cast<const hsa_amd_barrier_value_packet_t*>(&packet);
        metadata.packet.event_id = GetSignalEventId(barrier->completion_signal);
      }
      break;
    default:
      break;
  }
}

template <typename Overflow>
void AppendOverflowPackets(Overflow& overflow, const AqlPacket* packets, uint64_t begin,
                           uint64_t end, const MetadataPacket* metadata) {
  for (uint64_t i = begin; i < end; ++i) {
    typename Overflow::value_type packet{};
    packet.aql = packets[i];
    if (metadata)
      packet.metadata = metadata[i];
    else
      FillNoopMetadata(packet.metadata);
    overflow.push_back(packet);
  }
}

template <typename Overflow>
void CopyOverflowToScratch(const Overflow& overflow, std::vector<AqlPacket>& aql,
                           std::vector<MetadataPacket>& metadata) {
  aql.resize(overflow.size());
  metadata.resize(overflow.size());
  for (uint64_t i = 0; i < overflow.size(); ++i) {
    aql[i] = overflow[i].aql;
    metadata[i] = overflow[i].metadata;
  }
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

void InterceptQueue::Initialize() {
  // Initial retry_index_ value must ensure that
  // InterceptQueue::IsPendingRetryPoint will return false before the first
  // retry barrier packet is inserted.
  assert(!IsPendingRetryPoint(next_packet_) &&
         "Packet intercept error: initial retry index is incompatible with IsPendingRetryPoint.\n");
  buffer_ = SharedArray<AqlPacket, 4096>(wrapped->amd_queue_.hsa_queue.size);
  amd_queue_.hsa_queue.base_address = reinterpret_cast<void*>(&buffer_[0]);

  // Pre-allocate staging buffer with queue size
  staging_buffer_.resize(wrapped->amd_queue_.hsa_queue.size);
  aql_scratch_.resize(wrapped->amd_queue_.hsa_queue.size);
  metadata_scratch_.resize(wrapped->amd_queue_.hsa_queue.size);

  // Fill the ring buffer with invalid packet headers.
  // Leave packet content uninitialized to help trigger application errors.
  for (uint32_t pkt_id = 0; pkt_id < wrapped->amd_queue_.hsa_queue.size; ++pkt_id) {
    buffer_[pkt_id].packet.header = HSA_PACKET_TYPE_INVALID;
  }

  uint64_t metadata_ring = 0;
  if (wrapped->GetInfo(HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER, &metadata_ring) ==
          HSA_STATUS_SUCCESS &&
      metadata_ring != 0) {
    metadata_ring_buf_ = reinterpret_cast<void*>(metadata_ring);
    proxy_metadata_buf_.resize(wrapped->amd_queue_.hsa_queue.size);
    for (auto& metadata : proxy_metadata_buf_) FillNoopMetadata(metadata);
  }

  // Install copy submission interceptor.
  AddInterceptor(Submit, this);
}

#ifndef ROCR_INTERCEPT_QUEUE_TESTING
InterceptQueue::InterceptQueue(std::unique_ptr<Queue> queue)
    : QueueProxy(std::move(queue)),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      next_packet_(0),
      retry_index_(0),
      async_doorbell_(nullptr),
      quit_(false),
      active_(true) {
  Initialize();

  // Match the queue's signal ABI block to async_doorbell_'s.
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

  sigGuard.Dismiss();
}
#endif

#ifdef ROCR_INTERCEPT_QUEUE_TESTING
InterceptQueue::InterceptQueue(std::unique_ptr<Queue> queue, SharedQueue* shared_queue)
    : QueueProxy(std::move(queue), shared_queue),
      LocalSignal(0),
      DoorbellSignal(signal()),
      next_packet_(0),
      retry_index_(0),
      async_doorbell_(nullptr),
      quit_(false),
      active_(true) {
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);
  Initialize();
}
#endif

InterceptQueue::~InterceptQueue() {
  active_ = false;

#ifndef ROCR_INTERCEPT_QUEUE_TESTING
  // Kill the async doorbell handler
  // Doorbell may not be used during or after queue destroy, however an interrupt may be in flight.
  // Ensure doorbell value is not 0, mark for exit, wake handler and wait for termination value.
  async_doorbell_->StoreRelaxed(DOORBELL_MAX);
  quit_ = true;
  hsa_signal_value_t val = async_doorbell_->ExchRelaxed(1);
  if (val != 0)
    async_doorbell_->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 0, -1, HSA_WAIT_STATE_BLOCKED);
  async_doorbell_->DestroySignal();
#endif
}

void InterceptQueue::NotifyWrappedDoorbell(uint64_t new_index) {
#ifndef ROCR_INTERCEPT_QUEUE_TESTING
  HSA::hsa_signal_store_screlease(wrapped->amd_queue_.hsa_queue.doorbell_signal, new_index);
#endif
}

hsa_signal_t InterceptQueue::GetRetryCompletionSignal() {
  return async_doorbell_ ? Signal::Convert(async_doorbell_) : hsa_signal_t{0};
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

  const MetadataPacket* metadata = nullptr;
  if (queue->metadata_ring_buf_) {
    queue->metadata_scratch_.resize(pkt_count);
    for (uint64_t i = 0; i < pkt_count; ++i) {
      OriginalPacketInfo* match = nullptr;
      for (auto& original : queue->originals_) {
        if (!original.matched && MaskedPacketMatch(packets[i], original.aql) &&
            CompletionSignalMatch(packets[i], original.aql)) {
          match = &original;
          break;
        }
      }
      if (match == nullptr) {
        for (auto& original : queue->originals_) {
          if (!original.matched && MaskedPacketMatch(packets[i], original.aql)) {
            match = &original;
            break;
          }
        }
      }
      if (match != nullptr) {
        queue->metadata_scratch_[i] = match->metadata;
        match->matched = true;
      } else {
        FillNoopMetadata(queue->metadata_scratch_[i]);
      }
      UpdateMetadataEventId(packets[i], queue->metadata_scratch_[i]);
    }
    metadata = queue->metadata_scratch_.data();
  }

  // Submit final packet transform to hardware.
  uint64_t submitted_count = queue->Submit(packets, pkt_count, metadata);
  if (submitted_count == pkt_count) return;

  // Could not submit all the final packets, stash unsubmitted ones for later.
  assert(queue->overflow_.empty() && "Packet intercept error: overflow buffer not empty.\n");
  AppendOverflowPackets(queue->overflow_, packets, submitted_count, pkt_count, metadata);
}

uint64_t InterceptQueue::Submit(const AqlPacket* packets, uint64_t count,
                                const AqlMetadataPrefetchPacket* metadata) {
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
      // If we're in overflow processing (retry mechanism) and still can't fit all packets,
      // submit as many as possible to make progress and avoid infinite retry loops
      if (!overflow_.empty() && free_slots > (pending_retry_point ? 1 : 2)) {
        submitted_count = free_slots - (pending_retry_point ? 0 : 1);
      } else {
        submitted_count = 0;
      }
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
      if (metadata_ring_buf_) {
        MetadataPacket noop;
        FillNoopMetadata(noop);
        WriteMetadataSlot(metadata_ring_buf_, barrier, mask, noop);
      }
      ring[barrier & mask].barrier_and.completion_signal = GetRetryCompletionSignal();
      if (wrapped->IsDeviceMemRingBuf() && needsPcieOrdering()) {
        // Ensure the packet body is written as header may get reordered when writing over PCIE
        _mm_sfence();
      }
      atomic::Store(&ring[barrier & mask].barrier_and.header, kBarrierHeader,
                    std::memory_order_release);
      // Update the wrapped queue's doorbell so it knows there is a new packet in the queue.
      NotifyWrappedDoorbell(barrier);

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
      MetadataPacket noop;
      FillNoopMetadata(noop);
      while (submitted_count > 0 ||
             (packets_index < count && IsInterceptMarkerPacket(&packets[packets_index]))) {
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
            if (metadata_ring_buf_)
              WriteMetadataSlot(metadata_ring_buf_, write + write_index, mask,
                                metadata ? metadata[packets_index] : noop);
            first_written_packet_index = packets_index;
          } else {
            if (metadata_ring_buf_)
              WriteMetadataSlot(metadata_ring_buf_, write + write_index, mask,
                                metadata ? metadata[packets_index] : noop);
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
        atomic::Store(&ring[write & mask].packet.header,
                      packets[first_written_packet_index].packet.header, std::memory_order_release);
        NotifyWrappedDoorbell(write + write_index - 1);
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
    CopyOverflowToScratch(overflow_, aql_scratch_, metadata_scratch_);
    uint64_t submitted_count = Submit(aql_scratch_.data(), overflow_.size(),
                                      metadata_ring_buf_ ? metadata_scratch_.data() : nullptr);

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
    const AqlPacket* interceptor_input;
    if ((next_packet_ + packet_count) > ((next_packet_ & ~mask) + amd_queue_.hsa_queue.size)) {
      // Packets wrap around - use pre-allocated staging buffer
      for (uint64_t j = 0; j < packet_count; ++j) {
        staging_buffer_[j] = ring[(next_packet_ + j) & mask];
      }
      interceptor_input = staging_buffer_.data();
    } else {
      // Packets are contiguous in the ring buffer
      interceptor_input = &ring[next_packet_ & mask];
    }

    if (metadata_ring_buf_) {
      originals_.resize(packet_count);
      for (uint64_t j = 0; j < packet_count; ++j) {
        originals_[j].aql = interceptor_input[j];
        originals_[j].metadata = proxy_metadata_buf_[(next_packet_ + j) & mask];
        originals_[j].matched = false;
      }
    }

    handler.first(interceptor_input, packet_count, next_packet_, handler.second, PacketWriter);

    if (IsDeviceMemRingBuf() && needsPcieOrdering()) {
      // Ensure the packet body is written as header may get reordered when writing over PCIE
      _mm_sfence();
    }
  }
  i = next_packet_;
  while (i < std::min(end, invalid_header_i)) {
    // Invalidate consumed packets.
    if (metadata_ring_buf_) FillNoopMetadata(proxy_metadata_buf_[i & mask]);
    atomic::Store(&ring[i & mask].packet.header, kInvalidHeader, std::memory_order_release);
    // Packet has now been processed so advance the read index.
    ++i;
  }

  next_packet_ = i;
  Cursor.queue = nullptr;
  atomic::Store(&amd_queue_.read_dispatch_id, next_packet_, std::memory_order_release);
}

hsa_status_t InterceptQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER:
      if (metadata_ring_buf_ == nullptr) return wrapped->GetInfo(attribute, value);
      *reinterpret_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(proxy_metadata_buf_.data());
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_DISPATCH_PKT_VERSION_MAJOR:
    case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_DISPATCH_PKT_VERSION_MINOR:
    case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_BARRIER_PKT_VERSION_MAJOR:
    case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_BARRIER_PKT_VERSION_MINOR:
    case HSA_AMD_QUEUE_INFO_PROPERTIES:
      return wrapped->GetInfo(attribute, value);
    case HSA_AMD_QUEUE_INFO_AGENT:
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
    case HSA_QUEUE_INFO_USE_COUNT:
    case HSA_QUEUE_INFO_HW_ID: {
      if (!AMD::AqlQueue::IsType(wrapped.get())) return HSA_STATUS_ERROR_INVALID_QUEUE;

      AMD::AqlQueue* aqlQueue = static_cast<AMD::AqlQueue*>(wrapped.get());
      return aqlQueue->GetInfo(attribute, value);
    }
    default:
      break;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

}  // namespace core
}  // namespace rocr
