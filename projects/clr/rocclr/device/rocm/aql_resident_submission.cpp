// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_submission.hpp"
#include "aql_resident_storage.hpp"
#include "rocdevice.hpp"
#include "rocvirtual.hpp"
#include "rocrctx.hpp"
#include "aql_packet_publication.hpp"
#include <cstring>

namespace amd::roc::aql_resident {
static_assert(sizeof(hsa_amd_aql_ib_jump_packet_t) == 64);
static_assert(offsetof(hsa_amd_aql_ib_jump_packet_t, target_base_addr) == 8);
static_assert(offsetof(hsa_amd_aql_ib_jump_packet_t, target_flags) == 16);
static_assert(offsetof(hsa_amd_aql_ib_jump_packet_t, completion_signal) == 48);

SubmitResult QueueSubmitter::submit(VirtualGPU& gpu, const ResidentImage& image,
                                    uint32_t packetCount,
                                    const hsa_kernel_dispatch_packet_t* preparedFixup,
                                    const std::shared_ptr<amd::AqlBatchImage>& resources,
                                    bool blocking) {
  auto* queue = gpu.gpu_queue_;
  if (!queue || queue->size < 4 || !resources || &image.device() != &gpu.dev() ||
      gpu.timestamp_ || gpu.profiling_ || gpu.cooperative_ ||
      gpu.metadata_preloader_.HasMetadataQueue() || gpu.device_mem_ring_buf_ ||
      !packetCount || packetCount > 65535 || uint64_t(packetCount)*64 > image.size()) {
    return SubmitResult::Unsupported;
  }
  if (preparedFixup && ((preparedFixup->header & 0xff) != HSA_PACKET_TYPE_KERNEL_DISPATCH ||
      !(preparedFixup->header & (1u << HSA_PACKET_HEADER_BARRIER)) ||
      preparedFixup->completion_signal.handle || preparedFixup->private_segment_size)) {
    return SubmitResult::Unsupported;
  }
  hsa_amd_aql_ib_jump_packet_t root{};
  root.target_base_addr = reinterpret_cast<uintptr_t>(image.base());
  root.target_size_packets = packetCount;
  root.target_flags = preparedFixup ? HSA_AMD_AQL_IB_JUMP_TARGET_FLAG_DIRTY : 0;
  root.completion_signal = gpu.Barriers().ActiveSignal();
  {
    auto* signal = gpu.Barriers().GetLastSignal();
    std::scoped_lock lock(signal->LockSignalOps());
    signal->retained_graph_image_ = resources;
  }
  constexpr uint16_t rootHeader = HSA_PACKET_TYPE_VENDOR_SPECIFIC |
      (1u << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  const uint64_t count = preparedFixup ? 2 : 1;
  // One reservation prevents another producer from entering between fixup and
  // target. Complete all bodies, publish the tail, then publish the head last.
  // Use the same physical-queue reservation and stream-order tracking as
  // ordinary AQL. Do not optimize away these explicit mutation/root barriers.
  const auto reservation = gpu.ReserveAqlSlots(count);
  const uint64_t first = reservation.start_slot;
  const uint64_t last = first + count - 1;
  while (last - Hsa::queue_load_read_index_scacquire(queue) >= queue->size - 1) {
    amd::Os::yield();
  }
  auto* rootSlot = static_cast<hsa_amd_aql_ib_jump_packet_t*>(queue->base_address) +
                   (last & (queue->size-1));
  root.header.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  std::memcpy(rootSlot, &root, sizeof(root));
  hsa_kernel_dispatch_packet_t* fixupSlot = nullptr;
  if (preparedFixup) {
    fixupSlot = static_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address) +
                (first & (queue->size-1));
    auto fixup = *preparedFixup;
    fixup.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
    std::memcpy(fixupSlot, &fixup, sizeof(fixup));
  }
  gpu.TrackQueueProgress(root, last);
  gpu.RecordAqlPacketHeader(reservation, count - 1, rootHeader);
  if (preparedFixup) gpu.RecordAqlPacketHeader(reservation, 0, preparedFixup->header);
  gpu.CompleteAqlSubmission(reservation);
  packet_store_release(reinterpret_cast<uint32_t*>(rootSlot), rootHeader,
                       HSA_AMD_PACKET_TYPE_AQL_IB_JUMP);
  if (fixupSlot) {
    packet_store_release(reinterpret_cast<uint32_t*>(fixupSlot),
                         preparedFixup->header, preparedFixup->setup);
  }
  gpu.ringQueueDoorbell(last);
  gpu.addSystemScope_ = false;
  gpu.fence_state_ = Device::kCacheStateSystem;
  gpu.setFenceDirty(false);
  gpu.hasPendingDispatch_ = false;
  if (blocking && !gpu.Barriers().WaitCurrent()) return SubmitResult::Failed;
  return SubmitResult::Submitted;
}
}  // namespace amd::roc::aql_resident
