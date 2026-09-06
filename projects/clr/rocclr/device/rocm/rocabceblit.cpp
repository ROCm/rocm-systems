/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rocabceblit.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "abce_decode.h"
#include "device/rocm/rocabce.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocvirtual.hpp"

namespace amd::roc {

void AbceBlitManager::normalizePeerEndpoints(hsa_agent_t& srcAgent, uint32_t& srcDriverNodeId,
                                             hsa_agent_t& dstAgent,
                                             uint32_t& dstDriverNodeId) const {
  const hsa_agent_t cpu_agent = dev().getCpuAgent();
  const hsa_agent_t backend_device = dev().getBackendDevice();
  if (srcAgent.handle != dstAgent.handle && srcAgent.handle != cpu_agent.handle &&
      dstAgent.handle != cpu_agent.handle) {
    const bool source_is_backend = srcAgent.handle == backend_device.handle;
    const hsa_agent_t peer_agent = source_is_backend ? dstAgent : srcAgent;
    const uint32_t peer_driver_node_id =
        source_is_backend ? dstDriverNodeId : srcDriverNodeId;
    srcAgent = backend_device;
    srcDriverNodeId = dev().info().driverNodeId_;
    dstAgent = peer_agent;
    dstDriverNodeId = peer_driver_node_id;
  }
}

void AbceBlitManager::logFramePackets(const abce_plan_frame_t& frame, uint32_t frame_idx,
                                      const char* label, amd::LogMask mask) const {
  const uint32_t* packet_words = reinterpret_cast<const uint32_t*>(frame.packets);
  const uint32_t num_words = frame.bytes / sizeof(uint32_t);
  const abce_isa_version_t isa{dev().isa().versionMajor(), dev().isa().versionMinor(),
                               dev().isa().versionStepping()};
  uint32_t packet_idx = 0;
  uint32_t packet_base = 0;
  while (packet_base < num_words) {
    const abce_decoded_packet_t packet =
        abce_decode_sdma_packet(packet_words + packet_base, num_words - packet_base, isa);
    ClPrint(amd::LOG_DETAIL_DEBUG, mask, "%s frame[%u] packet[%u]: %s op=%u sub_op=%u dword=%u count=%u",
            label, frame_idx, packet_idx, packet.name, packet.op, packet.sub_op, packet_base,
            packet.dwords);
    for (uint32_t packet_word_base = 0; packet_word_base < packet.dwords; packet_word_base += 8) {
      char dump[256];
      int dump_offset = std::snprintf(dump, sizeof(dump), "%s frame[%u] packet[%u] +%02u:", label,
                                      frame_idx, packet_idx, packet_word_base);
      const uint32_t line_end = std::min(packet_word_base + 8, packet.dwords);
      for (uint32_t packet_word = packet_word_base;
           packet_word < line_end && dump_offset > 0 &&
           static_cast<size_t>(dump_offset) < sizeof(dump);
           ++packet_word) {
        dump_offset += std::snprintf(dump + dump_offset, sizeof(dump) - dump_offset, " %08x",
                                     packet_words[packet_base + packet_word]);
      }
      ClPrint(amd::LOG_DETAIL_DEBUG, mask, "%s", dump);
    }
    packet_base += packet.dwords;
    ++packet_idx;
  }
}

AbceBlitManager::CopyResult AbceBlitManager::submitCopy(
    address dst, hsa_agent_t dstAgent, uint32_t dstDriverNodeId, const_address src,
    hsa_agent_t srcAgent, uint32_t srcDriverNodeId, size_t size) const {
  if (size == 0) {
    return CopyResult::kNotSubmitted;
  }

  normalizePeerEndpoints(srcAgent, srcDriverNodeId, dstAgent, dstDriverNodeId);
  const hsa_agent_t cpu_agent = dev().getCpuAgent();
  const bool host_src = (srcAgent.handle == cpu_agent.handle);
  const bool host_dst = (dstAgent.handle == cpu_agent.handle);

  HwQueueEngine engine;
  if (srcAgent.handle == dstAgent.handle) {
    engine = HwQueueEngine::SdmaD2D;
  } else if (host_src) {
    engine = HwQueueEngine::SdmaH2D;
  } else if (host_dst) {
    engine = HwQueueEngine::SdmaD2H;
  } else {
    engine = HwQueueEngine::SdmaP2P;
  }

  abce::CopyOp copy;
  copy.kind = ABCE_OP_KIND_LINEAR;
  copy.src = src;
  copy.dst = dst;
  copy.size = size;
  copy.src_end.kind = host_src ? ABCE_ENDPOINT_KIND_HOST : ABCE_ENDPOINT_KIND_DEVICE;
  copy.dst_end.kind = host_dst ? ABCE_ENDPOINT_KIND_HOST : ABCE_ENDPOINT_KIND_DEVICE;
  if (!host_src && srcDriverNodeId == UINT32_MAX) {
    return CopyResult::kNotSubmitted;
  }
  if (!host_dst && dstDriverNodeId == UINT32_MAX) {
    return CopyResult::kNotSubmitted;
  }
  copy.src_end.device_id = srcDriverNodeId;
  copy.dst_end.device_id = dstDriverNodeId;

  AbceEngineSet& abce_engines = *dev().abceEngineSet();
  abce::CopyOrchestrator& orchestrator = abce_engines.orchestrator();

  // Non-peer copies retain one legal ring per stream. The tracker may elide the
  // previous ABCE signal only when this exact coordinator ring is unchanged.
  const bool reserve_ring = engine != HwQueueEngine::SdmaP2P;
  uint32_t chosen_ring = VirtualGPU::HwQueueTracker::kInvalidAbceRing;
  const HwQueueEngine tracked_engine = HwQueueEngine::SdmaAbce;
  if (reserve_ring) {
    uint32_t ranked_engines[AbceEngineSet::kMaxEngines]{};
    const uint32_t num_ranked =
        orchestrator.RankLegalEngines(copy, ranked_engines, AbceEngineSet::kMaxEngines);
    const uint32_t selected_ring =
        abce_engines.engine_allocator().Acquire(&gpu(), ranked_engines, num_ranked);
    if (selected_ring == AbceEngineSet::kNoEngine) {
      ClPrint(amd::LOG_WARNING, amd::LOG_COPY,
              "ABCE ring unavailable: engine=%s ranked=%u size=%zu", EngineOpName(engine),
              num_ranked, size);
      return CopyResult::kNotSubmitted;
    }
    chosen_ring = selected_ring;
  }

  std::vector<hsa_signal_t>& wait_events =
      gpu().Barriers().WaitingSignal(tracked_engine, chosen_ring);
  hsa_signal_t completion = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());

  // Each waiting signal counts down to zero on completion. ABCE ignores deps
  // whose observed value is already zero (nothing left to wait on).
  std::vector<abce_dep_signal_t> deps;
  deps.reserve(wait_events.size());
  for (hsa_signal_t wait_signal : wait_events) {
    const uint64_t observed = static_cast<uint64_t>(Hsa::signal_load_relaxed(wait_signal));
    deps.push_back(abce_hsa_dep_signal(reinterpret_cast<amd_signal_t*>(wait_signal.handle),
                                       observed, /*reference=*/0));
  }

  abce::CopyMetadata metadata;
  metadata.deps = deps.empty() ? nullptr : deps.data();
  metadata.num_deps = static_cast<uint32_t>(deps.size());
  metadata.coherency.flags.emit_hdp_flush = !amd::IS_HIP;
  // Completion takes the HSA signal one below its armed value and rings its
  // event mailbox, so the existing HwQueueTracker wait machinery wakes normally.
  // A blocking callback arms it one higher and takes the last step itself, so
  // the engine writes the value outright only when the signal is armed to one.
  const uint64_t completion_pending_value =
      static_cast<uint64_t>(Hsa::signal_load_relaxed(completion));
  auto* completion_signal = reinterpret_cast<amd_signal_t*>(completion.handle);
  metadata.out = abce_hsa_signal_ref(completion_signal, completion_pending_value - 1);
  metadata.out.sole_update = completion_pending_value == kInitSignalValueOne;
  if (gpu().timestamp() != nullptr) {
    completion_signal->start_ts = 0;
    completion_signal->end_ts = 0;
    metadata.timestamps.start_value = &completion_signal->start_ts;
    metadata.timestamps.end_value = &completion_signal->end_ts;
  }
  abce::Plan plan;
  const abce_status_t map_status = orchestrator.MapCopy(&copy, 1, metadata, &plan);
  if (!abce::IsOk(map_status)) {
    ClPrint(amd::LOG_WARNING, amd::LOG_COPY,
            "ABCE map failed: status=%s engine=%s src_node=%u dst_node=%u size=%zu",
            abce_status_name(map_status), EngineOpName(engine), copy.src_end.device_id,
            copy.dst_end.device_id, size);
    gpu().Barriers().ResetCurrentSignal();
    gpu().Barriers().SetActiveEngine(HwQueueEngine::Unknown);
    return CopyResult::kNotSubmitted;
  }

  if (reserve_ring) {
    plan.frames[0].engine = chosen_ring;
  } else {
    chosen_ring = plan.frames[0].engine;
  }

  if (AMD_LOG_LEVEL >= amd::LOG_DETAIL_DEBUG) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE,
            "ABCE signals: completion_handle=0x%zx initial=%lld event_id=%u deps=%u "
            "chosen_ring=%u",
            completion.handle, static_cast<long long>(kInitSignalValueOne), metadata.out.event_id,
            metadata.num_deps, chosen_ring);
    for (uint32_t dep_idx = 0; dep_idx < metadata.num_deps; ++dep_idx) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE,
              "ABCE dep[%u]: handle=0x%zx observed=%llu ref=%llu",
              dep_idx, wait_events[dep_idx].handle,
              static_cast<unsigned long long>(metadata.deps[dep_idx].observed_value),
              static_cast<unsigned long long>(metadata.deps[dep_idx].reference_value));
    }
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
      const abce_plan_frame_t& frame = plan.frames[frame_idx];
      char choices[128];
      int choices_offset = std::snprintf(choices, sizeof(choices), "[");
      for (uint32_t choice_idx = 0;
           choice_idx < frame.num_ranked_engines && choices_offset > 0 &&
           static_cast<size_t>(choices_offset) < sizeof(choices);
           ++choice_idx) {
        choices_offset +=
            std::snprintf(choices + choices_offset, sizeof(choices) - choices_offset, "%s%u",
                          choice_idx == 0 ? "" : ",", frame.ranked_engines[choice_idx]);
      }
      if (choices_offset > 0 && static_cast<size_t>(choices_offset) < sizeof(choices))
        std::snprintf(choices + choices_offset, sizeof(choices) - choices_offset, "]");

      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE,
              "ABCE frame[%u]: engine=%u ops=%u bytes=%u coordinator=%u preferred=%u "
              "choices=%s active_signal=0x%zx",
              frame_idx, frame.engine, frame.num_ops, frame.bytes, frame.coordinator,
              frame.has_engine_preference ? 1u : 0u, choices, completion.handle);
      logFramePackets(frame, frame_idx, "ABCE", amd::LOG_QUEUE);
    }
  }

  const abce_status_t submit_status = orchestrator.Submit(plan);
  if (!abce::IsOk(submit_status)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_COPY,
            "ABCE submit failed: status=%s engine=%s size=%zu",
            abce_status_name(submit_status), EngineOpName(engine), size);
    gpu().Barriers().ResetCurrentSignal();
    gpu().Barriers().SetActiveEngine(HwQueueEngine::Unknown);
    return CopyResult::kFailed;
  }

  gpu().addSystemScope();
  gpu().setFenceDirty(false);
  return CopyResult::kSubmitted;
}

AbceBlitManager::CopyResult AbceBlitManager::copyHostBuffer(
    const_address hostSrc, address hostDst, size_t size, bool hostToDevice) const {
  gpu().releaseGpuMemoryFence(hostToDevice || !dev().settings().blocking_blit_);

  const hsa_agent_t cpu_agent = dev().getCpuAgent();
  const hsa_agent_t gpu_agent = dev().getBackendDevice();
  const uint32_t gpu_driver_node_id = dev().info().driverNodeId_;
  size_t copy_offset = 0;
  bool first_transfer = true;
  bool submitted = false;

  while (copy_offset < size) {
    const_address host_memory = hostToDevice ? hostSrc : hostDst;
    BufferState buffer = {};
    constexpr bool kEnablePinning = true;
    getBuffer(host_memory + copy_offset, size - copy_offset, kEnablePinning, first_transfer,
              buffer);
    if (buffer.buffer_ == 0) {
      return submitted ? CopyResult::kFailed : CopyResult::kNotSubmitted;
    }

    if (hostToDevice && buffer.pinnedMem_ == nullptr) {
      std::memcpy(buffer.buffer_, hostSrc + copy_offset, buffer.copySize_);
    }

    const address dst = hostToDevice ? hostDst + copy_offset : buffer.buffer_;
    const_address src = hostToDevice ? buffer.buffer_ : hostSrc + copy_offset;
    const CopyResult result = submitCopy(
        dst, hostToDevice ? gpu_agent : cpu_agent,
        hostToDevice ? gpu_driver_node_id : ABCE_ANY_DEVICE, src,
        hostToDevice ? cpu_agent : gpu_agent,
        hostToDevice ? ABCE_ANY_DEVICE : gpu_driver_node_id, buffer.copySize_);
    if (result != CopyResult::kSubmitted) {
      releaseBuffer(buffer);
      return submitted || result == CopyResult::kFailed ? CopyResult::kFailed
                                                        : CopyResult::kNotSubmitted;
    }
    submitted = true;

    if (!hostToDevice && buffer.pinnedMem_ == nullptr) {
      gpu().Barriers().WaitCurrent();
      std::memcpy(hostDst + copy_offset, buffer.buffer_, buffer.copySize_);
    }

    copy_offset += buffer.copySize_;
    first_transfer = false;
    releaseBuffer(buffer);
  }

  if (amd::IS_HIP && (gpu().command() != nullptr) && gpu().command()->IsMemoryPinned()) {
    gpu().Barriers().WaitCurrent();
    gpu().command()->ReleasePinnedMemory();
  }
  return CopyResult::kSubmitted;
}

bool AbceBlitManager::describeCopy(const amd::BatchCopyOp& op, abce_copy_op_t& copy) const {
  switch (op.metadata.copyOpType_) {
    case amd::CopyMetadata::kCopyOpLinear:
      copy.kind = ABCE_OP_KIND_LINEAR;
      break;
    case amd::CopyMetadata::kCopyOpSwap:
      copy.kind = ABCE_OP_KIND_SWAP;
      break;
    case amd::CopyMetadata::kCopyOpIndirectSrc:
      copy.kind = ABCE_OP_KIND_INDIRECT;
      copy.indirect_src = true;
      break;
    case amd::CopyMetadata::kCopyOpIndirectDst:
      copy.kind = ABCE_OP_KIND_INDIRECT;
      copy.indirect_dst = true;
      break;
    case amd::CopyMetadata::kCopyOpIndirectSrcDst:
      copy.kind = ABCE_OP_KIND_INDIRECT;
      copy.indirect_src = true;
      copy.indirect_dst = true;
      break;
    default:
      return false;
  }

  const Memory& source_memory =
      gpuMem(*op.srcMemory->getDeviceMemory(*op.srcMemory->getContext().devices()[0]));
  const Memory& destination_memory =
      gpuMem(*op.dstMemory->getDeviceMemory(*op.dstMemory->getContext().devices()[0]));

  address src = reinterpret_cast<address>(source_memory.getDeviceMemory()) + op.srcOffset;
  address dst = reinterpret_cast<address>(destination_memory.getDeviceMemory()) + op.dstOffset;
  // The true owner is cached when memory is created, including IPC imports, and
  // refreshed after VMM mappings become backed.
  hsa_agent_t src_agent = source_memory.getOwningAgent();
  hsa_agent_t dst_agent = destination_memory.getOwningAgent();
  uint32_t src_driver_node_id = source_memory.getOwningAgentDriverNodeId();
  uint32_t dst_driver_node_id = destination_memory.getOwningAgentDriverNodeId();
  const hsa_agent_t cpu_agent = dev().getCpuAgent();
  normalizePeerEndpoints(src_agent, src_driver_node_id, dst_agent, dst_driver_node_id);

  const bool host_src = src_agent.handle == cpu_agent.handle;
  const bool host_dst = dst_agent.handle == cpu_agent.handle;
  if (host_src && host_dst) {
    return false;
  }

  copy.src = src;
  copy.dst = dst;
  copy.size = op.size;
  copy.src_end.kind = host_src ? ABCE_ENDPOINT_KIND_HOST : ABCE_ENDPOINT_KIND_DEVICE;
  copy.dst_end.kind = host_dst ? ABCE_ENDPOINT_KIND_HOST : ABCE_ENDPOINT_KIND_DEVICE;
  if (!host_src && src_driver_node_id == UINT32_MAX) {
    return false;
  }
  if (!host_dst && dst_driver_node_id == UINT32_MAX) {
    return false;
  }
  copy.src_end.device_id = src_driver_node_id;
  copy.dst_end.device_id = dst_driver_node_id;
  return true;
}

bool AbceBlitManager::submitCopyBatch(const std::vector<amd::BatchCopyOp>& copyOps,
                                      const std::vector<hsa_signal_t>* externalWaitEvents,
                                      std::vector<ProfilingSignal*>* outBatchSignals) const {
  if (copyOps.empty()) {
    return true;
  }

  // Convert the complete batch before touching tracker state. MapCopy owns all
  // engine ranking and placement decisions; if an operation is not representable,
  // leave the complete batch on the existing ROCr path.
  std::vector<abce_copy_op_t> copies;
  copies.reserve(copyOps.size());
  for (const amd::BatchCopyOp& op : copyOps) {
    abce::CopyOp copy;
    if (!describeCopy(op, copy)) {
      return DmaBlitManager::submitCopyBatch(copyOps, externalWaitEvents, outBatchSignals);
    }
    copies.push_back(copy);
  }

  std::vector<hsa_signal_t> wait_events =
      externalWaitEvents != nullptr
          ? *externalWaitEvents
          : gpu().Barriers().WaitingSignal(HwQueueEngine::SdmaAbce,
                                           VirtualGPU::HwQueueTracker::kInvalidAbceRing);

  // Each waiting signal counts down to zero on completion. ABCE omits a poll
  // whose observed value is already zero.
  std::vector<abce_dep_signal_t> deps;
  deps.reserve(wait_events.size());
  for (hsa_signal_t wait_signal : wait_events) {
    const uint64_t observed = static_cast<uint64_t>(Hsa::signal_load_relaxed(wait_signal));
    deps.push_back(abce_hsa_dep_signal(reinterpret_cast<amd_signal_t*>(wait_signal.handle),
                                       observed, /*reference=*/0));
  }

  AbceEngineSet& abce_engines = *dev().abceEngineSet();
  abce::CopyOrchestrator& orchestrator = abce_engines.orchestrator();

  // A plan may span several rings, so no exact-ring identity describes it.
  // SdmaAbce preserves timestamp handling while the invalid ring forces a wait.
  gpu().Barriers().SetActiveEngine(HwQueueEngine::SdmaAbce);

  hsa_signal_t completion =
      gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());
  abce::CopyMetadata metadata;
  metadata.deps = deps.empty() ? nullptr : deps.data();
  metadata.num_deps = static_cast<uint32_t>(deps.size());
  metadata.coherency.flags.emit_hdp_flush = !amd::IS_HIP;
  const uint64_t completion_pending_value =
      static_cast<uint64_t>(Hsa::signal_load_relaxed(completion));
  auto* completion_signal = reinterpret_cast<amd_signal_t*>(completion.handle);
  metadata.out = abce_hsa_signal_ref(completion_signal, completion_pending_value - 1);
  metadata.out.sole_update = completion_pending_value == kInitSignalValueOne;
  if (gpu().timestamp() != nullptr) {
    completion_signal->start_ts = 0;
    completion_signal->end_ts = 0;
    metadata.timestamps.start_value = &completion_signal->start_ts;
    metadata.timestamps.end_value = &completion_signal->end_ts;
  }

  abce::Plan plan;
  const abce_status_t map_status = orchestrator.MapCopy(
      copies.data(), static_cast<uint32_t>(copies.size()), metadata, &plan);
  if (!abce::IsOk(map_status)) {
    ClPrint(amd::LOG_WARNING, amd::LOG_COPY,
            "ABCE batch map failed: status=%s failed_op=%u copies=%zu",
            abce_status_name(map_status), plan.failed_op, copies.size());
    gpu().Barriers().ResetCurrentSignal();
    return DmaBlitManager::submitCopyBatch(copyOps, &wait_events, outBatchSignals);
  }

  if (AMD_LOG_LEVEL >= amd::LOG_DETAIL_DEBUG) {
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
      const abce_plan_frame_t& frame = plan.frames[frame_idx];
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY,
              "ABCE batch frame[%u]: ring=%u copies=%u bytes=%u coordinator=%u",
              frame_idx, frame.engine, frame.num_ops, frame.bytes,
              frame.coordinator ? 1u : 0u);
      logFramePackets(frame, frame_idx, "ABCE batch", amd::LOG_COPY);
    }
  }

  const abce_status_t submit_status = orchestrator.Submit(plan);
  if (!abce::IsOk(submit_status)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_COPY, "ABCE batch submit failed: status=%s frames=%u",
            abce_status_name(submit_status), plan.num_frames);
    return false;
  }

  gpu().addSystemScope();
  gpu().setFenceDirty(false);
  return true;
}

bool AbceBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                 const amd::Coord3D& srcOrigin,
                                 const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                 bool entire, amd::CopyMetadata copyMetadata) const {
  Memory& source_memory = gpuMem(srcMemory);
  Memory& destination_memory = gpuMem(dstMemory);
  const bool use_shader =
      setup_.disableHwlCopyBuffer_ ||
      useShaderCopyBufferPath(source_memory, destination_memory, size[0], copyMetadata);
  const bool use_host =
      !HSA_ENABLE_DTIF_FAST_COPY &&
      (setup_.disableCopyBuffer_ ||
       (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached() &&
        (dev().agent_profile() != HSA_PROFILE_FULL) && dstMemory.isHostMemDirectAccess()));
  if (use_shader || use_host || size[0] == 0) {
    return KernelBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                         copyMetadata);
  }

  std::scoped_lock lock(*lockXfer());
  if (amd::IS_HIP) {
    if (srcMemory.isHostMemDirectAccess()) {
      gpu().SetCopyCommandType(CL_COMMAND_WRITE_BUFFER);
    }
    if (dstMemory.isHostMemDirectAccess()) {
      gpu().SetCopyCommandType(CL_COMMAND_READ_BUFFER);
    }
  }

  address src = source_memory.getDeviceMemory() + srcOrigin[0];
  address dst = destination_memory.getDeviceMemory() + dstOrigin[0];
  hsa_agent_t src_agent = source_memory.getOwningAgent();
  hsa_agent_t dst_agent = destination_memory.getOwningAgent();
  const uint32_t src_driver_node_id = source_memory.getOwningAgentDriverNodeId();
  const uint32_t dst_driver_node_id = destination_memory.getOwningAgentDriverNodeId();
  const bool skip_cpu_wait =
      copyMetadata.isAsync_ || srcMemory.isHostMemDirectAccess() ||
      !dstMemory.isHostMemDirectAccess();
  gpu().releaseGpuMemoryFence(skip_cpu_wait);

  const CopyResult result = submitCopy(dst, dst_agent, dst_driver_node_id, src, src_agent,
                                       src_driver_node_id, size[0]);
  if (result == CopyResult::kNotSubmitted) {
    return KernelBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                         copyMetadata);
  }
  if (result == CopyResult::kFailed) {
    return false;
  }
  synchronize();
  return true;
}

bool AbceBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                 const amd::Coord3D& origin, const amd::Coord3D& size,
                                 bool entire, amd::CopyMetadata copyMetadata) const {
  const bool use_host =
      dev().settings().blocking_blit_ &&
      (setup_.disableReadBuffer_ ||
       (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached()));
  const bool use_shader =
      setup_.disableHwlCopyBuffer_ || size[0] <= dev().settings().sdmaCopyThreshold_ ||
      copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::BLIT;
  if (use_host || use_shader || size[0] == 0) {
    return KernelBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
  }

  std::scoped_lock lock(*lockXfer());
  const_address src = gpuMem(srcMemory).getDeviceMemory() + origin[0];
  const CopyResult result =
      copyHostBuffer(src, reinterpret_cast<address>(dstHost), size[0], false);
  if (result == CopyResult::kNotSubmitted) {
    return KernelBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
  }
  if (result == CopyResult::kFailed) {
    return false;
  }
  synchronize();
  return true;
}

bool AbceBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  bool entire, amd::CopyMetadata copyMetadata) const {
  const bool use_host =
      dev().settings().blocking_blit_ &&
      (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess() ||
       gpuMem(dstMemory).IsPersistentDirectMap());
  const bool use_shader =
      setup_.disableHwlCopyBuffer_ || size[0] <= dev().settings().sdmaCopyThreshold_ ||
      copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::BLIT;
  if (use_host || use_shader || size[0] == 0) {
    return KernelBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
  }

  std::scoped_lock lock(*lockXfer());
  address dst = gpuMem(dstMemory).getDeviceMemory() + origin[0];
  const CopyResult result =
      copyHostBuffer(reinterpret_cast<const_address>(srcHost), dst, size[0], true);
  if (result == CopyResult::kNotSubmitted) {
    return KernelBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
  }
  if (result == CopyResult::kFailed) {
    return false;
  }
  synchronize();
  return true;
}

}  // namespace amd::roc
