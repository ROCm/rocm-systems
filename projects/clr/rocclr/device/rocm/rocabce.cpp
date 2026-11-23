/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rocabce.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "device/rocm/rocdevice.hpp"
#include "utils/flags.hpp"
#include "utils/nontemporal.hpp"

namespace amd::roc {
namespace {

constexpr uint32_t kHostRingBytes = 8 * Mi;
constexpr size_t kRingLineBytes = 64;

// Writes a device-memory ring region through the write-combined BAR in whole
// 64-byte lines, like the device-memory AQL rings. The ring pads every region
// to 64 bytes (align64), so the destination is line aligned and line sized.
// MOVDIR64B closes the WC buffer, which orders the lines ahead of the doorbell.
template <bool kUseMovdir64b> void WriteDeviceRingLines(void* /*user_data*/, char* dst,
                                                        const void* src, size_t src_bytes,
                                                        size_t dst_bytes) {
  assert((reinterpret_cast<uintptr_t>(dst) % kRingLineBytes) == 0 &&
         (dst_bytes % kRingLineBytes) == 0 && "ABCE device ring regions must be whole lines");
  const char* src_lines = static_cast<const char*>(src);
  alignas(kRingLineBytes) char partial_line[kRingLineBytes];
  for (size_t line_offset = 0; line_offset < dst_bytes; line_offset += kRingLineBytes) {
    const char* line = partial_line;
    if (line_offset + kRingLineBytes <= src_bytes) {
      line = src_lines + line_offset;
    } else {
      std::memset(partial_line, 0, kRingLineBytes);
      if (line_offset < src_bytes) {
        std::memcpy(partial_line, src_lines + line_offset, src_bytes - line_offset);
      }
    }
    if constexpr (kUseMovdir64b) {
      amd::movdir64b_copy64(dst + line_offset, line);
    } else {
      amd::nontemporalMemcpy(dst + line_offset, line, kRingLineBytes);
    }
  }
  if constexpr (!kUseMovdir64b) {
    amd::nontemporalStoreFence();
  }
}

}  // namespace

abce_hsa_queue_api_t MakeAbceQueueApi() {
  abce_hsa_queue_api_t api;
  api.queue_get_info = &Hsa::amd_queue_get_info;
  api.load_read_index = &Hsa::queue_load_read_index_relaxed;
  api.load_write_index = &Hsa::queue_load_write_index_relaxed;
  api.signal_store_release = &Hsa::signal_store_screlease;
  return api;
}

std::unique_ptr<AbceEngineSet> AbceEngineSet::Create(Device& device) {
  if (!DEBUG_CLR_ENABLE_ABCE) {
    return nullptr;
  }
  if (!Hsa::amd_queue_create_available()) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE disabled: runtime lacks hsa_amd_queue_create");
    return nullptr;
  }

  const abce_isa_version_t isa{device.isa().versionMajor(), device.isa().versionMinor(),
                               device.isa().versionStepping()};
  // Wrap in unique_ptr up front so a failed Init() cleans up any partial state
  // through the destructor.
  std::unique_ptr<AbceEngineSet> engines(new AbceEngineSet(isa));
  if (!engines->Init(device)) {
    return nullptr;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "ABCE enabled with %u SDMA engine(s)",
          engines->num_engines());
  return engines;
}

bool AbceEngineSet::Init(Device& device) {
  const abce_hsa_queue_api_t api = MakeAbceQueueApi();
  if (!abce_hsa_queue_api_complete(&api)) {
    return false;
  }

  const uint32_t host_sdma = device.info().numSDMAengines_;
  uint32_t xgmi_sdma = 0;
  if (Hsa::agent_get_info(
          device.getBackendDevice(),
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG),
          &xgmi_sdma) != HSA_STATUS_SUCCESS) {
    xgmi_sdma = 0;
  }
  const uint32_t total_sdma = host_sdma + xgmi_sdma;
  const uint32_t want =
      std::min({total_sdma, AbceEngineSet::kMaxEngines, static_cast<uint32_t>(GPU_MAX_SDMA_QUEUES)});
  if (want == 0) {
    return false;
  }

  orchestrator_ = std::make_unique<abce::CopyOrchestrator>(builder_);
  // Selects the host-copy heatmap profile from the device-wide SDMA engine count,
  // and seeds the P2P band from the host/xGMI split: KFD numbers the non-xGMI
  // engines first, so host_sdma is where the xGMI band starts. The KFD load below
  // overrides it with the driver's own count wherever sysfs is readable.
  orchestrator_->InitDeviceProfile(total_sdma, host_sdma);
  if (abce_sdma_engine_policy_t* policy = orchestrator_->sdma_policy();
      policy != nullptr && !abce_sdma_policy_load_topology_from_kfd(policy, nullptr)) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: KFD topology unavailable; P2P engine affinity is disabled");
  }

  // A device-memory ring spares the engine fetching packets across PCIe, so it only pays
  // where the host link is PCIe. Over xGMI, frames this small fetch as fast from system
  // memory, and writing them into device memory costs the host more.
  // It is filled through the BAR, so it needs Large BAR and the whole-line writer; regions
  // are padded to 64 bytes to keep every write a full line. The BAR stores pass through
  // HDP and must land before the doorbell: the SDMA engine runs whatever precedes the
  // write index, with no packet header to wait on.
  const bool device_ring =
      DEBUG_CLR_ABCE_DEVICE_RING && device.info().largeBar_ && device.PcieStoresNeedFlush();
  abce_hsa_ring_options_t ring_options;
  abce_hsa_ring_options_initialize(&ring_options);
  ring_options.direct_queue_words = DEBUG_CLR_DIRECT_DOORBELL;
  if (device_ring) {
    ring_options.align64 = true;
    ring_options.readback_before_doorbell = true;
    ring_options.write =
        device.info().movdir64b_ ? &WriteDeviceRingLines<true> : &WriteDeviceRingLines<false>;
  }

  for (uint32_t engine_idx = 0; engine_idx < want; ++engine_idx) {
    hsa_amd_queue_create_desc_t desc = {};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
    if (device_ring) {
      desc.flags |= HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF;
    }
    desc.queue_size_bytes = kHostRingBytes;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.engine.sdma.sdma_engine_id = engine_idx;  // pin one ring per hw engine

    hsa_status_t status = Hsa::amd_queue_create(device.getBackendDevice(), &desc, 1);
    if (status != HSA_STATUS_SUCCESS || desc.queue == nullptr) {
      // Older KFD cannot create by-engine-id SDMA queues, so a failure on engine 0
      // disables ABCE. A later failure keeps the rings registered so far; policy
      // ranking and mapping only pick engines in the orchestrator's registered mask.
      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
              "ABCE: SDMA queue create for engine %u failed (status %d)", engine_idx, status);
      break;
    }

    hsa_queue_t* queue = desc.queue;
    if (abce_hsa_queue_ring_initialize(&rings_[engine_idx], queue, &api, &ring_options) !=
            HSA_STATUS_SUCCESS ||
        !abce_hsa_register_engine(orchestrator_->get(), engine_idx, &rings_[engine_idx])) {
      Hsa::queue_destroy(queue);
      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
              "ABCE: ring init/register for engine %u failed", engine_idx);
      break;
    }

    queues_[engine_idx] = queue;
    ++num_engines_;
  }

  if (num_engines_ == 0) {
    orchestrator_.reset();
    return false;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "ABCE registered %u host-facing and %u xGMI SDMA engine(s), rings in %s memory%s",
          std::min(num_engines_, host_sdma),
          num_engines_ > host_sdma ? num_engines_ - host_sdma : 0,
          device_ring ? "device" : "system",
          ring_options.readback_before_doorbell ? ", read back before doorbell" : "");
  return true;
}

void AbceEngineSet::DestroyQueues() {
  for (uint32_t engine_idx = 0; engine_idx < num_engines_; ++engine_idx) {
    if (queues_[engine_idx] != nullptr) {
      abce_hsa_queue_ring_drain(&rings_[engine_idx]);
      Hsa::queue_destroy(queues_[engine_idx]);
      queues_[engine_idx] = nullptr;
    }
  }
  num_engines_ = 0;
}

AbceEngineSet::~AbceEngineSet() { DestroyQueues(); }

uint32_t AbceEngineSet::EngineAllocator::Acquire(const VirtualGPU* vgpu, const uint32_t* ranked,
                                                 uint32_t num_ranked) {
  if (num_ranked == 0) {
    return kNoEngine;
  }

  std::scoped_lock lock(lock_);

  // Keep the stream's exact ring while it remains legal. HwQueueTracker uses the
  // selected ring identity to decide whether in-order execution can replace a wait.
  auto reservation = vgpu_to_engine_.find(vgpu);
  const bool held_a_ring = reservation != vgpu_to_engine_.end();
  if (held_a_ring &&
      std::find(ranked, ranked + num_ranked, reservation->second) != ranked + num_ranked) {
    return reservation->second;
  }

  uint64_t taken = 0;
  for (const auto& entry : vgpu_to_engine_) {
    if (entry.first != vgpu) {
      taken |= uint64_t{1} << entry.second;
    }
  }

  uint32_t selected = kNoEngine;
  for (uint32_t choice_idx = 0; choice_idx < num_ranked; ++choice_idx) {
    if (!(taken & (uint64_t{1} << ranked[choice_idx]))) {
      selected = ranked[choice_idx];
      break;
    }
  }
  if (selected == kNoEngine) {
    selected = ranked[next_round_robin_++ % num_ranked];
  }

  vgpu_to_engine_[vgpu] = selected;
  ClPrint(amd::LOG_INFO, amd::LOG_COPY,
          "Reserved ABCE ring %u for VirtualGPU %p: ranked[0]=%u taken_mask=0x%llx", selected, vgpu,
          ranked[0], static_cast<unsigned long long>(taken));
  return selected;
}

void AbceEngineSet::EngineAllocator::Release(const VirtualGPU* vgpu) {
  std::scoped_lock lock(lock_);
  auto reservation = vgpu_to_engine_.find(vgpu);
  if (reservation != vgpu_to_engine_.end()) {
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Released ABCE ring %u for VirtualGPU %p",
            reservation->second, vgpu);
    vgpu_to_engine_.erase(reservation);
  }
}

}  // namespace amd::roc
