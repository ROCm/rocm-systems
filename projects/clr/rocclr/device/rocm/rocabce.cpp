/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rocabce.hpp"

#include <algorithm>
#include <cstring>

#include "device/rocm/rocdevice.hpp"
#include "utils/flags.hpp"

namespace amd::roc {
namespace {

constexpr size_t kDeviceRingControlBytes =
    sizeof(abce_ring_control_t) * AbceEngineSet::kMaxEngines;
constexpr uint32_t kHostRingBytes = 8 * Mi;
//! A kernel submits one small frame per copy, so its rings need far less room
//! than the host's batched ones.
constexpr uint32_t kDeviceRingBytes = 1 * Mi;

//! Device submission needs SDMA atomics for the completion packet plus a
//! doorbell the shader can write. Restricted to the parts we can test rather
//! than inferred from the ISA, so an untried target falls back to host copies.
bool SupportsDeviceSubmission(const abce_isa_version_t& isa) {
  return (isa.major == 9 && ((isa.minor == 0 && isa.stepping == 10) || isa.minor == 4)) ||
         (isa.major == 12 && isa.minor >= 5);
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

  for (uint32_t engine_idx = 0; engine_idx < want; ++engine_idx) {
    hsa_amd_queue_create_desc_t desc = {};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
    desc.queue_size_bytes = kHostRingBytes;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.engine.sdma.sdma_engine_id = engine_idx;  // pin one ring per hw engine

    hsa_status_t status = Hsa::amd_queue_create(device.getBackendDevice(), &desc, 1);
    if (status != HSA_STATUS_SUCCESS || desc.queue == nullptr) {
      // Older KFD cannot create by-engine-id SDMA queues; disable the feature
      // rather than partially enabling it.
      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
              "ABCE: SDMA queue create for engine %u failed (status %d)", engine_idx, status);
      break;
    }

    hsa_queue_t* queue = desc.queue;
    if (abce_hsa_queue_ring_initialize(&rings_[engine_idx], queue, &api, nullptr) !=
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

  InitDeviceEngines(device);
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "ABCE registered %u host-facing and %u xGMI SDMA engine(s)",
          std::min(num_engines_, host_sdma),
          num_engines_ > host_sdma ? num_engines_ - host_sdma : 0);
  return true;
}

bool AbceEngineSet::InitDeviceEngines(Device& device) {
  // Kernel-side copies are opt-in: their queues come out of the per-GPU SDMA
  // queue budget that every process shares.
  if (GPU_MAX_SDMA_DEVICE_QUEUES == 0 || !SupportsDeviceSubmission(builder_.isa())) {
    return false;
  }

  device_ = &device;
  local_driver_node_id_ = device.info().driverNodeId_;
  device_atomic_support_ = abce_detect_default_platform_caps(builder_.isa()).device_atomic_support;
  const size_t engines_bytes = sizeof(abce_device_engine_t) * kMaxEngines;
  device_engines_staging_ = static_cast<abce_device_engine_t*>(
      device.hostAlloc(engines_bytes, alignof(abce_device_engine_t),
                       Device::MemorySegment::kNoAtomics, nullptr, false));
  if (device_engines_staging_ != nullptr) {
    std::memset(device_engines_staging_, 0, engines_bytes);
  }
  device_engines_ = static_cast<abce_device_engine_t*>(device.deviceLocalAlloc(engines_bytes));
  if (device_engines_staging_ == nullptr || device_engines_ == nullptr) {
    ReleaseDeviceEngines();
    return false;
  }
  return true;
}

void* AbceEngineSet::AllocZeroedDeviceMemory(size_t bytes) const {
  // Device memory is not reliably host-writable without a large BAR, so a
  // zeroed host block seeds it.
  void* device_block = device_->deviceLocalAlloc(bytes);
  void* zeroed_block =
      device_->hostAlloc(bytes, 64, Device::MemorySegment::kNoAtomics, nullptr, false);
  bool seeded = false;
  if (device_block != nullptr && zeroed_block != nullptr) {
    std::memset(zeroed_block, 0, bytes);
    seeded = Hsa::memory_copy(device_block, zeroed_block, bytes) == HSA_STATUS_SUCCESS;
  }
  if (zeroed_block != nullptr) {
    device_->hostFree(zeroed_block, bytes);
  }
  if (!seeded && device_block != nullptr) {
    device_->memFree(device_block, bytes);
    device_block = nullptr;
  }
  return device_block;
}

bool AbceEngineSet::SeedDeviceRingControls() {
  if (device_controls_seeded_) {
    return device_ring_controls_ != nullptr;
  }
  device_controls_seeded_ = true;
  // Device memory, since only kernels touch these; zeroed, as a fresh queue
  // starts at write index zero.
  device_ring_controls_ =
      static_cast<abce_ring_control_t*>(AllocZeroedDeviceMemory(kDeviceRingControlBytes));
  if (device_ring_controls_ == nullptr) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: device ring cursors unavailable; device submission disabled");
  }
  return device_ring_controls_ != nullptr;
}

uint32_t AbceEngineSet::RankDeviceEngines(abce_copy_endpoint_t src, abce_copy_endpoint_t dst,
                                          uint32_t* ranked) const {
  if (src.kind == ABCE_ENDPOINT_KIND_DEVICE && dst.kind == ABCE_ENDPOINT_KIND_DEVICE &&
      src.device_id != dst.device_id) {
    const uint32_t peer = src.device_id == local_driver_node_id_ ? dst.device_id : src.device_id;
    src = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, local_driver_node_id_);
    dst = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, peer);
  }
  // The policy ranks the rings the orchestrator plans host copies on; one it
  // omits is reserved for other traffic.
  const uint64_t registered = (uint64_t{1} << num_engines_) - 1;
  if (const abce_sdma_engine_policy_t* policy = orchestrator_->sdma_policy(); policy != nullptr) {
    const abce_policy_result_t result =
        abce_sdma_policy_rank_engines(policy, &src, &dst, registered, ranked, kMaxEngines);
    if (result.disposition == ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE) {
      return 0;
    }
    if (result.disposition == ABCE_POLICY_DISPOSITION_RANKED) {
      return result.count;
    }
  }
  for (uint32_t engine_idx = 0; engine_idx < num_engines_; ++engine_idx) {
    ranked[engine_idx] = engine_idx;
  }
  return num_engines_;
}

bool AbceEngineSet::CreateDeviceRing(uint32_t engine_idx) {
  // Only kernels write this ring, so it lives in device memory whether or not
  // the CPU could reach it.
  hsa_amd_queue_create_desc_t desc = {};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.flags = HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF;
  desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
  desc.queue_size_bytes = kDeviceRingBytes;
  desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  desc.engine.sdma.sdma_engine_id = engine_idx;
  hsa_status_t status = Hsa::amd_queue_create(device_->getBackendDevice(), &desc, 1);
  if (status != HSA_STATUS_SUCCESS || desc.queue == nullptr) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: device ring create on engine %u failed (status %d)", engine_idx, status);
    return false;
  }

  hsa_queue_t* queue = desc.queue;
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  uint64_t doorbell_pointer = 0;
  // Only ROCr can hand out the queue's raw control words, and only for
  // directly submittable queue types.
  if (Hsa::amd_queue_get_info(queue, HSA_AMD_QUEUE_INFO_READ_POINTER, &read_pointer) !=
          HSA_STATUS_SUCCESS ||
      Hsa::amd_queue_get_info(queue, HSA_AMD_QUEUE_INFO_WRITE_POINTER, &write_pointer) !=
          HSA_STATUS_SUCCESS ||
      Hsa::amd_queue_get_info(queue, HSA_AMD_QUEUE_INFO_DOORBELL_ID, &doorbell_pointer) !=
          HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: device ring on engine %u does not expose its queue addresses", engine_idx);
    Hsa::queue_destroy(queue);
    return false;
  }

  abce_device_ring_t ring{};
  ring.base = static_cast<uint32_t*>(queue->base_address);
  ring.size = static_cast<uint64_t>(queue->size);
  ring.read_ptr = reinterpret_cast<volatile uint64_t*>(read_pointer);
  ring.write_ptr = reinterpret_cast<volatile uint64_t*>(write_pointer);
  ring.doorbell = reinterpret_cast<volatile uint64_t*>(doorbell_pointer);
  ring.control = &device_ring_controls_[engine_idx];
  abce_device_engine_t& engine = device_engines_staging_[engine_idx];
  const auto abandon = [&](const char* reason) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "ABCE: device ring on engine %u: %s", engine_idx,
            reason);
    std::memset(&engine, 0, sizeof(engine));
    Hsa::queue_destroy(queue);
    return false;
  };
  if (!abce_status_is_ok(abce_device_engine_initialize(builder_.get(), &ring, engine_idx,
                                                       device_atomic_support_, &engine))) {
    return abandon("this ISA's copies need GCR packets");
  }
  // A blocking copy, and no handle names the engine until it returns, so a
  // kernel never observes a half-written descriptor.
  if (Hsa::memory_copy(&device_engines_[engine_idx], &engine, sizeof(engine)) !=
      HSA_STATUS_SUCCESS) {
    return abandon("publishing its descriptor failed");
  }
  device_queues_[engine_idx] = queue;
  ++num_device_rings_;
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "ABCE device ring %u of %u created on engine %u",
          num_device_rings_, static_cast<uint32_t>(GPU_MAX_SDMA_DEVICE_QUEUES), engine_idx);
  return true;
}

uint32_t AbceEngineSet::GetDeviceHandles(abce_copy_endpoint_t src, abce_copy_endpoint_t dst,
                                         uint32_t max_handles, abce_device_handle_t* handles) {
  std::scoped_lock lock(device_lock_);
  if (device_engines_ == nullptr || !SeedDeviceRingControls()) {
    return 0;
  }
  uint32_t ranked[kMaxEngines];
  const uint32_t num_ranked = RankDeviceEngines(src, dst, ranked);
  uint32_t num_handles = 0;
  for (uint32_t rank = 0; rank < num_ranked && num_handles < max_handles; ++rank) {
    const uint32_t engine_idx = ranked[rank];
    if (device_queues_[engine_idx] == nullptr) {
      if (((failed_device_engines_ >> engine_idx) & 1u) ||
          num_device_rings_ >= GPU_MAX_SDMA_DEVICE_QUEUES) {
        continue;
      }
      if (!CreateDeviceRing(engine_idx)) {
        failed_device_engines_ |= uint64_t{1} << engine_idx;
        continue;
      }
    }
    abce_device_handle_t& handle = handles[num_handles++];
    handle.engine = &device_engines_[engine_idx];
    handle.abi_version = ABCE_DEVICE_ABI_VERSION;
  }
  return num_handles;
}

void AbceEngineSet::ReleaseDeviceEngines() {
  if (device_ == nullptr) {
    return;
  }
  const size_t engines_bytes = sizeof(abce_device_engine_t) * kMaxEngines;
  if (device_engines_ != nullptr) {
    device_->memFree(device_engines_, engines_bytes);
    device_engines_ = nullptr;
  }
  if (device_engines_staging_ != nullptr) {
    device_->hostFree(device_engines_staging_, engines_bytes);
    device_engines_staging_ = nullptr;
  }
  if (device_ring_controls_ != nullptr) {
    device_->memFree(device_ring_controls_, kDeviceRingControlBytes);
    device_ring_controls_ = nullptr;
  }
}

void AbceEngineSet::DrainDeviceRing(uint32_t engine_idx) const {
  // The ring's cursors are in device memory, out of the host's reach. The
  // queue's own pointers are host-visible and answer the same question, since a
  // kernel advances the write pointer before ringing the doorbell and nothing is
  // left to submit by the time this runs. A wait packet whose condition never
  // comes true would hold it forever, so give up after a while.
  constexpr uint64_t kDrainLimitNanos = 2ull * 1000 * 1000 * 1000;
  const abce_device_ring_t& ring = device_engines_staging_[engine_idx].ring;
  const uint64_t start = amd::Os::timeNanos();
  while (*ring.read_ptr < *ring.write_ptr) {
    if (amd::Os::timeNanos() - start > kDrainLimitNanos) {
      ClPrint(amd::LOG_WARNING, amd::LOG_QUEUE,
              "ABCE: device ring on engine %u still busy at teardown (read 0x%lx, write 0x%lx)",
              engine_idx, *ring.read_ptr, *ring.write_ptr);
      return;
    }
    abce_ring_pause();
  }
}

void AbceEngineSet::DestroyQueues() {
  for (uint32_t engine_idx = 0; engine_idx < num_engines_; ++engine_idx) {
    // A kernel may have published packets the engine has not drained yet, and
    // nothing else joins device submissions before the queue goes away.
    if (device_queues_[engine_idx] != nullptr) {
      DrainDeviceRing(engine_idx);
      Hsa::queue_destroy(device_queues_[engine_idx]);
      device_queues_[engine_idx] = nullptr;
    }
    if (queues_[engine_idx] != nullptr) {
      abce_hsa_queue_ring_drain(&rings_[engine_idx]);
      Hsa::queue_destroy(queues_[engine_idx]);
      queues_[engine_idx] = nullptr;
    }
  }
  num_engines_ = 0;
}

AbceEngineSet::~AbceEngineSet() {
  DestroyQueues();
  // After the queues: the device descriptors reference their rings.
  ReleaseDeviceEngines();
}

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
