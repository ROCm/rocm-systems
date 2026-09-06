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

//! Fills @p engine_order with the first @p num_engines engines in the order
//! kernel copies favor them: the preferred engine of the local, host and each
//! published peer route, then the rest ascending. Returns how many it wrote. A
//! device-queue limit then keeps the engines kernel copies are steered to.
uint32_t DeviceRingOrder(const abce_device_engine_set_t& view, uint32_t num_engines,
                         std::array<uint32_t, AbceEngineSet::kMaxEngines>& engine_order) {
  uint32_t num_ordered = 0;
  uint64_t ordered_engines = 0;
  auto append = [&](uint32_t engine_idx) {
    if (engine_idx >= num_engines || ((ordered_engines >> engine_idx) & 1u)) return;
    ordered_engines |= uint64_t{1} << engine_idx;
    engine_order[num_ordered++] = engine_idx;
  };
  append(view.local_to_local.preferred);
  append(view.host_to_local.preferred);
  append(view.local_to_host.preferred);
  for (uint32_t ordinal = 0; ordinal < ABCE_DEVICE_MAX_PEERS; ++ordinal) {
    const int8_t peer_agent = view.agent_for_device[ordinal];
    if (peer_agent >= 0 && peer_agent != view.local_agent) {
      append(view.local_to_peer[peer_agent].preferred);
    }
  }
  for (uint32_t engine_idx = 0; engine_idx < num_engines; ++engine_idx) append(engine_idx);
  return num_ordered;
}

//! Agent id ABCE knows a KFD node by. The host path names devices by driver node
//! id directly, so the two spaces agree by construction; ids beyond the routes'
//! reach become ABCE_DEVICE_AGENT_UNKNOWN, which places by ring availability.
int8_t AgentIdFor(uint32_t driver_node_id) {
  return driver_node_id < ABCE_DEVICE_MAX_AGENTS ? static_cast<int8_t>(driver_node_id)
                                                 : ABCE_DEVICE_AGENT_UNKNOWN;
}

//! The route for one transfer class, from the policy's ranking over the rings
//! registered with the orchestrator. A registered ring the ranking omits is
//! forbidden, which is how the policy expresses a reservation; a class it has no
//! opinion on stays empty and so leaves every ring open.
abce_device_route_t RouteFor(const abce_sdma_engine_policy_t* policy, uint64_t registered,
                             abce_copy_endpoint_t src, abce_copy_endpoint_t dst) {
  abce_device_route_t route;
  abce_device_route_initialize(&route);
  uint32_t ranked[ABCE_MAX_ENGINES];
  const abce_policy_result_t result =
      abce_sdma_policy_rank_engines(policy, &src, &dst, registered, ranked, ABCE_MAX_ENGINES);
  switch (result.disposition) {
    case ABCE_POLICY_DISPOSITION_NO_PREFERENCE:
      break;
    case ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE:
      route.forbidden = ~uint64_t{0};
      break;
    case ABCE_POLICY_DISPOSITION_RANKED: {
      uint64_t ranked_mask = 0;
      for (uint32_t rank = 0; rank < result.count; ++rank) {
        ranked_mask |= uint64_t{1} << ranked[rank];
      }
      route.forbidden = registered & ~ranked_mask;
      route.preferred = result.count != 0 ? ranked[0] : ABCE_DEVICE_MAX_ENGINES;
      break;
    }
  }
  return route;
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

  // Routes come from the registered host rings, so the view follows them.
  if (InitDeviceView(device)) {
    BuildDeviceRoutes();
    engine_set_staging_->num_engines = num_engines_;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "ABCE registered %u host-facing and %u xGMI SDMA engine(s)",
          std::min(num_engines_, host_sdma),
          num_engines_ > host_sdma ? num_engines_ - host_sdma : 0);
  return true;
}

bool AbceEngineSet::InitDeviceView(Device& device) {
  // Kernel-side copies are opt-in: their queues come out of the per-GPU SDMA
  // queue budget that every process shares.
  if (GPU_MAX_SDMA_DEVICE_QUEUES == 0 || !SupportsDeviceSubmission(builder_.isa())) {
    return false;
  }

  device_ = &device;
  // The runtime builds the view here in host memory, then publishes it into
  // device memory for kernels to read.
  engine_set_staging_ = static_cast<abce_device_engine_set_t*>(
      device.hostAlloc(sizeof(abce_device_engine_set_t), alignof(abce_device_engine_set_t),
                       Device::MemorySegment::kNoAtomics, nullptr, false));
  if (engine_set_staging_ == nullptr) {
    ReleaseDeviceView();
    return false;
  }
  abce_device_engine_set_initialize(builder_.get(), engine_set_staging_);
  engine_set_staging_->device_atomic_support =
      abce_detect_default_platform_caps(builder_.isa()).device_atomic_support;
  engine_set_staging_->local_agent = AgentIdFor(device.info().driverNodeId_);

  engine_set_device_ = static_cast<abce_device_engine_set_t*>(
      device.deviceLocalAlloc(sizeof(abce_device_engine_set_t)));
  if (engine_set_device_ == nullptr) {
    ReleaseDeviceView();
    return false;
  }
  return true;
}

void AbceEngineSet::BuildDeviceRoutes() {
  const abce_sdma_engine_policy_t* policy = orchestrator_->sdma_policy();
  // With no policy there is nothing to route by; the initialized routes leave
  // every usable ring open.
  if (policy == nullptr) {
    return;
  }
  // The rings the orchestrator plans host copies on.
  const uint64_t registered = (uint64_t{1} << num_engines_) - 1;
  // A copy whose ends a kernel does not name gets what the host planner gives a
  // copy between unknown device ids, so it still avoids every reserved ring.
  const abce_copy_endpoint_t unknown_device =
      abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, ABCE_ANY_DEVICE);
  engine_set_staging_->unknown = RouteFor(policy, registered, unknown_device, unknown_device);

  const int8_t local_agent = engine_set_staging_->local_agent;
  // The class routes are relative to this device, so without a nameable local
  // agent every copy takes the unknown route above.
  if (local_agent < 0) {
    return;
  }
  const abce_copy_endpoint_t host =
      abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
  const abce_copy_endpoint_t local =
      abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, static_cast<uint32_t>(local_agent));

  engine_set_staging_->host_to_local = RouteFor(policy, registered, host, local);
  engine_set_staging_->local_to_host = RouteFor(policy, registered, local, host);
  engine_set_staging_->local_to_local = RouteFor(policy, registered, local, local);
  for (uint32_t peer = 0; peer < ABCE_DEVICE_MAX_AGENTS; ++peer) {
    if (peer == static_cast<uint32_t>(local_agent)) {
      continue;
    }
    const abce_copy_endpoint_t peer_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, peer);
    // The copy runs on this device's engines whichever way the data flows, so
    // both directions take the engine this device favors for the link, as
    // AbceBlitManager::normalizePeerEndpoints does for host-issued copies.
    const abce_device_route_t link_route = RouteFor(policy, registered, local, peer_end);
    engine_set_staging_->local_to_peer[peer] = link_route;
    engine_set_staging_->peer_to_local[peer] = link_route;
  }
}

bool AbceEngineSet::CreateDeviceRings() {
  device_rings_created_ = true;

  // Device memory, since only kernels touch these. It is not reliably
  // host-writable without a large BAR, so a zeroed host block seeds it — a fresh
  // queue starts at write index zero.
  device_ring_controls_ =
      static_cast<abce_ring_control_t*>(device_->deviceLocalAlloc(kDeviceRingControlBytes));
  void* zeroed_controls =
      device_->hostAlloc(kDeviceRingControlBytes, alignof(abce_ring_control_t),
                         Device::MemorySegment::kNoAtomics, nullptr, false);
  bool seeded = false;
  if (device_ring_controls_ != nullptr && zeroed_controls != nullptr) {
    std::memset(zeroed_controls, 0, kDeviceRingControlBytes);
    seeded = Hsa::memory_copy(device_ring_controls_, zeroed_controls, kDeviceRingControlBytes) ==
             HSA_STATUS_SUCCESS;
  }
  if (zeroed_controls != nullptr) {
    device_->hostFree(zeroed_controls, kDeviceRingControlBytes);
  }
  if (!seeded) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: device ring cursors unavailable; device submission disabled");
    return false;
  }

  abce_device_engine_set_t& view = *engine_set_staging_;
  std::array<uint32_t, kMaxEngines> engine_order{};
  const uint32_t num_ordered = DeviceRingOrder(view, num_engines_, engine_order);
  uint32_t num_device_rings = 0;
  for (uint32_t order_idx = 0;
       order_idx < num_ordered && num_device_rings < GPU_MAX_SDMA_DEVICE_QUEUES; ++order_idx) {
    const uint32_t engine_idx = engine_order[order_idx];
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
      continue;
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
      continue;
    }

    abce_device_ring_t& device_ring = view.engines[engine_idx];
    device_ring.base = static_cast<uint32_t*>(queue->base_address);
    device_ring.size = static_cast<uint64_t>(queue->size);
    device_ring.read_ptr = reinterpret_cast<volatile uint64_t*>(read_pointer);
    device_ring.write_ptr = reinterpret_cast<volatile uint64_t*>(write_pointer);
    device_ring.doorbell = reinterpret_cast<volatile uint64_t*>(doorbell_pointer);
    device_ring.control = &device_ring_controls_[engine_idx];
    view.device_usable_engines |= uint64_t{1} << engine_idx;
    device_queues_[engine_idx] = queue;
    ++num_device_rings;
  }

  if (view.device_usable_engines == 0) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: no engine could take a device ring; device submission disabled");
    return false;
  }
  if (!PublishDeviceView()) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: publishing the device view failed; device submission disabled");
    return false;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "ABCE device submission enabled on %d of %u engine(s) (mask 0x%lx)",
          abce_popcount_u64(view.device_usable_engines), num_engines_,
          view.device_usable_engines);
  return true;
}

bool AbceEngineSet::PublishDeviceView() {
  // A blocking copy, and the handle is not handed out until it returns, so a
  // kernel never observes a half-written view.
  return Hsa::memory_copy(engine_set_device_, engine_set_staging_,
                          sizeof(abce_device_engine_set_t)) == HSA_STATUS_SUCCESS;
}

bool AbceEngineSet::GetDeviceHandle(abce_device_handle_t& handle) {
  // Shares the lock with ConfigureDevicePeers, which re-publishes the view.
  std::scoped_lock lock(device_view_lock_);
  handle = {};
  if (engine_set_staging_ == nullptr) {
    return false;
  }
  if (!device_rings_created_) {
    device_view_published_ = CreateDeviceRings();
  }
  if (!device_view_published_) {
    return false;
  }
  handle.engine_set = engine_set_device_;
  handle.abi_version = ABCE_DEVICE_ABI_VERSION;
  return true;
}

void AbceEngineSet::ConfigureDevicePeers(const uint32_t* driver_node_ids, uint32_t num_devices) {
  std::scoped_lock lock(device_view_lock_);
  if (engine_set_staging_ == nullptr || device_peers_configured_) {
    return;
  }
  for (uint32_t device_id = 0; device_id < std::min(num_devices, ABCE_DEVICE_MAX_PEERS);
       ++device_id) {
    engine_set_staging_->agent_for_device[device_id] = AgentIdFor(driver_node_ids[device_id]);
  }
  // Before the first handle the device rings' creation publishes this. After
  // it, the peer map only steers placement, so a failed re-publish costs
  // affinity and leaves the rest of the view intact and usable.
  if (device_view_published_ && !PublishDeviceView()) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "ABCE: publishing the peer map failed; kernels place by ring availability alone");
  }
  device_peers_configured_ = true;
}

void AbceEngineSet::ReleaseDeviceView() {
  if (device_ == nullptr) {
    return;
  }
  if (engine_set_device_ != nullptr) {
    device_->memFree(engine_set_device_, sizeof(abce_device_engine_set_t));
    engine_set_device_ = nullptr;
  }
  if (engine_set_staging_ != nullptr) {
    device_->hostFree(engine_set_staging_, sizeof(abce_device_engine_set_t));
    engine_set_staging_ = nullptr;
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
  // left to submit by the time this runs.
  const abce_device_ring_t& ring = engine_set_staging_->engines[engine_idx];
  while (*ring.read_ptr < *ring.write_ptr) {
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
  // After the queues: the device view's descriptors reference their rings.
  ReleaseDeviceView();
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
