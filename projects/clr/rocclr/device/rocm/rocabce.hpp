/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Device-level ABCE (Accelerated Blit Copy Engine) plumbing for the ROCm CLR
// backend. Owns one host-memory ring per host-facing and xGMI SDMA engine and a
// single CopyOrchestrator shared by every stream on the device, plus, once a
// kernel asks for them, a separate device-memory ring per engine for kernels.

// rocrctx.hpp pulls in the HSA headers (including amd_hsa_signal.h, which defines
// AMD_HSA_SIGNAL_H) that the ABCE adapters rely on, so it must precede them.
#include "device/rocm/rocrctx.hpp"

#include <array>
#include <memory>
#include <unordered_map>

#include "abce_cxx.h"
#include "abce_device.h"
#include "abce_hsa.h"
#include "thread/monitor.hpp"

namespace amd::roc {

class Device;
class VirtualGPU;

//! Builds an abce_hsa_queue_api_t from CLR's dynamically-loaded ROCr entry
//! points. CLR may load ROCr at runtime, so ABCE's abce_hsa_queue_api_direct()
//! (which references the hsa_* symbols directly) cannot be used.
abce_hsa_queue_api_t MakeAbceQueueApi();

//! Device-owned set of ABCE SDMA rings plus the orchestrator that drives them.
//!
//! Created once per roc::Device (like the AQL queue pool) and shared across all
//! VirtualGPUs. Ring reservation/commit is CAS-based, so concurrent producers
//! from multiple streams are safe on the host rings.
class AbceEngineSet {
 public:
  //! Upper bound on SDMA engines wired into ABCE (host + xGMI).
  static constexpr uint32_t kMaxEngines = ABCE_MAX_ENGINES;

  AbceEngineSet(const AbceEngineSet&) = delete;
  AbceEngineSet& operator=(const AbceEngineSet&) = delete;

  ~AbceEngineSet();

  //! Creates the engine set for @p device. Returns nullptr when the feature is
  //! disabled, the runtime lacks by-engine-id SDMA queue creation, or no ring
  //! could be brought up. The device keeps running on the default path in that
  //! case.
  static std::unique_ptr<AbceEngineSet> Create(Device& device);

  //! Returned when a copy has no legal ring at all.
  static constexpr uint32_t kNoEngine = kMaxEngines;

  //! The device-shared orchestrator. Valid for the lifetime of this object.
  abce::CopyOrchestrator& orchestrator() { return *orchestrator_; }

  //! Number of SDMA rings successfully registered with the orchestrator.
  uint32_t num_engines() const { return num_engines_; }

  //! Prefers to keep a stream on one legal ring. HwQueueTracker may elide the
  //! preceding ABCE signal only when its stored coordinator ring is identical.
  //!
  //! Mirrors Device::SdmaEngineAllocator, which does the same for the ROCr blit
  //! path and likewise drops the reservation when the stream goes idle.
  class EngineAllocator {
   public:
    //! Ring for @p vgpu to run this copy on, chosen from @p ranked (best first, as
    //! returned by MapCopy). Keeps the stream's current ring whenever it appears in
    //! @p ranked; otherwise prefers one no other stream holds and falls back to
    //! round-robin. Returns kNoEngine when @p num_ranked is 0.
    uint32_t Acquire(const VirtualGPU* vgpu, const uint32_t* ranked, uint32_t num_ranked);

    //! Drops @p vgpu's reservation, freeing the ring for other streams.
    void Release(const VirtualGPU* vgpu);

   private:
    amd::Monitor lock_;
    std::unordered_map<const VirtualGPU*, uint32_t> vgpu_to_engine_;
    uint32_t next_round_robin_ = 0;
  };

  EngineAllocator& engine_allocator() { return engine_allocator_; }

  //! Handle a kernel submits copies through (abce_device.h).
  //!
  //! The first call creates the device rings, so a process that never submits
  //! from a kernel never pays for them. False, leaving @p handle empty, when
  //! device submission is unavailable: it needs a supported ISA, and at least one
  //! engine whose extra queue ROCr can create, exposes the read/write/doorbell
  //! addresses of, and gives a doorbell a shader can ring. Host copies keep
  //! working either way.
  bool GetDeviceHandle(abce_device_handle_t& handle);

  //! Publishes the client's device-ordinal -> agent-id correspondence, so a
  //! kernel can name a copy's endpoints by the client's own ordinals. Only ring
  //! affinity depends on it, so leaving it uncalled costs bandwidth and never
  //! correctness.
  //!
  //! Idempotent, and ignores a second call: the mapping is a property of the
  //! process's device set, not of the caller.
  void ConfigureDevicePeers(const uint32_t* driver_node_ids, uint32_t num_devices);

 private:
  explicit AbceEngineSet(const abce_isa_version_t& isa) : builder_(isa) {}

  bool Init(Device& device);
  void DestroyQueues();

  //! Waits out a device ring at teardown through the queue's own pointers, since
  //! its cursors are in device memory where the host cannot read them.
  void DrainDeviceRing(uint32_t engine_idx) const;

  //! Creates a device ring on every engine that can have one — a second SDMA
  //! queue beside the host ring — and publishes the view. Runs once, under
  //! device_view_lock_. Returns whether kernels can submit.
  bool CreateDeviceRings();

  //! Allocates the device-readable view. Returns false, leaving it null, when
  //! device submission is unsupported; the host path is unaffected either way.
  bool InitDeviceView(Device& device);
  void ReleaseDeviceView();

  //! Asks the orchestrator's SDMA policy which rings serve each transfer class,
  //! so a kernel reaches the ring MapCopy would have chosen and never one the
  //! policy reserves. Must run after every ring is registered: the policy ranks
  //! the registered set.
  void BuildDeviceRoutes();

  //! Copies the staged view into device memory, making the runtime's edits
  //! visible to kernels. Cheap and idempotent, so a later edit just re-publishes.
  bool PublishDeviceView();

  abce::Builder builder_;
  std::unique_ptr<abce::CopyOrchestrator> orchestrator_;
  std::array<hsa_queue_t*, kMaxEngines> queues_{};
  std::array<abce_hsa_queue_ring_t, kMaxEngines> rings_{};
  uint32_t num_engines_ = 0;
  EngineAllocator engine_allocator_;

  //! Owning device that allocated the blocks below; needed at teardown.
  Device* device_ = nullptr;
  //! Device rings by engine; null where an engine has none.
  std::array<hsa_queue_t*, kMaxEngines> device_queues_{};
  bool device_rings_created_ = false;
  bool device_view_published_ = false;
  //! Cursors for the device rings, indexed by engine. Only kernels touch them,
  //! so they sit in device memory and a submit's cursor traffic stays on the
  //! device.
  abce_ring_control_t* device_ring_controls_ = nullptr;
  //! The view a kernel reads, in DEVICE memory: a submit makes several dependent
  //! reads of it, and from host memory each costs roughly ten times as much.
  abce_device_engine_set_t* engine_set_device_ = nullptr;
  //! The runtime's own copy, built and later edited here and published to
  //! engine_set_device_. Kept because ConfigureDevicePeers() edits it after
  //! Init() has returned.
  abce_device_engine_set_t* engine_set_staging_ = nullptr;
  //! Guards the peer map and the device rings against concurrent publication:
  //! unlike the routes they are written after Init() by whichever thread asks
  //! for a handle first.
  amd::Monitor device_view_lock_;
  bool device_peers_configured_ = false;
};

}  // namespace amd::roc
