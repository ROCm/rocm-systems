/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Device-level ABCE (Accelerated Blit Copy Engine) plumbing for the ROCm CLR
// backend. Owns one host-memory ring per host-facing and xGMI SDMA engine and a
// single CopyOrchestrator shared by every stream on the device, plus, once a
// kernel asks for one, a separate device-memory ring on an engine for kernels.

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

  //! Fills @p handles with up to @p max_handles handles a kernel submits copies
  //! from @p src to @p dst through (abce_device.h), one device ring each, on the
  //! engines the SDMA policy ranks for those ends, best first, among those that
  //! have or can get one. Returns how many it wrote. A device ring is created on
  //! first use, up to GPU_MAX_SDMA_DEVICE_QUEUES in all, so a process that never
  //! submits from a kernel never pays for one. Zero when no ranked engine can
  //! take a device ring: it needs a supported ISA, and an extra queue ROCr can
  //! create, exposes the read/write/doorbell addresses of, and gives a doorbell
  //! a shader can ring. Host copies keep working either way.
  uint32_t GetDeviceHandles(abce_copy_endpoint_t src, abce_copy_endpoint_t dst,
                            uint32_t max_handles, abce_device_handle_t* handles);

 private:
  explicit AbceEngineSet(const abce_isa_version_t& isa) : builder_(isa) {}

  bool Init(Device& device);
  void DestroyQueues();

  //! Waits out a device ring at teardown, for a bounded time, through the
  //! queue's own pointers, since its cursors are in device memory where the
  //! host cannot read them.
  void DrainDeviceRing(uint32_t engine_idx) const;

  //! Allocates the device engine descriptors. Returns false, leaving them null,
  //! when device submission is off or unsupported; the host path is unaffected
  //! either way.
  bool InitDeviceEngines(Device& device);
  void ReleaseDeviceEngines();

  //! Zeroes the device rings' cursors once, before the first ring is created.
  bool SeedDeviceRingControls();

  //! Device-local block of @p bytes, zeroed; null on failure.
  void* AllocZeroedDeviceMemory(size_t bytes) const;

  //! Fills @p ranked with the engines, best first, the SDMA policy allows for a
  //! copy from @p src to @p dst; returns how many. A peer copy ranks as this
  //! device to the peer, as AbceBlitManager::normalizePeerEndpoints does.
  uint32_t RankDeviceEngines(abce_copy_endpoint_t src, abce_copy_endpoint_t dst,
                             uint32_t* ranked) const;

  //! Creates a device ring on @p engine_idx — a second SDMA queue beside the
  //! host ring — and publishes its descriptor. Runs under device_lock_.
  bool CreateDeviceRing(uint32_t engine_idx);

  abce::Builder builder_;
  std::unique_ptr<abce::CopyOrchestrator> orchestrator_;
  std::array<hsa_queue_t*, kMaxEngines> queues_{};
  std::array<abce_hsa_queue_ring_t, kMaxEngines> rings_{};
  uint32_t num_engines_ = 0;
  EngineAllocator engine_allocator_;

  //! Owning device that allocated the blocks below; needed at teardown.
  Device* device_ = nullptr;
  uint32_t local_driver_node_id_ = 0;
  bool device_atomic_support_ = false;
  //! Device rings by engine; null where an engine has none.
  std::array<hsa_queue_t*, kMaxEngines> device_queues_{};
  uint32_t num_device_rings_ = 0;
  //! Engines whose device ring could not be created, so it is not retried.
  uint64_t failed_device_engines_ = 0;
  bool device_controls_seeded_ = false;
  //! Cursors for the device rings, indexed by engine. Only kernels touch them,
  //! so they sit in device memory and a submit's cursor traffic stays on the
  //! device.
  abce_ring_control_t* device_ring_controls_ = nullptr;
  //! The descriptors a kernel reads, by engine, in DEVICE memory: a submit makes
  //! several dependent reads of one, and from host memory each costs roughly
  //! ten times as much.
  abce_device_engine_t* device_engines_ = nullptr;
  //! The runtime's copy, built here and published to device_engines_.
  abce_device_engine_t* device_engines_staging_ = nullptr;
  //! Serializes device ring creation, which whichever thread first asks for a
  //! handle on an engine does.
  amd::Monitor device_lock_;
};

}  // namespace amd::roc
