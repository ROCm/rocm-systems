/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Device-level ABCE (Accelerated Blit Copy Engine) plumbing for the ROCm CLR
// backend. Owns one host-memory ring per host-facing and xGMI SDMA engine and a
// single CopyOrchestrator shared by every stream on the device.

// rocrctx.hpp pulls in the HSA headers (including amd_hsa_signal.h, which defines
// AMD_HSA_SIGNAL_H) that the ABCE adapters rely on, so it must precede them.
#include "device/rocm/rocrctx.hpp"

#include <array>
#include <memory>
#include <unordered_map>

#include "abce_cxx.h"
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

 private:
  explicit AbceEngineSet(const abce_isa_version_t& isa) : builder_(isa) {}

  bool Init(Device& device);
  void DestroyQueues();

  abce::Builder builder_;
  std::unique_ptr<abce::CopyOrchestrator> orchestrator_;
  std::array<hsa_queue_t*, kMaxEngines> queues_{};
  std::array<abce_hsa_queue_ring_t, kMaxEngines> rings_{};
  uint32_t num_engines_ = 0;
  EngineAllocator engine_allocator_;
};

}  // namespace amd::roc
