/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — optional C++ conveniences.
//
// ABCE's interface is C (abce_host.h and below). This header is the C++ layer
// over it, in the same relationship libhrx's hrx_runtime_cxx.h has to
// hrx_runtime.h: it adds nothing to the ABI and every type here is
// layout-identical to the C struct it derives from, so a wrapper converts to its
// C counterpart by plain reference.
//
// It exists for two reasons:
//   * Several ABCE structs have non-zero defaults (a copy endpoint defaults to
//     ABCE_ANY_DEVICE, metadata defaults both coherency triggers and
//     prefer_fused on), so zero-initializing them in C++ is silently wrong. The
//     wrappers below run the matching abce_*_initialize() in their constructor.
//   * The objects with an initialize/use lifecycle read better as C++ types that
//     cannot be copied out from under the callbacks that capture their address.
//
// Nothing here is required: a C++ caller may use the C API directly.

#ifndef ABCE_CXX_H_
#define ABCE_CXX_H_

#include <string>

#include "abce_builder.h"
#include "abce_config.h"
#include "abce_frame.h"
#include "abce_host.h"
#include "abce_ring_host.h"

namespace abce {

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

inline bool IsOk(abce_status_t status) { return abce_status_is_ok(status); }

inline std::string FormatStatus(abce_status_t status) { return abce_status_name(status); }

//===----------------------------------------------------------------------===//
// Default-initialized value types
//
// Each derives from its C struct purely to run the initializer; the layout and
// the field names are the C ones, so `abce_copy_op_t& op = my_copy_op;` and
// passing `&my_copy_op` to a C entry point both work.
//===----------------------------------------------------------------------===//

struct BuilderConfig : abce_builder_config_t {
  BuilderConfig() { abce_builder_config_initialize(this); }
};

struct CopyOp : abce_copy_op_t {
  CopyOp() { abce_copy_op_initialize(this); }
};

struct CopyMetadata : abce_copy_metadata_t {
  CopyMetadata() { abce_copy_metadata_initialize(this); }
};

struct EngineAffinity : abce_engine_affinity_t {
  EngineAffinity() { abce_engine_affinity_initialize(this); }
};

struct Plan : abce_plan_t {
  Plan() { abce_plan_initialize(this); }
};

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

class Builder {
 public:
  explicit Builder(abce_isa_version_t isa, const abce_builder_config_t* config = nullptr)
      : status_(abce_builder_initialize(isa, config, &builder_)) {}

  // What abce_builder_initialize() returned. Anything but ABCE_STATUS_OK means
  // |config| asked for limits the packets cannot carry and the builder is unusable;
  // the default config always succeeds.
  abce_status_t status() const { return status_; }
  const abce_builder_t* get() const { return &builder_; }
  const abce_isa_version_t& isa() const { return builder_.isa; }
  bool is_gfx125plus() const { return abce_builder_is_gfx125plus(&builder_); }
  size_t max_copy_size() const { return abce_builder_max_copy_size(&builder_); }
  bool requires_gcr() const { return abce_builder_requires_gcr(&builder_); }

 private:
  // Declared ahead of status_ so its zeroing runs before the initialize call that
  // status_'s initializer makes.
  abce_builder_t builder_{};
  abce_status_t status_;
};

//===----------------------------------------------------------------------===//
// CopyOrchestrator
//
// Non-copyable and non-movable: registered rings and the built-in policy are
// reached through pointers into this object, so it must keep its address.
//===----------------------------------------------------------------------===//

class CopyOrchestrator {
 public:
  explicit CopyOrchestrator(const Builder& builder) {
    abce_copy_orchestrator_initialize(builder.get(), &orchestrator_);
  }

  CopyOrchestrator(const Builder& builder, abce_platform_caps_t caps) {
    abce_copy_orchestrator_initialize_with_caps(builder.get(), caps, &orchestrator_);
  }

  CopyOrchestrator(const CopyOrchestrator&) = delete;
  CopyOrchestrator& operator=(const CopyOrchestrator&) = delete;
  CopyOrchestrator(CopyOrchestrator&&) = delete;
  CopyOrchestrator& operator=(CopyOrchestrator&&) = delete;

  abce_copy_orchestrator_t* get() { return &orchestrator_; }

  abce_sdma_engine_policy_t* sdma_policy() {
    return abce_copy_orchestrator_sdma_policy(&orchestrator_);
  }

  void InitDeviceProfile(uint32_t total_sdma_engines, uint32_t num_non_xgmi_sdma_engines = 0) {
    abce_copy_orchestrator_init_device_profile(&orchestrator_, total_sdma_engines,
                                               num_non_xgmi_sdma_engines);
  }

  bool RegisterEngine(uint32_t index, abce_ring_t* ring,
                      const abce_engine_affinity_t* affinity = nullptr) {
    return abce_copy_orchestrator_register_engine(&orchestrator_, index, ring, affinity);
  }

  void SetEnginePolicy(abce_engine_policy_fn_t policy, void* user_data = nullptr) {
    abce_copy_orchestrator_set_engine_policy(&orchestrator_, policy, user_data);
  }

  const abce_engine_affinity_t& EngineAffinityFor(uint32_t index) const {
    return *abce_copy_orchestrator_engine_affinity(&orchestrator_, index);
  }

  uint32_t RankLegalEngines(const abce_copy_op_t& copy, uint32_t* engines,
                            uint32_t capacity) const {
    return abce_copy_orchestrator_rank_legal_engines(&orchestrator_, &copy, engines, capacity);
  }

  uint32_t DescribeExecution(const abce_plan_t& plan) const {
    return abce_copy_orchestrator_describe_execution(&orchestrator_, &plan);
  }

  abce_status_t MapCopy(const abce_copy_op_t* copies, uint32_t num_copies,
                        const abce_copy_metadata_t& metadata, abce_plan_t* out_plan) {
    return abce_copy_orchestrator_map_copy(&orchestrator_, copies, num_copies, &metadata,
                                           out_plan);
  }

  abce_status_t Submit(abce_plan_t& plan) {
    return abce_copy_orchestrator_submit(&orchestrator_, &plan);
  }

  abce_status_t Dispatch(const abce_copy_op_t* copies, uint32_t num_copies,
                         const abce_copy_metadata_t& metadata) {
    return abce_copy_orchestrator_dispatch(&orchestrator_, copies, num_copies, &metadata);
  }

 private:
  abce_copy_orchestrator_t orchestrator_;
};

}  // namespace abce

#endif  // ABCE_CXX_H_
