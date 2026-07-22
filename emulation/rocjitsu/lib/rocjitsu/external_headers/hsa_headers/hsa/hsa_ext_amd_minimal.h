// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_ext_amd_minimal.h
/// @brief Minimal AMD HSA extension ABI mirror used by rocjitsu hooks.

#pragma once

#include "hsa/hsa.h"

#include <cstddef>
#include <cstdint>

/// @brief Minimal AMD extension memory-pool handle mirror.
struct hsa_amd_memory_pool_s {
  uint64_t handle;
};
using hsa_amd_memory_pool_t = hsa_amd_memory_pool_s;

using hsa_amd_memory_pool_info_t = uint32_t;
using hsa_amd_agent_memory_pool_info_t = uint32_t;
using hsa_amd_sdma_engine_id_t = uint32_t;
using hsa_amd_pointer_type_t = uint32_t;
using hsa_amd_copy_direction_t = uint32_t;

/// @brief Minimal profiling dispatch-time result mirror.
struct hsa_amd_profiling_dispatch_time_t {
  uint64_t start;
  uint64_t end;
};

// Public hsa_ext_amd.h enum values mirrored by name so ABI renumbering is visible.
inline constexpr hsa_status_t HSA_STATUS_ERROR_INVALID_MEMORY_POOL = static_cast<hsa_status_t>(40);
inline constexpr uint32_t HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 0xA004;
inline constexpr hsa_amd_agent_memory_pool_info_t HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0;
inline constexpr hsa_amd_memory_pool_info_t HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0;
inline constexpr hsa_amd_memory_pool_info_t HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1;
inline constexpr hsa_amd_memory_pool_info_t HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5;
inline constexpr hsa_amd_memory_pool_info_t HSA_AMD_MEMORY_POOL_INFO_LOCATION = 17;

/// @brief Minimal pitched pointer mirror for hsa_amd_memory_async_copy_rect.
struct hsa_pitched_ptr_t {
  void *base;
  size_t pitch;
  size_t slice;
};

/// @brief Minimal AMD virtual-memory access descriptor mirror.
struct hsa_amd_memory_access_desc_t {
  hsa_access_permission_t permissions;
  hsa_agent_t agent_handle;
};

/// @brief Minimal AMD pointer info mirror used to rewrite app-facing agent fields.
struct hsa_amd_pointer_info_t {
  uint32_t size;
  hsa_amd_pointer_type_t type;
  void *agentBaseAddress;
  void *hostBaseAddress;
  size_t sizeInBytes;
  void *userData;
  hsa_agent_t agentOwner;
  uint32_t global_flags;
  bool registered;
  uint32_t alloc_flags;
};
static_assert(offsetof(hsa_amd_pointer_info_t, registered) == 52);
static_assert(offsetof(hsa_amd_pointer_info_t, alloc_flags) == 56);
static_assert(sizeof(hsa_amd_pointer_info_t) == 64);

/// @brief Operation type values used by hsa_amd_memory_async_batch_copy.
enum hsa_amd_memory_copy_op_type_t : uint16_t {
  HSA_AMD_MEMORY_COPY_OP_LINEAR = 0,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST = 1,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP = 2,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC = 3,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST = 4,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST = 5,
};

/// @brief Minimal layout mirror of ROCR's batch-copy operation descriptor.
///
/// @details The HSA hook copies these descriptors only to rewrite agent
/// handles from guest to host before forwarding the call. Field order and
/// widths must match hsa_ext_amd.h.
struct hsa_amd_memory_copy_op_t {
  uint16_t version;
  uint16_t type;
  uint16_t num_entries;
  uint16_t traffic_class;
  hsa_signal_t completion_signal;
  union {
    void *src;
    void **src_list;
  };
  union {
    hsa_agent_t src_agent;
    hsa_agent_t *src_agent_list;
  };
  union {
    hsa_agent_t dst_agent;
    hsa_agent_t *dst_agent_list;
  };
  union {
    void *dst;
    void **dst_list;
  };
  union {
    struct {
      size_t size;
      size_t unused_size;
    };
    struct {
      size_t src_size;
      size_t dst_size;
    };
    struct {
      size_t *size_list;
      size_t reserved0;
    };
  };
  struct {
    uint16_t function;
    uint16_t scope;
    uint32_t reserved;
    void *addr;
    uint64_t value;
    uint64_t mask;
  } wait;
  struct {
    uint16_t operation;
    uint16_t scope;
    uint32_t reserved;
    void *addr;
    uint64_t data;
  } signal;
  uint64_t reserved1[1];
};
static_assert(offsetof(hsa_amd_memory_copy_op_t, version) == 0);
static_assert(offsetof(hsa_amd_memory_copy_op_t, type) == 2);
static_assert(offsetof(hsa_amd_memory_copy_op_t, num_entries) == 4);
static_assert(offsetof(hsa_amd_memory_copy_op_t, traffic_class) == 6);
static_assert(offsetof(hsa_amd_memory_copy_op_t, completion_signal) == 8);
static_assert(offsetof(hsa_amd_memory_copy_op_t, src) == 16);
static_assert(offsetof(hsa_amd_memory_copy_op_t, src_agent) == 24);
static_assert(offsetof(hsa_amd_memory_copy_op_t, dst_agent) == 32);
static_assert(offsetof(hsa_amd_memory_copy_op_t, dst) == 40);
static_assert(offsetof(hsa_amd_memory_copy_op_t, size) == 48);
static_assert(offsetof(hsa_amd_memory_copy_op_t, wait) == 64);
static_assert(offsetof(hsa_amd_memory_copy_op_t, signal) == 96);
static_assert(offsetof(hsa_amd_memory_copy_op_t, reserved1) == 120);
static_assert(sizeof(hsa_amd_memory_copy_op_t) == 128);

using hsa_amd_agent_iterate_memory_pools_fn_t = hsa_status_t(HSA_API *)(
    hsa_agent_t, hsa_status_t (*)(hsa_amd_memory_pool_t memory_pool, void *data), void *);
using hsa_amd_memory_pool_get_info_fn_t = hsa_status_t(HSA_API *)(hsa_amd_memory_pool_t,
                                                                  hsa_amd_memory_pool_info_t,
                                                                  void *);
using hsa_amd_agent_memory_pool_get_info_fn_t = hsa_status_t(HSA_API *)(
    hsa_agent_t, hsa_amd_memory_pool_t, hsa_amd_agent_memory_pool_info_t, void *);
using hsa_amd_memory_pool_allocate_fn_t = hsa_status_t(HSA_API *)(hsa_amd_memory_pool_t, size_t,
                                                                  uint32_t, void **);
using hsa_amd_memory_pool_free_fn_t = hsa_status_t(HSA_API *)(void *);
using hsa_amd_profiling_set_profiler_enabled_fn_t = hsa_status_t(HSA_API *)(hsa_queue_t *, int);
using hsa_amd_profiling_get_dispatch_time_fn_t =
    hsa_status_t(HSA_API *)(hsa_agent_t, hsa_signal_t, hsa_amd_profiling_dispatch_time_t *);
using hsa_amd_profiling_convert_tick_to_system_domain_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t,
                                                                                     uint64_t,
                                                                                     uint64_t *);
using hsa_amd_agents_allow_access_fn_t = hsa_status_t(HSA_API *)(uint32_t, const hsa_agent_t *,
                                                                 const uint32_t *, const void *);
using hsa_amd_memory_async_copy_fn_t = hsa_status_t(HSA_API *)(void *, hsa_agent_t, const void *,
                                                               hsa_agent_t, size_t, uint32_t,
                                                               const hsa_signal_t *, hsa_signal_t);
using hsa_amd_memory_async_copy_on_engine_fn_t =
    hsa_status_t(HSA_API *)(void *, hsa_agent_t, const void *, hsa_agent_t, size_t, uint32_t,
                            const hsa_signal_t *, hsa_signal_t, hsa_amd_sdma_engine_id_t, bool);
using hsa_amd_memory_copy_engine_status_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t, hsa_agent_t,
                                                                       uint32_t *);
using hsa_amd_memory_get_preferred_copy_engine_fn_t =
    hsa_status_t(HSA_API *)(hsa_agent_t, hsa_agent_t, hsa_amd_sdma_engine_id_t *);
using hsa_amd_memory_lock_fn_t = hsa_status_t(HSA_API *)(void *, size_t, hsa_agent_t *, int,
                                                         void **);
using hsa_amd_memory_fill_fn_t = hsa_status_t(HSA_API *)(void *, uint32_t, size_t);
using hsa_amd_pointer_info_fn_t = hsa_status_t(HSA_API *)(const void *, hsa_amd_pointer_info_t *,
                                                          void *(*)(size_t), uint32_t *,
                                                          hsa_agent_t **);
using hsa_amd_memory_lock_to_pool_fn_t = hsa_status_t(HSA_API *)(void *, size_t, hsa_agent_t *, int,
                                                                 hsa_amd_memory_pool_t, uint32_t,
                                                                 void **);
using hsa_amd_svm_prefetch_async_fn_t = hsa_status_t(HSA_API *)(void *, size_t, hsa_agent_t,
                                                                uint32_t, const hsa_signal_t *,
                                                                hsa_signal_t);
using hsa_amd_vmem_set_access_fn_t = hsa_status_t(HSA_API *)(void *, size_t,
                                                             const hsa_amd_memory_access_desc_t *,
                                                             size_t);
using hsa_amd_vmem_get_access_fn_t = hsa_status_t(HSA_API *)(void *, hsa_access_permission_t *,
                                                             hsa_agent_t);
using hsa_amd_memory_async_copy_rect_fn_t = hsa_status_t(HSA_API *)(
    const hsa_pitched_ptr_t *, const hsa_dim3_t *, const hsa_pitched_ptr_t *, const hsa_dim3_t *,
    const hsa_dim3_t *, hsa_agent_t, hsa_amd_copy_direction_t, uint32_t, const hsa_signal_t *,
    hsa_signal_t);
using hsa_amd_agent_set_async_scratch_limit_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t, size_t);
using hsa_amd_memory_async_batch_copy_fn_t = hsa_status_t(HSA_API *)(
    const hsa_amd_memory_copy_op_t *, uint32_t, uint32_t, const hsa_signal_t *);
using hsa_amd_agent_preload_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t, uint64_t);
