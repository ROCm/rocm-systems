// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_lds.h
/// @brief Hook-side state used to dispatch virtual-LDS kernel variants.

#pragma once

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu::hooks {

/// @brief Virtual-LDS policy and kernarg facts consumed by the HSA dispatch path.
struct VirtualLdsKernelRuntimeMetadata {
  std::string kernel_name;
  std::string sidecar_variant_name;
  uint32_t static_lds_bytes = 0;
  uint32_t normal_private_segment_size = 0;
  uint32_t virtual_private_segment_size = 0;
  uint32_t kernarg_size = 0;
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

/// @brief Independently parsed sidecar and virtual-LDS runtime records.
struct ParsedVirtualLdsHookMetadata {
  std::vector<SidecarVariantMetadata> sidecars;
  std::vector<VirtualLdsKernelRuntimeMetadata> virtual_lds;
};

/// @brief Parse and join the metadata needed by hook-side virtual LDS.
[[nodiscard]] std::optional<ParsedVirtualLdsHookMetadata>
parse_virtual_lds_hook_metadata(std::span<const uint8_t> image);

/// @brief Validate the load-time-determinable virtual-LDS invariants.
///
/// @details Every dispatch of a virtual-LDS kernel needs the runtime-state-block
/// ABI and a kernarg wrapper layout whose backing-pointer offset matches the
/// descriptor's. Those facts are fixed per kernel (independent of the AQL packet),
/// so a violation means the code object can never be dispatched correctly. It is
/// detected here, at load, and the load is failed — rather than at dispatch, where
/// the only recovery would be to submit a faulting packet or stall the queue.
/// Dispatch-time-only failures (backing-allocation OOM, per-packet grid overflow)
/// cannot be decided here and are handled on the dispatch path.
///
/// @returns The kernel name of the first unsupportable record, or std::nullopt
/// when every virtual-LDS record is load-time valid.
[[nodiscard]] std::optional<std::string>
first_unsupportable_virtual_lds_kernel(const ParsedVirtualLdsHookMetadata &metadata);

/// @brief Process-local virtual-LDS policy associated with HSA symbols.
class VirtualLdsRuntimeRegistry {
public:
  struct ResolvedKernel {
    uint64_t executable = 0;
    VirtualLdsKernelRuntimeMetadata metadata;
    uint64_t normal_kernel_object = 0;
    uint64_t virtual_kernel_object = 0;
  };

  static VirtualLdsRuntimeRegistry &instance();
  void record_load(hsa_executable_t executable, hsa_agent_t guest_agent, hsa_agent_t host_agent,
                   hsa_loaded_code_object_t loaded_code_object,
                   ParsedVirtualLdsHookMetadata metadata);
  void record_symbol(hsa_executable_t executable, std::string_view symbol_name,
                     hsa_executable_symbol_t symbol);
  void note_kernel_object(hsa_executable_symbol_t symbol, uint64_t kernel_object,
                          uint32_t private_segment_size);
  [[nodiscard]] std::optional<ResolvedKernel> find_by_kernel_object(uint64_t kernel_object);
  [[nodiscard]] std::optional<std::string> kernel_name_for_object(uint64_t kernel_object);
  [[nodiscard]] uint32_t private_segment_size_for_object(uint64_t kernel_object);
  void erase_executable(hsa_executable_t executable);
  void clear();

private:
  struct LoadEntry {
    uint64_t executable = 0;
    uint64_t guest_agent = 0;
    uint64_t host_agent = 0;
    uint64_t loaded_code_object = 0;
    std::vector<VirtualLdsKernelRuntimeMetadata> metadata;
  };

  std::mutex mutex_;
  std::vector<LoadEntry> loads_;
  std::unordered_map<uint64_t, ResolvedKernel> symbols_;
};

/// @brief Per-dispatch buffers owned by a queue slot after virtual-LDS rewrite.
struct VirtualLdsDispatchBuffers {
  void *kernarg = nullptr;
  void *backing = nullptr;
  uint64_t virtual_kernel_object = 0;
  hsa_signal_t completion_signal{};
  bool completion_signal_was_pending = false;
  /// @brief True after the application destroys its borrowed completion signal.
  ///
  /// @details A valid application only destroys a dispatch completion signal
  /// after the dispatch has completed. Remember that fact without retaining the
  /// now-invalid handle so later virtual-LDS cleanup never reads destroyed HSA
  /// signal storage.
  bool completion_signal_destroyed = false;
  bool owns_completion_signal = false;
};

/// @brief HSA entry points used by hook-side virtual-LDS resource management.
///
/// @details Keeping this narrow interface here avoids coupling the reusable
/// virtual-LDS state to the large HSA table-interposition implementation.
struct VirtualLdsRuntimeApi {
  decltype(&hsa_iterate_agents) iterate_agents = nullptr;
  decltype(&hsa_signal_create) signal_create = nullptr;
  decltype(&hsa_signal_destroy) signal_destroy = nullptr;
  decltype(&hsa_signal_load_scacquire) signal_load_scacquire = nullptr;
  hsa_amd_agent_iterate_memory_pools_fn_t iterate_memory_pools = nullptr;
  hsa_amd_memory_pool_get_info_fn_t memory_pool_get_info = nullptr;
  hsa_amd_memory_pool_allocate_fn_t memory_pool_allocate = nullptr;
  hsa_amd_memory_pool_free_fn_t memory_pool_free = nullptr;
  hsa_amd_agents_allow_access_fn_t agents_allow_access = nullptr;
};

/// @brief Install the original HSA entry points used by virtual-LDS dispatches.
void set_virtual_lds_runtime_api(VirtualLdsRuntimeApi api);

/// @brief Clear the installed HSA entry points on hook unload.
///
/// @details After this, virtual-LDS resource calls become no-ops until the next
/// install. Prevents a second OnLoad from racing readers against a stale table
/// and stops post-unload callbacks from using entry points ROCR may have reset.
void clear_virtual_lds_runtime_api();

[[nodiscard]] bool empty_virtual_lds_buffers(const VirtualLdsDispatchBuffers &buffers);
void release_virtual_lds_buffers(VirtualLdsDispatchBuffers &buffers);
void record_virtual_lds_completion_signal(VirtualLdsDispatchBuffers &buffers,
                                          hsa_kernel_dispatch_packet_t &packet);
[[nodiscard]] bool completed_virtual_lds_dispatch(const VirtualLdsDispatchBuffers &buffers);

/// @brief Allocator for virtual-LDS backing storage and wrapper kernargs.
class VirtualLdsDispatchAllocator {
public:
  static VirtualLdsDispatchAllocator &instance();
  [[nodiscard]] bool allocate(hsa_agent_t host_agent, size_t backing_bytes, size_t kernarg_bytes,
                              VirtualLdsDispatchBuffers &buffers);
  void clear();

private:
  struct Pools {
    hsa_amd_memory_pool_t backing_pool{};
    hsa_amd_memory_pool_t kernarg_pool{};
    bool has_backing_pool = false;
    bool has_kernarg_pool = false;
  };
  struct PoolSearch {
    hsa_amd_memory_pool_get_info_fn_t get_info = nullptr;
    Pools pools;
  };
  struct GlobalKernargPoolSearch {
    hsa_amd_agent_iterate_memory_pools_fn_t iterate_pools = nullptr;
    hsa_amd_memory_pool_get_info_fn_t get_info = nullptr;
    hsa_amd_memory_pool_t kernarg_pool{};
    bool found = false;
  };

  static hsa_status_t collect_pool(hsa_amd_memory_pool_t pool, void *data);
  static hsa_status_t collect_agent_kernarg_pool(hsa_agent_t agent, void *data);
  [[nodiscard]] static std::optional<hsa_amd_memory_pool_t> find_global_kernarg_pool();
  [[nodiscard]] std::optional<Pools> pools_for_agent(hsa_agent_t host_agent,
                                                     bool need_kernarg_pool);
  [[nodiscard]] static bool allocate_from_pool(hsa_amd_memory_pool_t pool, size_t size, void **ptr);
  [[nodiscard]] static bool allow_agent_access(hsa_agent_t host_agent, const void *ptr);

  std::mutex mutex_;
  std::unordered_map<uint64_t, Pools> pools_by_agent_;
};

} // namespace rocjitsu::hooks
