// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_lds.cpp
/// @brief Hook-side virtual-LDS metadata, symbol state, and allocations.

#include "rocjitsu/hooks/virtual_lds.h"

#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/hooks/sidecar_registry.h"

#include <algorithm>
#include <atomic>
#include <ranges>
#include <utility>

namespace rocjitsu::hooks {
namespace {

constexpr uint32_t kHsaAmdSegmentGlobal = 0;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagKernargInit = 1u;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagFineGrained = 2u;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagCoarseGrained = 4u;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagExtendedScopeFineGrained = 8u;

// The runtime API table is installed exactly once from OnLoad (before any queue
// exists) but read from the doorbell poll thread, the CP scanner thread, and the
// ROCR packet-interceptor thread. Publishing the table through an atomic pointer
// with release/acquire ordering gives those readers a well-defined view: the
// non-atomic writes into the storage below happen-before the release store, and
// each reader's acquire load establishes visibility of the whole table.
VirtualLdsRuntimeApi g_runtime_api_storage;
std::atomic<const VirtualLdsRuntimeApi *> g_runtime_api_ptr{nullptr};

[[nodiscard]] const VirtualLdsRuntimeApi &runtime_api() {
  static constexpr VirtualLdsRuntimeApi kUninstalled{};
  const VirtualLdsRuntimeApi *api = g_runtime_api_ptr.load(std::memory_order_acquire);
  return api != nullptr ? *api : kUninstalled;
}

[[nodiscard]] std::string normalize_kernel_symbol_name(std::string_view symbol_name) {
  constexpr std::string_view kDescriptorSuffix = ".kd";
  if (symbol_name.ends_with(kDescriptorSuffix))
    symbol_name.remove_suffix(kDescriptorSuffix.size());
  return std::string(symbol_name);
}

void free_allocation(void *ptr) {
  if (ptr != nullptr && runtime_api().memory_pool_free != nullptr)
    (void)runtime_api().memory_pool_free(ptr);
}

} // namespace

std::optional<ParsedVirtualLdsHookMetadata>
parse_virtual_lds_hook_metadata(std::span<const uint8_t> image) {
  auto virtual_bytes = read_metadata_section(image, kVirtualLdsMetadataSectionName);
  auto sidecars = parse_sidecar_metadata_section(image);
  auto kernarg_bytes = read_metadata_section(image, kKernargExtensionMetadataSectionName);
  if (!virtual_bytes || !sidecars || !kernarg_bytes)
    return std::nullopt;

  ParsedVirtualLdsHookMetadata parsed{.sidecars = std::move(*sidecars), .virtual_lds = {}};
  // Generic sidecars and kernarg extensions are independently useful. A
  // virtual-LDS record is the only feature that requires all three records.
  if (virtual_bytes->empty())
    return parsed;
  if (parsed.sidecars.empty() || kernarg_bytes->empty())
    return std::nullopt;

  auto plans = parse_virtual_lds_metadata(*virtual_bytes);
  auto extensions = parse_kernarg_extension_metadata(*kernarg_bytes);
  if (!plans || !extensions)
    return std::nullopt;

  parsed.virtual_lds.reserve(plans->size());
  for (const VirtualLdsKernelMetadata &plan : *plans) {
    const auto sidecar = std::ranges::find_if(parsed.sidecars, [&](const auto &candidate) {
      return candidate.kernel_name == plan.kernel_name &&
             candidate.variant_name == plan.sidecar_variant_name;
    });
    const auto extension = std::ranges::find_if(*extensions, [&](const auto &candidate) {
      return candidate.kernel_name == plan.kernel_name &&
             candidate.variant_name == plan.sidecar_variant_name;
    });
    if (sidecar == parsed.sidecars.end() || extension == extensions->end() ||
        extension->payloads.size() != 1 ||
        extension->payloads.front().name != kVirtualLdsRuntimeStatePayloadName ||
        extension->payloads.front().size != sizeof(VirtualLdsDispatchState) ||
        extension->payloads.front().alignment != alignof(uint64_t)) {
      return std::nullopt;
    }

    const KernargExtensionPayloadLayout payload{.size = extension->payloads.front().size,
                                                .alignment = extension->payloads.front().alignment};
    const auto layout =
        make_kernarg_extension_layout(extension->original_kernarg_size, std::span{&payload, 1});
    if (!layout || layout->payload_offsets.size() != 1)
      return std::nullopt;

    parsed.virtual_lds.push_back({.kernel_name = plan.kernel_name,
                                  .sidecar_variant_name = plan.sidecar_variant_name,
                                  .static_lds_bytes = plan.static_lds_bytes,
                                  .normal_private_segment_size = plan.normal_private_segment_size,
                                  .virtual_private_segment_size = plan.virtual_private_segment_size,
                                  .kernarg_size = extension->original_kernarg_size,
                                  .backing_pointer_kernarg_offset = layout->payload_offsets.front(),
                                  .virtual_lds_base_sgpr = plan.virtual_lds_base_sgpr,
                                  .flags = plan.flags});
  }
  return parsed;
}

std::optional<std::string>
first_unsupportable_virtual_lds_kernel(const ParsedVirtualLdsHookMetadata &metadata) {
  for (const VirtualLdsKernelRuntimeMetadata &kernel : metadata.virtual_lds) {
    // The dispatch path requires the runtime-state-block ABI for every virtual
    // LDS launch; without it there is no place to publish the backing pointer.
    if ((kernel.flags & kVirtualLdsFlagRuntimeStateBlock) == 0)
      return kernel.kernel_name;

    // The kernarg wrapper layout must be reconstructible and its backing-pointer
    // payload must land at the offset the descriptor was built for. This is a
    // pure function of the recorded kernarg size, so a mismatch is a permanent
    // property of the code object, not a per-dispatch condition.
    const KernargExtensionPayloadLayout payload{
        .size = static_cast<uint32_t>(sizeof(VirtualLdsDispatchState)),
        .alignment = alignof(uint64_t),
    };
    const auto wrapper_layout =
        make_kernarg_extension_layout(kernel.kernarg_size, std::span{&payload, 1});
    if (!wrapper_layout || wrapper_layout->payload_offsets.size() != 1 ||
        wrapper_layout->payload_offsets.front() != kernel.backing_pointer_kernarg_offset) {
      return kernel.kernel_name;
    }
  }
  return std::nullopt;
}

VirtualLdsRuntimeRegistry &VirtualLdsRuntimeRegistry::instance() {
  static VirtualLdsRuntimeRegistry registry;
  return registry;
}

void VirtualLdsRuntimeRegistry::record_load(hsa_executable_t executable, hsa_agent_t guest_agent,
                                            hsa_agent_t host_agent,
                                            hsa_loaded_code_object_t loaded_code_object,
                                            ParsedVirtualLdsHookMetadata metadata) {
  SidecarRegistry::instance().record_load(executable.handle, loaded_code_object.handle,
                                          std::move(metadata.sidecars));
  if (metadata.virtual_lds.empty())
    return;
  std::lock_guard lock(mutex_);
  loads_.push_back({.executable = executable.handle,
                    .guest_agent = guest_agent.handle,
                    .host_agent = host_agent.handle,
                    .loaded_code_object = loaded_code_object.handle,
                    .metadata = std::move(metadata.virtual_lds)});
}

void VirtualLdsRuntimeRegistry::record_symbol(hsa_executable_t executable,
                                              std::string_view symbol_name,
                                              hsa_executable_symbol_t symbol) {
  if (symbol.handle == 0 || symbol_name.empty())
    return;
  SidecarRegistry::instance().record_symbol(executable.handle, symbol_name, symbol.handle);
  const std::string kernel_name = normalize_kernel_symbol_name(symbol_name);
  std::lock_guard lock(mutex_);
  const auto associated = symbols_.find(symbol.handle);
  if (associated != symbols_.end()) {
    if (associated->second.executable == executable.handle &&
        associated->second.metadata.kernel_name == kernel_name) {
      return;
    }
    // HSA may reuse symbol handles after executable destruction. Never retain
    // feature metadata belonging to the old executable in that case.
    symbols_.erase(associated);
  }
  for (size_t load_index = loads_.size(); load_index > 0; --load_index) {
    LoadEntry &load = loads_[load_index - 1];
    if (load.executable != executable.handle)
      continue;
    auto record = std::ranges::find_if(
        load.metadata, [&](const auto &candidate) { return candidate.kernel_name == kernel_name; });
    if (record == load.metadata.end())
      continue;
    symbols_[symbol.handle] = {.executable = executable.handle, .metadata = *record};
    load.metadata.erase(record);
    if (load.metadata.empty())
      loads_.erase(loads_.begin() + static_cast<std::ptrdiff_t>(load_index - 1));
    return;
  }
}

void VirtualLdsRuntimeRegistry::note_kernel_object(hsa_executable_symbol_t symbol,
                                                   uint64_t kernel_object,
                                                   uint32_t private_segment_size) {
  if (symbol.handle != 0 && kernel_object != 0)
    SidecarRegistry::instance().note_kernel_object(symbol.handle, kernel_object,
                                                   private_segment_size);
}

std::optional<VirtualLdsRuntimeRegistry::ResolvedKernel>
VirtualLdsRuntimeRegistry::find_by_kernel_object(uint64_t kernel_object) {
  if (kernel_object == 0)
    return std::nullopt;
  const auto sidecar = SidecarRegistry::instance().find_by_kernel_object(kernel_object);
  if (!sidecar)
    return std::nullopt;
  std::lock_guard lock(mutex_);
  const auto feature = symbols_.find(sidecar->symbol);
  if (feature == symbols_.end() || feature->second.executable != sidecar->executable ||
      feature->second.metadata.kernel_name != sidecar->kernel_name) {
    return std::nullopt;
  }
  const auto variant = sidecar->variant_object(feature->second.metadata.sidecar_variant_name);
  if (!variant)
    return std::nullopt;
  ResolvedKernel resolved = feature->second;
  resolved.normal_kernel_object = sidecar->normal_kernel_object;
  resolved.virtual_kernel_object = *variant;
  return resolved;
}

std::optional<std::string>
VirtualLdsRuntimeRegistry::kernel_name_for_object(uint64_t kernel_object) {
  return SidecarRegistry::instance().kernel_name_for_object(kernel_object);
}

uint32_t VirtualLdsRuntimeRegistry::private_segment_size_for_object(uint64_t kernel_object) {
  return SidecarRegistry::instance().private_segment_size_for_object(kernel_object);
}

void VirtualLdsRuntimeRegistry::erase_executable(hsa_executable_t executable) {
  SidecarRegistry::instance().erase_executable(executable.handle);
  std::lock_guard lock(mutex_);
  std::erase_if(loads_,
                [&](const LoadEntry &load) { return load.executable == executable.handle; });
  std::erase_if(symbols_,
                [&](const auto &entry) { return entry.second.executable == executable.handle; });
}

void VirtualLdsRuntimeRegistry::clear() {
  SidecarRegistry::instance().clear();
  std::lock_guard lock(mutex_);
  loads_.clear();
  symbols_.clear();
}

void set_virtual_lds_runtime_api(VirtualLdsRuntimeApi api) {
  g_runtime_api_storage = api;
  g_runtime_api_ptr.store(&g_runtime_api_storage, std::memory_order_release);
}

void clear_virtual_lds_runtime_api() {
  g_runtime_api_ptr.store(nullptr, std::memory_order_release);
}

bool empty_virtual_lds_buffers(const VirtualLdsDispatchBuffers &buffers) {
  return buffers.kernarg == nullptr && buffers.backing == nullptr;
}

void release_virtual_lds_buffers(VirtualLdsDispatchBuffers &buffers) {
  free_allocation(buffers.kernarg);
  free_allocation(buffers.backing);
  if (buffers.owns_completion_signal && buffers.completion_signal.handle != 0 &&
      runtime_api().signal_destroy != nullptr) {
    (void)runtime_api().signal_destroy(buffers.completion_signal);
  }
  buffers = {};
}

void record_virtual_lds_completion_signal(VirtualLdsDispatchBuffers &buffers,
                                          hsa_kernel_dispatch_packet_t &packet) {
  if (packet.completion_signal.handle == 0) {
    if (runtime_api().signal_create == nullptr)
      return;
    hsa_signal_t signal{};
    if (runtime_api().signal_create(1, 0, nullptr, &signal) != HSA_STATUS_SUCCESS ||
        signal.handle == 0) {
      return;
    }
    packet.completion_signal = signal;
    buffers.completion_signal = signal;
    buffers.completion_signal_was_pending = true;
    buffers.owns_completion_signal = true;
    return;
  }
  buffers.completion_signal = packet.completion_signal;
  if (runtime_api().signal_load_scacquire != nullptr) {
    // Completion signals are decremented by CP. A zero value before enqueue is
    // not a useful lifetime fence because it already appears complete.
    buffers.completion_signal_was_pending =
        runtime_api().signal_load_scacquire(packet.completion_signal) > 0;
  }
}

bool completed_virtual_lds_dispatch(const VirtualLdsDispatchBuffers &buffers) {
  if (empty_virtual_lds_buffers(buffers))
    return true;
  if (buffers.completion_signal_destroyed)
    return true;
  if (buffers.completion_signal.handle == 0 || !buffers.completion_signal_was_pending ||
      runtime_api().signal_load_scacquire == nullptr) {
    return false;
  }
  return runtime_api().signal_load_scacquire(buffers.completion_signal) <= 0;
}

VirtualLdsDispatchAllocator &VirtualLdsDispatchAllocator::instance() {
  static VirtualLdsDispatchAllocator allocator;
  return allocator;
}

bool VirtualLdsDispatchAllocator::allocate(hsa_agent_t host_agent, size_t backing_bytes,
                                           size_t kernarg_bytes,
                                           VirtualLdsDispatchBuffers &buffers) {
  if (host_agent.handle == 0 || backing_bytes == 0 || kernarg_bytes == 0)
    return false;
  const auto pools = pools_for_agent(host_agent, true);
  if (!pools || !pools->has_backing_pool || !pools->has_kernarg_pool)
    return false;

  void *backing = nullptr;
  if (!allocate_from_pool(pools->backing_pool, backing_bytes, &backing))
    return false;
  if (!allow_agent_access(host_agent, backing)) {
    free_allocation(backing);
    return false;
  }
  void *kernarg = nullptr;
  if (!allocate_from_pool(pools->kernarg_pool, kernarg_bytes, &kernarg)) {
    free_allocation(backing);
    return false;
  }
  if (!allow_agent_access(host_agent, kernarg)) {
    free_allocation(kernarg);
    free_allocation(backing);
    return false;
  }
  buffers.backing = backing;
  buffers.kernarg = kernarg;
  return true;
}

void VirtualLdsDispatchAllocator::clear() {
  std::lock_guard lock(mutex_);
  pools_by_agent_.clear();
}

hsa_status_t VirtualLdsDispatchAllocator::collect_pool(hsa_amd_memory_pool_t pool, void *data) {
  auto *search = static_cast<PoolSearch *>(data);
  if (search == nullptr || search->get_info == nullptr)
    return HSA_STATUS_ERROR;
  uint32_t segment = 0;
  uint32_t flags = 0;
  bool runtime_alloc_allowed = false;
  if (search->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  (void)search->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  (void)search->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                         &runtime_alloc_allowed);
  if (segment != kHsaAmdSegmentGlobal || !runtime_alloc_allowed)
    return HSA_STATUS_SUCCESS;

  const bool kernarg = (flags & kHsaAmdMemoryPoolGlobalFlagKernargInit) != 0;
  const bool backing =
      (flags & (kHsaAmdMemoryPoolGlobalFlagFineGrained | kHsaAmdMemoryPoolGlobalFlagCoarseGrained |
                kHsaAmdMemoryPoolGlobalFlagExtendedScopeFineGrained)) != 0;
  if (kernarg && !search->pools.has_kernarg_pool) {
    search->pools.kernarg_pool = pool;
    search->pools.has_kernarg_pool = true;
  }
  if (!kernarg && backing && !search->pools.has_backing_pool) {
    search->pools.backing_pool = pool;
    search->pools.has_backing_pool = true;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t VirtualLdsDispatchAllocator::collect_agent_kernarg_pool(hsa_agent_t agent,
                                                                     void *data) {
  auto *search = static_cast<GlobalKernargPoolSearch *>(data);
  if (search == nullptr || search->iterate_pools == nullptr || search->get_info == nullptr ||
      search->found) {
    return HSA_STATUS_SUCCESS;
  }
  PoolSearch pools{.get_info = search->get_info, .pools = {}};
  (void)search->iterate_pools(agent, collect_pool, &pools);
  if (pools.pools.has_kernarg_pool) {
    search->kernarg_pool = pools.pools.kernarg_pool;
    search->found = true;
  }
  return HSA_STATUS_SUCCESS;
}

std::optional<hsa_amd_memory_pool_t> VirtualLdsDispatchAllocator::find_global_kernarg_pool() {
  if (runtime_api().iterate_agents == nullptr || runtime_api().iterate_memory_pools == nullptr ||
      runtime_api().memory_pool_get_info == nullptr) {
    return std::nullopt;
  }
  GlobalKernargPoolSearch search{.iterate_pools = runtime_api().iterate_memory_pools,
                                 .get_info = runtime_api().memory_pool_get_info};
  (void)runtime_api().iterate_agents(collect_agent_kernarg_pool, &search);
  if (!search.found)
    return std::nullopt;
  return search.kernarg_pool;
}

std::optional<VirtualLdsDispatchAllocator::Pools>
VirtualLdsDispatchAllocator::pools_for_agent(hsa_agent_t host_agent, bool need_kernarg_pool) {
  {
    std::lock_guard lock(mutex_);
    const auto found = pools_by_agent_.find(host_agent.handle);
    if (found != pools_by_agent_.end() && (!need_kernarg_pool || found->second.has_kernarg_pool)) {
      return found->second;
    }
  }
  if (runtime_api().iterate_memory_pools == nullptr ||
      runtime_api().memory_pool_get_info == nullptr) {
    return std::nullopt;
  }
  PoolSearch search{.get_info = runtime_api().memory_pool_get_info, .pools = {}};
  const hsa_status_t status = runtime_api().iterate_memory_pools(host_agent, collect_pool, &search);
  if (status == HSA_STATUS_SUCCESS && need_kernarg_pool && !search.pools.has_kernarg_pool) {
    if (auto global_pool = find_global_kernarg_pool()) {
      search.pools.kernarg_pool = *global_pool;
      search.pools.has_kernarg_pool = true;
    }
  }
  if (status != HSA_STATUS_SUCCESS || !search.pools.has_backing_pool ||
      (need_kernarg_pool && !search.pools.has_kernarg_pool)) {
    return std::nullopt;
  }
  std::lock_guard lock(mutex_);
  return pools_by_agent_.insert_or_assign(host_agent.handle, search.pools).first->second;
}

bool VirtualLdsDispatchAllocator::allocate_from_pool(hsa_amd_memory_pool_t pool, size_t size,
                                                     void **ptr) {
  if (runtime_api().memory_pool_allocate == nullptr || ptr == nullptr)
    return false;
  *ptr = nullptr;
  return runtime_api().memory_pool_allocate(pool, size, 0, ptr) == HSA_STATUS_SUCCESS &&
         *ptr != nullptr;
}

bool VirtualLdsDispatchAllocator::allow_agent_access(hsa_agent_t host_agent, const void *ptr) {
  return runtime_api().agents_allow_access != nullptr && ptr != nullptr &&
         runtime_api().agents_allow_access(1, &host_agent, nullptr, ptr) == HSA_STATUS_SUCCESS;
}

} // namespace rocjitsu::hooks
