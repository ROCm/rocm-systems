// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::optional<MoiEvidenceContainerView>
resolve_moi_evidence_container(const ProgramInventory &inventory, bool in_kernel,
                               std::string_view container_name,
                               std::span<const ConSanExecutionOwner> execution_owners) {
  const auto make_view = [in_kernel](const auto &container,
                                     std::optional<uint64_t> descriptor_file_offset,
                                     uint64_t code_size) {
    return MoiEvidenceContainerView{
        .qualified_name = std::string(in_kernel ? "kernel:" : "function:") + container.name,
        .ordinary_memory_sites = container.ordinary_memory_sites,
        .barrier_sites = container.barrier_sites,
        .atomic_sites = container.atomic_sites,
        .kernel_descriptor_file_offset = descriptor_file_offset,
        .entry_text_offset = container.entry_text_offset,
        .text_file_offset = container.text_file_offset,
        .code_size = code_size,
        .uses_cluster_workgroup_id = in_kernel && container.uses_cluster_workgroup_id,
    };
  };
  if (!in_kernel) {
    const ConSanFunctionInfo *function = inventory.find_function_by_name(container_name);
    if (function == nullptr)
      return std::nullopt;
    return make_view(*function, std::nullopt, function->code_size);
  }

  const ConSanKernelInfo *kernel = inventory.find_kernel_by_name(container_name);
  if (kernel == nullptr && execution_owners.size() == 1u) {
    kernel = inventory.find_kernel_by_descriptor(execution_owners.front().descriptor_file_offset);
  }
  if (kernel == nullptr || is_rocclr_runtime_kernel_name(kernel->name))
    return std::nullopt;

  std::optional<uint64_t> descriptor_file_offset = kernel->descriptor_file_offset;
  if (execution_owners.size() != 1u ||
      execution_owners.front().descriptor_file_offset != kernel->descriptor_file_offset) {
    descriptor_file_offset.reset();
  }
  return make_view(*kernel, descriptor_file_offset,
                   kernel->has_text_range ? kernel->code_size : 0u);
}

} // namespace rocjitsu::consan_moi_impl
