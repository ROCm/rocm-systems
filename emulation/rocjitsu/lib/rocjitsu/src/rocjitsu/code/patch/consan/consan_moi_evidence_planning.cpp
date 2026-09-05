// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::optional<MoiEvidenceContainerView>
resolve_moi_evidence_container(const ProgramInventory &inventory, bool in_kernel,
                               std::string_view container_name,
                               std::span<const ConSanExecutionOwner> execution_owners) {
  const auto make_view = [in_kernel](const auto &container, uint64_t code_size) {
    return MoiEvidenceContainerView{
        .qualified_name = std::string(in_kernel ? "kernel:" : "function:") + container.name,
        .entry_text_offset = container.entry_text_offset,
        .text_file_offset = container.text_file_offset,
        .code_size = code_size,
        .uses_cluster_workgroup_id = in_kernel && container.uses_cluster_workgroup_id,
    };
  };
  if (!in_kernel) {
    const ConSanProgramContainer *function = inventory.find_function_by_name(container_name);
    if (function == nullptr)
      return std::nullopt;
    return make_view(*function, function->code_size);
  }

  const ConSanProgramContainer *kernel = inventory.find_kernel_by_name(container_name);
  if (kernel == nullptr && execution_owners.size() == 1u) {
    kernel = inventory.kernel(execution_owners.front());
  }
  if (kernel == nullptr || is_rocclr_runtime_kernel_name(kernel->name))
    return std::nullopt;

  return make_view(*kernel, kernel->has_text_range ? kernel->code_size : 0u);
}

} // namespace rocjitsu::consan_moi_impl
