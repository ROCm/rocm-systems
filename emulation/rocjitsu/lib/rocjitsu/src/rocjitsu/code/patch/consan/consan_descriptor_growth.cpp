// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/spill_manager.h"

#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu {
namespace {

class ActiveDescriptorResolver {
public:
  ActiveDescriptorResolver(const ProgramInventory &inventory,
                           const AmdGpuCodeObject &active_code_object) {
    original_kernels_.reserve(inventory.kernels().size());
    for (const ConSanKernelInfo &kernel : inventory.kernels())
      original_kernels_.emplace(kernel.descriptor_file_offset, &kernel);
    active_kernels_.reserve(active_code_object.kernels().size());
    for (const AmdGpuKernelInfo &kernel : active_code_object.kernels())
      active_kernels_.emplace(kernel.name, &kernel);
  }

  [[nodiscard]] std::pair<const ConSanKernelInfo *, const AmdGpuKernelInfo *>
  resolve(uint64_t original_descriptor_offset) const {
    const auto original = original_kernels_.find(original_descriptor_offset);
    if (original == original_kernels_.end())
      return {};
    const auto active = active_kernels_.find(original->second->name);
    return {original->second, active == active_kernels_.end() ? nullptr : active->second};
  }

private:
  std::unordered_map<uint64_t, const ConSanKernelInfo *> original_kernels_;
  std::unordered_map<std::string_view, const AmdGpuKernelInfo *> active_kernels_;
};

template <typename CurrentImage, typename CommitDescriptor>
[[nodiscard]] bool apply_descriptor_mutations(
    CurrentImage current_image, CommitDescriptor commit_descriptor,
    const ProgramInventory &inventory, const AmdGpuCodeObject &active_code_object,
    const ConSanDescriptorMutationBatch &batch, const ConSanDescriptorMutationPolicy &policy,
    rj_code_arch_t arch, std::string_view subject, std::vector<std::string> &errors) {
  std::unordered_set<uint64_t> owners;
  owners.reserve(batch.vgprs.size() + batch.sgprs.size() + batch.private_segment_bytes.size() +
                 batch.group_segment_bytes.size());
  const auto collect = [&](const auto &requirements) {
    for (const auto &[owner, required] : requirements) {
      if (required != 0u)
        owners.insert(owner);
    }
  };
  collect(batch.vgprs);
  collect(batch.sgprs);
  collect(batch.private_segment_bytes);
  collect(batch.group_segment_bytes);

  const ActiveDescriptorResolver resolver(inventory, active_code_object);
  for (uint64_t owner : owners) {
    const auto [original_kernel, active_kernel] = resolver.resolve(owner);
    if (original_kernel == nullptr || active_kernel == nullptr) {
      errors.emplace_back(std::string(subject) + " could not resolve an active descriptor owner");
      return false;
    }
    const uint64_t active_offset = active_kernel->descriptor_file_offset;
    auto descriptor = read_kernel_descriptor(current_image(), active_offset);
    if (!descriptor) {
      errors.emplace_back(std::string(subject) + " descriptor exceeds ELF bytes");
      return false;
    }

    if (const auto it = batch.vgprs.find(owner); it != batch.vgprs.end() && it->second != 0u) {
      if (!grow_descriptor_vgpr_allocation(
              *descriptor,
              {.required_ordinary_count = it->second,
               .maximum_ordinary_count = policy.maximum_ordinary_vgpr_count,
               .accumulator_bank_is_proven_empty = policy.inventory_proves_empty_accumulator_bank &&
                                                   original_kernel->agpr_count == 0u},
              arch)) {
        errors.emplace_back(std::string(subject) + " could not grow descriptor VGPR allocation");
        return false;
      }
    }
    if (const auto it = batch.sgprs.find(owner); it != batch.sgprs.end() && it->second != 0u) {
      if (!grow_descriptor_sgpr_allocation(*descriptor, it->second, arch)) {
        errors.emplace_back(std::string(subject) + " could not grow descriptor SGPR allocation");
        return false;
      }
    }
    if (const auto it = batch.private_segment_bytes.find(owner);
        it != batch.private_segment_bytes.end() && it->second != 0u) {
      const SpillDescriptorUpdate update =
          update_kernel_descriptor_for_spills(*descriptor, it->second);
      if (update != SpillDescriptorUpdate::Updated && update != SpillDescriptorUpdate::Unchanged) {
        errors.emplace_back(std::string(subject) + " could not grow descriptor private segment");
        return false;
      }
    }
    if (const auto it = batch.group_segment_bytes.find(owner);
        it != batch.group_segment_bytes.end() && it->second != 0u) {
      if (!policy.maximum_group_segment_bytes || it->second > *policy.maximum_group_segment_bytes ||
          descriptor->group_segment_fixed_size > it->second) {
        errors.emplace_back(std::string(subject) +
                            " descriptor has an incompatible fixed-LDS size");
        return false;
      }
      descriptor->group_segment_fixed_size = it->second;
    }
    if (!commit_descriptor(active_offset, *descriptor)) {
      errors.emplace_back(std::string(subject) + " could not publish descriptor mutation");
      return false;
    }
  }
  return true;
}

} // namespace

bool apply_consan_descriptor_mutations_to_patcher(CodeObjectPatcher &patcher,
                                                  const ProgramInventory &inventory,
                                                  const AmdGpuCodeObject &active_code_object,
                                                  const ConSanDescriptorMutationBatch &batch,
                                                  const ConSanDescriptorMutationPolicy &policy,
                                                  rj_code_arch_t arch, std::string_view subject,
                                                  std::vector<std::string> &errors) {
  return apply_descriptor_mutations([&]() { return patcher.image_bytes(); },
                                    [&](uint64_t offset, const auto &descriptor) {
                                      return patcher.patch_kernel_descriptor(offset, descriptor);
                                    },
                                    inventory, active_code_object, batch, policy, arch, subject,
                                    errors);
}

bool apply_consan_descriptor_mutations_to_bytes(std::span<uint8_t> image,
                                                const ProgramInventory &inventory,
                                                const ConSanDescriptorMutationBatch &batch,
                                                const ConSanDescriptorMutationPolicy &policy,
                                                rj_code_arch_t arch, std::string_view subject,
                                                std::vector<std::string> &errors) {
  const AmdGpuCodeObject active_code_object(image.data(), image.size());
  if (!active_code_object.is_valid()) {
    errors.emplace_back(std::string(subject) + " cannot parse the active descriptor image");
    return false;
  }
  return apply_descriptor_mutations([&]() { return std::span<const uint8_t>(image); },
                                    [&](uint64_t offset, const auto &descriptor) {
                                      return write_kernel_descriptor(image, offset, descriptor);
                                    },
                                    inventory, active_code_object, batch, policy, arch, subject,
                                    errors);
}

} // namespace rocjitsu
