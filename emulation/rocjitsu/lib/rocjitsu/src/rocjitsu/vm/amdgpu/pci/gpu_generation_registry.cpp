// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/gpu_generation_registry.h"

#include <algorithm>
#include <format>

namespace rocjitsu {
namespace {

/// @brief KFD target version gfx1250 reports.
constexpr uint32_t kGfx1250TargetVersions[] = {120500};

/// @details One row per generation this image can present. Adding a part is a
/// row plus its profile, which is the whole point of the indirection: nothing
/// here needs editing anywhere else to answer for a new target.
constexpr GpuGenerationDescriptor kGenerations[] = {
    {.id = "gfx1250",
     .gfx_target_versions = kGfx1250TargetVersions,
     .discovery_factory = &gfx1250_discovery_spec},
};

} // namespace

GpuGenerationRegistryError
GpuGenerationRegistry::validate(std::span<const GpuGenerationDescriptor> descriptors) {
  for (std::size_t descriptor_index = 0; descriptor_index < descriptors.size();
       ++descriptor_index) {
    const GpuGenerationDescriptor &descriptor = descriptors[descriptor_index];
    if (descriptor.id.empty()) {
      return std::format("generation {} has no name", descriptor_index);
    }
    if (descriptor.discovery_factory == nullptr) {
      return std::format("generation {} describes no blocks to publish", descriptor.id);
    }
    if (descriptor.gfx_target_versions.empty()) {
      return std::format("generation {} answers for no gfx target", descriptor.id);
    }
    for (const uint32_t version : descriptor.gfx_target_versions) {
      // Zero is what an unset config reads as, so a generation claiming it would
      // answer for every configuration that forgot to say which part it is.
      if (version == 0) {
        return std::format("generation {} claims gfx target 0, which is an unset one",
                           descriptor.id);
      }
    }
    // Two generations answering for one target is the defect this registry
    // exists to make impossible: the winner would be whichever was registered
    // first, and a config would silently get a part it did not name.
    for (std::size_t other_index = descriptor_index + 1; other_index < descriptors.size();
         ++other_index) {
      const GpuGenerationDescriptor &other = descriptors[other_index];
      if (descriptor.id == other.id) {
        return std::format("two generations are both named {}", descriptor.id);
      }
      for (const uint32_t version : descriptor.gfx_target_versions) {
        if (std::ranges::find(other.gfx_target_versions, version) !=
            other.gfx_target_versions.end()) {
          return std::format("generations {} and {} both answer for gfx target {}", descriptor.id,
                             other.id, version);
        }
      }
    }
  }
  return std::nullopt;
}

const GpuGenerationDescriptor *GpuGenerationRegistry::find(uint32_t gfx_target_version) const {
  if (!ok() || gfx_target_version == 0) {
    return nullptr;
  }
  for (const GpuGenerationDescriptor &descriptor : descriptors_) {
    if (std::ranges::find(descriptor.gfx_target_versions, gfx_target_version) !=
        descriptor.gfx_target_versions.end()) {
      return &descriptor;
    }
  }
  return nullptr;
}

const GpuGenerationDescriptor *GpuGenerationRegistry::find(std::string_view id) const {
  if (!ok() || id.empty()) {
    return nullptr;
  }
  const std::span<const GpuGenerationDescriptor>::iterator found =
      std::ranges::find(descriptors_, id, &GpuGenerationDescriptor::id);
  return found == descriptors_.end() ? nullptr : &*found;
}

std::string GpuGenerationRegistry::known_ids() const {
  std::string names;
  for (const GpuGenerationDescriptor &descriptor : descriptors_) {
    if (!names.empty()) {
      names += ", ";
    }
    names += descriptor.id;
  }
  return names;
}

const GpuGenerationRegistry &gpu_generations() {
  static const GpuGenerationRegistry registry(kGenerations);
  return registry;
}

} // namespace rocjitsu
