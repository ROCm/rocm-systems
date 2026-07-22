// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

std::string duplicate_message(std::string_view kind, std::string_view value) {
  return "duplicate ISA target " + std::string(kind) + " '" + std::string(value) + "'";
}

IsaTargetRegistryError registry_error(std::string message) {
  return IsaTargetRegistryError{std::move(message)};
}

template <typename Key> constexpr auto enum_value(Key key) {
  return static_cast<std::underlying_type_t<Key>>(key);
}

constexpr bool is_public_architecture_key(rj_code_arch_t key) {
  const auto value = enum_value(key);
  return value >= enum_value(ROCJITSU_CODE_ARCH_CDNA1) &&
         value < enum_value(ROCJITSU_CODE_ARCH_NUM_ARCHS);
}

constexpr bool is_public_gpu_target_key(rj_code_target_id_t key) {
  const auto value = enum_value(key);
  return value >= enum_value(ROCJITSU_CODE_TARGET_GFX90A) &&
         value < enum_value(ROCJITSU_CODE_TARGET_INVALID);
}

} // namespace

IsaTargetRegistryError IsaTargetRegistry::add(IsaTargetDescriptor descriptor) {
  if (frozen_)
    return registry_error("cannot modify a frozen ISA target registry");
  if (descriptor.id.empty())
    return registry_error("ISA target canonical ID must not be empty");
  if (descriptor.decoder_factory == nullptr)
    return registry_error("ISA target '" + descriptor.id + "' has no decoder factory");
  std::unordered_set<std::string_view> new_ids;
  new_ids.emplace(descriptor.id);
  for (const std::string &alias : descriptor.aliases) {
    if (alias.empty())
      return registry_error("ISA target '" + descriptor.id + "' has an empty alias");
    if (!new_ids.emplace(alias).second)
      return registry_error(duplicate_message("ID", alias));
  }
  if (descriptor.architecture_id != ROCJITSU_CODE_ARCH_INVALID &&
      !is_public_architecture_key(descriptor.architecture_id))
    return registry_error("ISA target contains an unallocated architecture enum value");

  for (auto binding = descriptor.gpu_targets.begin(); binding != descriptor.gpu_targets.end();
       ++binding) {
    if (!is_public_gpu_target_key(binding->public_id))
      return registry_error("ISA target contains an unallocated GPU target enum value");
    if (binding->code_object_id.empty())
      return registry_error("ISA target '" + descriptor.id + "' has an empty GPU code-object ID");
    if (binding->code_object_id != descriptor.id &&
        std::find(descriptor.aliases.begin(), descriptor.aliases.end(), binding->code_object_id) ==
            descriptor.aliases.end())
      return registry_error("ISA target '" + descriptor.id + "' has GPU code-object ID '" +
                            binding->code_object_id + "' that is not an ID or alias");
    if (binding->elf_machine == 0)
      return registry_error("ISA target '" + descriptor.id +
                            "' has an empty GPU ELF machine value");

    for (auto previous = descriptor.gpu_targets.begin(); previous != binding; ++previous) {
      if (previous->public_id == binding->public_id)
        return registry_error(
            duplicate_message("GPU target", std::to_string(enum_value(binding->public_id))));
      if (previous->code_object_id == binding->code_object_id)
        return registry_error(duplicate_message("GPU code-object ID", binding->code_object_id));
      if (previous->elf_machine == binding->elf_machine)
        return registry_error(
            duplicate_message("GPU ELF machine", std::to_string(binding->elf_machine)));
    }
  }
  if (!descriptor.gpu_targets.empty() && descriptor.architecture_id == ROCJITSU_CODE_ARCH_INVALID)
    return registry_error("ISA target '" + descriptor.id +
                          "' with GPU bindings must have an architecture ID");

  for (const IsaTargetDescriptor &existing : targets_) {
    auto conflicts_with = [&](std::string_view id) {
      if (new_ids.contains(id))
        return registry_error(duplicate_message("ID", id));
      return IsaTargetRegistryError{};
    };
    if (auto error = conflicts_with(existing.id))
      return error;
    for (const std::string &alias : existing.aliases) {
      if (auto error = conflicts_with(alias))
        return error;
    }

    if (descriptor.architecture_id != ROCJITSU_CODE_ARCH_INVALID &&
        existing.architecture_id == descriptor.architecture_id)
      return registry_error(duplicate_message(
          "architecture", std::to_string(enum_value(descriptor.architecture_id))));
    for (const IsaGpuTargetBinding &gpu_target : descriptor.gpu_targets) {
      for (const IsaGpuTargetBinding &existing_gpu_target : existing.gpu_targets) {
        if (existing_gpu_target.public_id == gpu_target.public_id)
          return registry_error(
              duplicate_message("GPU target", std::to_string(enum_value(gpu_target.public_id))));
        if (existing_gpu_target.code_object_id == gpu_target.code_object_id)
          return registry_error(duplicate_message("GPU code-object ID", gpu_target.code_object_id));
        if (existing_gpu_target.elf_machine == gpu_target.elf_machine)
          return registry_error(
              duplicate_message("GPU ELF machine", std::to_string(gpu_target.elf_machine)));
      }
    }
  }

  targets_.push_back(std::move(descriptor));
  return std::nullopt;
}

IsaTargetRegistryError IsaTargetRegistry::freeze() {
  if (frozen_)
    return registry_error("ISA target registry is already frozen");
  std::sort(targets_.begin(), targets_.end(),
            [](const IsaTargetDescriptor &lhs, const IsaTargetDescriptor &rhs) {
              return lhs.id < rhs.id;
            });
  frozen_ = true;
  return std::nullopt;
}

std::span<const IsaTargetDescriptor> IsaTargetRegistry::targets() const {
  if (!frozen_ || !ok())
    return {};
  return targets_;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(std::string_view id) const {
  if (!frozen_ || !ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    if (target.id == id ||
        std::find(target.aliases.begin(), target.aliases.end(), id) != target.aliases.end())
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_arch_t architecture_id) const {
  if (!frozen_ || !ok() || !is_public_architecture_key(architecture_id))
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    if (target.architecture_id == architecture_id)
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_target_id_t gpu_target_id) const {
  if (!frozen_ || !ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaGpuTargetBinding &gpu_target : target.gpu_targets) {
      if (gpu_target.public_id == gpu_target_id)
        return &target;
    }
  }
  return nullptr;
}

const IsaGpuTargetBinding *
IsaTargetRegistry::find_gpu_target_by_elf_machine(uint32_t elf_machine) const {
  if (!frozen_ || !ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaGpuTargetBinding &gpu_target : target.gpu_targets) {
      if (gpu_target.elf_machine == elf_machine)
        return &gpu_target;
    }
  }
  return nullptr;
}

const IsaGpuTargetBinding *
IsaTargetRegistry::find_gpu_target_by_code_object_id(std::string_view id) const {
  if (!frozen_ || !ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaGpuTargetBinding &gpu_target : target.gpu_targets) {
      if (gpu_target.code_object_id == id)
        return &gpu_target;
    }
  }
  return nullptr;
}

IsaTargetRegistry make_isa_target_registry(std::span<const IsaTargetProvider> providers) {
  IsaTargetRegistry registry;
  for (IsaTargetProvider provider : providers) {
    if (provider == nullptr) {
      registry.initialization_error_ = "ISA target provider must not be null";
      return registry;
    }
    if (auto error = provider(registry)) {
      registry.initialization_error_ = std::move(*error);
      return registry;
    }
  }
  if (auto error = registry.freeze())
    registry.initialization_error_ = std::move(*error);
  return registry;
}

} // namespace rocjitsu
