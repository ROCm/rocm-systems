// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/target_registry.h"

#include <type_traits>
#include <utility>

namespace rocjitsu {
namespace {

std::string duplicate_message(std::string_view kind, std::string_view value) {
  return "duplicate ISA target " + std::string(kind) + " '" + std::string(value) + "'";
}

IsaTargetRegistryError registry_error(std::string message) {
  return IsaTargetRegistryError{std::move(message)};
}

template <typename Key> constexpr std::underlying_type_t<Key> enum_value(Key key) {
  return static_cast<std::underlying_type_t<Key>>(key);
}

constexpr bool is_public_architecture_key(rj_code_arch_t key) {
  const std::underlying_type_t<rj_code_arch_t> value = enum_value(key);
  return value >= enum_value(ROCJITSU_CODE_ARCH_CDNA1) &&
         value < enum_value(ROCJITSU_CODE_ARCH_NUM_ARCHS);
}

constexpr bool is_public_gpu_target_key(rj_code_target_id_t key) {
  const std::underlying_type_t<rj_code_target_id_t> value = enum_value(key);
  return value >= enum_value(ROCJITSU_CODE_TARGET_GFX90A) &&
         value < enum_value(ROCJITSU_CODE_TARGET_INVALID);
}

bool contains_id(const IsaTargetDescriptor &descriptor, std::string_view id) {
  if (descriptor.id == id)
    return true;
  for (std::string_view alias : descriptor.aliases) {
    if (alias == id)
      return true;
  }
  return false;
}

IsaTargetRegistryError validate_descriptor(const IsaTargetDescriptor &descriptor) {
  if (descriptor.id.empty())
    return registry_error("ISA target canonical ID must not be empty");
  if (descriptor.decoder_factory == nullptr)
    return registry_error("ISA target '" + std::string(descriptor.id) + "' has no decoder factory");
  for (size_t alias_index = 0; alias_index < descriptor.aliases.size(); ++alias_index) {
    const std::string_view alias = descriptor.aliases[alias_index];
    if (alias.empty())
      return registry_error("ISA target '" + std::string(descriptor.id) + "' has an empty alias");
    if (alias == descriptor.id)
      return registry_error(duplicate_message("ID", alias));
    for (size_t previous_index = 0; previous_index < alias_index; ++previous_index) {
      if (descriptor.aliases[previous_index] == alias)
        return registry_error(duplicate_message("ID", alias));
    }
  }
  if (descriptor.architecture_id != ROCJITSU_CODE_ARCH_INVALID &&
      !is_public_architecture_key(descriptor.architecture_id))
    return registry_error("ISA target contains an unallocated architecture enum value");

  for (size_t binding_index = 0; binding_index < descriptor.gpu_targets.size(); ++binding_index) {
    const IsaGpuTargetDescription &binding = descriptor.gpu_targets[binding_index];
    if (!is_public_gpu_target_key(binding.public_id))
      return registry_error("ISA target contains an unallocated GPU target enum value");
    if (binding.code_object_id.empty())
      return registry_error("ISA target '" + std::string(descriptor.id) +
                            "' has an empty GPU code-object ID");
    if (!contains_id(descriptor, binding.code_object_id))
      return registry_error("ISA target '" + std::string(descriptor.id) +
                            "' has GPU code-object ID '" + std::string(binding.code_object_id) +
                            "' that is not an ID or alias");
    if (binding.elf_machine == 0)
      return registry_error("ISA target '" + std::string(descriptor.id) +
                            "' has an empty GPU ELF machine value");

    for (size_t previous_index = 0; previous_index < binding_index; ++previous_index) {
      const IsaGpuTargetDescription &previous = descriptor.gpu_targets[previous_index];
      if (previous.public_id == binding.public_id)
        return registry_error(
            duplicate_message("GPU target", std::to_string(enum_value(binding.public_id))));
      if (previous.code_object_id == binding.code_object_id)
        return registry_error(duplicate_message("GPU code-object ID", binding.code_object_id));
      if (previous.elf_machine == binding.elf_machine)
        return registry_error(
            duplicate_message("GPU ELF machine", std::to_string(binding.elf_machine)));
    }
  }
  if (!descriptor.gpu_targets.empty() && descriptor.architecture_id == ROCJITSU_CODE_ARCH_INVALID)
    return registry_error("ISA target '" + std::string(descriptor.id) +
                          "' with GPU bindings must have an architecture ID");
  return std::nullopt;
}

IsaTargetRegistryError validate_descriptors(std::span<const IsaTargetDescriptor> descriptors) {
  for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
    const IsaTargetDescriptor &descriptor = descriptors[descriptor_index];
    if (IsaTargetRegistryError error = validate_descriptor(descriptor))
      return error;
    for (size_t existing_index = 0; existing_index < descriptor_index; ++existing_index) {
      const IsaTargetDescriptor &existing = descriptors[existing_index];
      auto conflicts_with = [&](std::string_view id) {
        if (contains_id(descriptor, id))
          return registry_error(duplicate_message("ID", id));
        return IsaTargetRegistryError{};
      };
      if (IsaTargetRegistryError error = conflicts_with(existing.id))
        return error;
      for (std::string_view alias : existing.aliases) {
        if (IsaTargetRegistryError error = conflicts_with(alias))
          return error;
      }
      // Code-object IDs are IDs or aliases, whose global uniqueness is checked above.

      if (descriptor.architecture_id != ROCJITSU_CODE_ARCH_INVALID &&
          existing.architecture_id == descriptor.architecture_id)
        return registry_error(duplicate_message(
            "architecture", std::to_string(enum_value(descriptor.architecture_id))));
      for (const IsaGpuTargetDescription &gpu_target : descriptor.gpu_targets) {
        for (const IsaGpuTargetDescription &existing_gpu_target : existing.gpu_targets) {
          if (existing_gpu_target.public_id == gpu_target.public_id)
            return registry_error(
                duplicate_message("GPU target", std::to_string(enum_value(gpu_target.public_id))));
          if (existing_gpu_target.elf_machine == gpu_target.elf_machine)
            return registry_error(
                duplicate_message("GPU ELF machine", std::to_string(gpu_target.elf_machine)));
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace

IsaTargetRegistry::IsaTargetRegistry(std::span<const IsaTargetDescriptor> targets)
    : targets_(targets), initialization_error_(validate_descriptors(targets)) {}

std::span<const IsaTargetDescriptor> IsaTargetRegistry::targets() const {
  return ok() ? targets_ : std::span<const IsaTargetDescriptor>{};
}

const IsaTargetDescriptor *IsaTargetRegistry::find(std::string_view id) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    if (contains_id(target, id))
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_arch_t architecture_id) const {
  if (!ok() || !is_public_architecture_key(architecture_id))
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    if (target.architecture_id == architecture_id)
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_target_id_t gpu_target_id) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaGpuTargetDescription &gpu_target : target.gpu_targets) {
      if (gpu_target.public_id == gpu_target_id)
        return &target;
    }
  }
  return nullptr;
}

const IsaGpuTargetDescription *
IsaTargetRegistry::find_gpu_target_by_elf_machine(uint32_t elf_machine) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaGpuTargetDescription &gpu_target : target.gpu_targets) {
      if (gpu_target.elf_machine == elf_machine)
        return &gpu_target;
    }
  }
  return nullptr;
}

const IsaGpuTargetDescription *
IsaTargetRegistry::find_gpu_target_by_code_object_id(std::string_view id) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaGpuTargetDescription &gpu_target : target.gpu_targets) {
      if (gpu_target.code_object_id == id)
        return &gpu_target;
    }
  }
  return nullptr;
}

} // namespace rocjitsu
