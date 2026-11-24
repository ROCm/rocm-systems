// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/isa/isa_family.h"

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

constexpr bool is_family_key(rocjitsu::IsaFamily key) {
  const std::underlying_type_t<rocjitsu::IsaFamily> value = enum_value(key);
  return value >= enum_value(rocjitsu::IsaFamily::Cdna1) &&
         value < enum_value(rocjitsu::IsaFamily::Count);
}

constexpr bool is_public_target_key(rj_code_target_id_t key) {
  const std::underlying_type_t<rj_code_target_id_t> value = enum_value(key);
  return value >= enum_value(ROCJITSU_CODE_TARGET_GFX90A) &&
         value < enum_value(ROCJITSU_CODE_TARGET_NUM_TARGETS);
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
  if (descriptor.family != rocjitsu::IsaFamily::Invalid && !is_family_key(descriptor.family))
    return registry_error("ISA target contains an unallocated ISA family value");

  for (size_t binding_index = 0; binding_index < descriptor.bindings.size(); ++binding_index) {
    const IsaTargetBinding &binding = descriptor.bindings[binding_index];
    if (!is_public_target_key(binding.public_id))
      return registry_error("ISA target contains an unallocated target enum value");
    if (binding.id.empty())
      return registry_error("ISA target '" + std::string(descriptor.id) +
                            "' has an empty target ID");
    if (!contains_id(descriptor, binding.id))
      return registry_error("ISA target '" + std::string(descriptor.id) + "' has target ID '" +
                            std::string(binding.id) + "' that is not an ID or alias");

    for (size_t previous_index = 0; previous_index < binding_index; ++previous_index) {
      const IsaTargetBinding &previous = descriptor.bindings[previous_index];
      if (previous.public_id == binding.public_id)
        return registry_error(
            duplicate_message("concrete target", std::to_string(enum_value(binding.public_id))));
      if (previous.id == binding.id)
        return registry_error(duplicate_message("target ID", binding.id));
      if (binding.elf_machine != 0 && previous.elf_machine == binding.elf_machine)
        return registry_error(
            duplicate_message("GPU ELF machine", std::to_string(binding.elf_machine)));
    }
  }
  if (!descriptor.bindings.empty() && descriptor.family == rocjitsu::IsaFamily::Invalid)
    return registry_error("ISA target '" + std::string(descriptor.id) +
                          "' with target bindings must have an ISA family");
  if (descriptor.variant_decoder_factory != nullptr) {
    if (descriptor.default_target == ROCJITSU_CODE_TARGET_INVALID)
      return registry_error("ISA target '" + std::string(descriptor.id) +
                            "' with a variant decoder factory has no default concrete target");
    bool found_default = false;
    for (const IsaTargetBinding &binding : descriptor.bindings)
      found_default |= binding.public_id == descriptor.default_target;
    if (!found_default)
      return registry_error("ISA target '" + std::string(descriptor.id) +
                            "' default concrete target is not one of its bindings");
  } else if (descriptor.default_target != ROCJITSU_CODE_TARGET_INVALID) {
    return registry_error("ISA target '" + std::string(descriptor.id) +
                          "' has a default concrete target but no variant decoder factory");
  }
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

      if (descriptor.family != rocjitsu::IsaFamily::Invalid && existing.family == descriptor.family)
        return registry_error(
            duplicate_message("architecture", std::to_string(enum_value(descriptor.family))));
      for (const IsaTargetBinding &binding : descriptor.bindings) {
        for (const IsaTargetBinding &existing_binding : existing.bindings) {
          if (existing_binding.public_id == binding.public_id)
            return registry_error(duplicate_message("concrete target",
                                                    std::to_string(enum_value(binding.public_id))));
          if (binding.elf_machine != 0 && existing_binding.elf_machine == binding.elf_machine)
            return registry_error(
                duplicate_message("GPU ELF machine", std::to_string(binding.elf_machine)));
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

const IsaTargetDescriptor *IsaTargetRegistry::find(rocjitsu::IsaFamily family) const {
  if (!ok() || !is_family_key(family))
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    if (target.family == family)
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_target_id_t target_id) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaTargetBinding &binding : target.bindings) {
      if (binding.public_id == target_id)
        return &target;
    }
  }
  return nullptr;
}

const IsaTargetBinding *IsaTargetRegistry::find_target_by_elf_machine(uint32_t elf_machine) const {
  if (!ok() || elf_machine == 0)
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaTargetBinding &binding : target.bindings) {
      if (binding.elf_machine == elf_machine)
        return &binding;
    }
  }
  return nullptr;
}

const IsaTargetBinding *
IsaTargetRegistry::find_target_by_code_object_id(std::string_view id) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaTargetBinding &binding : target.bindings) {
      if (binding.elf_machine != 0 && binding.id == id)
        return &binding;
    }
  }
  return nullptr;
}

const IsaTargetBinding *IsaTargetRegistry::find_target(rj_code_target_id_t target_id) const {
  if (!ok() || !is_public_target_key(target_id))
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaTargetBinding &binding : target.bindings) {
      if (binding.public_id == target_id)
        return &binding;
    }
  }
  return nullptr;
}

const IsaTargetBinding *IsaTargetRegistry::find_target(std::string_view id) const {
  if (!ok())
    return nullptr;
  for (const IsaTargetDescriptor &target : targets_) {
    for (const IsaTargetBinding &binding : target.bindings) {
      if (binding.id == id)
        return &binding;
    }
  }
  return nullptr;
}

const IsaTargetBinding *
IsaTargetRegistry::find_default_target(const IsaTargetDescriptor &descriptor) const {
  if (!ok() || descriptor.default_target == ROCJITSU_CODE_TARGET_INVALID)
    return nullptr;
  for (const IsaTargetDescriptor &registered : targets_) {
    if (&registered != &descriptor)
      continue;
    for (const IsaTargetBinding &binding : registered.bindings) {
      if (binding.public_id == registered.default_target)
        return &binding;
    }
    return nullptr;
  }
  return nullptr;
}

} // namespace rocjitsu
