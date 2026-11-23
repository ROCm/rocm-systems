// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_registry.h
/// @brief Scoped registry for statically selected ISA target providers.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/isa_family.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu {

class Decoder;

/// @brief Immutable semantic capabilities of one concrete target.
///
/// Capability bits are defined by the ISA family that consumes them. They
/// describe instruction/encoding legality and target-specific semantics rather
/// than asking consumers to name a GPU revision.
struct IsaTargetCapabilities {
  uint64_t instruction_features = 0;
  /// Whether S_SETREG writes to MODE have the VGPR-MSB clobber and
  /// S_SETREG_IMM32_B32(MODE) has the adjacency hazard that require the
  /// target-specific software fixup.
  bool setreg_vgpr_msb_fixup = false;
  /// Whether the selected provider implements execution for this target.
  bool execution_implemented = false;
};

/// @brief Concrete target identity and capabilities published by an ISA provider.
struct IsaTargetBinding {
  rj_code_target_id_t public_id;
  /// Canonical target name, also used in AMDGPU offload bundle entry IDs.
  std::string_view id;
  /// AMDGPU ELF machine value, or zero for a target without AMDGPU code objects.
  uint32_t elf_machine = 0;
  /// KFD packed target version (for example 120501 for gfx1251), or zero.
  uint32_t gfx_target_version = 0;
  /// Semantic capabilities selected by this concrete target.
  IsaTargetCapabilities capabilities = {};
};

/// @brief Static, non-owning description contributed by one ISA target.
struct IsaTargetDescriptor {
  using DecoderFactory = std::unique_ptr<Decoder> (*)();
  using VariantDecoderFactory = std::unique_ptr<Decoder> (*)(const IsaTargetBinding &gpu_target);

  /// Canonical target identity (for example ``gfx1250``).
  std::string_view id;
  /// Additional string identities accepted by lookup.
  std::span<const std::string_view> aliases = {};
  /// Shared implementation family, or Invalid for a source-integrated provider.
  rocjitsu::IsaFamily family = rocjitsu::IsaFamily::Invalid;
  /// Concrete target keys, names, and optional AMDGPU code-object identities.
  std::span<const IsaTargetBinding> bindings = {};
  /// Concrete target selected by architecture-only lookup, or INVALID when
  /// this provider does not have target-specific decoder behavior.
  rj_code_target_id_t default_target = ROCJITSU_CODE_TARGET_INVALID;
  /// Factory used to construct this target's decoder.
  DecoderFactory decoder_factory = nullptr;
  /// Optional factory used when lookup selected a concrete GPU binding.
  VariantDecoderFactory variant_decoder_factory = nullptr;
  /// Whether decoded instructions include execution callbacks in this image.
  bool supports_execution = false;
};

/// @brief Recoverable composition error; ``std::nullopt`` means success.
using IsaTargetRegistryError = std::optional<std::string>;

/// @brief An immutable view over a consumer's statically selected ISA targets.
///
/// There is intentionally no singleton or dynamic-loading entry point. A final
/// tool or shared object constructs its own instance over the exact static
/// descriptor array selected by its build.
class IsaTargetRegistry final {
public:
  /// @pre @p targets and all storage referenced by its spans remain alive and
  /// unchanged for this registry's lifetime. Generated compositions satisfy
  /// this with static constexpr storage.
  template <size_t Size>
  explicit IsaTargetRegistry(const std::array<IsaTargetDescriptor, Size> &targets)
      : IsaTargetRegistry(std::span<const IsaTargetDescriptor>{targets}) {}
  template <size_t Size>
  explicit IsaTargetRegistry(const IsaTargetDescriptor (&targets)[Size])
      : IsaTargetRegistry(std::span<const IsaTargetDescriptor>{targets}) {}
  template <size_t Size> IsaTargetRegistry(std::array<IsaTargetDescriptor, Size> &) = delete;
  template <size_t Size> IsaTargetRegistry(std::array<IsaTargetDescriptor, Size> &&) = delete;
  template <size_t Size> IsaTargetRegistry(const std::array<IsaTargetDescriptor, Size> &&) = delete;
  template <size_t Size> IsaTargetRegistry(IsaTargetDescriptor (&)[Size]) = delete;
  template <size_t Size> IsaTargetRegistry(IsaTargetDescriptor (&&)[Size]) = delete;
  template <size_t Size> IsaTargetRegistry(const IsaTargetDescriptor (&&)[Size]) = delete;

  IsaTargetRegistry(IsaTargetRegistry &&) = delete;
  IsaTargetRegistry &operator=(IsaTargetRegistry &&) = delete;
  IsaTargetRegistry(const IsaTargetRegistry &) = delete;
  IsaTargetRegistry &operator=(const IsaTargetRegistry &) = delete;

  /// @brief Whether static descriptor validation completed successfully.
  bool ok() const noexcept { return !initialization_error_.has_value(); }
  /// @brief Static descriptor-composition diagnostic, empty when ok().
  std::string_view error() const noexcept {
    return initialization_error_ ? std::string_view(*initialization_error_) : std::string_view{};
  }

  /// @brief Enumerate targets in composition order.
  std::span<const IsaTargetDescriptor> targets() const;

  /// @brief Look up a canonical ID or string alias.
  const IsaTargetDescriptor *find(std::string_view id) const;
  /// @brief Look up an internal implementation family.
  const IsaTargetDescriptor *find(rocjitsu::IsaFamily family) const;
  /// @brief Look up a descriptor-bound public target key.
  const IsaTargetDescriptor *find(rj_code_target_id_t target_id) const;
  /// @brief Look up the binding for a standalone ELF machine value.
  const IsaTargetBinding *find_target_by_elf_machine(uint32_t elf_machine) const;
  /// @brief Look up the binding for a Clang offload bundle processor ID.
  const IsaTargetBinding *find_target_by_code_object_id(std::string_view id) const;
  /// @brief Look up a binding by its public target enum.
  const IsaTargetBinding *find_target(rj_code_target_id_t target_id) const;
  /// @brief Look up a concrete binding by its canonical target name.
  const IsaTargetBinding *find_target(std::string_view id) const;
  /// @brief Resolve a descriptor's explicit architecture-only default binding.
  const IsaTargetBinding *find_default_target(const IsaTargetDescriptor &descriptor) const;

private:
  explicit IsaTargetRegistry(std::span<const IsaTargetDescriptor> targets);

  const std::span<const IsaTargetDescriptor> targets_;
  const IsaTargetRegistryError initialization_error_;
};

/// @brief Registry selected for a component's public enum and C entry points.
///
/// Its function-local registry is owned by one final linked image; it is not a
/// process-wide registry or shared with independently linked DSOs.
const IsaTargetRegistry &default_isa_target_registry();

} // namespace rocjitsu
