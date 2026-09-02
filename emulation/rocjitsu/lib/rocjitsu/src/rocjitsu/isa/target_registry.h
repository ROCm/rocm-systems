// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_registry.h
/// @brief Scoped registry for statically selected ISA target providers.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu {

class Decoder;

/// @brief Immutable semantic capabilities of one concrete GPU target.
///
/// Capability bits are defined by the ISA family that consumes them. They
/// describe instruction/encoding legality, rather than naming a GPU revision.
struct IsaTargetCapabilities {
  uint64_t instruction_features = 0;
  /// Whether S_SETREG writes to MODE have the VGPR-MSB clobber and adjacency
  /// hazard that require the target-specific software fixup.
  bool setreg_vgpr_msb_fixup = false;
  /// Whether the selected provider implements execution for this target.
  bool execution_implemented = false;
};

/// @brief Non-owning code-object identity published by an AMDGPU ISA target.
struct IsaGpuTargetDescription {
  rj_code_target_id_t public_id;
  /// Processor name used in Clang offload bundle entry IDs (for example ``gfx90a``).
  std::string_view code_object_id;
  /// ELF ``e_flags & EF_AMDGPU_MACH`` value for standalone code objects.
  uint32_t elf_machine;
  /// KFD packed target version (for example 120501 for gfx1251), or zero.
  uint32_t gfx_target_version = 0;
  /// Semantic capabilities selected by this concrete target.
  IsaTargetCapabilities capabilities = {};
};

/// @brief Static, non-owning description contributed by one ISA target.
struct IsaTargetDescriptor {
  using DecoderFactory = std::unique_ptr<Decoder> (*)();
  using VariantDecoderFactory =
      std::unique_ptr<Decoder> (*)(const IsaGpuTargetDescription &gpu_target);

  /// Canonical target identity (for example ``gfx1250``).
  std::string_view id;
  /// Additional string identities accepted by lookup.
  std::span<const std::string_view> aliases = {};
  /// Public architecture enum key accepted by lookup, or INVALID when unbound.
  rj_code_arch_t architecture_id = ROCJITSU_CODE_ARCH_INVALID;
  /// Public GPU target keys and the code-object identities bound to them.
  std::span<const IsaGpuTargetDescription> gpu_targets = {};
  /// Concrete target selected by architecture-only lookup, or INVALID when
  /// this provider does not have target-specific decoder behavior.
  rj_code_target_id_t default_gpu_target = ROCJITSU_CODE_TARGET_INVALID;
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
  /// @brief Look up a descriptor-bound public architecture key.
  const IsaTargetDescriptor *find(rj_code_arch_t architecture_id) const;
  /// @brief Look up a descriptor-bound public GPU target key.
  const IsaTargetDescriptor *find(rj_code_target_id_t gpu_target_id) const;
  /// @brief Look up the binding for a standalone ELF machine value.
  const IsaGpuTargetDescription *find_gpu_target_by_elf_machine(uint32_t elf_machine) const;
  /// @brief Look up the binding for a Clang offload bundle processor ID.
  const IsaGpuTargetDescription *find_gpu_target_by_code_object_id(std::string_view id) const;
  /// @brief Look up a binding by its public target enum.
  const IsaGpuTargetDescription *find_gpu_target(rj_code_target_id_t gpu_target_id) const;
  /// @brief Resolve a descriptor's explicit architecture-only default binding.
  const IsaGpuTargetDescription *
  find_default_gpu_target(const IsaTargetDescriptor &descriptor) const;

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
