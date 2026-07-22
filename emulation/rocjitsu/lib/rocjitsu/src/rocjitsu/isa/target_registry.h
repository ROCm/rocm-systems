// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_registry.h
/// @brief Scoped registry for statically selected ISA target providers.

#ifndef ROCJITSU_ISA_TARGET_REGISTRY_H_
#define ROCJITSU_ISA_TARGET_REGISTRY_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/execution_backend.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class Decoder;

/// @brief Non-owning code-object identity published by an AMDGPU ISA target.
struct IsaGpuTargetDescription {
  rj_code_target_id_t public_id;
  /// Processor name used in Clang offload bundle entry IDs (for example ``gfx90a``).
  std::string_view code_object_id;
  /// ELF ``e_flags & EF_AMDGPU_MACH`` value for standalone code objects.
  uint32_t elf_machine;
};

/// @brief Non-owning, compile-time description published by an ISA target.
struct IsaTargetDescription {
  std::string_view id;
  std::span<const std::string_view> aliases = {};
  /// Public architecture key bound by this provider, or INVALID when unbound.
  rj_code_arch_t architecture_id = ROCJITSU_CODE_ARCH_INVALID;
  std::span<const IsaGpuTargetDescription> gpu_targets = {};
  /// Whether decoded instructions carry executable callbacks.
  bool supports_execution = false;
};

/// @brief Owning code-object identity bound by one ISA provider.
struct IsaGpuTargetBinding {
  rj_code_target_id_t public_id;
  std::string code_object_id;
  uint32_t elf_machine;
};

/// @brief Immutable description contributed by one ISA provider.
struct IsaTargetDescriptor {
  using DecoderFactory = std::unique_ptr<Decoder> (*)(const IsaExecutionBackend *backend);

  /// Canonical target identity (for example ``gfx1250``).
  std::string id;
  /// Additional string identities accepted by lookup.
  std::vector<std::string> aliases = {};
  /// Public architecture enum key accepted by lookup, or INVALID when unbound.
  rj_code_arch_t architecture_id = ROCJITSU_CODE_ARCH_INVALID;
  /// Public GPU target keys and the code-object identities bound to them.
  std::vector<IsaGpuTargetBinding> gpu_targets = {};
  DecoderFactory decoder_factory = nullptr;
  bool supports_execution = false;
  /// Null for model-only targets and for unsplit ISAs with inline execution.
  const IsaExecutionBackend *execution_backend = nullptr;
};

/// @brief Recoverable registration error; ``std::nullopt`` means success.
using IsaTargetRegistryError = std::optional<std::string>;

/// @brief Ordinary provider function selected explicitly by a consumer's build.
using IsaTargetProvider = IsaTargetRegistryError (*)(class IsaTargetRegistry &registry);

/// @brief A consumer-owned registry that becomes immutable before use.
///
/// There is intentionally no singleton or dynamic-loading entry point. A final
/// tool or shared object constructs its own instance from the exact provider
/// list selected by its build, then freezes it before lookup.
class IsaTargetRegistry final {
public:
  IsaTargetRegistry() = default;
  IsaTargetRegistry(IsaTargetRegistry &&) noexcept = default;
  IsaTargetRegistry &operator=(IsaTargetRegistry &&) = delete;
  IsaTargetRegistry(const IsaTargetRegistry &) = delete;
  IsaTargetRegistry &operator=(const IsaTargetRegistry &) = delete;

  /// @brief Add one provider contribution during initialization.
  /// @returns ``std::nullopt`` on success or a diagnostic on failure.
  [[nodiscard]] IsaTargetRegistryError add(IsaTargetDescriptor descriptor);

  /// @brief Sort deterministically and make the registry read-only.
  /// @returns ``std::nullopt`` on success or a diagnostic on failure.
  [[nodiscard]] IsaTargetRegistryError freeze();
  /// @brief Whether the registry is immutable and ready for lookup.
  bool frozen() const noexcept { return frozen_; }
  /// @brief Whether static provider composition completed successfully.
  bool ok() const noexcept { return !initialization_error_.has_value(); }
  /// @brief Static provider-composition diagnostic, empty when ok().
  std::string_view error() const noexcept {
    return initialization_error_ ? std::string_view(*initialization_error_) : std::string_view{};
  }

  /// @brief Enumerate targets by canonical ID. Requires a frozen registry.
  std::span<const IsaTargetDescriptor> targets() const;

  /// @brief Look up a canonical ID or string alias. Requires freeze().
  const IsaTargetDescriptor *find(std::string_view id) const;
  /// @brief Look up a provider-bound public architecture key. Requires freeze().
  const IsaTargetDescriptor *find(rj_code_arch_t architecture_id) const;
  /// @brief Look up a provider-bound public GPU target key. Requires freeze().
  const IsaTargetDescriptor *find(rj_code_target_id_t gpu_target_id) const;
  /// @brief Look up the binding for a standalone ELF machine value. Requires freeze().
  const IsaGpuTargetBinding *find_gpu_target_by_elf_machine(uint32_t elf_machine) const;
  /// @brief Look up the binding for a Clang offload bundle processor ID. Requires freeze().
  const IsaGpuTargetBinding *find_gpu_target_by_code_object_id(std::string_view id) const;

private:
  friend IsaTargetRegistry make_isa_target_registry(std::span<const IsaTargetProvider> providers);

  std::vector<IsaTargetDescriptor> targets_;
  IsaTargetRegistryError initialization_error_;
  bool frozen_ = false;
};

/// @brief Construct a scoped registry from explicit provider calls.
/// @details On success the returned registry is frozen and ok(). On failure it
/// is not frozen, ok() is false, and error() contains the provider diagnostic.
IsaTargetRegistry make_isa_target_registry(std::span<const IsaTargetProvider> providers);

/// @brief Registry selected for a component's public enum and C entry points.
///
/// Its function-local registry is owned by one final linked image; it is not a
/// process-wide registry or shared with independently linked DSOs.
const IsaTargetRegistry &default_isa_target_registry();

} // namespace rocjitsu

#endif // ROCJITSU_ISA_TARGET_REGISTRY_H_
