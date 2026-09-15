// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_generation_registry.h
/// @brief The GPU generations a device model can present, and how one is chosen.
///
/// @details A generation is everything that differs between parts: which IP
/// blocks the device publishes, at which versions, with which register-base
/// segments. Selecting one used to be an equality test against a single
/// hardcoded target, which answered "is this the one part we model" rather than
/// "which part is this" -- a shape that cannot express two.
///
/// This deliberately mirrors @ref rocjitsu::IsaTargetRegistry rather than
/// inventing a second registry idiom: static descriptors validated once at
/// construction, lookup by key, no singleton and no dynamic loading. What it
/// does *not* copy is that registry's generated-header machinery, which exists
/// so separately linked images can ship different subsets of ISA targets. There
/// is no equivalent need here -- one image presents whichever part its config
/// names -- so the descriptors are a plain checked-in array.
///
/// A descriptor carries a factory rather than a built @ref IpDiscoverySpec: a
/// spec owns vectors, so it is not a literal type and could not live in a
/// constant array. That indirection is the same one @ref IsaTargetDescriptor
/// uses for its decoders, and for the same reason.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu {

/// @brief One GPU generation this device model can present.
class GpuGenerationDescriptor {
public:
  /// @brief Builds the blocks this generation publishes.
  using DiscoverySpecFactory = IpDiscoverySpec (*)(const GpuDiscoveryTopology &);

  /// @brief Canonical name, as a person writes it (for example ``gfx1250``).
  ///
  /// @details For humans and for tools that take a generation on a command
  /// line. Deliberately not what a config selects with -- see @ref
  /// gfx_target_versions.
  std::string_view id;

  /// @brief KFD target versions this generation answers for (for example
  /// 120500).
  ///
  /// @details The selection key, because it is already the identity a config
  /// states and the one KFD reports. Adding a second identifier to the config
  /// for this would create two statements of the same fact that could disagree,
  /// which is the failure the one-profile rule exists to prevent.
  ///
  /// A span rather than a single value, because one modelled generation can
  /// legitimately answer for several closely related targets.
  std::span<const uint32_t> gfx_target_versions;

  /// @brief Builds this generation's discovery spec. Never null.
  DiscoverySpecFactory discovery_factory = nullptr;
};

/// @brief Recoverable composition error; ``std::nullopt`` means success.
using GpuGenerationRegistryError = std::optional<std::string>;

/// @brief An immutable view over the generations an image was built with.
///
/// @details Holds a non-owning view, so the descriptor array must outlive it.
/// In practice that array is a constant with static storage duration.
class GpuGenerationRegistry final {
public:
  /// @brief Compose over @p descriptors and validate them.
  /// @param[in] descriptors Generations to offer; must outlive this registry.
  template <std::size_t DescriptorCount>
  explicit GpuGenerationRegistry(const GpuGenerationDescriptor (&descriptors)[DescriptorCount])
      : GpuGenerationRegistry(std::span<const GpuGenerationDescriptor>{descriptors}) {}
  template <std::size_t DescriptorCount>
  explicit GpuGenerationRegistry(
      const std::array<GpuGenerationDescriptor, DescriptorCount> &descriptors)
      : GpuGenerationRegistry(std::span<const GpuGenerationDescriptor>{descriptors}) {}

  // Only a lasting array may be registered: a temporary would leave every
  // lookup reading freed descriptors.
  template <std::size_t DescriptorCount>
  GpuGenerationRegistry(std::array<GpuGenerationDescriptor, DescriptorCount> &) = delete;
  template <std::size_t DescriptorCount>
  GpuGenerationRegistry(std::array<GpuGenerationDescriptor, DescriptorCount> &&) = delete;
  template <std::size_t DescriptorCount>
  GpuGenerationRegistry(const std::array<GpuGenerationDescriptor, DescriptorCount> &&) = delete;
  template <std::size_t DescriptorCount>
  GpuGenerationRegistry(GpuGenerationDescriptor (&)[DescriptorCount]) = delete;
  template <std::size_t DescriptorCount>
  GpuGenerationRegistry(GpuGenerationDescriptor (&&)[DescriptorCount]) = delete;
  template <std::size_t DescriptorCount>
  GpuGenerationRegistry(const GpuGenerationDescriptor (&&)[DescriptorCount]) = delete;

  GpuGenerationRegistry(const GpuGenerationRegistry &) = delete;
  GpuGenerationRegistry &operator=(const GpuGenerationRegistry &) = delete;
  GpuGenerationRegistry(GpuGenerationRegistry &&) = delete;
  GpuGenerationRegistry &operator=(GpuGenerationRegistry &&) = delete;
  ~GpuGenerationRegistry() = default;

  /// @brief Whether the descriptors were consistent.
  [[nodiscard]] bool ok() const { return !initialization_error_.has_value(); }

  /// @brief Why composition failed, when it did.
  [[nodiscard]] const GpuGenerationRegistryError &error() const { return initialization_error_; }

  /// @brief The generations offered, in the order they were registered.
  [[nodiscard]] std::span<const GpuGenerationDescriptor> generations() const {
    return descriptors_;
  }

  /// @brief Find the generation answering for @p gfx_target_version.
  /// @returns The descriptor, or nullptr when no generation claims that target
  ///          or the registry did not compose.
  [[nodiscard]] const GpuGenerationDescriptor *find(uint32_t gfx_target_version) const;

  /// @brief Find the generation named @p id.
  /// @returns The descriptor, or nullptr when nothing is named that or the
  ///          registry did not compose.
  [[nodiscard]] const GpuGenerationDescriptor *find(std::string_view id) const;

  /// @brief The registered names, comma separated, for a diagnostic.
  [[nodiscard]] std::string known_ids() const;

private:
  explicit GpuGenerationRegistry(std::span<const GpuGenerationDescriptor> descriptors)
      : descriptors_(descriptors), initialization_error_(validate(descriptors_)) {}

  [[nodiscard]] static GpuGenerationRegistryError
  validate(std::span<const GpuGenerationDescriptor> descriptors);

  std::span<const GpuGenerationDescriptor> descriptors_;
  GpuGenerationRegistryError initialization_error_;
};

/// @brief The generations this image was built with.
///
/// @details One composition per image, over the checked-in descriptor array.
/// Nothing about it is dynamic; it is a function rather than a constant only so
/// that validation runs once and its result can be reported.
[[nodiscard]] const GpuGenerationRegistry &gpu_generations();

} // namespace rocjitsu
