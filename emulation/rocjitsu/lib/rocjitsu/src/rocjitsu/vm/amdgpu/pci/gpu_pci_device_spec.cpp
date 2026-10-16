// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"

#include "rocjitsu/vm/amdgpu/pci/gpu_generation_registry.h"
#include "util/log.h"

#include <algorithm>
#include <format>

namespace rocjitsu {
namespace {

/// @brief Largest power of two no larger than @p limit, or zero if there is
/// none.
///
/// @details A limit of zero has no answer, and returning one would turn a
/// missing memory size into a one-byte aperture that looks valid.
uint64_t largest_power_of_two_within(uint64_t limit) {
  if (limit == 0) {
    return 0;
  }
  uint64_t size = 1;
  while (size <= limit / 2) {
    size *= 2;
  }
  return size;
}

/// @brief Largest window onto video memory chosen when a config asks for none.
///
/// @details Memory capacities are rarely powers of two and are often enormous,
/// so exposing all of one as a BAR is neither legal nor necessary: the indirect
/// window reaches whatever the aperture does not.
constexpr uint64_t kDefaultVramApertureBytes = 256 * 1024 * 1024;

bool has_discovery_override(const config::KfdDiscoveryOverrides &overrides) {
  return overrides.num_shader_engines.has_value() ||
         overrides.num_shader_arrays_per_engine.has_value() ||
         overrides.num_cu_per_sh.has_value() || overrides.wave_front_size.has_value() ||
         overrides.max_waves_per_simd.has_value() || overrides.max_slots_scratch_cu.has_value() ||
         overrides.lds_size_kb.has_value();
}

bool has_default_direct_topology(const config::KfdDeviceConfig &device) {
  const config::KfdDeviceConfig defaults;
  return device.num_shader_engines == defaults.num_shader_engines &&
         device.num_shader_arrays_per_engine == defaults.num_shader_arrays_per_engine &&
         device.num_cu_per_sh == defaults.num_cu_per_sh &&
         device.wave_front_size == defaults.wave_front_size &&
         device.max_waves_per_simd == defaults.max_waves_per_simd &&
         device.max_slots_scratch_cu == defaults.max_slots_scratch_cu &&
         device.lds_size_kb == defaults.lds_size_kb;
}

/// @brief Choose the IP blocks to describe for @p device.
///
/// @details Resolved through the generation registry rather than compared
/// against one modelled part, so the question asked is "which part is this"
/// rather than "is this the part". A target no generation answers for gets no
/// blocks: publishing one part's table for a configuration that models another
/// would have the guest driver bind support for hardware the rest of the
/// simulation is not, which fails later and further away than refusing here. An
/// empty profile makes the device refuse to become usable, and says why.
/// @param[in] device The configured device.
/// @returns The blocks to describe, empty if this part has no profile.
[[nodiscard]] IpDiscoverySpec discovery_spec_for(const config::KfdDeviceConfig &device) {
  const GpuGenerationRegistry &generations = gpu_generations();
  // A registry that did not compose is a build-time mistake rather than a
  // configuration one, and it would otherwise present as every part being
  // unknown.
  if (!generations.ok()) {
    util::Logger::warn(std::format("the GPU generations this build offers are inconsistent: {}",
                                   *generations.error()));
    return {};
  }
  const GpuGenerationDescriptor *generation = generations.find(device.gfx_target_version);
  if (generation == nullptr) {
    util::Logger::warn(std::format(
        "gfx target {} has no IP discovery profile, so this device cannot describe itself to a "
        "guest driver; modelled generations are {}",
        device.gfx_target_version, generations.known_ids()));
    return {};
  }
  // Public config loaders resolve omitted fields from the generation and copy
  // explicit overrides into these KFD-facing values. Direct C++ callers may
  // still provide only a target, so preserve that historical shorthand unless
  // a loader recorded an explicit zero override. A caller that supplies a
  // complete topology remains authoritative.
  GpuDiscoveryTopology topology = generation->topology_defaults;
  const bool complete_topology =
      device.num_shader_engines != 0 && device.num_shader_arrays_per_engine != 0 &&
      device.num_cu_per_sh != 0 && device.wave_front_size != 0 && device.max_waves_per_simd != 0 &&
      device.max_slots_scratch_cu != 0 && device.lds_size_kb != 0;
  const config::KfdDiscoveryOverrides &overrides = device.discovery_overrides;
  const auto explicitly_zero = [](const std::optional<uint32_t> &value) {
    return value.has_value() && *value == 0;
  };
  const bool has_explicit_zero =
      explicitly_zero(overrides.num_shader_engines) ||
      explicitly_zero(overrides.num_shader_arrays_per_engine) ||
      explicitly_zero(overrides.num_cu_per_sh) || explicitly_zero(overrides.wave_front_size) ||
      explicitly_zero(overrides.max_waves_per_simd) ||
      explicitly_zero(overrides.max_slots_scratch_cu) || explicitly_zero(overrides.lds_size_kb);
  if (complete_topology || has_explicit_zero) {
    topology.shader_engines = device.num_shader_engines;
    topology.shader_arrays_per_engine = device.num_shader_arrays_per_engine;
    topology.compute_units_per_shader_array = device.num_cu_per_sh;
    topology.wavefront_size = device.wave_front_size;
    topology.max_waves_per_simd = device.max_waves_per_simd;
    topology.max_scratch_slots_per_cu = device.max_slots_scratch_cu;
    topology.lds_size_kb = device.lds_size_kb;
  } else if (has_discovery_override(overrides) || !has_default_direct_topology(device)) {
    util::Logger::warn(std::format(
        "gfx target {} has a partially specified direct topology; resolve generation defaults "
        "before constructing its PCI discovery table",
        device.gfx_target_version));
    return {};
  }
  return generation->discovery_factory(topology);
}

} // namespace

GpuPciDeviceSpec gpu_pci_spec_from_config(const config::KfdDeviceConfig &device,
                                          const config::PciDeviceConfig &pci) {
  GpuPciDeviceSpec spec;
  spec.id.vendor = static_cast<uint16_t>(device.vendor_id);
  spec.id.device = static_cast<uint16_t>(device.device_id);
  // A subsystem that names itself after the device is the common case for a
  // reference board, so an unset value follows the device rather than reading
  // as an unrelated vendor.
  spec.id.subsys_vendor = static_cast<uint16_t>(
      pci.subsystem_vendor_id != 0 ? pci.subsystem_vendor_id : device.vendor_id);
  spec.id.subsys =
      static_cast<uint16_t>(pci.subsystem_id != 0 ? pci.subsystem_id : device.device_id);
  spec.id.cls = static_cast<uint8_t>((pci.class_code >> 16) & 0xff);
  spec.id.subcls = static_cast<uint8_t>((pci.class_code >> 8) & 0xff);
  spec.id.prog_if = static_cast<uint8_t>(pci.class_code & 0xff);
  spec.id.revision = static_cast<uint8_t>(device.pci_revision_id);

  spec.vram_bytes = device.local_mem_size;
  // An unset aperture is resolved here rather than left for each consumer to
  // interpret, so everything downstream sees the same explicit window.
  spec.vram_aperture_bytes =
      pci.vram_aperture_bytes != 0
          ? pci.vram_aperture_bytes
          : largest_power_of_two_within(std::min(device.local_mem_size, kDefaultVramApertureBytes));
  spec.doorbell_aperture_bytes = pci.doorbell_aperture_bytes;
  spec.register_aperture_bytes = pci.register_aperture_bytes;
  // The blocks are chosen here, next to the identity, so the two describe one
  // GPU.
  spec.discovery = discovery_spec_for(device);
  return spec;
}

} // namespace rocjitsu
