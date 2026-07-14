// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file config_loader.h
/// @brief Configuration loading from JSON via FlatBuffers schema.

#ifndef ROCJITSU_CONFIG_CONFIG_LOADER_H_
#define ROCJITSU_CONFIG_CONFIG_LOADER_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/config/kfd_device_config.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
class SoC;
namespace amdgpu {
class GpuMemory;
class Xcd;
} // namespace amdgpu

namespace config {

/// @brief Result of building a declarative topology.
///
/// Provides convenience accessors for navigating the GPU component hierarchy.
/// These pointers are non-owning — the component tree owns the objects.
struct TopologyBuildResult {
  std::unique_ptr<simdojo::CompositeComponent> root;
  amdgpu::GpuMemory *memory = nullptr;
  std::vector<simdojo::LinkSpec> link_specs; ///< Deferred link specs for wiring.

  /// @brief Convenience: number of XCDs in the topology.
  uint32_t num_xcds = 0;
  /// @brief Convenience: XCD pointers (non-owning, into root's subtree).
  std::vector<amdgpu::Xcd *> xcds;
};

/// @brief Result of loading a simulation configuration.
///
/// Contains the engine configuration and the built component tree. Provides
/// convenience accessors for the SoC and GPU memory. After loading, wire
/// the topology into a SimulationEngine:
/// @code
///   auto loaded = load_config("config.json", kEmbeddedSchema);
///   SimulationEngine engine(loaded.engine_config);
///   engine.topology().set_root(loaded.take_root());
///   loaded.wire_links(engine.topology());
///   engine.build();
/// @endcode
struct LoadedConfig {
  simdojo::SimulationEngine::Config engine_config;
  TopologyBuildResult build_result;
  std::vector<TopologyBuildResult>
      extra_gpu_builds; ///< Additional GPU SoC trees (for num_gpus > 1).
  simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL;
  KfdDeviceConfig device;               ///< KFD device identity from vm.gpu.device.
  DbtGuestConfig dbt_guest;             ///< Optional DBT guest-GPU discovery config.
  uint32_t num_gpus = 1;                ///< Number of simulated GPU instances.
  std::vector<KfdDeviceConfig> devices; ///< Per-GPU configs (populated when num_gpus > 1).

  /// @brief Return the SoC from the topology root.
  SoC *soc();

  /// @brief Return GPU memory.
  amdgpu::GpuMemory *memory() { return build_result.memory; }

  /// @brief Transfer ownership of the root component to the caller.
  std::unique_ptr<simdojo::CompositeComponent> take_root() { return std::move(build_result.root); }

  /// @brief Wire deferred link specs into a topology. Call after set_root().
  void wire_links(simdojo::Topology &topo) { topo.wire_links(build_result.link_specs, exec_mode); }
};

/// @brief Parse an architecture name string to an rj_code_arch_t enum value.
rj_code_arch_t parse_arch(const std::string &arch_str);

/// @brief Convert an rj_code_arch_t enum to its string name.
const char *arch_to_string(rj_code_arch_t arch);

/// @brief Load simulation config from a JSON file.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
LoadedConfig load_config(const std::string &json_path, const std::string &schema_text);

/// @brief Load simulation config from a JSON string.
/// @param json JSON configuration string.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on parse errors or invalid config.
LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_text);

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_LOADER_H_
