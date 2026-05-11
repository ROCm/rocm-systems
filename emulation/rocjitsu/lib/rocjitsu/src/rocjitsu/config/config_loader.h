// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file config_loader.h
/// @brief Configuration loading from JSON via FlatBuffers schema.

#ifndef ROCJITSU_CONFIG_CONFIG_LOADER_H_
#define ROCJITSU_CONFIG_CONFIG_LOADER_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
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
///   auto loaded = load_config("config.json", "schema.fbs");
///   SimulationEngine engine(loaded.engine_config);
///   engine.topology().set_root(loaded.take_root());
///   loaded.wire_links(engine.topology());
///   engine.build();
/// @endcode
/// @brief KFD device identity extracted from vm.gpu.device in the config.
/// All values use the same names as Sysfs::GpuInfo fields for easy mapping.
struct KfdDeviceConfig {
  uint32_t gpu_id = 0;
  uint32_t gfx_target_version = 0;
  uint32_t vendor_id = 0x1002;
  uint32_t device_id = 0;
  uint32_t family_id = 0;
  uint64_t unique_id = 0;
  std::string marketing_name;
  uint32_t drm_render_minor = 128;
  uint32_t simd_count = 0;
  uint32_t max_waves_per_simd = 10;
  uint32_t num_shader_engines = 0;
  uint32_t num_shader_arrays_per_engine = 1;
  uint32_t num_cu_per_sh = 0;
  uint32_t simd_per_cu = 4;
  uint32_t wave_front_size = 64;
  uint32_t max_slots_scratch_cu = 32;
  uint64_t local_mem_size = 0;
  uint32_t lds_size_kb = 64;
  uint32_t mem_width = 4096;
  uint32_t mem_clk_max = 1200;
  uint32_t l1_size_kb = 32;
  uint32_t l1_line_size = 128;
  uint32_t l1_assoc = 4;
  uint32_t l2_size_kb = 4096;
  uint32_t l2_line_size = 128;
  uint32_t l2_assoc = 16;
  uint32_t num_sdma_engines = 2;
  uint32_t num_sdma_xgmi_engines = 0;
  uint32_t num_cp_queues = 128;
  uint32_t max_engine_clk_fcompute = 2100;
  bool present = false; ///< True if device section existed in config.
};

struct LoadedConfig {
  simdojo::SimulationEngine::Config engine_config;
  TopologyBuildResult build_result;
  simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL;
  KfdDeviceConfig device; ///< KFD device identity from vm.gpu.device.

  /// @brief Return the SoC (root component, typed).
  SoC *soc() { return dynamic_cast<SoC *>(build_result.root.get()); }

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
/// @param schema_path Path to the simulation_config.fbs schema file.
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
LoadedConfig load_config(const std::string &json_path, const std::string &schema_path);

/// @brief Load simulation config from a JSON string.
/// @param json JSON configuration string.
/// @param schema_path Path to the simulation_config.fbs schema file.
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on parse errors or invalid config.
LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_path);

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_LOADER_H_
