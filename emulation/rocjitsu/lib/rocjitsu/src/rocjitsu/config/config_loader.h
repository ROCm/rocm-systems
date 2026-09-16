// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file config_loader.h
/// @brief Configuration loading from JSON via FlatBuffers schema.

#ifndef ROCJITSU_CONFIG_CONFIG_LOADER_H_
#define ROCJITSU_CONFIG_CONFIG_LOADER_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/config/kfd_device_config.h"
#include "rocjitsu/config/pci_device_config.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <cstddef>
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

/// Default host-wide thread cap for automatic functional CU dispatch.
///
/// This conservative policy limit bounds persistent worker allocation on
/// large hosts while retaining substantial CU parallelism. It is not a
/// hardware limit; embedding callers can override it.
inline constexpr uint32_t kDefaultCpuDispatchThreadCap = 32;

/// Resolve the requested functional CU-dispatch width for each SoC.
///
/// A nonzero request is applied independently to every SoC. Zero selects one
/// capped host-wide budget that is divided as evenly as possible across the
/// SoCs. Every SoC retains a minimum width of one (serial dispatch).
/// A zero SoC count returns an empty vector. This is a defensive path for an
/// empty or already-consumed LoadedConfig; VM creation rejects a missing
/// primary SoC before applying the policy.
///
/// @code
/// resolve_cpu_dispatch_thread_budgets(0, 128, 4, 12); // {3, 3, 3, 3}
/// resolve_cpu_dispatch_thread_budgets(7, 2, 3, 12);   // {7, 7, 7}
/// @endcode
std::vector<uint32_t>
resolve_cpu_dispatch_thread_budgets(uint32_t requested_threads, uint32_t hardware_threads,
                                    size_t soc_count,
                                    uint32_t automatic_thread_cap = kDefaultCpuDispatchThreadCap);

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
/// the topology into a SimulationEngine. Multi-threaded rocjitsu topologies
/// use the XCD-aware policy from rocjitsu/vm/amdgpu/partitioning.h.
///
/// load_config() has already resolved engine_config.num_threads by the time it
/// returns: a config that omits the field (or sets it to 0) comes back holding
/// min(available host threads, XCD count). Overriding it here is what the clamp
/// is for; pin 1 when the consumer needs SimulationEngine::step(), which
/// rejects a multi-partition engine.
/// @code
///   #include "rocjitsu/vm/amdgpu/partitioning.h"
///
///   config::LoadedConfig loaded = load_config("config.json", kEmbeddedSchema);
///   SoC *soc = loaded.soc();
///   loaded.apply_cpu_dispatch_threads();
///
///   // Only needed to override the resolved default; requested_threads may
///   // exceed the XCD count, and the clamp brings it back into range.
///   loaded.engine_config.num_threads =
///       rocjitsu::amdgpu::clamp_xcd_partition_count(soc, requested_threads);
///
///   simdojo::SimulationEngine engine(loaded.engine_config);
///   engine.topology().set_root(loaded.take_root());
///   loaded.wire_links(engine.topology());
///
///   // A multi-partition engine must have a partition policy installed before
///   // create(), which otherwise throws.
///   if (loaded.engine_config.num_threads > 1 &&
///       !rocjitsu::amdgpu::partition_topology_by_xcds(
///           engine.topology(), soc, loaded.engine_config.num_threads))
///     throw std::invalid_argument("multi-threaded topology requires an XCD");
///   engine.create();
/// @endcode
struct LoadedConfig {
  simdojo::SimulationEngine::Config engine_config;
  TopologyBuildResult build_result;
  std::vector<TopologyBuildResult>
      extra_gpu_builds; ///< Additional GPU SoC trees (for num_gpus > 1).
  simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL;
  KfdDeviceConfig device;               ///< KFD device identity from vm.gpu.device.
  PciDeviceConfig pci;                  ///< PCI bus shape from vm.gpu.pci.
  DbtGuestConfig dbt_guest;             ///< Optional DBT guest-GPU discovery config.
  uint32_t num_gpus = 1;                ///< Number of simulated GPU instances.
  std::vector<KfdDeviceConfig> devices; ///< Per-GPU configs (populated when num_gpus > 1).
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  /// Requested functional dispatch width. Automatic mode uses a host-wide
  /// budget capped at 32; each SoC's effective width is CU-capacity-clamped.
  uint32_t cpu_dispatch_threads = 1;

  /// @brief Apply the requested functional dispatch policy to every loaded SoC.
  ///
  /// Call this before transferring topology ownership with take_root(). The
  /// no-argument overload detects the current host's hardware-thread count.
  void apply_cpu_dispatch_threads();

  /// @brief Apply the dispatch policy using supplied automatic-mode inputs.
  /// @details This overload makes host-dependent policy explicit for embedding
  /// callers and deterministic tests.
  /// @param hardware_threads Host-thread count available for automatic sizing.
  /// @param automatic_thread_cap Maximum host-wide width in automatic mode;
  /// values below one are treated as one.
  void apply_cpu_dispatch_threads(uint32_t hardware_threads,
                                  uint32_t automatic_thread_cap = kDefaultCpuDispatchThreadCap);

  /// @brief Return the SoC from the topology root.
  SoC *soc();

  /// @brief Return GPU memory.
  amdgpu::GpuMemory *memory() { return build_result.memory; }

  /// @brief Transfer ownership of the root component to the caller.
  std::unique_ptr<simdojo::CompositeComponent> take_root() { return std::move(build_result.root); }

  /// @brief Wire deferred link specs into a topology. Call after set_root().
  void wire_links(simdojo::Topology &topo) { topo.wire_links(build_result.link_specs, exec_mode); }
};

/// @brief The identity and bus shape of the GPU a config describes.
struct DeviceIdentityConfig {
  KfdDeviceConfig device; ///< GPU identity, as KFD also reports it.
  PciDeviceConfig pci;    ///< Bus shape for PCI-attached front ends.
};

/// @brief Read only a config's device identity and bus shape.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @returns The identity and bus shape.
/// @throws std::runtime_error when the file cannot be read or parsed.
/// @details Building the topology allocates the whole simulated machine, which
/// for a large part is gigabytes of register files. A front end that only needs
/// to know which GPU to present should not pay for a machine it never runs.
DeviceIdentityConfig load_device_identity(const std::string &json_path,
                                          const std::string &schema_text);

/// @brief Parse an architecture name string to an rj_code_arch_t enum value.
rj_code_arch_t parse_arch(const std::string &arch_str);

/// @brief Convert an rj_code_arch_t enum to its string name.
const char *arch_to_string(rj_code_arch_t arch);

/// @brief Parse an execution mode name, defaulting to FUNCTIONAL.
simdojo::ExecMode parse_exec_mode(const std::string &mode_str);

/// @brief Load simulation config from a JSON file.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
LoadedConfig load_config(const std::string &json_path, const std::string &schema_text);

/// @brief Load simulation config from a JSON file against a stated host width.
///
/// @details Like the two-argument overload, but resolves an unset (or zero)
/// `num_threads` against @p host_threads instead of the process's own affinity,
/// making host-dependent policy explicit for embedding callers and
/// deterministic tests. @p host_threads is always the width used, including 0,
/// which means indeterminate and resolves to a single partition.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param host_threads Host width to resolve the default partition count
/// against.
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
LoadedConfig load_config(const std::string &json_path, const std::string &schema_text,
                         uint32_t host_threads);

/// @brief Load simulation config from a JSON string.
/// @param json JSON configuration string.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on parse errors or invalid config.
LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_text);

/// @brief Load simulation config from a JSON string against a stated host width.
/// @details See the three-argument @ref load_config overload.
/// @param json JSON configuration string.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param host_threads Host width to resolve the default partition count
/// against; 0 means indeterminate and resolves to a single partition.
/// @returns LoadedConfig with engine parameters and built topology.
/// @throws std::runtime_error on parse errors or invalid config.
LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_text,
                                     uint32_t host_threads);

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_LOADER_H_
