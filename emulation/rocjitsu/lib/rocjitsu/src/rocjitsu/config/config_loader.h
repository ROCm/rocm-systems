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
#include "rocjitsu/config/system_graph.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"
#include "util/diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
class SoC;
namespace amdgpu {
class GpuMemory;
class Xcd;
namespace matrix_coexecution {
class ExecutionResources;
} // namespace matrix_coexecution
} // namespace amdgpu

namespace config {

/// Default clocked-engine and legacy dispatch-only ceiling.
///
/// These paths do not use the functional allocation table. Bound their
/// automatic worker counts on large hosts; explicit requests can override this
/// policy. Functional limits instead come from affinity and the target table.
inline constexpr uint32_t kDefaultExecutionThreadCap = 32;
/// Compatibility name for the older dispatch-only override API.
inline constexpr uint32_t kDefaultCpuDispatchThreadCap = kDefaultExecutionThreadCap;

/// A single VM-wide budget includes engines, retained workers from every SoC's
/// dispatch pool, and one shared async-helper pool. Explicit knobs take priority.
struct ExecutionThreadRequest {
  uint32_t budget = 0;   ///< Zero uses affinity and the target allocation table.
  uint32_t engines = 0;  ///< Zero selects from the table.
  uint32_t dispatch = 0; ///< Inclusive width per SoC; zero selects automatic sizing.
  int32_t helpers = -1;  ///< -1 selects automatic sizing; zero disables helpers.
};

/// One target-configured granule, with an inclusive dispatch width per GPU.
/// engines maps to num_threads, dispatch to cpu_dispatch_threads, and helpers
/// to async_helper_threads in JSON.
struct ExecutionThreadChoice {
  uint32_t engines = 1;
  uint32_t dispatch = 1;
  uint32_t helpers = 0;
  bool operator==(const ExecutionThreadChoice &) const = default;
};

struct ExecutionThreadAllocation {
  uint32_t engines = 1;
  std::vector<uint32_t> dispatch;
  uint32_t helpers = 0;
};

/// Pure table selector. Choose the largest effective allocation fitting the budget,
/// after topology clamps and explicit overrides; later entries break ties.
/// Explicit knobs may exceed the budget. With no fitting entry, unspecified knobs use
/// serial defaults. Clocked mode only allocates engine partitions.
ExecutionThreadAllocation resolve_execution_threads(const ExecutionThreadRequest &request,
                                                    uint32_t host_threads, uint32_t xcds,
                                                    std::span<const uint32_t> dispatch_capacities,
                                                    bool async_supported,
                                                    std::span<const ExecutionThreadChoice> choices,
                                                    bool clocked = false);

/// Metadata needed to evaluate the pure allocation rule without building a VM.
struct ExecutionThreadSettings {
  ExecutionThreadRequest request;
  std::vector<ExecutionThreadChoice> choices;
  uint32_t xcds = 1;
  std::vector<uint32_t> dispatch_capacities;
  bool async_supported = false;
  bool clocked = false;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;

  ExecutionThreadAllocation resolve(uint32_t host_threads) const {
    return resolve_execution_threads(request, host_threads, xcds, dispatch_capacities,
                                     async_supported, choices, clocked);
  }
};

/// @brief Read only a config's thread requests and topology dimensions.
/// @details Allocates no simulator components or worker threads. File-reading
/// and parsing failures emit a diagnostic and return failure.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @returns Settings for evaluating allocation, or failure after emitting a diagnostic.
/// @throws std::invalid_argument when required allocation metadata is invalid.
util::FailureOr<ExecutionThreadSettings>
load_execution_thread_settings(const std::string &json_path, const std::string &schema_text,
                               util::DiagnosticEmitter emit_error);

/// @brief Read thread requests and topology dimensions from config JSON text.
/// @details Same as the path overload, for a config a launch has already rewritten
/// in memory and does not need to place on disk. Parsing failures emit a diagnostic
/// and return failure.
/// @param json JSON configuration string.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @returns Settings for evaluating allocation, or failure after emitting a diagnostic.
/// @throws std::invalid_argument when required allocation metadata is invalid.
util::FailureOr<ExecutionThreadSettings>
load_execution_thread_settings_from_string(const std::string &json, const std::string &schema_text,
                                           util::DiagnosticEmitter emit_error);

/// Read thread dimensions for one selected node. An empty selector uses the
/// config's local_node field or defaults only when the graph has one node.
util::FailureOr<ExecutionThreadSettings>
load_execution_thread_settings_for_node(const std::string &json_path,
                                        const std::string &schema_text, std::string_view node_id,
                                        util::DiagnosticEmitter emit_error);
util::FailureOr<ExecutionThreadSettings> load_execution_thread_settings_from_string_for_node(
    const std::string &json, const std::string &schema_text, std::string_view node_id,
    util::DiagnosticEmitter emit_error);

/// @brief Check whether the ISA supports asynchronous MMA execution.
bool configured_async_mma_supported(rj_code_arch_t arch);
/// @brief Create lazy VM-owned resources for the requested helper count.
std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>
make_async_execution_resources(uint32_t helpers);

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
/// the shared execution-budget allocation. Overriding it here is what the clamp
/// is for; pin 1 when the consumer needs SimulationEngine::step(), which
/// rejects a multi-partition engine.
/// @code
///   #include "rocjitsu/vm/amdgpu/partitioning.h"
///
///   util::StringDiagnostic diagnostic;
///   util::FailureOr<config::LoadedConfig> result =
///       load_config("config.json", kEmbeddedSchema, diagnostic.emitter());
///   if (result.failed()) {
///     // Report diagnostic.message() and stop before accessing the value.
///     return;
///   }
///   config::LoadedConfig loaded = std::move(result.value());
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
  SystemGraph system_graph;             ///< Validated graph, including remote descriptions.
  std::string local_node_id;            ///< Node used for this VM's local projection.
  std::unordered_map<std::string, uint32_t>
      global_to_local_ordinals; ///< Global instance ID to local gpuN ordinal.
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  /// Requested dispatch width. Omitted/zero selects from thread_allocations;
  /// each SoC's upper bound is one plus the sum of CUs-1 across nonempty XCDs.
  uint32_t cpu_dispatch_threads = 0;
  uint32_t cpu_thread_budget = 0; ///< Zero uses affinity and the target allocation table.
  /// Preserve old automatic-dispatch checkpoint metadata when saving again.
  bool legacy_auto_dispatch = false;
  int32_t async_helper_threads = -1;           ///< -1 selects from the table; zero disables.
  uint32_t requested_engine_threads = 0;       ///< Original request, retained for checkpoints.
  ExecutionThreadAllocation execution_threads; ///< Effective allocation resolved during loading.
  std::vector<ExecutionThreadChoice> thread_allocations; ///< Target-preferred granules.
  std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>
      async_resources; ///< Shared by CUs.

  /// @brief Apply the requested functional dispatch policy to every loaded SoC.
  ///
  /// Call this before transferring topology ownership with take_root(). The
  /// method applies the allocation resolved during loading.
  void apply_cpu_dispatch_threads();

  /// @brief Explicitly override dispatch sizing independently of the total budget.
  /// @details This dispatch-only override ignores the allocation table and does not
  /// reallocate engines or helpers. Use resolve_execution_threads() to evaluate
  /// the joint plan for another host width.
  /// @param hardware_threads Host-thread count available for automatic sizing.
  /// @param automatic_thread_cap Maximum host-wide width in automatic mode;
  /// values below one are treated as one.
  void override_cpu_dispatch_threads(uint32_t hardware_threads,
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
/// @details Building the topology allocates the whole simulated machine, which
/// for a large part is gigabytes of register files. A front end that only needs
/// to know which GPU to present should not pay for a machine it never runs.
/// File-reading, parsing, and device conversion failures emit a diagnostic
/// and return failure. Other validation and allocation exceptions propagate.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @returns The identity and bus shape, or failure after emitting a diagnostic.
util::FailureOr<DeviceIdentityConfig> load_device_identity(const std::string &json_path,
                                                           const std::string &schema_text,
                                                           util::DiagnosticEmitter emit_error);

/// @brief Parse an architecture name string to an rj_code_arch_t enum value.
rj_code_arch_t parse_arch(const std::string &arch_str);

/// @brief Convert an rj_code_arch_t enum to its string name.
const char *arch_to_string(rj_code_arch_t arch);

/// @brief Parse an execution mode name, defaulting to FUNCTIONAL.
simdojo::ExecMode parse_exec_mode(const std::string &mode_str);

/// @brief Validate configuration through the loader's shared component traversal.
/// @details Shares CU resource and ISA checks with construction, using placeholders
/// for CUs. CU-dependent topology checks must use the declarative configuration;
/// execution wiring and its geometry warnings only run during full construction.
/// @param json_path Path to the JSON configuration file.
/// @param schema_text FlatBuffers schema text.
/// @param emit_error Destination for failure diagnostics.
/// @returns Success, or failure with a diagnostic for file-reading, parsing, or
/// KFD/DBT device conversion errors. Other validation and allocation exceptions propagate.
util::Result validate_config(const std::string &json_path, const std::string &schema_text,
                             util::DiagnosticEmitter emit_error);

/// @brief Load simulation config from a JSON file.
/// @details File-reading, parsing, and KFD/DBT device conversion failures emit a
/// diagnostic and return failure. Other validation and allocation exceptions propagate.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @returns LoadedConfig with engine parameters and built topology, or failure.
util::FailureOr<LoadedConfig> load_config(const std::string &json_path,
                                          const std::string &schema_text,
                                          util::DiagnosticEmitter emit_error);

/// @brief Load simulation config from a JSON file against a stated host width.
///
/// @details Like the three-argument overload, but uses @p host_threads as the
/// affinity input to the shared execution budget. Zero means indeterminate and
/// gives an automatic budget of one. An explicit cpu_thread_budget overrides it.
/// File-reading, parsing, and KFD/DBT device conversion failures emit a diagnostic
/// and return failure. Other validation and allocation exceptions propagate.
/// @param json_path Path to the JSON config file.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @param host_threads Affinity width used for automatic budget selection.
/// @returns LoadedConfig with engine parameters and built topology, or failure.
util::FailureOr<LoadedConfig> load_config(const std::string &json_path,
                                          const std::string &schema_text,
                                          util::DiagnosticEmitter emit_error,
                                          uint32_t host_threads);

/// @brief Load simulation config from a JSON string.
/// @details Parsing and KFD/DBT device conversion failures emit a diagnostic and
/// return failure. Other validation and allocation exceptions propagate.
/// @param json JSON configuration string.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @returns LoadedConfig with engine parameters and built topology, or failure.
util::FailureOr<LoadedConfig> load_config_from_string(const std::string &json,
                                                      const std::string &schema_text,
                                                      util::DiagnosticEmitter emit_error);

/// @brief Load simulation config from a JSON string against a stated host width.
/// @details See the four-argument @ref load_config overload.
/// Parsing and KFD/DBT device conversion failures emit a diagnostic and
/// return failure. Other validation and allocation exceptions propagate.
/// @param json JSON configuration string.
/// @param schema_text FlatBuffers schema text (the .fbs content).
/// @param emit_error Destination for failure diagnostics.
/// @param host_threads Affinity width for automatic selection; zero gives a budget of one.
/// @returns LoadedConfig with engine parameters and built topology, or failure.
util::FailureOr<LoadedConfig> load_config_from_string(const std::string &json,
                                                      const std::string &schema_text,
                                                      util::DiagnosticEmitter emit_error,
                                                      uint32_t host_threads);

/// Build only the selected local node. A nonempty node_id must agree with a
/// local_node field in the JSON, if one exists. An empty node_id uses local_node;
/// if neither selects a node, the graph must contain exactly one node. This
/// contract applies to both path and string overloads below. The graph remains
/// owned in the result for later topology and route projections.
util::FailureOr<LoadedConfig> load_config_for_node(const std::string &json_path,
                                                   const std::string &schema_text,
                                                   std::string_view node_id,
                                                   util::DiagnosticEmitter emit_error,
                                                   uint32_t host_threads);
util::FailureOr<LoadedConfig> load_config_for_node(const std::string &json_path,
                                                   const std::string &schema_text,
                                                   std::string_view node_id,
                                                   util::DiagnosticEmitter emit_error);
util::FailureOr<LoadedConfig> load_config_from_string_for_node(const std::string &json,
                                                               const std::string &schema_text,
                                                               std::string_view node_id,
                                                               util::DiagnosticEmitter emit_error,
                                                               uint32_t host_threads);
util::FailureOr<LoadedConfig> load_config_from_string_for_node(const std::string &json,
                                                               const std::string &schema_text,
                                                               std::string_view node_id,
                                                               util::DiagnosticEmitter emit_error);

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_LOADER_H_
