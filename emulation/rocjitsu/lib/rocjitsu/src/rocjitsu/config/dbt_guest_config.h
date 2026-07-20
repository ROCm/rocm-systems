// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_guest_config.h
/// @brief DBT guest-GPU configuration shared by the launcher, KMD interposer, and HSA hook.

#ifndef ROCJITSU_CONFIG_DBT_GUEST_CONFIG_H_
#define ROCJITSU_CONFIG_DBT_GUEST_CONFIG_H_

#include "rocjitsu/config/kfd_device_config.h"

#include <cstdint>
#include <optional>
#include <string>

namespace rocjitsu::fb {
struct DbtGuestConfig;
} // namespace rocjitsu::fb

namespace rocjitsu::config {

/// @brief Execution target used by DBT guest mode.
enum class DbtExecutionBackend {
  Hardware,  ///< Forward execution-facing operations to a real host GPU.
  Simulator, ///< Forward execution-facing operations to a RocJITsu simulated GPU.
};

/// @brief Host target selected for DBT translation and execution.
struct DbtHostConfig {
  std::string isa;     ///< Host ISA used for DBT output and ROCR execution.
  uint32_t gpu_id = 0; ///< Host KFD topology gpu_id; 0 matches topology to isa.
  DbtExecutionBackend backend = DbtExecutionBackend::Hardware; ///< Hardware or simulator execution.
  std::string simulator_config_path; ///< Optional external simulator host config.
};

/// @brief DBT guest-GPU discovery configuration.
///
/// @details When enabled, the Linux KFD interposer exposes one synthetic guest
/// GPU. The hardware backend forwards host-facing KFD operations to real
/// `/dev/kfd`; the simulator backend delegates them to SimulatedKfd. The HSA
/// tools hook translates guest code and maps guest-agent execution calls to the
/// selected hardware or simulated host agent.
struct DbtGuestConfig {
  bool enabled = false;          ///< True when GuestKfd mode is active.
  std::string guest_isa;         ///< Guest ISA advertised by the synthetic agent.
  DbtHostConfig host;            ///< Host translation and execution target.
  int log_level = 0;             ///< DBT hook logging level loaded from the config file.
  bool signal_backtrace = false; ///< Install a best-effort HSA-hook crash backtrace handler.
  KfdDeviceConfig guest_device;  ///< Synthetic guest device appended to KFD topology.
};

/// @brief Resolve the simulator host config selected by a DBT guest config.
/// @details An empty host_config_path selects dbt_config_path itself. Relative
/// external paths are resolved beside the DBT guest config. A non-empty path
/// selects that external file instead of VM/topology in the DBT guest file.
std::string resolve_dbt_host_config_path(const std::string &dbt_config_path,
                                         const std::string &host_config_path);

/// @brief Reject guest limits that exceed a simulator execution target.
/// @details Simulator-backed discovery must not advertise resource limits that
/// the selected target cannot execute. Limits not represented in KFD device
/// topology, such as per-kernel VGPR usage, remain the translator/runtime's
/// responsibility.
/// @throws std::runtime_error when an execution-relevant guest limit is not
/// supported by the simulator device.
void validate_dbt_simulator_device_limits(const DbtGuestConfig &guest,
                                          const KfdDeviceConfig &simulator_device);

/// @brief Convert a generated FlatBuffers DBT guest table into runtime config.
///
/// @details Shared by the DBT-only loader and the full simulation config
/// loader so both paths interpret `dbt_guest` identically.
DbtGuestConfig dbt_guest_from_fb(const fb::DbtGuestConfig *guest);

/// @brief Load only dbt_guest from a rocjitsu JSON config.
///
/// @details DBT guest mode does not instantiate the simulation VM. This helper
/// intentionally accepts configs that contain only the top-level `dbt_guest`
/// table, so guest discovery configs do not need unused `vm` or `topology`
/// sections.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
DbtGuestConfig load_dbt_guest_config_from_file(const std::string &path);

/// @brief Load only dbt_guest from the rocjitsu child-process runtime config file.
///
/// @details HSA tools run inside ROCR initialization and must not depend on the
/// full simulation topology builder. This helper parses the same JSON schema as
/// the main config loader but copies only the DBT guest-GPU block.
/// @returns DbtGuestConfig when the runtime config path file exists; std::nullopt otherwise.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
std::optional<DbtGuestConfig> load_dbt_guest_config_from_runtime_config();

} // namespace rocjitsu::config

#endif // ROCJITSU_CONFIG_DBT_GUEST_CONFIG_H_
