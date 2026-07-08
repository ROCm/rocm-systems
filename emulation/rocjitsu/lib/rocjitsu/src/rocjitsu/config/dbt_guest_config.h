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

/// @brief DBT guest-GPU discovery configuration.
///
/// @details When enabled, the Linux KFD interposer exposes one synthetic guest
/// GPU alongside the real host GPUs and the HSA tools hook maps guest-agent
/// execution calls to the host agent. The KFD layer only owns discovery; DBT and
/// HSA forwarding happen in the HSA hook.
struct DbtGuestConfig {
  bool enabled = false;          ///< True when GuestKfd mode is active.
  std::string guest_isa;         ///< Guest ISA advertised by the synthetic agent.
  std::string host_isa;          ///< Host ISA used for actual ROCR execution.
  uint32_t host_gpu_id = 0;      ///< Host KFD topology gpu_id; 0 matches topology to host_isa.
  int log_level = 0;             ///< DBT hook logging level loaded from the config file.
  bool signal_backtrace = false; ///< Install a best-effort HSA-hook crash backtrace handler.
  KfdDeviceConfig guest_device;  ///< Synthetic guest device appended to KFD topology.
};

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
