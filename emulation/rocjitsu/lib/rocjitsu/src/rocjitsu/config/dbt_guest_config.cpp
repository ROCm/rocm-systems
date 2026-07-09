// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/dbt_guest_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "flatbuffers/idl.h"
#include "simulation_config_generated.h"

#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace rocjitsu {
namespace config {
namespace {

void validate_guest_device_geometry(const KfdDeviceConfig &device) {
  if (!device.present || device.simd_count == 0)
    return;

  const uint64_t expected_simds =
      static_cast<uint64_t>(device.num_shader_engines) * device.num_cu_per_sh * device.simd_per_cu;
  if (expected_simds == device.simd_count)
    return;

  // DBT guest configs are written verbatim into synthetic KFD sysfs. Reject
  // internally inconsistent CU/SIMD geometry before ROCR observes properties
  // that disagree with each other during guest-agent discovery.
  throw std::runtime_error("dbt_guest.guest_device simd_count (" +
                           std::to_string(device.simd_count) +
                           ") must equal num_shader_engines * num_cu_per_sh * simd_per_cu (" +
                           std::to_string(expected_simds) + ")");
}

} // namespace

DbtGuestConfig dbt_guest_from_fb(const fb::DbtGuestConfig *guest) {
  DbtGuestConfig config;
  if (guest == nullptr)
    return config;

  config.enabled = guest->enabled();
  if (guest->guest_isa())
    config.guest_isa = guest->guest_isa()->str();
  if (guest->host_isa())
    config.host_isa = guest->host_isa()->str();
  config.host_gpu_id = guest->host_gpu_id();
  config.log_level = guest->log_level();
  config.signal_backtrace = guest->signal_backtrace();
  config.guest_device = kfd_device_from_fb(guest->guest_device());
  validate_guest_device_geometry(config.guest_device);
  return config;
}

DbtGuestConfig load_dbt_guest_config_from_file(const std::string &path) {
  return with_parsed_simulation_config_json(
      read_config_file(path), rocjitsu::kEmbeddedSchema,
      [](const fb::SimulationConfig *config) { return dbt_guest_from_fb(config->dbt_guest()); });
}

std::optional<DbtGuestConfig> load_dbt_guest_config_from_runtime_config() {
  std::ifstream file(rocjitsu::rpc_default_config_file_path());
  if (!file.is_open())
    return std::nullopt;

  std::string path;
  std::getline(file, path);
  if (path.empty())
    return std::nullopt;
  return load_dbt_guest_config_from_file(path);
}

} // namespace config
} // namespace rocjitsu
