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
  return config;
}

DbtGuestConfig load_dbt_guest_config_from_file(const std::string &path) {
  flatbuffers::Parser parser;
  const fb::SimulationConfig *config =
      parse_simulation_config_json(read_config_file(path), rocjitsu::kEmbeddedSchema, parser);
  return dbt_guest_from_fb(config->dbt_guest());
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
