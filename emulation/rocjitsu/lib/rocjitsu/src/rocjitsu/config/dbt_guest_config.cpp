// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/dbt_guest_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "flatbuffers/idl.h"
#include "simulation_config_generated.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace rocjitsu {
namespace config {
namespace {

DbtExecutionBackend parse_execution_backend(const fb::DbtGuestConfig *guest) {
  if (!guest->execution_backend())
    return DbtExecutionBackend::Hardware;
  const std::string value = guest->execution_backend()->str();
  if (value == "hardware")
    return DbtExecutionBackend::Hardware;
  if (value == "simulator")
    return DbtExecutionBackend::Simulator;
  throw std::runtime_error("dbt_guest.execution_backend must be 'hardware' or 'simulator', got '" +
                           value + "'");
}

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

const char *dbt_execution_backend_name(DbtExecutionBackend backend) {
  switch (backend) {
  case DbtExecutionBackend::Hardware:
    return "hardware";
  case DbtExecutionBackend::Simulator:
    return "simulator";
  }
  return "unknown";
}

std::string resolve_dbt_simulator_config_path(const std::string &dbt_config_path,
                                              const std::string &simulator_config) {
  std::filesystem::path resolved(simulator_config);
  if (resolved.is_relative())
    resolved = std::filesystem::path(dbt_config_path).parent_path() / resolved;
  return resolved.lexically_normal().string();
}

void validate_dbt_simulator_device_limits(const DbtGuestConfig &guest,
                                          const KfdDeviceConfig &simulator_device) {
  if (!guest.enabled || guest.execution_backend != DbtExecutionBackend::Simulator)
    return;
  if (!guest.guest_device.present || !simulator_device.present)
    throw std::runtime_error("simulator-backed dbt_guest requires guest and simulator devices");

  const auto require_at_most = [](const char *name, uint32_t guest_value,
                                  uint32_t simulator_value) {
    if (guest_value <= simulator_value)
      return;
    throw std::runtime_error("dbt_guest.guest_device." + std::string(name) + " (" +
                             std::to_string(guest_value) + ") exceeds simulator device capacity (" +
                             std::to_string(simulator_value) + ")");
  };
  require_at_most("lds_size_kb", guest.guest_device.lds_size_kb, simulator_device.lds_size_kb);
  require_at_most("max_slots_scratch_cu", guest.guest_device.max_slots_scratch_cu,
                  simulator_device.max_slots_scratch_cu);
  require_at_most("max_waves_per_simd", guest.guest_device.max_waves_per_simd,
                  simulator_device.max_waves_per_simd);
  if (guest.guest_device.wave_front_size != simulator_device.wave_front_size)
    throw std::runtime_error("dbt_guest.guest_device.wave_front_size (" +
                             std::to_string(guest.guest_device.wave_front_size) +
                             ") must match simulator device wave_front_size (" +
                             std::to_string(simulator_device.wave_front_size) + ")");
}

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
  config.execution_backend = parse_execution_backend(guest);
  if (guest->simulator_config())
    config.simulator_config = guest->simulator_config()->str();
  config.log_level = guest->log_level();
  config.signal_backtrace = guest->signal_backtrace();
  config.guest_device = kfd_device_from_fb(guest->guest_device());
  validate_guest_device_geometry(config.guest_device);
  if (config.enabled && config.execution_backend == DbtExecutionBackend::Simulator &&
      config.simulator_config.empty())
    throw std::runtime_error(
        "dbt_guest.simulator_config is required when execution_backend is 'simulator'");
  if (config.enabled && config.execution_backend == DbtExecutionBackend::Hardware &&
      !config.simulator_config.empty())
    throw std::runtime_error("dbt_guest.simulator_config requires execution_backend 'simulator'");
  return config;
}

DbtGuestConfig load_dbt_guest_config_from_file(const std::string &path) {
  const std::string json = read_config_file(path);
  bool has_dbt_guest = false;
  DbtGuestConfig parsed = with_parsed_simulation_config_json(
      json, rocjitsu::kEmbeddedSchema, [&has_dbt_guest](const fb::SimulationConfig *config) {
        has_dbt_guest = config->dbt_guest() != nullptr;
        return dbt_guest_from_fb(config->dbt_guest());
      });
  if (!has_dbt_guest)
    return parsed;

  // Simulation configs remain forward-compatible with unknown fields, but a
  // DBT guest block selects execution behavior and must reject misspelled keys
  // instead of silently falling back to the hardware backend.
  return with_parsed_simulation_config_json(
      json, rocjitsu::kEmbeddedSchema,
      [](const fb::SimulationConfig *config) { return dbt_guest_from_fb(config->dbt_guest()); },
      false);
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
