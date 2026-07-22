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
#include <unistd.h>
#include <vector>

namespace rocjitsu {
namespace config {
namespace {

DbtExecutionBackend execution_backend_from_fb(fb::DbtExecutionBackend backend) {
  switch (backend) {
  case fb::DbtExecutionBackend_hardware:
    return DbtExecutionBackend::Hardware;
  case fb::DbtExecutionBackend_simulator:
    return DbtExecutionBackend::Simulator;
  }
  throw std::runtime_error("dbt_guest.execution_backend is invalid");
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

void validate_dbt_simulator_device_limits(const DbtGuestConfig &guest,
                                          const KfdDeviceConfig &simulator_device) {
  if (!guest.enabled || guest.host.backend != DbtExecutionBackend::Simulator)
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
    config.host.isa = guest->host_isa()->str();
  config.host.gpu_id = guest->host_gpu_id();
  config.host.backend = execution_backend_from_fb(guest->execution_backend());
  if (guest->simulator_config())
    config.host.simulator_config_path = guest->simulator_config()->str();
  config.log_level = guest->log_level();
  config.signal_backtrace = guest->signal_backtrace();
  config.guest_device = kfd_device_from_fb(guest->guest_device());
  validate_guest_device_geometry(config.guest_device);
  if (config.enabled && config.host.backend == DbtExecutionBackend::Hardware &&
      !config.host.simulator_config_path.empty())
    throw std::runtime_error("dbt_guest.simulator_config requires execution_backend=\"simulator\"");
  return config;
}

std::string resolve_dbt_host_config_path(const std::string &dbt_config_path,
                                         const std::string &host_config_path) {
  const std::filesystem::path dbt_path(dbt_config_path);
  if (host_config_path.empty())
    return dbt_path.lexically_normal().string();

  const std::filesystem::path host_path(host_config_path);
  if (host_path.is_absolute())
    return host_path.lexically_normal().string();
  return (dbt_path.parent_path() / host_path).lexically_normal().string();
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
  // Try the handoff tiers in priority order, opening the first that exists:
  //   1. $ROCJITSU_INVOCATION_DIR/config_path — the launcher exports this dir before
  //      execvp so every descendant (incl. grandchildren via ctest, whose PID differs)
  //      finds it. Treat an empty value as unset (dir && *dir), matching interposer
  //      init(); an empty value would otherwise build "/config_path".
  //   2. this process's PID-scoped path (execvp preserves the launcher's PID for the
  //      direct child) — also the fallback if the env var is set but stale/misdirected.
  //   3. the well-known location for attach / daemon-only scenarios.
  // Falling straight from tier 1 to tier 3 (skipping tier 2) would miss a valid
  // per-PID handoff when the env var is set but its config_path is absent.
  std::vector<std::string> candidates;
  if (const char *dir = getenv(rocjitsu::kRpcInvocationDirEnv); dir && *dir)
    candidates.push_back(std::string(dir) + "/config_path");
  candidates.push_back(rocjitsu::rpc_invocation_config_file_path(getpid()));
  candidates.push_back(rocjitsu::rpc_default_config_file_path());

  std::ifstream file;
  for (const auto &candidate : candidates) {
    file.open(candidate);
    if (file.is_open())
      break;
  }
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
