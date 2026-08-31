// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/dbt_guest_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "flatbuffers/idl.h"
#include "simulation_config_generated.h"

#include <charconv>
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

DbtSiliconRevision silicon_revision_from_fb(fb::DbtSiliconRevision revision) {
  switch (revision) {
  case fb::DbtSiliconRevision_unspecified:
    return DbtSiliconRevision::Unspecified;
  case fb::DbtSiliconRevision_gfx1250_a0:
    return DbtSiliconRevision::Gfx1250A0;
  case fb::DbtSiliconRevision_gfx1250_b0:
    return DbtSiliconRevision::Gfx1250B0;
  }
  throw std::runtime_error("dbt_guest silicon revision is invalid");
}

void validate_guest_device_geometry(const KfdDeviceConfig &device) {
  if (!device.present || device.simd_count == 0)
    return;

  // num_cu_per_sh counts CUs per shader *array*, so the engine count only
  // yields the CU total once it is multiplied out by the arrays each engine
  // carries -- the same product KFD reports as node_props.array_count.
  const uint32_t arrays_per_engine =
      device.num_shader_arrays_per_engine == 0 ? 1u : device.num_shader_arrays_per_engine;
  const uint64_t expected_simds = static_cast<uint64_t>(device.num_shader_engines) *
                                  arrays_per_engine * device.num_cu_per_sh * device.simd_per_cu;
  if (expected_simds == device.simd_count)
    return;

  // DBT guest configs are written verbatim into synthetic KFD sysfs. Reject
  // internally inconsistent CU/SIMD geometry before ROCR observes properties
  // that disagree with each other during guest-agent discovery.
  throw std::runtime_error(
      "dbt_guest.guest_device simd_count (" + std::to_string(device.simd_count) +
      ") must equal num_shader_engines * num_shader_arrays_per_engine * num_cu_per_sh * "
      "simd_per_cu (" +
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
  config.guest_device = kfd_device_from_fb(guest->guest_device(), "dbt_guest.guest_device");
  config.guest_revision = silicon_revision_from_fb(guest->guest_revision());
  config.host_revision = silicon_revision_from_fb(guest->host_revision());
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

void apply_resolved_dbt_host_gpu_id(DbtGuestConfig &config, std::string_view value) {
  if (!config.enabled || config.host.gpu_id != 0)
    return;

  uint32_t gpu_id = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto [ptr, error] = std::from_chars(begin, end, gpu_id);
  if (error != std::errc{} || ptr != end || gpu_id == 0)
    throw std::runtime_error("runtime config handoff must contain a nonzero KFD gpu_id");
  config.host.gpu_id = gpu_id;
}

bool write_dbt_runtime_config_handoff(const std::string &config_path, const DbtGuestConfig &config,
                                      pid_t pid) {
  if (config.enabled && config.host.gpu_id == 0)
    return false;

  const std::string handoff_file = rpc_invocation_config_file_path(pid);
  std::error_code directory_error;
  std::filesystem::create_directories(std::filesystem::path(handoff_file).parent_path(),
                                      directory_error);
  if (directory_error)
    return false;
  const std::string temp_file = handoff_file + ".tmp";
  std::ofstream output(temp_file);
  if (!output)
    return false;
  output << config_path << '\n';
  if (config.enabled)
    output << config.host.gpu_id << '\n';
  output.close();
  if (!output.good()) {
    std::filesystem::remove(temp_file);
    return false;
  }

  std::error_code rename_error;
  std::filesystem::rename(temp_file, handoff_file, rename_error);
  if (rename_error)
    std::filesystem::remove(temp_file);
  return !rename_error;
}

std::optional<DbtRuntimeConfigHandoff> parse_dbt_runtime_config_handoff(std::string_view contents) {
  const size_t first_newline = contents.find('\n');
  std::string_view config_path = contents.substr(0, first_newline);
  if (!config_path.empty() && config_path.back() == '\r')
    config_path.remove_suffix(1);
  if (config_path.empty())
    return std::nullopt;

  std::optional<std::string> resolved_gpu_id;
  if (first_newline != std::string_view::npos) {
    std::string_view second_line = contents.substr(first_newline + 1);
    if (!second_line.empty()) {
      const size_t second_newline = second_line.find_first_of("\r\n");
      resolved_gpu_id = second_line.substr(0, second_newline);
    }
  }
  return DbtRuntimeConfigHandoff{std::string(config_path), std::move(resolved_gpu_id)};
}

DbtGuestConfig load_dbt_guest_config_from_handoff(const DbtRuntimeConfigHandoff &handoff) {
  DbtGuestConfig config = load_dbt_guest_config_from_file(handoff.config_path);
  if (handoff.resolved_gpu_id) {
    apply_resolved_dbt_host_gpu_id(config, *handoff.resolved_gpu_id);
  } else if (config.enabled && config.host.gpu_id == 0) {
    throw std::runtime_error("runtime config handoff must contain a resolved KFD gpu_id for "
                             "automatic DBT host selection");
  }
  return config;
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

  const std::string contents((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  std::optional<DbtRuntimeConfigHandoff> handoff = parse_dbt_runtime_config_handoff(contents);
  if (!handoff)
    return std::nullopt;
  return load_dbt_guest_config_from_handoff(*handoff);
}

} // namespace config
} // namespace rocjitsu
