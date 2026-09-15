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
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace config {
namespace {

FailureOr<DbtExecutionBackend> execution_backend_from_fb(fb::DbtExecutionBackend backend,
                                                         util::DiagnosticEmitter emit_error) {
  switch (backend) {
  case fb::DbtExecutionBackend_hardware:
    return DbtExecutionBackend::Hardware;
  case fb::DbtExecutionBackend_simulator:
    return DbtExecutionBackend::Simulator;
  }
  return emit_error.emit() << "dbt_guest.execution_backend is invalid";
}

FailureOr<DbtSiliconRevision> silicon_revision_from_fb(fb::DbtSiliconRevision revision,
                                                       util::DiagnosticEmitter emit_error) {
  switch (revision) {
  case fb::DbtSiliconRevision_unspecified:
    return DbtSiliconRevision::Unspecified;
  case fb::DbtSiliconRevision_gfx1250_a0:
    return DbtSiliconRevision::Gfx1250A0;
  case fb::DbtSiliconRevision_gfx1250_b0:
    return DbtSiliconRevision::Gfx1250B0;
  }
  return emit_error.emit() << "dbt_guest silicon revision is invalid";
}

Result validate_guest_device_geometry(const KfdDeviceConfig &device,
                                      util::DiagnosticEmitter emit_error) {
  if (!device.present || device.simd_count == 0)
    return Result::success();

  // num_cu_per_sh counts CUs per shader *array*, so the engine count only
  // yields the CU total once it is multiplied out by the arrays each engine
  // carries -- the same product KFD reports as node_props.array_count.
  const uint32_t arrays_per_engine =
      device.num_shader_arrays_per_engine == 0 ? 1u : device.num_shader_arrays_per_engine;
  const uint64_t expected_simds = static_cast<uint64_t>(device.num_shader_engines) *
                                  arrays_per_engine * device.num_cu_per_sh * device.simd_per_cu;
  if (expected_simds == device.simd_count)
    return Result::success();

  // DBT guest configs are written verbatim into synthetic KFD sysfs. Reject
  // internally inconsistent CU/SIMD geometry before ROCR observes properties
  // that disagree with each other during guest-agent discovery.
  return emit_error.emit()
         << "dbt_guest.guest_device simd_count (" << device.simd_count
         << ") must equal num_shader_engines * num_shader_arrays_per_engine * num_cu_per_sh * "
            "simd_per_cu ("
         << expected_simds << ')';
}

DbtGuestConfigResult parse_dbt_guest_config_json(const std::string &json,
                                                 bool skip_unexpected_fields, bool *has_dbt_guest,
                                                 util::DiagnosticEmitter emit_error) {
  flatbuffers::Parser parser;
  parser.opts.skip_unexpected_fields_in_json = skip_unexpected_fields;
  if (!parser.Parse(rocjitsu::kEmbeddedSchema))
    return emit_error.emit() << "Failed to parse schema: " << parser.error_;
  if (!parser.Parse(json.c_str()))
    return emit_error.emit() << "Failed to parse JSON config: " << parser.error_;

  const fb::SimulationConfig *config =
      flatbuffers::GetRoot<fb::SimulationConfig>(parser.builder_.GetBufferPointer());
  if (has_dbt_guest != nullptr)
    *has_dbt_guest = config->dbt_guest() != nullptr;
  return dbt_guest_from_fb(config->dbt_guest(), emit_error);
}

} // namespace

Result validate_dbt_simulator_device_limits(const DbtGuestConfig &guest,
                                            const KfdDeviceConfig &simulator_device,
                                            util::DiagnosticEmitter emit_error) {
  if (!guest.enabled || guest.host.backend != DbtExecutionBackend::Simulator)
    return Result::success();
  if (!guest.guest_device.present || !simulator_device.present)
    return emit_error.emit() << "simulator-backed dbt_guest requires guest and simulator devices";

  const auto require_at_most = [&emit_error](const char *name, uint32_t guest_value,
                                             uint32_t simulator_value) -> Result {
    if (guest_value <= simulator_value)
      return Result::success();
    return emit_error.emit() << "dbt_guest.guest_device." << name << " (" << guest_value
                             << ") exceeds simulator device capacity (" << simulator_value << ')';
  };
  if (require_at_most("lds_size_kb", guest.guest_device.lds_size_kb, simulator_device.lds_size_kb)
          .failed())
    return Result::failure();
  if (require_at_most("max_slots_scratch_cu", guest.guest_device.max_slots_scratch_cu,
                      simulator_device.max_slots_scratch_cu)
          .failed())
    return Result::failure();
  if (require_at_most("max_waves_per_simd", guest.guest_device.max_waves_per_simd,
                      simulator_device.max_waves_per_simd)
          .failed())
    return Result::failure();
  if (guest.guest_device.wave_front_size != simulator_device.wave_front_size)
    return emit_error.emit() << "dbt_guest.guest_device.wave_front_size ("
                             << guest.guest_device.wave_front_size
                             << ") must match simulator device wave_front_size ("
                             << simulator_device.wave_front_size << ')';
  return Result::success();
}

DbtGuestConfigResult dbt_guest_from_fb(const fb::DbtGuestConfig *guest,
                                       util::DiagnosticEmitter emit_error) {
  DbtGuestConfig config;
  if (guest == nullptr)
    return config;

  config.enabled = guest->enabled();
  if (guest->guest_isa())
    config.guest_isa = guest->guest_isa()->str();
  if (guest->host_isa())
    config.host.isa = guest->host_isa()->str();
  config.host.gpu_id = guest->host_gpu_id();
  FailureOr<DbtExecutionBackend> backend =
      execution_backend_from_fb(guest->execution_backend(), emit_error);
  if (backend.failed())
    return Result::failure();
  config.host.backend = backend.value();
  if (guest->simulator_config())
    config.host.simulator_config_path = guest->simulator_config()->str();
  config.log_level = guest->log_level();
  config.signal_backtrace = guest->signal_backtrace();
  FailureOr<KfdDeviceConfig> guest_device =
      kfd_device_from_fb(guest->guest_device(), "dbt_guest.guest_device", emit_error);
  if (guest_device.failed())
    return Result::failure();
  config.guest_device = std::move(guest_device).value();
  FailureOr<DbtSiliconRevision> guest_revision =
      silicon_revision_from_fb(guest->guest_revision(), emit_error);
  if (guest_revision.failed())
    return Result::failure();
  config.guest_revision = guest_revision.value();
  FailureOr<DbtSiliconRevision> host_revision =
      silicon_revision_from_fb(guest->host_revision(), emit_error);
  if (host_revision.failed())
    return Result::failure();
  config.host_revision = host_revision.value();
  if (validate_guest_device_geometry(config.guest_device, emit_error).failed())
    return Result::failure();
  if (config.enabled && config.host.backend == DbtExecutionBackend::Hardware &&
      !config.host.simulator_config_path.empty())
    return emit_error.emit()
           << "dbt_guest.simulator_config requires execution_backend=\"simulator\"";
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

DbtGuestConfigResult load_dbt_guest_config_from_file(const std::string &path,
                                                     util::DiagnosticEmitter emit_error) {
  std::ifstream file(path);
  if (!file.is_open())
    return emit_error.emit() << "Cannot open file: " << path;

  std::ostringstream contents;
  contents << file.rdbuf();
  if (file.bad())
    return emit_error.emit() << "Failed to read file: " << path;
  const std::string json = contents.str();

  bool has_dbt_guest = false;
  DbtGuestConfigResult parsed = parse_dbt_guest_config_json(json, true, &has_dbt_guest, emit_error);
  if (parsed.failed() || !has_dbt_guest)
    return parsed;

  // Simulation configs remain forward-compatible with unknown fields, but a
  // DBT guest block selects execution behavior and must reject misspelled keys
  // instead of silently falling back to the hardware backend.
  return parse_dbt_guest_config_json(json, false, nullptr, emit_error);
}

Result apply_resolved_dbt_host_gpu_id(DbtGuestConfig &config, std::string_view value,
                                      util::DiagnosticEmitter emit_error) {
  if (!config.enabled || config.host.gpu_id != 0)
    return Result::success();

  uint32_t gpu_id = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto [ptr, error] = std::from_chars(begin, end, gpu_id);
  if (error != std::errc{} || ptr != end || gpu_id == 0)
    return emit_error.emit() << "runtime config handoff must contain a nonzero KFD gpu_id";
  config.host.gpu_id = gpu_id;
  return Result::success();
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

DbtGuestConfigResult load_dbt_guest_config_from_handoff(const DbtRuntimeConfigHandoff &handoff,
                                                        util::DiagnosticEmitter emit_error) {
  DbtGuestConfigResult loaded = load_dbt_guest_config_from_file(handoff.config_path, emit_error);
  if (loaded.failed())
    return Result::failure();

  DbtGuestConfig config = std::move(loaded).value();
  if (handoff.resolved_gpu_id &&
      apply_resolved_dbt_host_gpu_id(config, *handoff.resolved_gpu_id, emit_error).failed())
    return Result::failure();
  if (!handoff.resolved_gpu_id && config.enabled && config.host.gpu_id == 0)
    return emit_error.emit() << "runtime config handoff must contain a resolved KFD gpu_id for "
                                "automatic DBT host selection";
  return config;
}

DbtGuestConfigResult load_dbt_guest_config_from_runtime_config(util::DiagnosticEmitter emit_error) {
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
    return emit_error.emit() << "runtime config handoff was not found";

  const std::string contents((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  if (file.bad())
    return emit_error.emit() << "failed to read runtime config handoff";
  std::optional<DbtRuntimeConfigHandoff> handoff = parse_dbt_runtime_config_handoff(contents);
  if (!handoff)
    return emit_error.emit() << "runtime config handoff does not contain a config path";

  return load_dbt_guest_config_from_handoff(*handoff, emit_error);
}

} // namespace config
} // namespace rocjitsu
