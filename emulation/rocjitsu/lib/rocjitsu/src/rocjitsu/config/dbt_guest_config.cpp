// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/dbt_guest_config.h"

#include "rocjitsu/vm/amdgpu/pci/gpu_generation_registry.h"

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
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace config {
namespace {

util::FailureOr<DbtExecutionBackend> execution_backend_from_fb(fb::DbtExecutionBackend backend,
                                                               util::DiagnosticEmitter emit_error) {
  switch (backend) {
  case fb::DbtExecutionBackend_hardware:
    return DbtExecutionBackend::Hardware;
  case fb::DbtExecutionBackend_simulator:
    return DbtExecutionBackend::Simulator;
  }
  return emit_error.emit() << "dbt_guest.execution_backend is invalid";
}

util::FailureOr<DbtSiliconRevision> silicon_revision_from_fb(fb::DbtSiliconRevision revision,
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

bool is_filler(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

void skip_filler(std::string_view json, size_t &at) {
  while (at < json.size()) {
    if (is_filler(json[at])) {
      ++at;
    } else if (json.compare(at, 2, "//") == 0) {
      const size_t line_end = json.find_first_of("\r\n", at + 2);
      at = line_end == std::string_view::npos ? json.size() : line_end + 1;
    } else if (json.compare(at, 2, "/*") == 0) {
      const size_t block_end = json.find("*/", at + 2);
      at = block_end == std::string_view::npos ? json.size() : block_end + 2;
    } else {
      return;
    }
  }
}

bool skip_quoted_string(std::string_view json, size_t &at) {
  if (at >= json.size() || (json[at] != '"' && json[at] != '\''))
    return false;
  const char quote = json[at++];
  while (at < json.size()) {
    const char c = json[at++];
    if (c == quote)
      return true;
    if (c == '\\' && at < json.size())
      ++at;
  }
  return false;
}

bool skip_json_value(std::string_view json, size_t &at) {
  skip_filler(json, at);
  if (at >= json.size())
    return false;
  if (json[at] == '"' || json[at] == '\'')
    return skip_quoted_string(json, at);
  if (json[at] == '{' || json[at] == '[') {
    std::vector<char> closers;
    closers.push_back(json[at++] == '{' ? '}' : ']');
    while (at < json.size()) {
      skip_filler(json, at);
      if (at >= json.size())
        return false;
      if (json[at] == '"' || json[at] == '\'') {
        if (!skip_quoted_string(json, at))
          return false;
        continue;
      }
      const char c = json[at++];
      if (c == '{') {
        closers.push_back('}');
      } else if (c == '[') {
        closers.push_back(']');
      } else if (c == '}' || c == ']') {
        if (closers.empty() || c != closers.back())
          return false;
        closers.pop_back();
        if (closers.empty())
          return true;
      }
    }
    return false;
  }

  while (at < json.size() && json[at] != ',' && json[at] != '}' && json[at] != ']' &&
         !is_filler(json[at])) {
    if (json.compare(at, 2, "//") == 0 || json.compare(at, 2, "/*") == 0)
      break;
    ++at;
  }
  return true;
}

std::optional<uint32_t> parse_hex_digits(std::string_view json, size_t &at, size_t count) {
  if (json.size() - at < count)
    return std::nullopt;
  const std::string_view digits = json.substr(at, count);
  uint32_t value = 0;
  const auto [ptr, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), value, 16);
  if (ec != std::errc{} || ptr != digits.data() + digits.size())
    return std::nullopt;
  at += count;
  return value;
}

bool append_utf8(uint32_t code_point, std::string &decoded) {
  if (code_point <= 0x7f) {
    decoded += static_cast<char>(code_point);
  } else if (code_point <= 0x7ff) {
    decoded += static_cast<char>(0xc0 | (code_point >> 6));
    decoded += static_cast<char>(0x80 | (code_point & 0x3f));
  } else if (code_point <= 0xffff) {
    decoded += static_cast<char>(0xe0 | (code_point >> 12));
    decoded += static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
    decoded += static_cast<char>(0x80 | (code_point & 0x3f));
  } else if (code_point <= 0x10ffff) {
    decoded += static_cast<char>(0xf0 | (code_point >> 18));
    decoded += static_cast<char>(0x80 | ((code_point >> 12) & 0x3f));
    decoded += static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
    decoded += static_cast<char>(0x80 | (code_point & 0x3f));
  } else {
    return false;
  }
  return true;
}

std::optional<std::string> parse_quoted_field_name(std::string_view json, size_t &at) {
  if (at >= json.size() || (json[at] != '"' && json[at] != '\''))
    return std::nullopt;
  const char quote = json[at++];
  std::string decoded;
  int unicode_high_surrogate = -1;
  while (at < json.size()) {
    const char c = json[at++];
    if (c == quote) {
      if (unicode_high_surrogate != -1)
        return std::nullopt;
      return decoded;
    }
    if (static_cast<unsigned char>(c) < ' ')
      return std::nullopt;
    if (c != '\\') {
      if (unicode_high_surrogate != -1)
        return std::nullopt;
      decoded += c;
      continue;
    }
    if (at >= json.size())
      return std::nullopt;
    const char escape = json[at++];
    if (unicode_high_surrogate != -1 && escape != 'u')
      return std::nullopt;
    switch (escape) {
    case 'n':
      decoded += '\n';
      break;
    case 't':
      decoded += '\t';
      break;
    case 'r':
      decoded += '\r';
      break;
    case 'b':
      decoded += '\b';
      break;
    case 'f':
      decoded += '\f';
      break;
    case '"':
    case '\'':
    case '\\':
    case '/':
      decoded += escape;
      break;
    case 'u': {
      std::optional<uint32_t> digits = parse_hex_digits(json, at, 4);
      if (!digits)
        return std::nullopt;
      const uint32_t value = *digits;
      if (value >= 0xd800 && value <= 0xdbff) {
        if (unicode_high_surrogate != -1)
          return std::nullopt;
        unicode_high_surrogate = static_cast<int>(value);
      } else if (value >= 0xdc00 && value <= 0xdfff) {
        if (unicode_high_surrogate == -1)
          return std::nullopt;
        const uint32_t code_point =
            0x10000 + ((static_cast<uint32_t>(unicode_high_surrogate) & 0x3ff) << 10) +
            (value & 0x3ff);
        if (!append_utf8(code_point, decoded))
          return std::nullopt;
        unicode_high_surrogate = -1;
      } else {
        if (unicode_high_surrogate != -1 || !append_utf8(value, decoded))
          return std::nullopt;
      }
      break;
    }
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<std::string> parse_field_name(std::string_view json, size_t &at) {
  skip_filler(json, at);
  if (at >= json.size())
    return std::nullopt;

  if (json[at] == '"' || json[at] == '\'')
    return parse_quoted_field_name(json, at);

  const size_t name_begin = at;
  while (at < json.size()) {
    const unsigned char c = static_cast<unsigned char>(json[at]);
    const bool identifier_char =
        (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    if (!identifier_char)
      break;
    ++at;
  }
  if (name_begin == at)
    return std::nullopt;
  return std::string(json.substr(name_begin, at - name_begin));
}

bool is_ignored_dbt_strict_top_level_field(std::string_view name) {
  // The DBT strict pass catches misspelled dbt_guest keys before handoff. The
  // plugin loader owns its top-level policy fields, so they bypass only this
  // DBT-specific validation and are still parsed by the plugin layer.
  return name == "plugins" || name == "require_all_plugins" || name == "required_plugins";
}

std::string json_without_dbt_strict_ignored_top_level_fields(std::string_view json) {
  size_t at = 0;
  skip_filler(json, at);
  if (at >= json.size() || json[at] != '{')
    return std::string(json);
  ++at;

  std::string rewritten = "{";
  bool first_field = true;
  while (at < json.size()) {
    skip_filler(json, at);
    if (at < json.size() && json[at] == '}')
      break;

    const size_t field_begin = at;
    std::optional<std::string> name = parse_field_name(json, at);
    if (!name)
      return std::string(json);
    skip_filler(json, at);
    if (at >= json.size() || json[at] != ':')
      return std::string(json);
    ++at;
    if (!skip_json_value(json, at))
      return std::string(json);
    const size_t field_end = at;

    if (!is_ignored_dbt_strict_top_level_field(*name)) {
      if (!first_field)
        rewritten += ",";
      rewritten.append(json.substr(field_begin, field_end - field_begin));
      first_field = false;
    }

    skip_filler(json, at);
    if (at < json.size() && json[at] == ',')
      ++at;
  }

  rewritten += "}";
  return rewritten;
}

util::Result validate_guest_device_geometry(const KfdDeviceConfig &device,
                                            util::DiagnosticEmitter emit_error) {
  if (!device.present || device.simd_count == 0)
    return util::Result::success();

  // num_cu_per_sh counts CUs per shader *array*, so the engine count only
  // yields the CU total once it is multiplied out by the arrays each engine
  // carries -- the same product KFD reports as node_props.array_count.
  const uint32_t arrays_per_engine =
      device.num_shader_arrays_per_engine == 0 ? 1u : device.num_shader_arrays_per_engine;
  const uint64_t expected_simds = static_cast<uint64_t>(device.num_shader_engines) *
                                  arrays_per_engine * device.num_cu_per_sh * device.simd_per_cu;
  if (expected_simds == device.simd_count)
    return util::Result::success();

  // DBT guest configs are written verbatim into synthetic KFD sysfs. Reject
  // internally inconsistent CU/SIMD geometry before ROCR observes properties
  // that disagree with each other during guest-agent discovery.
  return emit_error.emit()
         << "dbt_guest.guest_device simd_count (" << device.simd_count
         << ") must equal num_shader_engines * num_shader_arrays_per_engine * num_cu_per_sh * "
            "simd_per_cu ("
         << expected_simds << ")";
}

} // namespace

util::Result validate_dbt_simulator_device_limits(const DbtGuestConfig &guest,
                                                  const KfdDeviceConfig &simulator_device,
                                                  util::DiagnosticEmitter emit_error) {
  if (!guest.enabled || guest.host.backend != DbtExecutionBackend::Simulator)
    return util::Result::success();
  if (!guest.guest_device.present || !simulator_device.present)
    return emit_error.emit() << "simulator-backed dbt_guest requires guest and simulator devices";

  const auto require_at_most = [emit_error](const char *name, uint32_t guest_value,
                                            uint32_t simulator_value) -> util::Result {
    if (guest_value <= simulator_value)
      return util::Result::success();
    return emit_error.emit() << "dbt_guest.guest_device." << name << " (" << guest_value
                             << ") exceeds simulator device capacity (" << simulator_value << ")";
  };
  if (require_at_most("lds_size_kb", guest.guest_device.lds_size_kb, simulator_device.lds_size_kb)
          .failed() ||
      require_at_most("max_slots_scratch_cu", guest.guest_device.max_slots_scratch_cu,
                      simulator_device.max_slots_scratch_cu)
          .failed() ||
      require_at_most("max_waves_per_simd", guest.guest_device.max_waves_per_simd,
                      simulator_device.max_waves_per_simd)
          .failed())
    return util::Result::failure();
  if (guest.guest_device.wave_front_size != simulator_device.wave_front_size)
    return emit_error.emit() << "dbt_guest.guest_device.wave_front_size ("
                             << guest.guest_device.wave_front_size
                             << ") must match simulator device wave_front_size ("
                             << simulator_device.wave_front_size << ")";
  return util::Result::success();
}

util::FailureOr<DbtGuestConfig> dbt_guest_from_fb(const fb::DbtGuestConfig *guest,
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
  util::FailureOr<DbtExecutionBackend> backend =
      execution_backend_from_fb(guest->execution_backend(), emit_error);
  if (backend.failed())
    return util::Result::failure();
  config.host.backend = backend.value();
  if (guest->simulator_config())
    config.host.simulator_config_path = guest->simulator_config()->str();
  config.log_level = guest->log_level();
  config.signal_backtrace = guest->signal_backtrace();
  auto guest_device =
      kfd_device_from_fb(guest->guest_device(), "dbt_guest.guest_device", emit_error);
  if (guest_device.failed())
    return util::Result::failure();
  config.guest_device = std::move(guest_device).value();
  if (config.guest_device.present)
    (void)resolve_gpu_generation_topology(config.guest_device);
  util::FailureOr<DbtSiliconRevision> guest_revision =
      silicon_revision_from_fb(guest->guest_revision(), emit_error);
  if (guest_revision.failed())
    return util::Result::failure();
  config.guest_revision = guest_revision.value();
  util::FailureOr<DbtSiliconRevision> host_revision =
      silicon_revision_from_fb(guest->host_revision(), emit_error);
  if (host_revision.failed())
    return util::Result::failure();
  config.host_revision = host_revision.value();
  if (validate_guest_device_geometry(config.guest_device, emit_error).failed())
    return util::Result::failure();
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

util::FailureOr<DbtGuestConfig>
load_dbt_guest_config_from_file(const std::string &path, util::DiagnosticEmitter emit_error) {
  auto json = read_config_file(path, emit_error);
  if (json.failed())
    return util::Result::failure();
  bool has_dbt_guest = false;
  auto parsed = with_parsed_simulation_config_json(
      json.value(), rocjitsu::kEmbeddedSchema, emit_error,
      [&has_dbt_guest,
       emit_error](const fb::SimulationConfig *config) -> util::FailureOr<DbtGuestConfig> {
        has_dbt_guest = config->dbt_guest() != nullptr;
        if (config->system() && config->dbt_guest() && config->dbt_guest()->enabled())
          return emit_error.emit() << "system cannot be combined with dbt_guest mode";
        return dbt_guest_from_fb(config->dbt_guest(), emit_error);
      });
  if (parsed.failed())
    return util::Result::failure();
  if (!has_dbt_guest)
    return parsed;

  // Simulation configs remain forward-compatible with unknown fields, but a
  // DBT guest block selects execution behavior and must reject misspelled keys
  // instead of silently falling back to the hardware backend.
  const std::string strict_json = json_without_dbt_strict_ignored_top_level_fields(json.value());
  return with_parsed_simulation_config_json(
      strict_json, rocjitsu::kEmbeddedSchema, emit_error,
      [emit_error](const fb::SimulationConfig *config) {
        return dbt_guest_from_fb(config->dbt_guest(), emit_error);
      },
      false);
}

util::Result apply_resolved_dbt_host_gpu_id(DbtGuestConfig &config, std::string_view value,
                                            util::DiagnosticEmitter emit_error) {
  if (!config.enabled || config.host.gpu_id != 0)
    return util::Result::success();

  uint32_t gpu_id = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto [ptr, error] = std::from_chars(begin, end, gpu_id);
  if (error != std::errc{} || ptr != end || gpu_id == 0)
    return emit_error.emit() << "runtime config handoff must contain a nonzero KFD gpu_id";
  config.host.gpu_id = gpu_id;
  return util::Result::success();
}

bool write_dbt_runtime_config_handoff(const std::string &config_path, const DbtGuestConfig &config,
                                      const std::filesystem::path &invocation_runtime_dir) {
  if (config.enabled && config.host.gpu_id == 0)
    return false;

  const std::filesystem::path handoff_file = invocation_runtime_dir / "config_path";
  std::error_code directory_error;
  std::filesystem::create_directories(handoff_file.parent_path(), directory_error);
  if (directory_error)
    return false;
  const std::filesystem::path temp_file = handoff_file.string() + ".tmp";
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

util::FailureOr<DbtGuestConfig>
load_dbt_guest_config_from_handoff(const DbtRuntimeConfigHandoff &handoff,
                                   util::DiagnosticEmitter emit_error) {
  auto loaded = load_dbt_guest_config_from_file(handoff.config_path, emit_error);
  if (loaded.failed())
    return util::Result::failure();
  DbtGuestConfig config = std::move(loaded).value();
  if (handoff.resolved_gpu_id) {
    if (apply_resolved_dbt_host_gpu_id(config, *handoff.resolved_gpu_id, emit_error).failed())
      return util::Result::failure();
  } else if (config.enabled && config.host.gpu_id == 0) {
    return emit_error.emit() << "runtime config handoff must contain a resolved KFD gpu_id for "
                                "automatic DBT host selection";
  }
  return config;
}

util::FailureOr<std::optional<DbtGuestConfig>>
load_dbt_guest_config_from_runtime_config(util::DiagnosticEmitter emit_error) {
  // Try the handoff tiers in priority order, opening the first that exists:
  //   1. $ROCJITSU_INVOCATION_DIR/config_path — the launcher exports this dir before
  //      execvp so every descendant (incl. grandchildren via ctest, whose PID differs)
  //      finds it. Treat an empty value as unset (dir && *dir), matching interposer
  //      init(); an empty value would otherwise build "/config_path".
  //   2. this process's PID-scoped path (execvp preserves the launcher's PID for the
  //      direct child) — also the fallback if the env var is set but stale/misdirected.
  //   3. the well-known location for direct daemon-only or legacy attach scenarios.
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
  auto loaded = load_dbt_guest_config_from_handoff(*handoff, emit_error);
  if (loaded.failed())
    return util::Result::failure();
  return std::optional<DbtGuestConfig>(std::move(loaded).value());
}

} // namespace config
} // namespace rocjitsu
