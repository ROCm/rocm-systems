// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file config_common.h
/// @brief Shared FlatBuffers config parsing helpers.

#ifndef ROCJITSU_CONFIG_CONFIG_COMMON_H_
#define ROCJITSU_CONFIG_CONFIG_COMMON_H_

#include "rocjitsu/config/kfd_device_config.h"
#include "rocjitsu/config/pci_device_config.h"
#include "rocjitsu/result.h"

#include "flatbuffers/idl.h"
#include "simulation_config_generated.h"
#include "util/diagnostic.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace rocjitsu {
namespace config {

/// @brief Launch override: disable checks, enable ordinary checks, or also enable XCNT.
enum class WaitChecking { Off, On, All };

/// @brief Parse the on/off/all launch modes, distinct from per-CU warn/off policies.
inline std::optional<WaitChecking> parse_wait_checking(std::string_view value) {
  if (value == "off")
    return WaitChecking::Off;
  if (value == "on")
    return WaitChecking::On;
  if (value == "all")
    return WaitChecking::All;
  return std::nullopt;
}

/// @brief Parse a complete unsigned 32-bit decimal value.
inline std::optional<uint32_t> parse_uint32(std::string_view text) {
  uint32_t value = 0;
  const std::from_chars_result result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    return std::nullopt;
  return value;
}

inline FailureOr<uint32_t> parse_json_hex_digits(std::string_view json, size_t &at, size_t count,
                                                 const util::DiagnosticEmitter &emit_error) {
  if (json.size() - at < count)
    return emit_error.emit() << "unterminated escape sequence";

  const std::string_view digits = json.substr(at, count);
  uint32_t value = 0;
  const auto [ptr, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), value, 16);
  if (ec != std::errc{} || ptr != digits.data() + digits.size())
    return emit_error.emit() << "escape code must be followed by hexadecimal digits";

  at += count;
  return value;
}

inline Result append_json_utf8(uint32_t code_point, std::string &decoded,
                               const util::DiagnosticEmitter &emit_error) {
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
    return emit_error.emit() << "Unicode code point is out of range";
  }
  return Result::success();
}

/// @brief Decode a quoted string and advance past it.
inline FailureOr<std::string> parse_json_quoted_string(std::string_view json, size_t &at,
                                                       const util::DiagnosticEmitter &emit_error) {
  if (at >= json.size() || (json[at] != '"' && json[at] != '\''))
    return emit_error.emit() << "expected a quoted string";

  const char quote = json[at++];
  std::string decoded;
  int unicode_high_surrogate = -1;
  while (at < json.size()) {
    const char c = json[at++];
    if (c == quote) {
      if (unicode_high_surrogate != -1)
        return emit_error.emit() << "illegal Unicode sequence";
      return decoded;
    }

    if (static_cast<unsigned char>(c) < ' ')
      return emit_error.emit() << "illegal character in string constant";
    if (c != '\\') {
      if (unicode_high_surrogate != -1)
        return emit_error.emit() << "illegal Unicode sequence";
      decoded += c;
      continue;
    }

    if (at >= json.size())
      return emit_error.emit() << "unterminated escape sequence";
    const char escape = json[at++];
    if (unicode_high_surrogate != -1 && escape != 'u')
      return emit_error.emit() << "illegal Unicode sequence";

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
      decoded += '"';
      break;
    case '\'':
      decoded += '\'';
      break;
    case '\\':
      decoded += '\\';
      break;
    case '/':
      decoded += '/';
      break;
    case 'x': {
      FailureOr<uint32_t> digits = parse_json_hex_digits(json, at, 2, emit_error);
      if (digits.failed())
        return Result::failure();
      decoded += static_cast<char>(digits.value());
      break;
    }
    case 'u': {
      FailureOr<uint32_t> digits = parse_json_hex_digits(json, at, 4, emit_error);
      if (digits.failed())
        return Result::failure();
      const uint32_t value = digits.value();
      if (value >= 0xd800 && value <= 0xdbff) {
        if (unicode_high_surrogate != -1)
          return emit_error.emit() << "illegal Unicode sequence";
        unicode_high_surrogate = static_cast<int>(value);
      } else if (value >= 0xdc00 && value <= 0xdfff) {
        if (unicode_high_surrogate == -1)
          return emit_error.emit() << "illegal Unicode sequence";
        const uint32_t code_point =
            0x10000 + ((static_cast<uint32_t>(unicode_high_surrogate) & 0x3ff) << 10) +
            (value & 0x3ff);
        Result appended = append_json_utf8(code_point, decoded, emit_error);
        if (appended.failed())
          return Result::failure();
        unicode_high_surrogate = -1;
      } else {
        if (unicode_high_surrogate != -1)
          return emit_error.emit() << "illegal Unicode sequence";
        Result appended = append_json_utf8(value, decoded, emit_error);
        if (appended.failed())
          return Result::failure();
      }
      break;
    }
    default:
      return emit_error.emit() << "unknown escape code in string constant";
    }
  }

  return emit_error.emit() << "unterminated string constant";
}

inline std::string read_config_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Cannot open file: " + path);

  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

/// @brief Parse simulation-config JSON and consume the FlatBuffer while parser storage is alive.
///
/// @details FlatBuffers returns pointers into `Parser::builder_`. Keeping the
/// parser local to this helper makes the lifetime explicit: callers must copy
/// any values they need inside @p callback rather than retaining raw pointers.
/// @param skip_unexpected_fields Preserve the general simulation config's
/// forward-compatible unknown-field behavior when true; use false for strict
/// sub-configs whose misspelled keys could change execution behavior.
template <typename Callback>
decltype(auto)
with_parsed_simulation_config_json(const std::string &json, const std::string &schema_text,
                                   Callback &&callback, bool skip_unexpected_fields = true) {
  flatbuffers::IDLOptions options;
  // Preserve an explicit scalar default (notably cpu_dispatch_threads=1),
  // whose meaning differs from an omitted automatic setting in JSON.
  options.force_defaults = true;
  flatbuffers::Parser parser(options);
  parser.opts.skip_unexpected_fields_in_json = skip_unexpected_fields;
  if (!parser.Parse(schema_text.c_str()))
    throw std::runtime_error("Failed to parse schema: " + std::string(parser.error_));
  if (!parser.Parse(json.c_str()))
    throw std::runtime_error("Failed to parse JSON config: " + std::string(parser.error_));
  const fb::SimulationConfig *config =
      flatbuffers::GetRoot<fb::SimulationConfig>(parser.builder_.GetBufferPointer());
  return std::forward<Callback>(callback)(config);
}

/// @brief Convert a FlatBuffers PCI bus table into the runtime config form.
///
/// @details A config that says nothing about the bus gets the defaults, which
/// describe a compute GPU with apertures big enough for the driver to work with.
inline PciDeviceConfig pci_device_from_fb(const fb::PciDeviceInfo *pci) {
  PciDeviceConfig config;
  if (pci == nullptr)
    return config;

  config.class_code = pci->class_code();
  config.subsystem_vendor_id = pci->subsystem_vendor_id();
  config.subsystem_id = pci->subsystem_id();
  config.vram_aperture_bytes = pci->vram_aperture_bytes();
  config.doorbell_aperture_bytes = pci->doorbell_aperture_bytes();
  config.register_aperture_bytes = pci->register_aperture_bytes();
  return config;
}

/// @brief Convert a FlatBuffers KFD identity table into the runtime config form.
///
/// @details The resulting config owns its strings and IP versions so it is safe
/// after the FlatBuffers parser storage goes out of scope.
/// @param json_path JSON path of @p device, used in validation diagnostics.
inline KfdDeviceConfig kfd_device_from_fb(const fb::KfdDeviceInfo *device,
                                          std::string_view json_path) {
  KfdDeviceConfig config;
  if (device == nullptr)
    return config;

  config.present = true;
  config.gpu_id = device->gpu_id();
  config.gfx_target_version = device->gfx_target_version();
  config.vendor_id = device->vendor_id();
  config.device_id = device->device_id();
  config.family_id = device->family_id();
  config.unique_id = device->unique_id();
  if (device->marketing_name())
    config.marketing_name = device->marketing_name()->str();
  config.drm_render_minor = device->drm_render_minor();
  config.revision_id = device->revision_id();
  config.pci_revision_id = device->pci_revision_id();
  config.simd_count = device->simd_count();
  if (const auto value = device->max_waves_per_simd()) {
    config.max_waves_per_simd = *value;
    config.discovery_overrides.max_waves_per_simd = *value;
  }
  if (const auto value = device->num_shader_engines()) {
    config.num_shader_engines = *value;
    config.discovery_overrides.num_shader_engines = *value;
  }
  if (const auto value = device->num_shader_arrays_per_engine()) {
    config.num_shader_arrays_per_engine = *value;
    config.discovery_overrides.num_shader_arrays_per_engine = *value;
  }
  if (const auto value = device->num_cu_per_sh()) {
    config.num_cu_per_sh = *value;
    config.discovery_overrides.num_cu_per_sh = *value;
  }
  config.simd_per_cu = device->simd_per_cu();
  if (const auto value = device->wave_front_size()) {
    config.wave_front_size = *value;
    config.discovery_overrides.wave_front_size = *value;
  }
  if (const auto value = device->max_slots_scratch_cu()) {
    config.max_slots_scratch_cu = *value;
    config.discovery_overrides.max_slots_scratch_cu = *value;
  }
  config.local_mem_size = device->local_mem_size();
  config.vram_type = device->vram_type();
  if (const auto value = device->lds_size_kb()) {
    config.lds_size_kb = *value;
    config.discovery_overrides.lds_size_kb = *value;
  }
  config.mem_width = device->mem_width();
  config.mem_clk_max = device->mem_clk_max();
  config.l1_size_kb = device->l1_size_kb();
  config.l1_line_size = device->l1_line_size();
  config.l1_assoc = device->l1_assoc();
  config.l2_size_kb = device->l2_size_kb();
  config.l2_line_size = device->l2_line_size();
  config.l2_assoc = device->l2_assoc();
  config.num_sdma_engines = device->num_sdma_engines();
  config.num_sdma_xgmi_engines = device->num_sdma_xgmi_engines();
  config.num_sdma_queues_per_engine = device->num_sdma_queues_per_engine();
  if (config.num_sdma_engines != 0 && config.num_sdma_queues_per_engine == 0)
    throw std::runtime_error(std::string(json_path) +
                             ".num_sdma_queues_per_engine must be nonzero when " +
                             std::string(json_path) + ".num_sdma_engines is nonzero");
  config.num_cp_queues = device->num_cp_queues();
  config.max_engine_clk_fcompute = device->max_engine_clk_fcompute();
  config.location_id = device->location_id();
  config.hive_id = device->hive_id();
  config.domain = device->domain();
  config.capability = device->capability();
  config.capability2 = device->capability2();
  config.debug_prop = device->debug_prop();
  if (const auto *versions = device->ip_versions()) {
    for (const auto *version : *versions) {
      if (version->hardware_id() == 0 || version->major() == 0)
        throw std::runtime_error(std::string(json_path) +
                                 ".ip_versions requires nonzero hardware_id and major");
      if (std::ranges::any_of(config.ip_versions, [&](const DeviceIpVersion &existing) {
            return existing.hardware_id == version->hardware_id();
          }))
        throw std::runtime_error(std::string(json_path) +
                                 ".ip_versions contains a duplicate hardware_id");
      config.ip_versions.push_back(
          {version->hardware_id(), version->major(), version->minor(), version->revision()});
    }
  }
  return config;
}

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_COMMON_H_
