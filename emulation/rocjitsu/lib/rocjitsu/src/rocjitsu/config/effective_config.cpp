// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/effective_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/system_graph.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "simulation_config_generated.h"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace rocjitsu {
namespace config {
namespace {

constexpr std::string_view kBudgetField = "cpu_thread_budget";
constexpr std::string_view kNodeField = "local_node";

std::string json_string(std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 0x20) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 0x0f];
    } else {
      result += static_cast<char>(c);
    }
  }
  result += '"';
  return result;
}

bool is_identifier_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool is_identifier_char(char c) { return is_identifier_start(c) || (c >= '0' && c <= '9'); }

bool starts_comment(std::string_view json, size_t at) {
  return json[at] == '/' && at + 1 < json.size() && (json[at + 1] == '/' || json[at + 1] == '*');
}

/// @brief Advance past whitespace and the comment forms the config parser accepts.
void skip_filler(std::string_view json, size_t &at) {
  while (at < json.size()) {
    const char c = json[at];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
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

FailureOr<uint32_t> parse_hex_digits(std::string_view json, size_t &at, size_t count,
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

Result append_utf8(uint32_t code_point, std::string &decoded,
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
FailureOr<std::string> parse_quoted_string(std::string_view json, size_t &at,
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
      FailureOr<uint32_t> digits = parse_hex_digits(json, at, 2, emit_error);
      if (digits.failed())
        return Result::failure();
      decoded += static_cast<char>(digits.value());
      break;
    }
    case 'u': {
      FailureOr<uint32_t> digits = parse_hex_digits(json, at, 4, emit_error);
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
        Result appended = append_utf8(code_point, decoded, emit_error);
        if (appended.failed())
          return Result::failure();
        unicode_high_surrogate = -1;
      } else {
        if (unicode_high_surrogate != -1)
          return emit_error.emit() << "illegal Unicode sequence";
        Result appended = append_utf8(value, decoded, emit_error);
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

/// @brief Advance past a quoted string, honoring the loader's escapes.
Result skip_string(std::string_view json, size_t &at, const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> parsed = parse_quoted_string(json, at, emit_error);
  if (parsed.failed())
    return Result::failure();
  return Result::success();
}

/// @brief Advance past one value, descending through nested objects and arrays.
Result skip_value(std::string_view json, size_t &at, const util::DiagnosticEmitter &emit_error) {
  if (at >= json.size())
    return Result::success();

  if (json[at] == '"' || json[at] == '\'')
    return skip_string(json, at, emit_error);

  if (json[at] == '{' || json[at] == '[') {
    size_t depth = 0;
    while (at < json.size()) {
      const char c = json[at];
      if (c == '"' || c == '\'') {
        Result skipped = skip_string(json, at, emit_error);
        if (skipped.failed())
          return skipped;
        continue;
      }
      if (starts_comment(json, at)) {
        skip_filler(json, at);
        continue;
      }
      if (c == '{' || c == '[') {
        ++depth;
      } else if (c == '}' || c == ']') {
        --depth;
        if (depth == 0) {
          ++at;
          return Result::success();
        }
      }
      ++at;
    }
    return Result::success();
  }

  while (at < json.size() && json[at] != ',' && json[at] != '}' && json[at] != ']' &&
         json[at] != ' ' && json[at] != '\t' && json[at] != '\r' && json[at] != '\n') {
    if (starts_comment(json, at))
      break;
    ++at;
  }
  return Result::success();
}

struct FieldRange {
  size_t begin;
  size_t end;
  bool insertion_needs_comma;
};

/// @brief Locate a top-level value, or the insertion point for an absent field.
FailureOr<FieldRange> find_top_level_field(std::string_view json, std::string_view field,
                                           const util::DiagnosticEmitter &emit_error) {
  size_t at = 0;
  skip_filler(json, at);
  if (at >= json.size() || json[at] != '{')
    return emit_error.emit() << "simulation config must be a JSON object";
  const size_t fields_begin = ++at;

  skip_filler(json, at);
  const bool has_fields = at < json.size() && json[at] != '}';

  while (at < json.size()) {
    skip_filler(json, at);
    if (at >= json.size() || json[at] == '}')
      break;

    std::string name;
    if (json[at] == '"' || json[at] == '\'') {
      FailureOr<std::string> parsed_name = parse_quoted_string(json, at, emit_error);
      if (parsed_name.failed())
        return Result::failure();
      name = std::move(parsed_name.value());
    } else {
      const size_t name_begin = at;
      if (at >= json.size() || !is_identifier_start(json[at]))
        return emit_error.emit() << "simulation config has an unreadable field name";
      while (at < json.size() && is_identifier_char(json[at]))
        ++at;
      name.assign(json.substr(name_begin, at - name_begin));
    }
    if (name.empty())
      return emit_error.emit() << "simulation config has an empty field name";

    skip_filler(json, at);
    if (at >= json.size() || json[at] != ':')
      return emit_error.emit() << "simulation config field '" << name << "' has no value";
    ++at;

    skip_filler(json, at);
    const size_t value_begin = at;
    Result skipped_value = skip_value(json, at, emit_error);
    if (skipped_value.failed())
      return Result::failure();
    if (name == field) {
      return FieldRange{value_begin, at, false};
    }

    skip_filler(json, at);
    if (at < json.size() && json[at] == ',')
      ++at;
  }

  return FieldRange{fields_begin, fields_begin, has_fields};
}

/// @brief Replace one top-level value, preserving all unrelated configuration text.
FailureOr<std::string> set_top_level_field(std::string_view json, std::string_view field,
                                           std::string_view value,
                                           const util::DiagnosticEmitter &emit_error) {
  auto found = find_top_level_field(json, field, emit_error);
  if (found.failed())
    return Result::failure();
  const auto range = found.value();
  std::string rewritten(json.substr(0, range.begin));
  if (range.begin == range.end)
    rewritten.append("\"").append(field).append("\": ");
  rewritten.append(value);
  if (range.insertion_needs_comma)
    rewritten.append(",");
  rewritten.append(json.substr(range.end));
  return rewritten;
}

} // namespace

FailureOr<std::string> json_top_level_field(std::string_view json, std::string_view field,
                                            const util::DiagnosticEmitter &emit_error) {
  auto found = find_top_level_field(json, field, emit_error);
  if (found.failed())
    return Result::failure();
  const auto range = found.value();
  return std::string(json.substr(range.begin, range.end - range.begin));
}

FailureOr<std::string> json_with_cpu_thread_budget(std::string_view json, uint32_t budget,
                                                   const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> rewritten =
      set_top_level_field(json, kBudgetField, std::to_string(budget), emit_error);
  if (rewritten.failed())
    return Result::failure();

  FailureOr<uint32_t> applied = with_parsed_simulation_config_json(
      rewritten.value(), rocjitsu::kEmbeddedSchema, emit_error,
      [](const fb::SimulationConfig *config) -> FailureOr<uint32_t> {
        return config->cpu_thread_budget();
      });
  if (applied.failed())
    return Result::failure();
  if (applied.value() != budget)
    return emit_error.emit() << "cannot apply cpu_thread_budget to the simulation config";
  return std::move(rewritten.value());
}

FailureOr<std::string> json_with_launch_overrides(std::string_view json,
                                                  const LaunchOverrides &overrides,
                                                  const util::DiagnosticEmitter &emit_error) {
  std::string rewritten(json);
  if (overrides.cpu_thread_budget) {
    auto result = json_with_cpu_thread_budget(rewritten, *overrides.cpu_thread_budget, emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
  if (overrides.wait_checking) {
    const char *value = *overrides.wait_checking == WaitChecking::Off  ? "\"off\""
                        : *overrides.wait_checking == WaitChecking::On ? "\"on\""
                                                                       : "\"all\"";
    auto result = set_top_level_field(rewritten, "wait_checking", value, emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
    auto applied = with_parsed_simulation_config_json(
        rewritten, rocjitsu::kEmbeddedSchema, emit_error,
        [](const fb::SimulationConfig *config) -> FailureOr<std::optional<WaitChecking>> {
          return config->wait_checking()
                     ? parse_wait_checking(config->wait_checking()->string_view())
                     : std::nullopt;
        });
    if (applied.failed())
      return Result::failure();
    if (applied.value() != overrides.wait_checking)
      return emit_error.emit() << "cannot apply wait_checking to the simulation config";
  }
  if (!overrides.local_node.empty()) {
    auto result = json_with_local_node(rewritten, overrides.local_node, emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
  return rewritten;
}

FailureOr<std::string> json_with_local_node(std::string_view json, std::string_view node_id,
                                            const util::DiagnosticEmitter &emit_error) {
  if (node_id.empty())
    return emit_error.emit() << "local node ID must be nonempty";
  auto configured = with_parsed_simulation_config_json(
      std::string(json), rocjitsu::kEmbeddedSchema, emit_error,
      [](const fb::SimulationConfig *config) -> FailureOr<std::string> {
        return config->local_node() ? config->local_node()->str() : std::string{};
      });
  if (configured.failed())
    return Result::failure();
  if (!configured.value().empty() && configured.value() != node_id)
    return emit_error.emit() << "local node argument disagrees with config.local_node";
  FailureOr<std::string> rewritten =
      set_top_level_field(json, kNodeField, json_string(node_id), emit_error);
  if (rewritten.failed())
    return Result::failure();
  try {
    auto graph =
        load_system_graph_from_string(rewritten.value(), rocjitsu::kEmbeddedSchema, emit_error);
    if (graph.failed())
      return Result::failure();
    (void)graph.value().select_node(node_id);
    auto applied = with_parsed_simulation_config_json(
        rewritten.value(), rocjitsu::kEmbeddedSchema, emit_error,
        [](const fb::SimulationConfig *config) -> FailureOr<std::string> {
          return config->local_node() ? config->local_node()->str() : std::string{};
        });
    if (applied.failed())
      return Result::failure();
    if (applied.value() != node_id)
      return emit_error.emit() << "cannot apply local_node to the simulation config";
  } catch (const std::exception &error) {
    return emit_error.emit() << error.what();
  }
  return std::move(rewritten.value());
}

FailureOr<std::string> write_effective_config(const std::string &source_path,
                                              const LaunchOverrides &overrides, pid_t pid,
                                              const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> source_json = read_config_file(source_path, emit_error);
  if (source_json.failed())
    return Result::failure();
  FailureOr<std::string> json =
      json_with_launch_overrides(source_json.value(), overrides, emit_error);
  if (json.failed())
    return Result::failure();
  std::string effective_json = std::move(json.value());

  const std::filesystem::path directory(rocjitsu::rpc_invocation_runtime_dir(pid));
  std::error_code directory_error;
  std::filesystem::create_directories(directory, directory_error);
  if (directory_error)
    return emit_error.emit() << "cannot create runtime directory " << directory.string() << ": "
                             << directory_error.message();

  const std::filesystem::path target = directory / kEffectiveConfigName;
  const std::filesystem::path temporary = target.string() + ".tmp";
  std::ofstream output(temporary);
  output << effective_json;
  output.close();
  if (!output.good()) {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    return emit_error.emit() << "cannot write effective config " << target.string();
  }

  std::error_code rename_error;
  std::filesystem::rename(temporary, target, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    return emit_error.emit() << "cannot publish effective config " << target.string() << ": "
                             << rename_error.message();
  }
  return target.string();
}

FailureOr<std::string> write_effective_config(const std::string &source_path,
                                              std::optional<uint32_t> budget,
                                              std::string_view node_id, pid_t pid,
                                              const util::DiagnosticEmitter &emit_error) {
  LaunchOverrides overrides;
  overrides.cpu_thread_budget = budget;
  overrides.local_node = node_id;
  return write_effective_config(source_path, overrides, pid, emit_error);
}

} // namespace config
} // namespace rocjitsu
