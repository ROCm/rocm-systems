// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/effective_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "simulation_config_generated.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace rocjitsu {
namespace config {
namespace {

constexpr std::string_view kBudgetField = "cpu_thread_budget";

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
      size_t line_end = at + 2;
      while (line_end < json.size() && json[line_end] != '\n' && json[line_end] != '\r')
        ++line_end;
      at = line_end == json.size() ? json.size() : line_end + 1;
    } else if (json.compare(at, 2, "/*") == 0) {
      const size_t block_end = json.find("*/", at + 2);
      at = block_end == std::string_view::npos ? json.size() : block_end + 2;
    } else {
      return;
    }
  }
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

uint32_t parse_hex_digits(std::string_view json, size_t &at, size_t count) {
  if (json.size() - at < count)
    throw std::runtime_error("unterminated escape sequence");

  uint32_t value = 0;
  for (size_t i = 0; i < count; ++i) {
    const int digit = hex_digit(json[at++]);
    if (digit < 0)
      throw std::runtime_error("escape code must be followed by hexadecimal digits");
    value = (value << 4) | static_cast<uint32_t>(digit);
  }
  return value;
}

void append_utf8(uint32_t code_point, std::string &decoded) {
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
    throw std::runtime_error("Unicode code point is out of range");
  }
}

/// @brief Decode a quoted string and advance past it.
std::string parse_quoted_string(std::string_view json, size_t &at) {
  if (at >= json.size() || (json[at] != '"' && json[at] != '\''))
    throw std::runtime_error("expected a quoted string");

  const char quote = json[at++];
  std::string decoded;
  int unicode_high_surrogate = -1;
  while (at < json.size()) {
    const char c = json[at++];
    if (c == quote) {
      if (unicode_high_surrogate != -1)
        throw std::runtime_error("illegal Unicode sequence");
      return decoded;
    }

    if (static_cast<unsigned char>(c) < ' ')
      throw std::runtime_error("illegal character in string constant");
    if (c != '\\') {
      if (unicode_high_surrogate != -1)
        throw std::runtime_error("illegal Unicode sequence");
      decoded += c;
      continue;
    }

    if (at >= json.size())
      throw std::runtime_error("unterminated escape sequence");
    const char escape = json[at++];
    if (unicode_high_surrogate != -1 && escape != 'u')
      throw std::runtime_error("illegal Unicode sequence");

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
    case 'x':
      decoded += static_cast<char>(parse_hex_digits(json, at, 2));
      break;
    case 'u': {
      const uint32_t value = parse_hex_digits(json, at, 4);
      if (value >= 0xd800 && value <= 0xdbff) {
        if (unicode_high_surrogate != -1)
          throw std::runtime_error("illegal Unicode sequence");
        unicode_high_surrogate = static_cast<int>(value);
      } else if (value >= 0xdc00 && value <= 0xdfff) {
        if (unicode_high_surrogate == -1)
          throw std::runtime_error("illegal Unicode sequence");
        const uint32_t code_point =
            0x10000 + ((static_cast<uint32_t>(unicode_high_surrogate) & 0x3ff) << 10) +
            (value & 0x3ff);
        append_utf8(code_point, decoded);
        unicode_high_surrogate = -1;
      } else {
        if (unicode_high_surrogate != -1)
          throw std::runtime_error("illegal Unicode sequence");
        append_utf8(value, decoded);
      }
      break;
    }
    default:
      throw std::runtime_error("unknown escape code in string constant");
    }
  }

  throw std::runtime_error("unterminated string constant");
}

/// @brief Advance past a quoted string, honoring the loader's escapes.
void skip_string(std::string_view json, size_t &at) { (void)parse_quoted_string(json, at); }

/// @brief Advance past one value, descending through nested objects and arrays.
void skip_value(std::string_view json, size_t &at) {
  if (at >= json.size())
    return;

  if (json[at] == '"' || json[at] == '\'') {
    skip_string(json, at);
    return;
  }

  if (json[at] == '{' || json[at] == '[') {
    size_t depth = 0;
    while (at < json.size()) {
      const char c = json[at];
      if (c == '"' || c == '\'') {
        skip_string(json, at);
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
          return;
        }
      }
      ++at;
    }
    return;
  }

  while (at < json.size() && json[at] != ',' && json[at] != '}' && json[at] != ']' &&
         json[at] != ' ' && json[at] != '\t' && json[at] != '\r' && json[at] != '\n') {
    if (starts_comment(json, at))
      break;
    ++at;
  }
}

/// @brief Return @p json with top-level @p field set to @p value, inserting it when absent.
///
/// @details Only the one field's text changes, so every other field keeps the
/// spelling, ordering and formatting the author gave it and stays readable when a
/// failing run is reproduced from the launch copy.
std::string set_top_level_field(std::string_view json, std::string_view field,
                                std::string_view value) {
  size_t at = 0;
  skip_filler(json, at);
  if (at >= json.size() || json[at] != '{')
    throw std::runtime_error("simulation config must be a JSON object");
  const size_t fields_begin = ++at;

  skip_filler(json, at);
  const bool has_fields = at < json.size() && json[at] != '}';

  while (at < json.size()) {
    skip_filler(json, at);
    if (at >= json.size() || json[at] == '}')
      break;

    std::string name;
    if (json[at] == '"' || json[at] == '\'') {
      name = parse_quoted_string(json, at);
    } else {
      const size_t name_begin = at;
      if (at >= json.size() || !is_identifier_start(json[at]))
        throw std::runtime_error("simulation config has an unreadable field name");
      while (at < json.size() && is_identifier_char(json[at]))
        ++at;
      name.assign(json.substr(name_begin, at - name_begin));
    }
    if (name.empty())
      throw std::runtime_error("simulation config has an empty field name");

    skip_filler(json, at);
    if (at >= json.size() || json[at] != ':')
      throw std::runtime_error("simulation config field '" + std::string(name) + "' has no value");
    ++at;

    skip_filler(json, at);
    const size_t value_begin = at;
    skip_value(json, at);
    if (name == field) {
      std::string rewritten(json.substr(0, value_begin));
      rewritten.append(value);
      rewritten.append(json.substr(at));
      return rewritten;
    }

    skip_filler(json, at);
    if (at < json.size() && json[at] == ',')
      ++at;
  }

  std::string rewritten(json.substr(0, fields_begin));
  rewritten.append("\"").append(field).append("\": ").append(value);
  if (has_fields)
    rewritten.append(",");
  rewritten.append(json.substr(fields_begin));
  return rewritten;
}

} // namespace

std::string json_with_cpu_thread_budget(std::string_view json, uint32_t budget) {
  std::string rewritten = set_top_level_field(json, kBudgetField, std::to_string(budget));

  // Rewriting text must never be able to start a simulation under a budget nobody
  // asked for, so the result is parsed and the field read back before it is used.
  const uint32_t applied = with_parsed_simulation_config_json(
      rewritten, rocjitsu::kEmbeddedSchema,
      [](const fb::SimulationConfig *config) { return config->cpu_thread_budget(); });
  if (applied != budget)
    throw std::runtime_error("cannot apply cpu_thread_budget to the simulation config");
  return rewritten;
}

std::string write_effective_config(const std::string &source_path, uint32_t budget, pid_t pid) {
  const std::string json = json_with_cpu_thread_budget(read_config_file(source_path), budget);

  const std::filesystem::path directory(rocjitsu::rpc_invocation_runtime_dir(pid));
  std::error_code directory_error;
  std::filesystem::create_directories(directory, directory_error);
  if (directory_error)
    throw std::runtime_error("cannot create runtime directory " + directory.string() + ": " +
                             directory_error.message());

  const std::filesystem::path target = directory / kEffectiveConfigName;
  const std::filesystem::path temporary = target.string() + ".tmp";
  std::ofstream output(temporary);
  output << json;
  output.close();
  if (!output.good()) {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    throw std::runtime_error("cannot write effective config " + target.string());
  }

  std::error_code rename_error;
  std::filesystem::rename(temporary, target, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    throw std::runtime_error("cannot publish effective config " + target.string() + ": " +
                             rename_error.message());
  }
  return target.string();
}

} // namespace config
} // namespace rocjitsu
