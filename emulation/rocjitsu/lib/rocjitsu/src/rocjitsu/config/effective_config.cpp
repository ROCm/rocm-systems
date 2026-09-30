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

bool is_field_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

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
      const size_t line_end = json.find('\n', at);
      at = line_end == std::string_view::npos ? json.size() : line_end + 1;
    } else if (json.compare(at, 2, "/*") == 0) {
      const size_t block_end = json.find("*/", at + 2);
      at = block_end == std::string_view::npos ? json.size() : block_end + 2;
    } else {
      return;
    }
  }
}

/// @brief Advance past a quoted string, honoring backslash escapes.
void skip_string(std::string_view json, size_t &at) {
  for (++at; at < json.size(); ++at) {
    if (json[at] == '\\') {
      ++at;
      continue;
    }
    if (json[at] == '"') {
      ++at;
      return;
    }
  }
}

/// @brief Advance past one value, descending through nested objects and arrays.
void skip_value(std::string_view json, size_t &at) {
  if (at >= json.size())
    return;

  if (json[at] == '"') {
    skip_string(json, at);
    return;
  }

  if (json[at] == '{' || json[at] == '[') {
    size_t depth = 0;
    while (at < json.size()) {
      const char c = json[at];
      if (c == '"') {
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
         json[at] != ' ' && json[at] != '\t' && json[at] != '\r' && json[at] != '\n')
    ++at;
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

    std::string_view name;
    if (json[at] == '"') {
      const size_t name_begin = at + 1;
      skip_string(json, at);
      if (at > name_begin)
        name = json.substr(name_begin, at - 1 - name_begin);
    } else {
      const size_t name_begin = at;
      while (at < json.size() && is_field_name_char(json[at]))
        ++at;
      name = json.substr(name_begin, at - name_begin);
    }
    if (name.empty())
      throw std::runtime_error("simulation config has an unreadable field name");

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
