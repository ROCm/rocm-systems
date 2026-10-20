// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/effective_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "simulation_config_generated.h"

#include <cctype>
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

bool is_identifier_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool is_identifier_char(char c) { return is_identifier_start(c) || (c >= '0' && c <= '9'); }

bool starts_comment(std::string_view json, size_t at) {
  return json[at] == '/' && at + 1 < json.size() && (json[at + 1] == '/' || json[at + 1] == '*');
}

bool is_valid_plugin_name(std::string_view name) {
  if (name.empty())
    return false;
  for (unsigned char c : name)
    if (!std::isalnum(c) && c != '_' && c != '-')
      return false;
  return true;
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

/// @brief Advance past a quoted string, honoring the loader's escapes.
Result skip_string(std::string_view json, size_t &at, const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> parsed = parse_json_quoted_string(json, at, emit_error);
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

struct ObjectField {
  bool found = false;
  size_t fields_begin = 0;
  bool has_fields = false;
  size_t value_begin = 0;
  size_t value_end = 0;
};

FailureOr<ObjectField> find_object_field(std::string_view json, size_t object_begin,
                                         std::string_view field,
                                         const util::DiagnosticEmitter &emit_error) {
  size_t at = object_begin;
  if (at >= json.size() || json[at] != '{')
    return emit_error.emit() << "simulation config must be a JSON object";
  ObjectField result;
  result.fields_begin = ++at;

  skip_filler(json, at);
  result.has_fields = at < json.size() && json[at] != '}';

  while (at < json.size()) {
    skip_filler(json, at);
    if (at >= json.size() || json[at] == '}')
      break;

    std::string name;
    if (json[at] == '"' || json[at] == '\'') {
      FailureOr<std::string> parsed_name = parse_json_quoted_string(json, at, emit_error);
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
    const size_t value_end = at;
    if (name == field) {
      result.found = true;
      result.value_begin = value_begin;
      result.value_end = value_end;
      return result;
    }

    skip_filler(json, at);
    if (at < json.size() && json[at] == ',')
      ++at;
  }

  return result;
}

FailureOr<std::string> set_field_in_object(std::string_view json, size_t object_begin,
                                           std::string_view field, std::string_view value,
                                           const util::DiagnosticEmitter &emit_error) {
  FailureOr<ObjectField> found = find_object_field(json, object_begin, field, emit_error);
  if (found.failed())
    return Result::failure();
  if (found.value().found) {
    std::string rewritten(json.substr(0, found.value().value_begin));
    rewritten.append(value);
    rewritten.append(json.substr(found.value().value_end));
    return rewritten;
  }

  std::string rewritten(json.substr(0, found.value().fields_begin));
  rewritten.append("\"").append(field).append("\": ").append(value);
  if (found.value().has_fields)
    rewritten.append(",");
  rewritten.append(json.substr(found.value().fields_begin));
  return rewritten;
}

/// @brief Return @p json with top-level @p field set to @p value, inserting it when absent.
///
/// @details Only the one field's text changes, so every other field keeps the
/// spelling, ordering and formatting the author gave it and stays readable when a
/// failing run is reproduced from the launch copy.
FailureOr<std::string> set_top_level_field(std::string_view json, std::string_view field,
                                           std::string_view value,
                                           const util::DiagnosticEmitter &emit_error) {
  size_t at = 0;
  skip_filler(json, at);
  return set_field_in_object(json, at, field, value, emit_error);
}

FailureOr<std::string> json_with_gpu_count(std::string_view json, uint32_t count,
                                           const util::DiagnosticEmitter &emit_error) {
  size_t at = 0;
  skip_filler(json, at);
  FailureOr<ObjectField> vm = find_object_field(json, at, "vm", emit_error);
  if (vm.failed())
    return Result::failure();
  if (!vm.value().found)
    return emit_error.emit() << "cannot apply gpus_per_node: simulation config has no vm";

  size_t vm_begin = vm.value().value_begin;
  skip_filler(json, vm_begin);
  if (vm_begin >= json.size() || json[vm_begin] != '{')
    return emit_error.emit()
           << "cannot apply gpus_per_node: simulation config vm must be an object";

  FailureOr<ObjectField> gpu = find_object_field(json, vm_begin, "gpu", emit_error);
  if (gpu.failed())
    return Result::failure();
  if (!gpu.value().found)
    return emit_error.emit() << "cannot apply gpus_per_node: simulation config has no vm.gpu";

  size_t gpu_begin = gpu.value().value_begin;
  skip_filler(json, gpu_begin);
  if (gpu_begin >= json.size() || json[gpu_begin] != '{')
    return emit_error.emit()
           << "cannot apply gpus_per_node: simulation config vm.gpu must be an object";

  FailureOr<std::string> rewritten =
      set_field_in_object(json, gpu_begin, "num_gpus", std::to_string(count), emit_error);
  if (rewritten.failed())
    return Result::failure();

  try {
    const uint32_t applied = with_parsed_simulation_config_json(
        rewritten.value(), rocjitsu::kEmbeddedSchema, [](const fb::SimulationConfig *config) {
          return config->vm() && config->vm()->gpu() ? config->vm()->gpu()->num_gpus() : 0;
        });
    if (applied != count)
      return emit_error.emit() << "cannot apply gpus_per_node to the simulation config";
  } catch (const std::runtime_error &error) {
    return emit_error.emit() << error.what();
  }
  return std::move(rewritten.value());
}

FailureOr<std::string> json_with_plugin_enabled(std::string_view json, std::string_view plugin,
                                                const util::DiagnosticEmitter &emit_error) {
  if (!is_valid_plugin_name(plugin))
    return emit_error.emit() << "invalid plugin name '" << plugin
                             << "'; allowed: letters, digits, '_', '-'";

  size_t at = 0;
  skip_filler(json, at);
  FailureOr<ObjectField> plugins = find_object_field(json, at, "plugins", emit_error);
  if (plugins.failed())
    return Result::failure();

  if (!plugins.value().found) {
    std::string plugin_object = "{\"";
    plugin_object.append(plugin).append("\": {}}");
    return set_top_level_field(json, "plugins", plugin_object, emit_error);
  }

  size_t plugins_begin = plugins.value().value_begin;
  skip_filler(json, plugins_begin);
  if (plugins_begin >= json.size() || json[plugins_begin] != '{')
    return emit_error.emit() << "cannot apply plugin: simulation config plugins must be an object";

  FailureOr<ObjectField> existing_plugin =
      find_object_field(json, plugins_begin, plugin, emit_error);
  if (existing_plugin.failed())
    return Result::failure();
  if (existing_plugin.value().found)
    return std::string(json);

  return set_field_in_object(json, plugins_begin, plugin, "{}", emit_error);
}

} // namespace

FailureOr<std::string> json_with_cpu_thread_budget(std::string_view json, uint32_t budget,
                                                   const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> rewritten =
      set_top_level_field(json, kBudgetField, std::to_string(budget), emit_error);
  if (rewritten.failed())
    return Result::failure();

  // The schema parser reports a bad document by throwing. Callers of this function
  // handle failure through the returned result, so that message is emitted here.
  uint32_t applied = 0;
  try {
    applied = with_parsed_simulation_config_json(
        rewritten.value(), rocjitsu::kEmbeddedSchema,
        [](const fb::SimulationConfig *config) { return config->cpu_thread_budget(); });
  } catch (const std::runtime_error &error) {
    return emit_error.emit() << error.what();
  }
  if (applied != budget)
    return emit_error.emit() << "cannot apply cpu_thread_budget to the simulation config";
  return std::move(rewritten.value());
}

FailureOr<std::string> json_with_launch_overrides(std::string_view json,
                                                  const LaunchOverrides &overrides,
                                                  const util::DiagnosticEmitter &emit_error) {
  std::string rewritten(json);
  if (overrides.exec_mode) {
    std::string value = "\"";
    value.append(*overrides.exec_mode).append("\"");
    auto result = set_top_level_field(rewritten, "exec_mode", value, emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
    try {
      const std::string applied = with_parsed_simulation_config_json(
          rewritten, rocjitsu::kEmbeddedSchema, [](const fb::SimulationConfig *config) {
            return config->exec_mode() ? config->exec_mode()->str() : std::string{};
          });
      if (applied != *overrides.exec_mode)
        return emit_error.emit() << "cannot apply exec_mode to the simulation config";
    } catch (const std::runtime_error &error) {
      return emit_error.emit() << error.what();
    }
  }
  if (overrides.gpus_per_node) {
    auto result = json_with_gpu_count(rewritten, *overrides.gpus_per_node, emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
  if (!overrides.plugins.empty()) {
    for (const std::string &plugin : overrides.plugins) {
      auto result = json_with_plugin_enabled(rewritten, plugin, emit_error);
      if (result.failed())
        return Result::failure();
      rewritten = std::move(result.value());
    }
    auto result = set_top_level_field(rewritten, "require_all_plugins", "true", emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
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
    try {
      const auto applied = with_parsed_simulation_config_json(
          rewritten, rocjitsu::kEmbeddedSchema, [](const fb::SimulationConfig *config) {
            return config->wait_checking()
                       ? parse_wait_checking(config->wait_checking()->string_view())
                       : std::nullopt;
          });
      if (applied != overrides.wait_checking)
        return emit_error.emit() << "cannot apply wait_checking to the simulation config";
    } catch (const std::runtime_error &error) {
      return emit_error.emit() << error.what();
    }
  }
  return rewritten;
}

FailureOr<std::string> write_effective_config(const std::string &source_path,
                                              const LaunchOverrides &overrides, pid_t pid,
                                              const util::DiagnosticEmitter &emit_error) {
  // Reading the source throws when the file cannot be opened. The launch path
  // reports that through the same result as a rewrite or publish failure.
  std::string source_json;
  try {
    source_json = read_config_file(source_path);
  } catch (const std::runtime_error &error) {
    return emit_error.emit() << error.what();
  }
  FailureOr<std::string> json = json_with_launch_overrides(source_json, overrides, emit_error);
  if (json.failed())
    return Result::failure();

  const std::filesystem::path directory(rocjitsu::rpc_invocation_runtime_dir(pid));
  std::error_code directory_error;
  std::filesystem::create_directories(directory, directory_error);
  if (directory_error)
    return emit_error.emit() << "cannot create runtime directory " << directory.string() << ": "
                             << directory_error.message();

  const std::filesystem::path target = directory / kEffectiveConfigName;
  const std::filesystem::path temporary = target.string() + ".tmp";
  std::ofstream output(temporary);
  output << json.value();
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

} // namespace config
} // namespace rocjitsu
