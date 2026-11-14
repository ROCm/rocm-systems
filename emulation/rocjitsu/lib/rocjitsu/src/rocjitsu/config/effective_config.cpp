// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/effective_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/json_fields.h"
#include "rocjitsu/config/system_graph.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "simulation_config_generated.h"

#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/idl.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

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

std::string json_string_array(const std::vector<std::string> &values) {
  std::string result = "[";
  bool first = true;
  for (const std::string &value : values) {
    if (!first)
      result += ", ";
    result += json_string(value);
    first = false;
  }
  result += "]";
  return result;
}

FailureOr<std::vector<std::string>>
required_plugins_from_json(std::string_view json, const util::DiagnosticEmitter &emit_error) {
  auto field = json_top_level_field(json, "required_plugins", emit_error);
  if (field.failed())
    return Result::failure();
  if (field.value().empty())
    return std::vector<std::string>{};

  flatbuffers::Parser parser;
  flexbuffers::Builder builder;
  const std::string value(field.value());
  if (!parser.ParseFlexBuffer(value.c_str(), nullptr, &builder))
    return emit_error.emit() << "top-level 'required_plugins' must be a JSON string array";
  const flexbuffers::Reference root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsVector())
    return emit_error.emit() << "top-level 'required_plugins' must be an array";
  const flexbuffers::Vector entries = root.AsVector();
  std::vector<std::string> plugins;
  plugins.reserve(entries.size());
  for (size_t index = 0; index < entries.size(); ++index) {
    const flexbuffers::Reference entry = entries[index];
    if (!entry.IsString())
      return emit_error.emit() << "top-level 'required_plugins' entries must be strings";
    plugins.emplace_back(entry.AsString().str());
  }
  return plugins;
}

FailureOr<std::vector<std::string>>
merged_required_plugins(std::string_view json, const std::vector<std::string> &launch_plugins,
                        const util::DiagnosticEmitter &emit_error) {
  auto merged = required_plugins_from_json(json, emit_error);
  if (merged.failed())
    return Result::failure();
  for (const std::string &plugin : launch_plugins) {
    if (std::ranges::find(merged.value(), plugin) == merged.value().end())
      merged.value().push_back(plugin);
  }
  return std::move(merged.value());
}

bool is_json_object(std::string_view value) {
  const size_t at = value.find_first_not_of(" \t\r\n");
  return at != std::string_view::npos && value[at] == '{';
}

bool is_valid_plugin_name(std::string_view name) {
  if (name.empty())
    return false;
  for (unsigned char c : name)
    if (!std::isalnum(c) && c != '_' && c != '-')
      return false;
  return true;
}

template <typename Value, typename ReadField>
FailureOr<std::string> json_with_top_level_number(std::string_view json, std::string_view field,
                                                  Value value, ReadField read_field,
                                                  const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> rewritten =
      set_top_level_field(json, field, std::to_string(value), emit_error);
  if (rewritten.failed())
    return Result::failure();

  FailureOr<Value> applied = with_parsed_simulation_config_json(
      rewritten.value(), rocjitsu::kEmbeddedSchema, emit_error, std::move(read_field));
  if (applied.failed())
    return Result::failure();
  if (applied.value() != value)
    return emit_error.emit() << "cannot apply " << field << " to the simulation config";
  return std::move(rewritten.value());
}

FailureOr<std::string> json_with_gpu_count(std::string_view json, uint32_t count,
                                           const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> vm = json_top_level_field(json, "vm", emit_error);
  if (vm.failed())
    return Result::failure();
  if (vm.value().empty())
    return emit_error.emit() << "cannot apply gpus_per_node: simulation config has no vm";
  if (!is_json_object(vm.value()))
    return emit_error.emit()
           << "cannot apply gpus_per_node: simulation config vm must be an object";

  FailureOr<std::string> gpu = json_top_level_field(vm.value(), "gpu", emit_error);
  if (gpu.failed())
    return Result::failure();
  if (gpu.value().empty())
    return emit_error.emit() << "cannot apply gpus_per_node: simulation config has no vm.gpu";
  if (!is_json_object(gpu.value()))
    return emit_error.emit()
           << "cannot apply gpus_per_node: simulation config vm.gpu must be an object";

  FailureOr<std::string> rewritten_gpu =
      set_top_level_field(gpu.value(), "num_gpus", std::to_string(count), emit_error);
  if (rewritten_gpu.failed())
    return Result::failure();
  FailureOr<std::string> rewritten_vm =
      set_top_level_field(vm.value(), "gpu", rewritten_gpu.value(), emit_error);
  if (rewritten_vm.failed())
    return Result::failure();
  FailureOr<std::string> rewritten =
      set_top_level_field(json, "vm", rewritten_vm.value(), emit_error);
  if (rewritten.failed())
    return Result::failure();

  FailureOr<uint32_t> applied = with_parsed_simulation_config_json(
      rewritten.value(), rocjitsu::kEmbeddedSchema, emit_error,
      [](const fb::SimulationConfig *config) -> FailureOr<uint32_t> {
        return config->vm() && config->vm()->gpu() ? config->vm()->gpu()->num_gpus() : 0;
      });
  if (applied.failed())
    return Result::failure();
  if (applied.value() != count)
    return emit_error.emit() << "cannot apply gpus_per_node to the simulation config";
  return std::move(rewritten.value());
}

FailureOr<std::string> json_with_plugin_enabled(std::string_view json, std::string_view plugin,
                                                const util::DiagnosticEmitter &emit_error) {
  if (!is_valid_plugin_name(plugin))
    return emit_error.emit() << "invalid plugin name '" << plugin
                             << "'; allowed: letters, digits, '_', '-'";

  FailureOr<std::string> plugins = json_top_level_field(json, "plugins", emit_error);
  if (plugins.failed())
    return Result::failure();
  if (plugins.value().empty()) {
    std::string plugin_object = "{\"";
    plugin_object.append(plugin).append("\": {}}");
    return set_top_level_field(json, "plugins", plugin_object, emit_error);
  }
  if (!is_json_object(plugins.value()))
    return emit_error.emit() << "cannot apply plugin: simulation config plugins must be an object";

  FailureOr<std::string> existing_plugin =
      json_top_level_field(plugins.value(), plugin, emit_error);
  if (existing_plugin.failed())
    return Result::failure();
  if (!existing_plugin.value().empty())
    return std::string(json);

  FailureOr<std::string> rewritten_plugins =
      set_top_level_field(plugins.value(), plugin, "{}", emit_error);
  if (rewritten_plugins.failed())
    return Result::failure();
  return set_top_level_field(json, "plugins", rewritten_plugins.value(), emit_error);
}

Result validate_system_gpu_count_override(std::string_view json, const LaunchOverrides &overrides,
                                          const util::DiagnosticEmitter &emit_error) {
  if (!overrides.gpus_per_node)
    return Result::success();
  auto graph =
      load_system_graph_from_string(std::string(json), rocjitsu::kEmbeddedSchema, emit_error);
  if (graph.failed())
    return Result::failure();
  if (!graph.value().explicit_manifest)
    return Result::success();
  auto gpus = graph.value().local_gpus(emit_error, overrides.local_node);
  if (gpus.failed())
    return Result::failure();
  if (gpus.value().size() != *overrides.gpus_per_node)
    return emit_error.emit() << "cannot apply gpus_per_node: selected system node has "
                             << gpus.value().size() << " GPUs";
  return Result::success();
}

} // namespace

FailureOr<std::string> json_with_cpu_thread_budget(std::string_view json, uint32_t budget,
                                                   const util::DiagnosticEmitter &emit_error) {
  return json_with_top_level_number(
      json, kBudgetField, budget,
      [](const fb::SimulationConfig *config) -> FailureOr<uint32_t> {
        return config->cpu_thread_budget();
      },
      emit_error);
}

FailureOr<std::string> json_with_launch_overrides(std::string_view json,
                                                  const LaunchOverrides &overrides,
                                                  const util::DiagnosticEmitter &emit_error) {
  std::string rewritten(json);
  if (overrides.num_threads) {
    auto result = json_with_top_level_number(
        rewritten, "num_threads", *overrides.num_threads,
        [](const fb::SimulationConfig *config) -> FailureOr<uint32_t> {
          return config->num_threads();
        },
        emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
  if (overrides.cpu_dispatch_threads) {
    auto result = json_with_top_level_number(
        rewritten, "cpu_dispatch_threads", *overrides.cpu_dispatch_threads,
        [](const fb::SimulationConfig *config) -> FailureOr<uint32_t> {
          return config->cpu_dispatch_threads();
        },
        emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
  if (overrides.async_helper_threads) {
    auto result = json_with_top_level_number(
        rewritten, "async_helper_threads", *overrides.async_helper_threads,
        [](const fb::SimulationConfig *config) -> FailureOr<int32_t> {
          return config->async_helper_threads();
        },
        emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
  }
  if (overrides.exec_mode) {
    auto result =
        set_top_level_field(rewritten, "exec_mode", json_string(*overrides.exec_mode), emit_error);
    if (result.failed())
      return Result::failure();
    rewritten = std::move(result.value());
    auto applied = with_parsed_simulation_config_json(
        rewritten, rocjitsu::kEmbeddedSchema, emit_error,
        [](const fb::SimulationConfig *config) -> FailureOr<std::string> {
          return config->exec_mode() ? config->exec_mode()->str() : std::string{};
        });
    if (applied.failed())
      return Result::failure();
    if (applied.value() != *overrides.exec_mode)
      return emit_error.emit() << "cannot apply exec_mode to the simulation config";
  }
  if (overrides.gpus_per_node) {
    if (validate_system_gpu_count_override(rewritten, overrides, emit_error).failed())
      return Result::failure();
    auto graph = load_system_graph_from_string(rewritten, rocjitsu::kEmbeddedSchema, emit_error);
    if (graph.failed())
      return Result::failure();
    if (!graph.value().explicit_manifest) {
      auto result = json_with_gpu_count(rewritten, *overrides.gpus_per_node, emit_error);
      if (result.failed())
        return Result::failure();
      rewritten = std::move(result.value());
    }
  }
  if (!overrides.plugins.empty()) {
    for (const std::string &plugin : overrides.plugins) {
      auto result = json_with_plugin_enabled(rewritten, plugin, emit_error);
      if (result.failed())
        return Result::failure();
      rewritten = std::move(result.value());
    }
    auto required = merged_required_plugins(rewritten, overrides.plugins, emit_error);
    if (required.failed())
      return Result::failure();
    auto result = set_top_level_field(rewritten, "required_plugins",
                                      json_string_array(required.value()), emit_error);
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
    auto node = graph.value().select_node(emit_error, node_id);
    if (node.failed())
      return Result::failure();
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
    // Graph construction and JSON parsing still allocate and may throw.
    // Keep their existing launch-override containment until those are migrated.
    return emit_error.emit() << error.what();
  }
  return std::move(rewritten.value());
}

FailureOr<std::string> write_effective_config(const std::string &source_path,
                                              const LaunchOverrides &overrides,
                                              const std::filesystem::path &invocation_runtime_dir,
                                              const util::DiagnosticEmitter &emit_error) {
  FailureOr<std::string> source_json = read_config_file(source_path, emit_error);
  if (source_json.failed())
    return Result::failure();
  FailureOr<std::string> json =
      json_with_launch_overrides(source_json.value(), overrides, emit_error);
  if (json.failed())
    return Result::failure();
  std::string effective_json = std::move(json.value());

  const std::filesystem::path directory(invocation_runtime_dir);
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
  return write_effective_config(source_path, overrides, rocjitsu::rpc_invocation_runtime_dir(pid),
                                emit_error);
}

} // namespace config
} // namespace rocjitsu
