// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/effective_config.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/json_fields.h"
#include "rocjitsu/config/system_graph.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "simulation_config_generated.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
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

} // namespace

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
