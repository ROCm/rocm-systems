// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace rocjitsu::test {

/// Restore an expected-valid checkpoint, preserving the caller's output on failure.
/// Call with ASSERT_TRUE so a failure stops the consuming test or fixture.
[[nodiscard]] inline testing::AssertionResult
restore_valid_checkpoint(const std::string &path, config::LoadedConfig &output) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<config::LoadedConfig> result =
      config::restore_checkpoint(path, diagnostic.emitter());
  if (result.failed())
    return testing::AssertionFailure() << diagnostic.message();
  output = std::move(result).value();
  return testing::AssertionSuccess();
}

/// Load an expected-valid config, preserving the caller's output on failure.
/// Call with ASSERT_TRUE so a failure stops the consuming test or fixture.
/// Omitting host_threads selects the loader's automatic host-affinity policy.
[[nodiscard]] inline testing::AssertionResult
load_valid_config(const std::string &path, config::LoadedConfig &output,
                  std::optional<uint32_t> host_threads = std::nullopt) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<config::LoadedConfig> result =
      host_threads ? config::load_config(path, kEmbeddedSchema, diagnostic.emitter(), *host_threads)
                   : config::load_config(path, kEmbeddedSchema, diagnostic.emitter());
  if (result.failed())
    return testing::AssertionFailure() << diagnostic.message();
  output = std::move(result).value();
  return testing::AssertionSuccess();
}

/// Select a node when a shipped system config describes more than one host.
[[nodiscard]] inline testing::AssertionResult
load_valid_config_for_node(const std::string &path, config::LoadedConfig &output,
                           std::string_view node_id,
                           std::optional<uint32_t> host_threads = std::nullopt) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<config::LoadedConfig> result =
      host_threads
          ? config::load_config_for_node(path, kEmbeddedSchema, node_id, diagnostic.emitter(),
                                         *host_threads)
          : config::load_config_for_node(path, kEmbeddedSchema, node_id, diagnostic.emitter());
  if (result.failed())
    return testing::AssertionFailure() << diagnostic.message();
  output = std::move(result).value();
  return testing::AssertionSuccess();
}

/// String variant with the same output and host-width semantics as the file helper.
[[nodiscard]] inline testing::AssertionResult
load_valid_config_from_string(const std::string &json, config::LoadedConfig &output,
                              std::optional<uint32_t> host_threads = std::nullopt) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<config::LoadedConfig> result =
      host_threads ? config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter(),
                                                     *host_threads)
                   : config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter());
  if (result.failed())
    return testing::AssertionFailure() << diagnostic.message();
  output = std::move(result).value();
  return testing::AssertionSuccess();
}

} // namespace rocjitsu::test
