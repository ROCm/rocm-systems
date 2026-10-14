// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_CONFIG_TEST_UTIL_H_
#define ROCJITSU_TESTS_CONFIG_TEST_UTIL_H_

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace rocjitsu::test {

/// Load an expected-valid config, preserving the caller's output on failure.
/// Call with ASSERT_TRUE so a failure stops the consuming test or fixture.
/// Omitting host_threads selects the loader's automatic host-affinity policy.
[[nodiscard]] inline testing::AssertionResult
load_valid_config(const std::string &path, config::LoadedConfig &output,
                  std::optional<uint32_t> host_threads = std::nullopt) {
  util::StringDiagnostic diagnostic;
  auto result =
      host_threads ? config::load_config(path, kEmbeddedSchema, diagnostic.emitter(), *host_threads)
                   : config::load_config(path, kEmbeddedSchema, diagnostic.emitter());
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
  auto result = host_threads
                    ? config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter(),
                                                      *host_threads)
                    : config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter());
  if (result.failed())
    return testing::AssertionFailure() << diagnostic.message();
  output = std::move(result).value();
  return testing::AssertionSuccess();
}

} // namespace rocjitsu::test

#endif // ROCJITSU_TESTS_CONFIG_TEST_UTIL_H_
