// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Functional tests for the NCCL_ENV_PLUGIN environment plugin feature.
// Validates that RCCL correctly loads external env plugins, falls back
// to the internal default, and routes ncclGetEnv() through the active plugin.

#include "env.h"
#include "param.h"
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "common/ProcessIsolatedTestRunner.hpp"

#include <cstdio>
#include <filesystem>
#include <string>

namespace RcclUnitTesting {

// Returns the absolute path to a plugin .so located next to the test binary
// in the standard output subdirectory:  <binary_dir>/unit/plugins/<soName>
// Returns an empty string if /proc/self/exe cannot be resolved.
static std::string getTestPluginPath(const char* soName) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (ec) return "";
  return (exe.parent_path() / "unit" / "plugins" / soName).string();
}

TEST(EnvPluginTests, InternalPlugin_FallbackOnNone) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "InternalPlugin_FallbackOnNone",
      []() {
        setenv("RCCL_TEST_ENV_VAR", "hello_from_env", 1);
        initEnv();
        const char* val = ncclGetEnv("RCCL_TEST_ENV_VAR");
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "hello_from_env");
        unsetenv("RCCL_TEST_ENV_VAR");
      },
      {{"NCCL_ENV_PLUGIN", "none"}});
}

TEST(EnvPluginTests, InternalPlugin_ReturnsNullForUnset) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "InternalPlugin_ReturnsNullForUnset",
      []() {
        unsetenv("RCCL_TEST_NONEXISTENT_12345");
        initEnv();
        const char* val = ncclGetEnv("RCCL_TEST_NONEXISTENT_12345");
        ASSERT_EQ(val, nullptr);
      },
      {{"NCCL_ENV_PLUGIN", "none"}});
}

TEST(EnvPluginTests, InternalPlugin_GetEnvUsedByParamMacro) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "InternalPlugin_GetEnvUsedByParamMacro",
      []() {
        setenv("NCCL_DEBUG", "INFO", 1);
        initEnv();
        const char* val = ncclGetEnv("NCCL_DEBUG");
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "INFO");
        unsetenv("NCCL_DEBUG");
      },
      {{"NCCL_ENV_PLUGIN", "none"}});
}

TEST(EnvPluginTests, GracefulFallback_NonexistentPlugin) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GracefulFallback_NonexistentPlugin",
      []() {
        setenv("RCCL_TEST_FALLBACK", "works", 1);
        initEnv();
        const char* val = ncclGetEnv("RCCL_TEST_FALLBACK");
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "works");
        unsetenv("RCCL_TEST_FALLBACK");
      },
      {{"NCCL_ENV_PLUGIN", "/nonexistent/path/libfake-env.so"}});
}

TEST(EnvPluginTests, PluginInitialized_ReportsTrue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "PluginInitialized_ReportsTrue",
      []() {
        ASSERT_FALSE(ncclEnvPluginInitialized());
        initEnv();
        (void)ncclGetEnv("NCCL_DEBUG");
        ASSERT_TRUE(ncclEnvPluginInitialized());
      },
      {{"NCCL_ENV_PLUGIN", "none"}});
}

TEST(EnvPluginTests, ConfFile_LoadedBeforePlugin) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "ConfFile_LoadedBeforePlugin",
      []() {
        initEnv();
        const char* val = ncclGetEnv("RCCL_CONF_TEST_KEY");
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "conf_value");
      },
      {{"NCCL_ENV_PLUGIN", "none"},
       {"NCCL_CONF_FILE", "EnvPluginTestsConfFile.txt"}});
}

// ---------------------------------------------------------------------------
// External-plugin tests (Goal 3): exercise the full dlopen path
//
// Each test:
//   1. Resolves the plugin .so path from /proc/self/exe at runtime.
//   2. Sets NCCL_ENV_PLUGIN to the absolute path so ncclEnvPluginLoad()
//      performs a real dlopen() + dlsym(ncclEnvPlugin_v1).
//   3. Calls ncclGetEnv() to drive ncclInitEnv() -> ncclEnvPluginInit()
//      -> ncclEnvPluginLoad() and verify the returned value.
//
// Tests are skipped (not failed) when BUILD_EXT_EXAMPLES=ON was not used.
// ---------------------------------------------------------------------------

// Verifies that libnccl-env-example.so is dlopened successfully and that
// ncclGetEnv() returns process-environment values through it.
TEST(EnvPluginTests, ExternalPlugin_ExamplePlugin_Loaded) {
  RUN_ISOLATED_TEST(
      "ExternalPlugin_ExamplePlugin_Loaded",
      []() {
        std::string pluginPath = getTestPluginPath("libnccl-env-example.so");
        if (!std::filesystem::exists(pluginPath)) {
          GTEST_SKIP() << "libnccl-env-example.so not found at " << pluginPath
                       << " — rebuild with -DBUILD_EXT_EXAMPLES=ON";
        }

        setenv("NCCL_ENV_PLUGIN", pluginPath.c_str(), 1);
        setenv("RCCL_TEST_EXAMPLE_KEY", "example_via_ext_plugin", 1);

        initEnv();
        ASSERT_FALSE(ncclEnvPluginInitialized());

        const char* val = ncclGetEnv("RCCL_TEST_EXAMPLE_KEY");
        ASSERT_TRUE(ncclEnvPluginInitialized());
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "example_via_ext_plugin");

        unsetenv("RCCL_TEST_EXAMPLE_KEY");
      });
}

// Verifies that libnccl-env-json.so is dlopened and that a value present in
// the JSON config file overrides the same key set in the process environment.
// This is the strongest proof that the external plugin — not the internal
// getenv() fallback — handled the lookup.
TEST(EnvPluginTests, ExternalPlugin_JsonPlugin_OverridesGetenv) {
  RUN_ISOLATED_TEST(
      "ExternalPlugin_JsonPlugin_OverridesGetenv",
      []() {
        std::string pluginPath = getTestPluginPath("libnccl-env-json.so");
        if (!std::filesystem::exists(pluginPath)) {
          GTEST_SKIP() << "libnccl-env-json.so not found at " << pluginPath
                       << " — rebuild with -DBUILD_EXT_EXAMPLES=ON";
        }

        std::string jsonPath = std::string("/tmp/rccl_test_json_override_") +
                               std::to_string(getpid()) + ".json";
        FILE* f = fopen(jsonPath.c_str(), "w");
        ASSERT_NE(f, nullptr) << "Failed to create temp JSON file: " << jsonPath;
        fprintf(f, "{\"NCCL_TEST_JSON_KEY\": \"json_value\"}");
        fclose(f);

        // Intentionally set a different value in the process env.
        // The JSON plugin must return "json_value" (JSON wins over getenv).
        setenv("NCCL_TEST_JSON_KEY", "process_value", 1);
        setenv("NCCL_ENV_PLUGIN", pluginPath.c_str(), 1);
        setenv("NCCL_ENV_JSON_FILE", jsonPath.c_str(), 1);

        initEnv();
        const char* val = ncclGetEnv("NCCL_TEST_JSON_KEY");
        ASSERT_TRUE(ncclEnvPluginInitialized());
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "json_value");

        unsetenv("NCCL_TEST_JSON_KEY");
        remove(jsonPath.c_str());
      });
}

// Verifies that libnccl-env-json.so falls back to getenv() for a key that is
// absent from the JSON file.
TEST(EnvPluginTests, ExternalPlugin_JsonPlugin_FallsBackToGetenv) {
  RUN_ISOLATED_TEST(
      "ExternalPlugin_JsonPlugin_FallsBackToGetenv",
      []() {
        std::string pluginPath = getTestPluginPath("libnccl-env-json.so");
        if (!std::filesystem::exists(pluginPath)) {
          GTEST_SKIP() << "libnccl-env-json.so not found at " << pluginPath
                       << " — rebuild with -DBUILD_EXT_EXAMPLES=ON";
        }

        std::string jsonPath = std::string("/tmp/rccl_test_json_fallback_") +
                               std::to_string(getpid()) + ".json";
        FILE* f = fopen(jsonPath.c_str(), "w");
        ASSERT_NE(f, nullptr) << "Failed to create temp JSON file: " << jsonPath;
        // JSON has a different key; the test key must be served by getenv().
        fprintf(f, "{\"OTHER_KEY\": \"other_value\"}");
        fclose(f);

        setenv("NCCL_TEST_JSON_FALLBACK", "process_value", 1);
        setenv("NCCL_ENV_PLUGIN", pluginPath.c_str(), 1);
        setenv("NCCL_ENV_JSON_FILE", jsonPath.c_str(), 1);

        initEnv();
        const char* val = ncclGetEnv("NCCL_TEST_JSON_FALLBACK");
        ASSERT_TRUE(ncclEnvPluginInitialized());
        ASSERT_NE(val, nullptr);
        ASSERT_STREQ(val, "process_value");

        unsetenv("NCCL_TEST_JSON_FALLBACK");
        remove(jsonPath.c_str());
      });
}

// Verifies that libnccl-env-json.so returns nullptr for a key absent from
// both the JSON file and the process environment.
TEST(EnvPluginTests, ExternalPlugin_JsonPlugin_UnsetKeyReturnsNull) {
  RUN_ISOLATED_TEST(
      "ExternalPlugin_JsonPlugin_UnsetKeyReturnsNull",
      []() {
        std::string pluginPath = getTestPluginPath("libnccl-env-json.so");
        if (!std::filesystem::exists(pluginPath)) {
          GTEST_SKIP() << "libnccl-env-json.so not found at " << pluginPath
                       << " — rebuild with -DBUILD_EXT_EXAMPLES=ON";
        }

        std::string jsonPath = std::string("/tmp/rccl_test_json_null_") +
                               std::to_string(getpid()) + ".json";
        FILE* f = fopen(jsonPath.c_str(), "w");
        ASSERT_NE(f, nullptr) << "Failed to create temp JSON file: " << jsonPath;
        fprintf(f, "{\"OTHER_KEY\": \"other_value\"}");
        fclose(f);

        unsetenv("NCCL_TEST_JSON_MISSING_12345");
        setenv("NCCL_ENV_PLUGIN", pluginPath.c_str(), 1);
        setenv("NCCL_ENV_JSON_FILE", jsonPath.c_str(), 1);

        initEnv();
        const char* val = ncclGetEnv("NCCL_TEST_JSON_MISSING_12345");
        ASSERT_TRUE(ncclEnvPluginInitialized());
        ASSERT_EQ(val, nullptr);

        remove(jsonPath.c_str());
      });
}

} // namespace RcclUnitTesting
