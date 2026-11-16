// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_loader.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#ifndef PLUGIN_LOADER_FIXTURE_DIR
#error "PLUGIN_LOADER_FIXTURE_DIR must be defined"
#endif

namespace {

class PluginLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(trace_.initialize("rocjitsu-plugin-loader-"));
    trace_environment_.emplace("ROCJITSU_PLUGIN_TEST_TRACE", trace_.path());
    ASSERT_STREQ(std::getenv("ROCJITSU_PLUGIN_TEST_TRACE"), trace_.path().c_str());
  }

  std::string trace() const {
    std::ifstream input(trace_.path());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  int load(const char *name, rocjitsu::ExecutionPluginGroup &group) const {
    const std::string config = std::string{"{\"plugins\":{\""} + name + "\":{}}}";
    return rocjitsu::PluginLoader::load_from_config(config, group, PLUGIN_LOADER_FIXTURE_DIR);
  }

  rocjitsu::test::ScopedTempFile trace_;
  std::optional<rocjitsu::test::ScopedEnvironmentVariable> trace_environment_;
};

TEST_F(PluginLoaderTest, LoadsPluginAndDispatchesLifecycle) {
  rocjitsu::ExecutionPluginGroup group(rocjitsu::PluginSinkConfig{});
  EXPECT_EQ(load("good", group), 1);
  EXPECT_EQ(group.num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
  group.onInit();
  EXPECT_NE(trace().find("good:init\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, WarnsAndLoadsWhenRemovedProfiledConfigIsPresent) {
  testing::internal::CaptureStderr();
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(R"({"profiled":false,"plugins":{"good":{}}})",
                                                     PLUGIN_LOADER_FIXTURE_DIR);
  const std::string error = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(group.succeeded());
  ASSERT_EQ(group.value()->num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
  EXPECT_NE(error.find("hook profiling was removed; ignoring top-level 'profiled' config"),
            std::string::npos);
}

TEST_F(PluginLoaderTest, RejectsMissingRequiredExport) {
  rocjitsu::ExecutionPluginGroup group(rocjitsu::PluginSinkConfig{});
  EXPECT_EQ(load("missing", group), 0);
  EXPECT_TRUE(group.empty());
  EXPECT_EQ(trace().find("missing:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, PluginFailuresRemainBestEffortByDefault) {
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(R"({"plugins":{"missing":{}}})",
                                                     PLUGIN_LOADER_FIXTURE_DIR);

  ASSERT_TRUE(group.succeeded());
  EXPECT_TRUE(group.value()->empty());
}

TEST_F(PluginLoaderTest, RequiredMissingPluginReturnsFailure) {
  util::StringDiagnostic diagnostic;
  const util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(
          R"({"require_all_plugins":true,"plugins":{"missing":{}}})", PLUGIN_LOADER_FIXTURE_DIR,
          diagnostic.emitter());
  EXPECT_TRUE(group.failed());
  EXPECT_EQ(diagnostic.message(),
            "required plugin loading failed: loaded 0 of 1 configured plugins");
}

TEST_F(PluginLoaderTest, RequiredPluginLoads) {
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(
          R"({"require_all_plugins":true,"plugins":{"good":{}}})", PLUGIN_LOADER_FIXTURE_DIR);

  ASSERT_TRUE(group.succeeded());
  ASSERT_EQ(group.value()->num_plugins(), 1u);
  EXPECT_NE(trace().find("good:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, RequiringAllPluginsAllowsNoPlugins) {
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(R"({"require_all_plugins":true})",
                                                     PLUGIN_LOADER_FIXTURE_DIR);

  ASSERT_TRUE(group.succeeded());
  EXPECT_TRUE(group.value()->empty());
}

TEST_F(PluginLoaderTest, EmptyInputConfiguresEmptyGroup) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group("", PLUGIN_LOADER_FIXTURE_DIR,
                                                     diagnostic.emitter());
  ASSERT_TRUE(group.succeeded());
  EXPECT_TRUE(group.value()->empty());
  EXPECT_TRUE(diagnostic.message().empty());
}

TEST_F(PluginLoaderTest, RequiringAllPluginsRejectsNonObjectPlugins) {
  for (const char *plugins : {"[]", "null", R"("good")"}) {
    SCOPED_TRACE(plugins);
    const std::string config =
        std::string{"{\"require_all_plugins\":true,\"plugins\":"} + plugins + "}";

    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(rocjitsu::PluginLoader::configure_plugin_group(config, PLUGIN_LOADER_FIXTURE_DIR,
                                                               diagnostic.emitter())
                    .failed());
    EXPECT_EQ(diagnostic.message(),
              "top-level 'plugins' must be an object when 'require_all_plugins' is true");
  }
}

TEST_F(PluginLoaderTest, BestEffortLoadingIgnoresNonObjectPlugins) {
  for (const char *plugins : {"[]", "null", R"("good")"}) {
    SCOPED_TRACE(plugins);
    const std::string config = std::string{"{\"plugins\":"} + plugins + "}";

    util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
        rocjitsu::PluginLoader::configure_plugin_group(config, PLUGIN_LOADER_FIXTURE_DIR);
    ASSERT_TRUE(group.succeeded());
    EXPECT_TRUE(group.value()->empty());
  }
}

TEST_F(PluginLoaderTest, RejectsNonBooleanRequireAllPlugins) {
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(rocjitsu::PluginLoader::configure_plugin_group(
                  R"({"require_all_plugins":"true","plugins":{"good":{}}})",
                  PLUGIN_LOADER_FIXTURE_DIR, diagnostic.emitter())
                  .failed());
  EXPECT_EQ(diagnostic.message(), "top-level 'require_all_plugins' must be a boolean");
  EXPECT_EQ(trace().find("good:create\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, ReportsParserErrorDetails) {
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(rocjitsu::PluginLoader::configure_plugin_group(R"({"execution_backend":hardware})",
                                                             PLUGIN_LOADER_FIXTURE_DIR,
                                                             diagnostic.emitter())
                  .failed());
  EXPECT_NE(diagnostic.message().find("invalid plugin configuration JSON:"), std::string::npos);
  EXPECT_NE(diagnostic.message().find("1:"), std::string::npos);
  EXPECT_NE(diagnostic.message().find("hardware"), std::string::npos);
}

TEST_F(PluginLoaderTest, AllowsTrailingComments) {
  for (const char *suffix : {"// note", "/* note */", "/* note */ // another note\n"}) {
    SCOPED_TRACE(suffix);
    const std::string config =
        std::string(R"({"require_all_plugins":true,"plugins":{"good":{}}})") + suffix;
    util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
        rocjitsu::PluginLoader::configure_plugin_group(config, PLUGIN_LOADER_FIXTURE_DIR);
    ASSERT_TRUE(group.succeeded());
    EXPECT_EQ(group.value()->num_plugins(), 1u);
  }
}

TEST_F(PluginLoaderTest, CApiRejectsAdmissionBeforeReplacingInstalledGroup) {
  const auto count = [](std::string_view text, std::string_view event) {
    size_t occurrences = 0;
    for (size_t offset = 0; (offset = text.find(event, offset)) != std::string_view::npos;
         offset += event.size())
      ++occurrences;
    return occurrences;
  };
  const char *valid = R"({"require_all_plugins":true,"plugins":{"good":{}}})";
  struct RejectionCase {
    const char *config;
    const char *reason;
    bool partial_group;
  };
  const RejectionCase cases[] = {
      {"{} true", "invalid plugin configuration JSON", false},
      {"[]", "plugin configuration must be an object", false},
      {"42", "plugin configuration must be an object", false},
      {R"({"require_all_plugins":true,"plugins":{"good":{}})", "invalid plugin configuration JSON",
       false},
      {R"({"require_all_plugins":true,"plugins":{"good":{},"missing":{}}})",
       "required plugin loading failed: loaded 1 of 2 configured plugins", true},
      {R"({"require_all_plugins":"true","plugins":{"good":{}}})",
       "top-level 'require_all_plugins' must be a boolean", false},
      {R"({"require_all_plugins":true,"plugins":[]})",
       "top-level 'plugins' must be an object when 'require_all_plugins' is true", false},
  };
  for (bool initially_active : {false, true}) {
    SCOPED_TRACE(initially_active);
    rocjitsu::SoC soc("plugin_loader_test");
    soc.set_plugin_group(nullptr);
    std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(new rj_vm_t, rj_vm_destroy);
    vm->soc = &soc;
    if (initially_active) {
      ASSERT_EQ(rj_vm_load_plugins(vm.get(), valid, PLUGIN_LOADER_FIXTURE_DIR),
                ROCJITSU_STATUS_SUCCESS);
    }
    rocjitsu::ExecutionPluginGroup *installed = &soc.plugin_group();
    for (const RejectionCase &test : cases) {
      SCOPED_TRACE(test.config);
      const std::string before = trace();
      testing::internal::CaptureStderr();
      const rj_status_t status =
          rj_vm_load_plugins(vm.get(), test.config, PLUGIN_LOADER_FIXTURE_DIR);
      const std::string diagnostic = testing::internal::GetCapturedStderr();
      EXPECT_EQ(status, ROCJITSU_STATUS_INVALID_ARGUMENT);
      const std::string warning = std::string("rj_vm_load_plugins failed: ") + test.reason;
      EXPECT_EQ(count(diagnostic, "rj_vm_load_plugins failed:"), 1u) << diagnostic;
      EXPECT_NE(diagnostic.find(warning), std::string::npos) << diagnostic;
      EXPECT_EQ(&soc.plugin_group(), installed);
      EXPECT_EQ(vm->plugin_group_active.load(), initially_active);
      const std::string after = trace();
      EXPECT_EQ(count(after, "good:init\n"), count(before, "good:init\n"));
      EXPECT_EQ(count(after, "good:shutdown\n"), count(before, "good:shutdown\n"));
      EXPECT_EQ(count(after, "good:create\n"),
                count(before, "good:create\n") + (test.partial_group ? 1u : 0u));
      EXPECT_EQ(count(after, "good:destroy\n"),
                count(before, "good:destroy\n") + (test.partial_group ? 1u : 0u));
    }
    const std::string before_retry = trace();
    const std::string valid_with_whitespace = std::string(valid) + " \t\r\n";
    ASSERT_EQ(
        rj_vm_load_plugins(vm.get(), valid_with_whitespace.c_str(), PLUGIN_LOADER_FIXTURE_DIR),
        ROCJITSU_STATUS_SUCCESS);
    EXPECT_TRUE(vm->plugin_group_active.load());
    const std::string after_retry = trace();
    EXPECT_EQ(count(after_retry, "good:init\n"), count(before_retry, "good:init\n") + 1);
    EXPECT_EQ(count(after_retry, "good:shutdown\n"),
              count(before_retry, "good:shutdown\n") + (initially_active ? 1u : 0u));
    EXPECT_EQ(count(after_retry, "good:destroy\n"),
              count(before_retry, "good:destroy\n") + (initially_active ? 1u : 0u));
  }
}

TEST_F(PluginLoaderTest, AdmissionRejectionPreservesActiveFileSink) {
  rocjitsu::test::ScopedTempDirectory output;
  ASSERT_TRUE(output.initialize("rocjitsu-plugin-sinks-"));
  rocjitsu::SoC soc("plugin_loader_test");
  soc.set_plugin_group(nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(new rj_vm_t, rj_vm_destroy);
  vm->soc = &soc;
  const std::string sinks = R"(,"sinks":{"types":["file"],"dir":")" + output.path() + R"("}})";
  const std::string valid = R"({"require_all_plugins":true,"plugins":{"good":{}})" + sinks;
  ASSERT_EQ(setenv("ROCJITSU_PLUGIN_TEST_LONG_SHUTDOWN", "1", 1), 0);
  const rj_status_t initial_status =
      rj_vm_load_plugins(vm.get(), valid.c_str(), PLUGIN_LOADER_FIXTURE_DIR);
  unsetenv("ROCJITSU_PLUGIN_TEST_LONG_SHUTDOWN");
  ASSERT_EQ(initial_status, ROCJITSU_STATUS_SUCCESS);
  const auto contents = [&] {
    std::ifstream input(output.path() + "/boundary.log");
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  };
  const std::string before = contents();
  ASSERT_EQ(before, "boundary:init\n");
  const std::string rejected =
      R"({"require_all_plugins":true,"plugins":{"good":{},"missing":{}})" + sinks;
  EXPECT_EQ(rj_vm_load_plugins(vm.get(), rejected.c_str(), PLUGIN_LOADER_FIXTURE_DIR),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(contents(), before);
  EXPECT_TRUE(vm->plugin_group_active.load());
  ASSERT_EQ(rj_vm_load_plugins(vm.get(), valid.c_str(), PLUGIN_LOADER_FIXTURE_DIR),
            ROCJITSU_STATUS_SUCCESS);
  vm.reset();
  soc.set_plugin_group(nullptr);
  EXPECT_EQ(contents(), "boundary:init\nboundary:shutdown\nboundary:destroy\n");
}

TEST_F(PluginLoaderTest, RejectedFileOnlyCandidateDiscardsSimulationSummary) {
  rocjitsu::test::ScopedTempDirectory output;
  ASSERT_TRUE(output.initialize("rocjitsu-plugin-sinks-"));
  const std::string plugin_dir = PLUGIN_LOADER_RACE_DIR;
  rocjitsu::ExecutionPluginGroup probe(rocjitsu::PluginSinkConfig{});
  ASSERT_EQ(
      rocjitsu::PluginLoader::load_from_config(R"({"plugins":{"race":{}}})", probe, plugin_dir), 1);
  const std::string config = R"({"require_all_plugins":true,"plugins":{"race":{},"missing":{}},)"
                             R"("sinks":{"types":["file"],"dir":")" +
                             output.path() + R"("}})";
  testing::internal::CaptureStderr();
  const bool rejected = rocjitsu::PluginLoader::configure_plugin_group(config, plugin_dir).failed();
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(rejected);
  EXPECT_NE(diagnostic.find("missing"), std::string::npos);
  EXPECT_EQ(diagnostic.find("No races detected."), std::string::npos);
}

TEST_F(PluginLoaderTest, CApiEmptyInputRemovesInstalledPlugins) {
  rocjitsu::SoC soc("plugin_loader_test");
  soc.set_plugin_group(nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(new rj_vm_t, rj_vm_destroy);
  vm->soc = &soc;
  ASSERT_EQ(rj_vm_load_plugins(vm.get(), R"({"plugins":{"good":{}}})", PLUGIN_LOADER_FIXTURE_DIR),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(rj_vm_load_plugins(vm.get(), "", PLUGIN_LOADER_FIXTURE_DIR), ROCJITSU_STATUS_SUCCESS);
  EXPECT_TRUE(soc.plugin_group().empty());
  EXPECT_NE(trace().find("good:shutdown\n"), std::string::npos);
  EXPECT_NE(trace().find("good:destroy\n"), std::string::npos);
}

TEST_F(PluginLoaderTest, DestroysRejectedDuplicateBeforeUnload) {
  rocjitsu::ExecutionPluginGroup group(rocjitsu::PluginSinkConfig{});
  ASSERT_EQ(load("good", group), 1);
  ASSERT_EQ(load("duplicate", group), 0);

  const std::string events = trace();
  const size_t created = events.find("duplicate:create\n");
  const size_t destroyed = events.find("duplicate:destroy\n");
  const size_t unloaded = events.find("duplicate:unload\n");
  ASSERT_NE(created, std::string::npos) << events;
  ASSERT_NE(destroyed, std::string::npos) << events;
  ASSERT_NE(unloaded, std::string::npos) << events;
  EXPECT_LT(created, destroyed) << events;
  EXPECT_LT(destroyed, unloaded) << events;
  EXPECT_EQ(group.num_plugins(), 1u);
}

TEST_F(PluginLoaderTest, PluginFileSinkFailureFallsBackToStderr) {
  testing::internal::CaptureStderr();
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(
          R"({"plugins":{"good":{}},"sinks":{"types":["file"],"dir":"/dev/null"}})",
          PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_TRUE(group.succeeded());
  std::shared_ptr<rocjitsu::ExecutionPluginGroup> plugins = std::move(group).value();
  ASSERT_EQ(plugins->num_plugins(), 1u);
  plugins->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/boundary.log'"), std::string::npos);
  EXPECT_NE(error.find("boundary:init"), std::string::npos);
}

TEST_F(PluginLoaderTest, FileSinkFailureDoesNotDuplicateConfiguredStderr) {
  testing::internal::CaptureStderr();
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(
          R"({"plugins":{"good":{}},"sinks":{"types":["stderr","file"],"dir":"/dev/null"}})",
          PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_TRUE(group.succeeded());
  group.value()->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("cannot open plugin sink '/dev/null/boundary.log'"), std::string::npos);
  const size_t output = error.find("boundary:init");
  ASSERT_NE(output, std::string::npos);
  EXPECT_EQ(error.find("boundary:init", output + 1), std::string::npos);
}

TEST_F(PluginLoaderTest, FileSinkWithoutDirectoryUsesDefaultStderrSink) {
  testing::internal::CaptureStderr();
  util::FailureOr<std::shared_ptr<rocjitsu::ExecutionPluginGroup>> group =
      rocjitsu::PluginLoader::configure_plugin_group(
          R"({"plugins":{"good":{}},"sinks":{"types":["file"]}})", PLUGIN_LOADER_FIXTURE_DIR);
  ASSERT_TRUE(group.succeeded());
  group.value()->onInit();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find("sink type 'file' requested but no 'dir' set"), std::string::npos);
  EXPECT_NE(error.find("boundary:init"), std::string::npos);
}

} // namespace
