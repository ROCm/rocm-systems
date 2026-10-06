// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_shutdown_exit_test.cpp
/// @brief The launcher delivers on_shutdown when a KFD descriptor is still open.

#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

std::string require_env(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    throw std::runtime_error(std::string(name) + " is not set");
  return value;
}

void skip_without_launcher_env() {
  for (const char *name : {"RJ_SHUTDOWN_CLI", "RJ_SHUTDOWN_BASE_CONFIG", "RJ_SHUTDOWN_FAKE_BACKEND",
                           "RJ_SHUTDOWN_CHILD"}) {
    const char *value = std::getenv(name);
    if (!value || !*value)
      GTEST_SKIP() << name << " is not set";
  }
}

std::string read_file(const std::string &path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string json_escape(const std::string &value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (ch == '\\' || ch == '"')
      out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

std::string config_with_fake_backend(const std::string &base_path, const std::string &backend) {
  std::string json = read_file(base_path);
  while (!json.empty() &&
         (json.back() == '\n' || json.back() == '\r' || json.back() == ' ' || json.back() == '\t'))
    json.pop_back();
  if (json.empty() || json.back() != '}')
    throw std::runtime_error("base config does not end with an object");
  json.pop_back();
  json += ",\n  \"require_all_plugins\": true,\n  \"plugins\": {\n    \"perfsim\": {\n";
  json += "      \"library_path\": \"" + json_escape(backend) + "\"\n    }\n  }\n}\n";
  return json;
}

std::vector<std::string> lines_of(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (!line.empty())
      lines.push_back(line);
  }
  return lines;
}

int count_exact(const std::vector<std::string> &lines, const std::string &needle) {
  int count = 0;
  for (const auto &line : lines)
    if (line == needle)
      ++count;
  return count;
}

struct RunResult {
  int status = -1;
  std::string trace;
};

RunResult run_under_launcher(bool kill) {
  const std::string cli = require_env("RJ_SHUTDOWN_CLI");
  const std::string base_config = require_env("RJ_SHUTDOWN_BASE_CONFIG");
  const std::string backend = require_env("RJ_SHUTDOWN_FAKE_BACKEND");
  const std::string child = require_env("RJ_SHUTDOWN_CHILD");

  rocjitsu::test::ScopedTempFile config_file("plugin-shutdown-config-");
  rocjitsu::test::ScopedTempFile trace_file("plugin-shutdown-trace-");
  config_file.write(config_with_fake_backend(base_config, backend));

  const pid_t pid = fork();
  if (pid < 0)
    throw std::runtime_error("fork failed");
  if (pid == 0) {
    setenv("ROCJITSU_PERFSIM_FAKE_TRACE", trace_file.path().c_str(), 1);
    unsetenv("ROCJITSU_PERFSIM_FAKE_MODE");
    if (kill)
      execl(cli.c_str(), cli.c_str(), "--config", config_file.path().c_str(), "--", child.c_str(),
            "--kill", static_cast<char *>(nullptr));
    else
      execl(cli.c_str(), cli.c_str(), "--config", config_file.path().c_str(), "--", child.c_str(),
            static_cast<char *>(nullptr));
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
    throw std::runtime_error("waitpid failed");
  return {status, read_file(trace_file.path())};
}

} // namespace

TEST(InterposerPluginShutdown, DeliversOnShutdownWhenKfdRefRemains) {
  skip_without_launcher_env();
  if (::testing::Test::IsSkipped())
    return;
  const RunResult run = run_under_launcher(false);
  ASSERT_TRUE(WIFEXITED(run.status)) << run.status;
  ASSERT_EQ(WEXITSTATUS(run.status), 0) << run.trace;

  const std::vector<std::string> lines = lines_of(run.trace);
  ASSERT_GE(lines.size(), 2u) << run.trace;
  EXPECT_EQ(lines[lines.size() - 2], "shutdown") << run.trace;
  EXPECT_EQ(lines.back(), "unload") << run.trace;
  EXPECT_EQ(count_exact(lines, "shutdown"), 1) << run.trace;
}

TEST(InterposerPluginShutdown, KilledProcessDoesNotDeliverOnShutdown) {
  skip_without_launcher_env();
  if (::testing::Test::IsSkipped())
    return;
  const RunResult run = run_under_launcher(true);
  ASSERT_TRUE(WIFSIGNALED(run.status)) << run.status;
  EXPECT_EQ(WTERMSIG(run.status), SIGKILL);
  const std::vector<std::string> lines = lines_of(run.trace);
  EXPECT_EQ(count_exact(lines, "shutdown"), 0) << run.trace;
  bool saw_init = false;
  for (const auto &line : lines) {
    if (line.rfind("init ", 0) == 0)
      saw_init = true;
  }
  EXPECT_TRUE(saw_init) << run.trace;
}
