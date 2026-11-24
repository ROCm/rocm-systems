// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/daemon/rj_daemon.h"
#include "scoped_temp.h"

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <pthread.h>
#include <string>

namespace {
thread_local unsigned remaining_creations = 0;
thread_local bool injected = false;
thread_local unsigned joins = 0;
} // namespace

extern "C" int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *),
                                     void *);
extern "C" int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attributes,
                                     void *(*entry)(void *), void *argument) {
  if (remaining_creations != 0 && --remaining_creations == 0) {
    injected = true;
    return EAGAIN;
  }
  return __real_pthread_create(thread, attributes, entry, argument);
}

extern "C" int __real_pthread_join(pthread_t, void **);
extern "C" int __wrap_pthread_join(pthread_t thread, void **result) {
  ++joins;
  return __real_pthread_join(thread, result);
}

namespace {

void check_startup_failure(unsigned creation) {
  rocjitsu::test::ScopedTempDirectory directory;
  ASSERT_TRUE(rocjitsu::test::initialize_runtime_directory(directory, "rj-dt-"));
  const std::filesystem::path socket = std::filesystem::path(directory.path()) / "daemon.sock";
  std::ifstream input(std::filesystem::path(CONFIG_DIR) / "gfx950_mi350x_kmd.json");
  ASSERT_TRUE(input.good());
  const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  rj_daemon_t *daemon = nullptr;
  injected = false;
  joins = 0;
  remaining_creations = creation;
  const rj_status_t status = rj_daemon_start(json.c_str(), socket.c_str(), &daemon);
  remaining_creations = 0;
  EXPECT_TRUE(injected);
  EXPECT_EQ(joins, creation - 1);
  EXPECT_EQ(status, ROCJITSU_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(daemon, nullptr);
  if (daemon)
    EXPECT_EQ(rj_daemon_stop(daemon), ROCJITSU_STATUS_SUCCESS);
  EXPECT_FALSE(std::filesystem::exists(socket));

  ASSERT_EQ(rj_daemon_start(json.c_str(), socket.c_str(), &daemon), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(daemon, nullptr);
  EXPECT_EQ(rj_daemon_status(daemon), RJ_DAEMON_STATUS_RUNNING);
  EXPECT_EQ(rj_daemon_stop(daemon), ROCJITSU_STATUS_SUCCESS);
  EXPECT_FALSE(std::filesystem::exists(socket));
}

TEST(DaemonThreadStartup, EngineCreationFailureRollsBack) { check_startup_failure(1); }

TEST(DaemonThreadStartup, AcceptCreationFailureJoinsEngineAndRollsBack) {
  check_startup_failure(2);
}

} // namespace
