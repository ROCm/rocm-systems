// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Every other test here is single-threaded, so a sanitizer run over the suite
// cannot see a data race. This one drives the public API from several threads;
// its real value is under -fsanitize=thread (with setarch -R to disable ASLR).

#include "unit/concurrency_test.h"

#include <gtest/gtest.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

#include "include/amd_cuid.h"

namespace {

constexpr int kThreads = 8;
constexpr int kIterations = 25;

// Every status the workers are allowed to see. Anything else means the API
// returned something undefined under concurrency.
bool acceptable(amdcuid_status_t status) {
  switch (status) {
    case AMDCUID_STATUS_SUCCESS:
    case AMDCUID_STATUS_DEVICE_NOT_FOUND:
    case AMDCUID_STATUS_UNSUPPORTED:
    case AMDCUID_STATUS_INSUFFICIENT_SIZE:
    case AMDCUID_STATUS_PERMISSION_DENIED:
    case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND:
    case AMDCUID_STATUS_FILE_NOT_FOUND:
      return true;
    default:
      return false;
  }
}

}  // namespace

TestConcurrentApi::TestConcurrentApi() {
  SetTitle("Concurrent Public API Use");
  SetDescription(
      "Drive amdcuid_get_all_handles, amdcuid_query_device_property and "
      "amdcuid_refresh from several threads at once. Guards the "
      "CuidDeviceManager locking; run under ThreadSanitizer to detect races.");
}

void TestConcurrentApi::SetUp() {}

void TestConcurrentApi::Run() {
  std::atomic<int> bad_status{0};
  std::atomic<int> observed{0};

  auto worker = [&](int id) {
    for (int i = 0; i < kIterations; ++i) {
      uint32_t count = 0;
      amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
      if (!acceptable(status)) {
        ++bad_status;
        continue;
      }

      if (status == AMDCUID_STATUS_SUCCESS && count > 0) {
        std::vector<amdcuid_id_t> handles(count);
        uint32_t fetched = count;
        if (amdcuid_get_all_handles(handles.data(), &fetched) == AMDCUID_STATUS_SUCCESS) {
          ++observed;
          const uint32_t examine = (fetched < 4) ? fetched : 4;
          for (uint32_t h = 0; h < examine; ++h) {
            amdcuid_id_t derived{};
            uint32_t len = sizeof(derived);
            if (!acceptable(amdcuid_query_device_property(handles[h], AMDCUID_QUERY_DERIVED_CUID,
                                                          &derived, &len))) {
              ++bad_status;
            }
            // Size-query form: data == nullptr must still return a defined
            // status rather than an uninitialized one.
            uint32_t path_len = 0;
            if (!acceptable(amdcuid_query_device_property(handles[h], AMDCUID_QUERY_DEVICE_PATH,
                                                          nullptr, &path_len))) {
              ++bad_status;
            }
          }
        }
      }

      // Force a rediscovery underneath the readers. This is what makes the
      // device list get replaced while other threads are walking it.
      if ((i % 5) == (id % 5)) {
        if (!acceptable(amdcuid_refresh())) {
          ++bad_status;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back(worker, i);
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(bad_status.load(), 0) << "public API returned an unexpected status under concurrency";

  IF_VERB(1) {
    printf("  %d threads x %d iterations, %d successful enumerations\n", kThreads, kIterations,
           observed.load());
  }
}

void TestConcurrentApi::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestConcurrentApi::DisplayResults() const { TestBase::DisplayResults(); }
void TestConcurrentApi::Close() {}

// A child forked while another thread is inside the library, as Python's
// multiprocessing does, must be able to use it.
TEST(cuidtstUnprivileged, ForkedChildCanUseTheLibrary) {
  constexpr int kChildren = 20;
  std::atomic<bool> stop{false};
  std::thread busy([&] {
    while (!stop) {
      uint32_t count = 0;
      amdcuid_get_all_handles(nullptr, &count);
      amdcuid_refresh();
    }
  });
  int stuck = 0;
  for (int i = 0; i < kChildren; ++i) {
    const pid_t child = fork();
    if (child == 0) {
      alarm(10);
      uint32_t count = 0;
      amdcuid_get_all_handles(nullptr, &count);
      _exit(0);
    }
    int status = 0;
    if (child < 0 || waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0)
      ++stuck;
  }
  stop = true;
  busy.join();
  EXPECT_EQ(stuck, 0) << "of " << kChildren << " children";
}

namespace {

// Two threads fork at once, the first holding the library's lock and the
// second not. Handlers registered before the library's run their prepare
// after it and their parent before it, so this one holds the first fork
// between the library's prepare and parent handlers until the second fork
// has run its prepare.
std::atomic<bool> overlap_armed{false};
std::atomic<bool> first_in_fork{false};
std::atomic<bool> second_prepared{false};
thread_local int overlap_role = 0;

void overlap_prepare() {
  if (!overlap_armed) return;
  if (overlap_role == 2) second_prepared = true;
  if (overlap_role != 1) return;
  first_in_fork = true;
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!second_prepared && std::chrono::steady_clock::now() < end) std::this_thread::yield();
}

__attribute__((constructor(101))) void register_overlap_handler() {
  pthread_atfork(overlap_prepare, nullptr, nullptr);
}

bool child_used_library(pid_t child) {
  int status = 0;
  return child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
         WEXITSTATUS(status) == 0;
}

pid_t fork_and_use_library() {
  const pid_t child = fork();
  if (child == 0) {
    alarm(10);
    uint32_t count = 0;
    amdcuid_get_all_handles(nullptr, &count);
    _exit(0);
  }
  return child;
}

// Runs in a process of its own, since a parent left locked would hang every
// later test: 0 when both children and then the parent used the library.
int overlapping_forks() {
  uint32_t count = 0;
  amdcuid_get_all_handles(nullptr, &count);
  overlap_armed = true;
  std::atomic<bool> first_ok{false}, second_ok{false};
  std::thread first([&] {
    overlap_role = 1;
    first_ok = child_used_library(fork_and_use_library());
  });
  std::thread second([&] {
    overlap_role = 2;
    while (!first_in_fork) std::this_thread::yield();
    second_ok = child_used_library(fork_and_use_library());
  });
  first.join();
  second.join();
  overlap_armed = false;
  if (!first_ok || !second_ok) return 2;
  auto parent = std::async(std::launch::async, [] {
    uint32_t n = 0;
    return amdcuid_get_all_handles(nullptr, &n);
  });
  if (parent.wait_for(std::chrono::seconds(10)) != std::future_status::ready) _exit(3);
  return 0;
}

}  // namespace

TEST(cuidtstUnprivileged, OverlappingForksLeaveTheParentUsable) {
  const pid_t scenario = fork();
  if (scenario == 0) _exit(overlapping_forks());
  ASSERT_GT(scenario, 0);
  int status = 0;
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  pid_t done = 0;
  while ((done = waitpid(scenario, &status, WNOHANG)) == 0 &&
         std::chrono::steady_clock::now() < end)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (done == 0) {
    kill(scenario, SIGKILL);
    waitpid(scenario, &status, 0);
  }
  ASSERT_TRUE(WIFEXITED(status)) << "scenario killed";
  EXPECT_NE(WEXITSTATUS(status), 2) << "a child could not use the library";
  EXPECT_NE(WEXITSTATUS(status), 3) << "the parent was left with the library locked";
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
