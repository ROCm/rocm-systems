/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Test hipEventIpc behavior.

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <hip_test_common.hh>
#include <dirent.h>
#include <string>
#include <set>
#include <regex>
#include <sys/wait.h>
#include <unistd.h>

/**
 * Test Description
 * ------------------------
 *  - Verify that all shared memory objects for IPC events used internally by HIP are properly
 *    cleaned up after use and do not leave persistent files in /dev/shm.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventIpc_shm_cleanup.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

// Helper to get current /dev/shm files that look like HIP IPC
std::set<std::string> get_hip_ipc_shm_files() {
  std::set<std::string> result;
  DIR* dir = opendir("/dev/shm");
  if (!dir) return result;

  struct dirent* entry;
  std::regex hip_pattern("^hip_.*");  // Matches typical HIP shm names

  while ((entry = readdir(dir)) != nullptr) {
    if (std::regex_match(entry->d_name, hip_pattern)) {
      result.insert(entry->d_name);
    }
  }
  closedir(dir);
  return result;
}

HIP_TEST_CASE(Unit_hipEventIpc_shm_cleanup) {
  auto before = get_hip_ipc_shm_files();

  int pipefd[2];
  REQUIRE(pipe(pipefd) == 0);  // Create pipe: pipefd[0] for reading, pipefd[1] for writing

  pid_t pid = fork();
  REQUIRE(pid >= 0);  // Ensure fork succeeded

  if (pid == 0) {
    // === Child Process ===
    close(pipefd[1]);  // Close write end

    hipIpcEventHandle_t handle;
    ssize_t total = 0;
    char* ptr = reinterpret_cast<char*>(&handle);
    while (total < sizeof(handle)) {
      ssize_t r = read(pipefd[0], ptr + total, sizeof(handle) - total);
      if (r <= 0) {
        perror("Child failed to read from pipe");
        exit(1);
      }
      total += r;
    }
    close(pipefd[0]);

    hipEvent_t event;
    if (hipIpcOpenEventHandle(&event, handle) != hipSuccess) {
      perror("Child failed to open event handle");
      exit(1);
    }

    if (hipEventDestroy(event) != hipSuccess) {
      perror("Child failed to destroy event");
      exit(1);
    }

    exit(0);
  } else {
    // === Parent Process ===
    close(pipefd[0]);  // Close read end

    hipEvent_t event;
    REQUIRE(hipEventCreateWithFlags(&event, hipEventInterprocess | hipEventDisableTiming) ==
            hipSuccess);

    hipIpcEventHandle_t ipcHandle;
    REQUIRE(hipIpcGetEventHandle(&ipcHandle, event) == hipSuccess);

    // Write handle to child
    ssize_t total = 0;
    const char* ptr = reinterpret_cast<const char*>(&ipcHandle);
    while (total < sizeof(ipcHandle)) {
      ssize_t w = write(pipefd[1], ptr + total, sizeof(ipcHandle) - total);
      if (w <= 0) {
        perror("Parent failed to write to pipe");
        exit(1);
      }
      total += w;
    }
    close(pipefd[1]);

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    REQUIRE(hipEventDestroy(event) == hipSuccess);

    auto after = get_hip_ipc_shm_files();
    std::cout << before.size() << after.size() << std::endl;
    for (const auto& file : after) {
      REQUIRE(before.count(file) == 1);  // No new HIP shm files left behind
    }
  }
}
