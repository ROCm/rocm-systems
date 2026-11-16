// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Exercise versioned public HSA dispatch after inheriting an initialized DSO. */
#include <hsa/hsa.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
  const int expect_tool = argc == 2 && strcmp(argv[1], "tool") == 0;
  const hsa_status_t initialized = hsa_init();
  if (initialized != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "HSA initialization unavailable: %u\n", (unsigned)initialized);
    return 77;
  }
  uint16_t major = 0;
  const hsa_status_t expected = expect_tool
                                    ? HSA_STATUS_ERROR_INVALID_ARGUMENT
                                    : HSA_STATUS_SUCCESS;
  if (hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major) != expected) {
    fprintf(stderr, "HSA dispatch did not match the expected pre-fork status\n");
    hsa_shut_down();
    return 1;
  }
  pid_t child = fork();
  if (child < 0) {
    perror("fork");
    return 2;
  }
  if (child == 0) {
    alarm(3);
    uint16_t major = 0;
    const int rejected =
        hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major) ==
            HSA_STATUS_ERROR_NOT_INITIALIZED &&
        hsa_shut_down() == HSA_STATUS_ERROR_NOT_INITIALIZED &&
        hsa_init() != HSA_STATUS_SUCCESS;
    _exit(rejected ? 0 : 1);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    fprintf(stderr, "fork child reached inherited HSA state or timed out\n");
    return 1;
  }
  if (hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major) != expected ||
      hsa_shut_down() != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "parent HSA runtime failed after child exit\n");
    return 1;
  }
  return 0;
}
