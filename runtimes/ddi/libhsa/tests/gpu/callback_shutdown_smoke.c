// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <limits.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

static atomic_uint callback_status;
static atomic_uint dormant_calls;

static bool dormant_handler(hsa_signal_value_t value, void *arg) {
  (void)value;
  (void)arg;
  atomic_fetch_add_explicit(&dormant_calls, 1, memory_order_relaxed);
  return true;
}

static bool shutdown_handler(hsa_signal_value_t value, void *arg) {
  (void)value;
  (void)arg;
  unsigned status = hsa_shut_down();
  atomic_store_explicit(&callback_status, status, memory_order_release);
  return false;
}

static int64_t monotonic_milliseconds(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
  return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int check(const char *operation, hsa_status_t status) {
  if (status == HSA_STATUS_SUCCESS) return 1;
  fprintf(stderr, "%s failed: 0x%x\n", operation, status);
  return 0;
}

int main(void) {
  if (!check("hsa_init", hsa_init())) return 1;
  for (unsigned cycle = 0; cycle < 10; ++cycle) {
    atomic_store_explicit(&callback_status, UINT_MAX, memory_order_relaxed);
    hsa_signal_t dormant = {0}, trigger = {0};
    if (!check("create dormant", hsa_signal_create(1, 0, NULL, &dormant)) ||
        !check("create trigger", hsa_signal_create(1, 0, NULL, &trigger)) ||
        !check("register dormant", hsa_amd_signal_async_handler(
            dormant, HSA_SIGNAL_CONDITION_EQ, 0, dormant_handler, NULL)) ||
        !check("register shutdown", hsa_amd_signal_async_handler(
            trigger, HSA_SIGNAL_CONDITION_EQ, 0, shutdown_handler, NULL)))
      return 2;
    hsa_signal_store_screlease(trigger, 0);
    int64_t deadline = monotonic_milliseconds() + 5000;
    unsigned status;
    do {
      status = atomic_load_explicit(&callback_status, memory_order_acquire);
      if (status != UINT_MAX) break;
      usleep(1000);
    } while (monotonic_milliseconds() < deadline);
    if (status != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "callback shutdown cycle %u: status=0x%x\n", cycle, status);
      return 3;
    }
    do {
      status = hsa_init();
      if (status == HSA_STATUS_SUCCESS) break;
      usleep(1000);
    } while (monotonic_milliseconds() < deadline);
    if (status != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "reinit cycle %u: status=0x%x\n", cycle, status);
      return 4;
    }
    printf("callback shutdown and reinit cycle %u: success\n", cycle + 1);
    fflush(stdout);
  }
  if (!check("final shutdown", hsa_shut_down())) return 5;
  printf("dormant callback invocations: %u\n",
         atomic_load_explicit(&dormant_calls, memory_order_relaxed));
  return atomic_load_explicit(&dormant_calls, memory_order_relaxed) == 0 ? 0 : 6;
}
