// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* CPU-only signal storage, lifetime, and prefetch completion rejection. */
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdbool.h>
#include <stdio.h>

#define REQUIRE(condition)                                                      \
  do {                                                                          \
    if (!(condition)) {                                                         \
      fprintf(stderr, "CPU signal check failed at line %d: %s\n", __LINE__,     \
              #condition);                                                      \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

struct cpu_info {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t pool;
};

static hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void *data) {
  bool capable = false;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SIGNAL_CAPABLE, &capable);
  if (status == HSA_STATUS_SUCCESS && capable) {
    *(hsa_amd_memory_pool_t *)data = pool;
  }
  return status;
}

static hsa_status_t find_cpu_pool(hsa_agent_t agent, void *data) {
  hsa_device_type_t kind;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &kind);
  if (status != HSA_STATUS_SUCCESS || kind != HSA_DEVICE_TYPE_CPU) {
    return HSA_STATUS_ERROR;
  }
  struct cpu_info *cpu = data;
  cpu->agent = agent;
  return hsa_amd_agent_iterate_memory_pools(agent, find_pool, &cpu->pool);
}

int main(void) {
  for (unsigned generation = 0; generation < 4; ++generation) {
    hsa_status_t initialized = hsa_init();
    if (generation == 0 && initialized != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CPU-only HSA initialization unavailable: %#x\n",
              (unsigned)initialized);
      return 77;
    }
    REQUIRE(initialized == HSA_STATUS_SUCCESS);
    hsa_signal_t signals[2049];
    for (unsigned i = 0; i < 2049; ++i) {
      REQUIRE(hsa_signal_create((hsa_signal_value_t)i, 0, NULL, &signals[i]) ==
              HSA_STATUS_SUCCESS);
    }
    for (unsigned i = 0; i < 2049; ++i) {
      hsa_signal_add_scacq_screl(signals[i], 7);
      REQUIRE(hsa_signal_load_scacquire(signals[i]) == (hsa_signal_value_t)i + 7);
      REQUIRE(hsa_signal_destroy(signals[i]) == HSA_STATUS_SUCCESS);
    }
    struct cpu_info cpu = {0};
    REQUIRE(hsa_iterate_agents(find_cpu_pool, &cpu) == HSA_STATUS_SUCCESS);
    REQUIRE(cpu.pool.handle != 0);
    const hsa_signal_value_t initial[] = {3, -4, 9};
    hsa_amd_signal_batch_t batch;
    REQUIRE(hsa_amd_signal_batch_create(3, initial, 0, NULL, 0, cpu.pool, &batch,
                                       signals) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_signal_destroy(signals[1]) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_amd_signal_batch_destroy(batch) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_signal_load_scacquire(signals[0]) == initial[0]);
    REQUIRE(hsa_signal_load_scacquire(signals[2]) == initial[2]);
    REQUIRE(hsa_signal_destroy(signals[0]) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_signal_destroy(signals[2]) == HSA_STATUS_SUCCESS);

    int backing = 0;
    REQUIRE(hsa_signal_create(1, 0, NULL, &signals[0]) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_signal_create(1, 0, NULL, &signals[1]) == HSA_STATUS_SUCCESS);
    /* This CPU-only runtime has no active migration provider. Reject before
       accepting work or changing either signal, including a pending dependency. */
    REQUIRE(hsa_amd_svm_prefetch_async(&backing, sizeof(backing), cpu.agent, 1,
                                      &signals[0], signals[1]) ==
            HSA_STATUS_ERROR_NOT_SUPPORTED);
    REQUIRE(hsa_signal_load_scacquire(signals[0]) == 1);
    REQUIRE(hsa_signal_load_scacquire(signals[1]) == 1);
    REQUIRE(hsa_signal_destroy(signals[0]) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_signal_destroy(signals[1]) == HSA_STATUS_SUCCESS);
    REQUIRE(hsa_shut_down() == HSA_STATUS_SUCCESS);
  }
  return 0;
}
