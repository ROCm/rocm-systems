// Minimal HSA stub for CPU-only RCCL unit tests.
#pragma once
#include <cstdint>
#include <cstddef>

typedef enum {
    HSA_STATUS_SUCCESS = 0,
    HSA_STATUS_ERROR = 1,
} hsa_status_t;

typedef struct { uint64_t handle; } hsa_signal_t;
typedef struct { uint64_t handle; } hsa_agent_t;
typedef struct { uint64_t handle; } hsa_amd_memory_pool_t;

inline hsa_status_t hsa_status_string(hsa_status_t status, const char** str) {
  static const char* msg = "stub";
  if (str) *str = msg;
  return HSA_STATUS_SUCCESS;
}

inline hsa_status_t hsa_init() { return HSA_STATUS_SUCCESS; }
