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
