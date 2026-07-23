#pragma once

#include <vector>

typedef struct {
    uint64_t gpu_timestamp;
    uint64_t system_timestamp;
} crosststamp_t;

typedef struct {
    uint32_t kfd_gpu_id;
    uint32_t node_id;
} gpu_context_t;

int kfd_enumerate_gpus(std::vector<gpu_context_t>& gpus);
int kfd_get_crosststamp(const gpu_context_t& gpu, crosststamp_t&);
