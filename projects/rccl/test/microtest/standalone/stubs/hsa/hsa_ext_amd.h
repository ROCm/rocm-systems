// Minimal stub for hsa_ext_amd.h
#pragma once
#include "hsa.h"

typedef enum {
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 0,
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 1,
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 2,
} hsa_amd_memory_pool_global_flag_t;
