#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <string>

// Include aqlprofile v2 header
#include "aqlprofile-sdk/aql_profile_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error checking macro
#define CHECK_HSA(x) do { \
    hsa_status_t status = (x); \
    if (status != HSA_STATUS_SUCCESS) { \
        fprintf(stderr, "HSA error at %s:%d - status: %d\n", __FILE__, __LINE__, status); \
        exit(-1); \
    } \
} while(0)

// Forward declarations
typedef struct aql_profiler_context aql_profiler_context_t;
typedef struct aql_counter_result aql_counter_result_t;

// Counter result structure
struct aql_counter_result {
    char counter_name[64];
    uint64_t counter_value;
    uint32_t block_index;
    uint32_t event_id;
    uint32_t block_name;
};

// Main profiler context
struct aql_profiler_context {
    hsa_agent_t agent;
    hsa_queue_t* queue;

    // AQLProfile v2 interface
    aqlprofile_handle_t handle;
    aqlprofile_agent_handle_t agent_handle;
    aqlprofile_pmc_aql_packets_t packets;

    // Events to profile
    std::vector<aqlprofile_pmc_event_t>* events;
    aqlprofile_pmc_profile_t profile;

    // Memory pools
    hsa_amd_memory_pool_t cpu_pool;
    hsa_amd_memory_pool_t kernarg_pool;

    // Results storage
    aql_counter_result_t* results;
    uint32_t result_count;
    uint32_t result_capacity;

    // State tracking
    bool initialized;
    bool profiling_active;
};

// Function declarations
aql_profiler_context_t* aql_profiler_create(hsa_agent_t agent, hsa_queue_t* queue);
void aql_profiler_destroy(aql_profiler_context_t* ctx);

// Add counter for monitoring
int aql_profiler_add_counter(aql_profiler_context_t* ctx,
                            const char* block_name,
                            uint32_t block_index,
                            uint32_t event_id);

// Create profiling packets (returns packets for external submission)
int aql_profiler_create_packets(aql_profiler_context_t* ctx);
int aql_profiler_get_packets(aql_profiler_context_t* ctx, aqlprofile_pmc_aql_packets_t* packets);

// Collect results (call after packets have been submitted and completed)
int aql_profiler_collect_results(aql_profiler_context_t* ctx);

// Get results
int aql_profiler_get_results(aql_profiler_context_t* ctx,
                            aql_counter_result_t** results,
                            uint32_t* count);

// Print results helper
void aql_profiler_print_results(aql_profiler_context_t* ctx);

// Simple interface - takes agent, queue, and pre-formatted events
int aql_profiler_create_packets_simple(hsa_agent_t agent,
                                       hsa_queue_t* queue,
                                       const aqlprofile_pmc_event_t* events,
                                       uint32_t event_count,
                                       aqlprofile_pmc_aql_packets_t* packets,
                                       aqlprofile_handle_t* handle);

// Cleanup for simple interface
void aql_profiler_cleanup_simple(aqlprofile_handle_t handle);

// Utility functions for common counters
int aql_profiler_add_basic_counters(aql_profiler_context_t* ctx, const char* gfxip);
int aql_profiler_add_sq_counters(aql_profiler_context_t* ctx);
int aql_profiler_add_memory_counters(aql_profiler_context_t* ctx, const char* gfxip);

#ifdef __cplusplus
}
#endif