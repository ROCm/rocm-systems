#include "aql_profiler.h"
#include <iostream>
#include <memory>
#include <cstring>
#include <thread>
#include <chrono>

// Memory allocation callback for aqlprofile
static hsa_status_t memory_alloc_callback(void** ptr, uint64_t size,
                                         aqlprofile_buffer_desc_flags_t flags,
                                         void* userdata) {
    aql_profiler_context_t* ctx = static_cast<aql_profiler_context_t*>(userdata);

    if (flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED) {
        // Use device memory pool (kernarg)
        hsa_status_t status = hsa_amd_memory_pool_allocate(ctx->kernarg_pool, size, 0, ptr);
        if (status == HSA_STATUS_SUCCESS) {
            hsa_amd_memory_fill(*ptr, 0u, size / sizeof(uint32_t));
            hsa_amd_agents_allow_access(1, &ctx->agent, nullptr, *ptr);
        }
        return status;
    } else {
        // Use CPU memory pool
        hsa_status_t status = hsa_amd_memory_pool_allocate(ctx->cpu_pool, size,
                                                          HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, ptr);
        if (status == HSA_STATUS_SUCCESS) {
            hsa_amd_memory_fill(*ptr, 0u, size / sizeof(uint32_t));
            hsa_amd_agents_allow_access(1, &ctx->agent, nullptr, *ptr);
        }
        return status;
    }
}

// Memory deallocation callback
static void memory_dealloc_callback(void* ptr, void* userdata) {
    (void)userdata; // Suppress unused parameter warning
    if (ptr) {
        hsa_amd_memory_pool_free(ptr);
    }
}

// Memory copy callback
static hsa_status_t memory_copy_callback(void* dst, const void* src, size_t size, void* userdata) {
    (void)userdata; // Suppress unused parameter warning
    if (size == 0) return HSA_STATUS_SUCCESS;
    return hsa_memory_copy(dst, src, size);
}

// Data callback to collect results
static hsa_status_t data_callback(aqlprofile_pmc_event_t event,
                                 uint64_t counter_id,
                                 uint64_t counter_value,
                                 void* userdata) {
    (void)counter_id; // Suppress unused parameter warning
    aql_profiler_context_t* ctx = static_cast<aql_profiler_context_t*>(userdata);

    if (ctx->result_count >= ctx->result_capacity) {
        // Expand results array
        ctx->result_capacity *= 2;
        ctx->results = static_cast<aql_counter_result_t*>(
            realloc(ctx->results, ctx->result_capacity * sizeof(aql_counter_result_t)));
    }

    aql_counter_result_t* result = &ctx->results[ctx->result_count++];
    result->counter_value = counter_value;
    result->block_index = event.block_index;
    result->event_id = event.event_id;
    result->block_name = event.block_name;

    // Generate counter name based on block and event
    const char* block_str = "UNKNOWN";
    switch (event.block_name) {
        case HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ: block_str = "SQ"; break;
        case HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA: block_str = "TA"; break;
        case HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP: block_str = "TCP"; break;
        case HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC: block_str = "TCC"; break;
        case HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM: block_str = "GRBM"; break;
        default:
            snprintf(result->counter_name, sizeof(result->counter_name),
                     "BLOCK_%u", event.block_name);
            return HSA_STATUS_SUCCESS;
    }

    snprintf(result->counter_name, sizeof(result->counter_name),
             "%s_%u_EVENT_%u", block_str, event.block_index, event.event_id);

    return HSA_STATUS_SUCCESS;
}

// Helper to find global pool (following aqlprofile pattern)
static hsa_status_t find_global_pool(hsa_amd_memory_pool_t pool, void* data, bool kern_arg) {
    if (!data) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    auto* pool_ptr = static_cast<hsa_amd_memory_pool_t*>(data);

    hsa_amd_segment_t segment;
    hsa_status_t status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (status != HSA_STATUS_SUCCESS) return status;

    if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

    uint32_t flags;
    status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (status != HSA_STATUS_SUCCESS) return status;

    uint32_t karg_flag = flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    if ((karg_flag == 0 && kern_arg) || (karg_flag != 0 && !kern_arg)) {
        return HSA_STATUS_SUCCESS;
    }

    *pool_ptr = pool;
    return HSA_STATUS_INFO_BREAK;
}

// Find standard pool (global, not kernarg)
static hsa_status_t find_standard_pool(hsa_amd_memory_pool_t pool, void* data) {
    return find_global_pool(pool, data, false);
}

// Find kernarg pool (global + kernarg)
static hsa_status_t find_kernarg_pool(hsa_amd_memory_pool_t pool, void* data) {
    return find_global_pool(pool, data, true);
}

// Helper to find memory pools for a specific agent
static hsa_status_t find_memory_pools(hsa_agent_t agent,
                                     hsa_amd_memory_pool_t* cpu_pool,
                                     hsa_amd_memory_pool_t* kernarg_pool) {
    // Check what type of agent this is
    hsa_device_type_t device_type;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != HSA_STATUS_SUCCESS) return status;

    if (device_type == HSA_DEVICE_TYPE_GPU) {
        // GPU agents typically only have a standard global pool
        status = hsa_amd_agent_iterate_memory_pools(agent, find_standard_pool, cpu_pool);
        if (status != HSA_STATUS_INFO_BREAK && status != HSA_STATUS_SUCCESS) {
            return status;
        }

        // For GPU, use the same pool for both (kernarg allocation will be handled by HSA)
        *kernarg_pool = *cpu_pool;
        return HSA_STATUS_SUCCESS;
    } else {
        // CPU agents need both standard and kernarg pools
        status = hsa_amd_agent_iterate_memory_pools(agent, find_standard_pool, cpu_pool);
        if (status != HSA_STATUS_INFO_BREAK && status != HSA_STATUS_SUCCESS) {
            return status;
        }

        status = hsa_amd_agent_iterate_memory_pools(agent, find_kernarg_pool, kernarg_pool);
        if (status != HSA_STATUS_INFO_BREAK && status != HSA_STATUS_SUCCESS) {
            return status;
        }

        return HSA_STATUS_SUCCESS;
    }
}

// Helper to convert block name string to enum
static hsa_ven_amd_aqlprofile_block_name_t string_to_block_name(const char* block_name) {
    if (strcmp(block_name, "SQ") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    if (strcmp(block_name, "TA") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA;
    if (strcmp(block_name, "TCP") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP;
    if (strcmp(block_name, "TCC") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC;
    if (strcmp(block_name, "GRBM") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM;
    if (strcmp(block_name, "TD") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD;
    // Note: DB and CB blocks may not be available in all versions
    // if (strcmp(block_name, "DB") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_DB;
    // if (strcmp(block_name, "CB") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CB;
    return HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ; // Default fallback
}

// Create profiler context
aql_profiler_context_t* aql_profiler_create(hsa_agent_t agent, hsa_queue_t* queue) {
    aql_profiler_context_t* ctx = static_cast<aql_profiler_context_t*>(
        calloc(1, sizeof(aql_profiler_context_t)));
    if (!ctx) return nullptr;

    ctx->agent = agent;
    ctx->queue = queue;
    ctx->result_capacity = 16;
    ctx->results = static_cast<aql_counter_result_t*>(
        malloc(ctx->result_capacity * sizeof(aql_counter_result_t)));
    ctx->events = new std::vector<aqlprofile_pmc_event_t>();

    // Find memory pools
    if (find_memory_pools(agent, &ctx->cpu_pool, &ctx->kernarg_pool) != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to find memory pools\n");
        delete ctx->events;
        free(ctx->results);
        free(ctx);
        return nullptr;
    }

    // Register agent with aqlprofile
    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name));

    uint32_t xcc_count = 1, se_count = 1, cu_count = 1, sa_per_se = 1;
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &xcc_count);
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES, &se_count);
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cu_count);
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE, &sa_per_se);

    aqlprofile_agent_info_t agent_info = {
        .agent_gfxip = agent_name,
        .xcc_num = xcc_count,
        .se_num = se_count,
        .cu_num = cu_count,
        .shader_arrays_per_se = sa_per_se
    };

    if (aqlprofile_register_agent(&ctx->agent_handle, &agent_info) != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to register agent with aqlprofile\n");
        delete ctx->events;
        free(ctx->results);
        free(ctx);
        return nullptr;
    }

    ctx->initialized = true;
    return ctx;
}

// Destroy profiler context
void aql_profiler_destroy(aql_profiler_context_t* ctx) {
    if (!ctx) return;

    if (ctx->initialized && ctx->handle.handle != 0) {
        aqlprofile_pmc_delete_packets(ctx->handle);
    }

    delete ctx->events;
    free(ctx->results);
    free(ctx);
}

// Add counter for monitoring
int aql_profiler_add_counter(aql_profiler_context_t* ctx,
                           const char* block_name,
                           uint32_t block_index,
                           uint32_t event_id) {
    if (!ctx || !ctx->initialized || ctx->profiling_active) {
        return -1;
    }

    aqlprofile_pmc_event_t event = {
        .block_index = block_index,
        .event_id = event_id,
        .flags = {.raw = 0},
        .block_name = string_to_block_name(block_name)
    };

    // Validate event
    bool valid = false;
    if (aqlprofile_validate_pmc_event(ctx->agent_handle, &event, &valid) == HSA_STATUS_SUCCESS) {
        if (!valid) {
            fprintf(stderr, "Warning: Event %s[%u]:%u may not be valid for this agent\n",
                    block_name, block_index, event_id);
        }
    }

    ctx->events->push_back(event);
    printf("Added counter: %s[%u] event %u\n", block_name, block_index, event_id);
    return 0;
}

// Create profiling packets
int aql_profiler_create_packets(aql_profiler_context_t* ctx) {
    if (!ctx || !ctx->initialized || ctx->profiling_active) {
        return -1;
    }

    if (ctx->events->empty()) {
        fprintf(stderr, "No events added for profiling\n");
        return -1;
    }

    // Initialize packet structures
    constexpr hsa_ext_amd_aql_pm4_packet_t null_packet = {
        .header = 0,
        .pm4_command = {0},
        .completion_signal = {.handle = 0}
    };

    ctx->packets.start_packet = null_packet;
    ctx->packets.stop_packet = null_packet;
    ctx->packets.read_packet = null_packet;

    // Setup profile
    ctx->profile.agent = ctx->agent_handle;
    ctx->profile.events = ctx->events->data();
    ctx->profile.event_count = static_cast<uint32_t>(ctx->events->size());

    // Create AQL packets
    hsa_status_t status = aqlprofile_pmc_create_packets(
        &ctx->handle,
        &ctx->packets,
        ctx->profile,
        memory_alloc_callback,
        memory_dealloc_callback,
        memory_copy_callback,
        ctx
    );

    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create AQL packets: %d\n", status);
        return -1;
    }

    // Set proper packet headers
    ctx->packets.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    ctx->packets.stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    ctx->packets.read_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

    printf("Created AQL profiling packets with %zu events\n", ctx->events->size());
    ctx->profiling_active = true;
    ctx->result_count = 0;
    return 0;
}

// Get packets for external submission
int aql_profiler_get_packets(aql_profiler_context_t* ctx, aqlprofile_pmc_aql_packets_t* packets) {
    if (!ctx || !packets || !ctx->profiling_active) {
        return -1;
    }

    *packets = ctx->packets;
    return 0;
}

// Collect results (call after packets have been submitted and completed)
int aql_profiler_collect_results(aql_profiler_context_t* ctx) {
    if (!ctx || !ctx->initialized || !ctx->profiling_active) {
        return -1;
    }

    // Collect results
    hsa_status_t status = aqlprofile_pmc_iterate_data(ctx->handle, data_callback, ctx);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to iterate counter data: %d\n", status);
        return -1;
    }

    printf("Collected %u profiling results\n", ctx->result_count);
    ctx->profiling_active = false;
    return 0;
}

// Get results
int aql_profiler_get_results(aql_profiler_context_t* ctx,
                           aql_counter_result_t** results,
                           uint32_t* count) {
    if (!ctx || !results || !count) {
        return -1;
    }

    *results = ctx->results;
    *count = ctx->result_count;
    return 0;
}

// Print results helper
void aql_profiler_print_results(aql_profiler_context_t* ctx) {
    if (!ctx) return;

    printf("\n=== AQL Profiling Results ===\n");
    for (uint32_t i = 0; i < ctx->result_count; i++) {
        printf("  %s: %lu\n", ctx->results[i].counter_name, ctx->results[i].counter_value);
    }
    if (ctx->result_count == 0) {
        printf("  No results collected\n");
    }
    printf("==============================\n\n");
}

// Utility functions for common counter sets
int aql_profiler_add_basic_counters(aql_profiler_context_t* ctx, const char* gfxip) {
    (void)gfxip; // Suppress unused parameter warning
    if (!ctx) return -1;

    // Use the specific events provided: block_name=6 (SQ block)
    aql_profiler_add_counter(ctx, "SQ", 0, 35);   // Event 35
    aql_profiler_add_counter(ctx, "SQ", 0, 37);   // Event 37
    aql_profiler_add_counter(ctx, "SQ", 0, 40);   // Event 40
    return 0;
}

int aql_profiler_add_sq_counters(aql_profiler_context_t* ctx) {
    if (!ctx) return -1;

    // Use all the specific SQ events provided
    aql_profiler_add_counter(ctx, "SQ", 0, 35);   // Event 35
    aql_profiler_add_counter(ctx, "SQ", 0, 37);   // Event 37
    aql_profiler_add_counter(ctx, "SQ", 0, 40);   // Event 40
    aql_profiler_add_counter(ctx, "SQ", 0, 54);   // Event 54
    aql_profiler_add_counter(ctx, "SQ", 0, 36);   // Event 36
    return 0;
}

int aql_profiler_add_memory_counters(aql_profiler_context_t* ctx, const char* gfxip) {
    (void)gfxip; // Suppress unused parameter warning
    if (!ctx) return -1;

    // Use a subset of the SQ events for memory-related monitoring
    aql_profiler_add_counter(ctx, "SQ", 0, 36);   // Event 36
    aql_profiler_add_counter(ctx, "SQ", 0, 54);   // Event 54
    return 0;
}

// Simple interface - takes agent, queue, and pre-formatted events
int aql_profiler_create_packets_simple(hsa_agent_t agent,
                                       hsa_queue_t* queue,
                                       const aqlprofile_pmc_event_t* events,
                                       uint32_t event_count,
                                       aqlprofile_pmc_aql_packets_t* packets,
                                       aqlprofile_handle_t* handle) {
    (void)queue; // Queue not needed for packet creation

    if (!events || event_count == 0 || !packets || !handle) {
        return -1;
    }

    // Register agent with aqlprofile
    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name));

    uint32_t xcc_count = 1, se_count = 1, cu_count = 1, sa_per_se = 1;
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &xcc_count);
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES, &se_count);
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cu_count);
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE, &sa_per_se);

    aqlprofile_agent_handle_t agent_handle;
    aqlprofile_agent_info_t agent_info = {
        .agent_gfxip = agent_name,
        .xcc_num = xcc_count,
        .se_num = se_count,
        .cu_num = cu_count,
        .shader_arrays_per_se = sa_per_se
    };

    if (aqlprofile_register_agent(&agent_handle, &agent_info) != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to register agent with aqlprofile\n");
        return -1;
    }

    // Create profile structure
    aqlprofile_pmc_profile_t profile = {
        .agent = agent_handle,
        .events = events,
        .event_count = event_count
    };

    // Initialize packet structures
    constexpr hsa_ext_amd_aql_pm4_packet_t null_packet = {
        .header = 0,
        .pm4_command = {0},
        .completion_signal = {.handle = 0}
    };

    packets->start_packet = null_packet;
    packets->stop_packet = null_packet;
    packets->read_packet = null_packet;

    // Create temporary context for memory callbacks
    aql_profiler_context_t temp_ctx = {};
    temp_ctx.agent = agent;

    // Find memory pools for the temporary context
    if (find_memory_pools(agent, &temp_ctx.cpu_pool, &temp_ctx.kernarg_pool) != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to find memory pools\n");
        return -1;
    }

    // Create AQL packets
    hsa_status_t status = aqlprofile_pmc_create_packets(
        handle,
        packets,
        profile,
        memory_alloc_callback,
        memory_dealloc_callback,
        memory_copy_callback,
        &temp_ctx
    );

    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create AQL packets: %d\n", status);
        return -1;
    }

    // Set proper packet headers
    packets->start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets->stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets->read_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

    printf("Created AQL profiling packets with %u events\n", event_count);
    return 0;
}

// Cleanup for simple interface
void aql_profiler_cleanup_simple(aqlprofile_handle_t handle) {
    if (handle.handle != 0) {
        aqlprofile_pmc_delete_packets(handle);
    }
}