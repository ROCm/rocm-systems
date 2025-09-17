/**
 * @file aql_pmc_interface.c
 * @brief Implementation of AQLProfile v2-compatible PMC interface
 *
 * This file implements the PMC packet creation functionality
 * compatible with AQLProfile v2's interface.
 */

#include "aql_pmc_interface.h"
#include "aql_arch_ops.h"
#include "aql_cmd_buffer.h"

// External function declaration
extern aql_result_t aql_populate_packet_from_buffer(const void* cmd_buffer, uint32_t cmd_size,
                                                   const aql_arch_ops_t* arch_ops,
                                                   aql_pm4_ib_packet_t* aql_packet);

#ifdef __KERNEL__
#include <linux/string.h>
#include <linux/slab.h>
#else
#include <string.h>
#include <stdlib.h>
#endif

// Map block names to block IDs
static aql_block_id_t get_block_id_from_name(const char* block_name) {
    if (strcmp(block_name, "CB") == 0) return AQL_BLOCK_CB;
    if (strcmp(block_name, "CPC") == 0) return AQL_BLOCK_CPC;
    if (strcmp(block_name, "CPF") == 0) return AQL_BLOCK_CPF;
    if (strcmp(block_name, "CPG") == 0) return AQL_BLOCK_CPG;
    if (strcmp(block_name, "DB") == 0) return AQL_BLOCK_DB;
    if (strcmp(block_name, "GDS") == 0) return AQL_BLOCK_GDS;
    if (strcmp(block_name, "GRBM") == 0) return AQL_BLOCK_GRBM;
    if (strcmp(block_name, "GRBM_SE") == 0) return AQL_BLOCK_GRBM_SE;
    if (strcmp(block_name, "SPI") == 0) return AQL_BLOCK_SPI;
    if (strcmp(block_name, "SQ") == 0) return AQL_BLOCK_SQ;
    if (strcmp(block_name, "TCP") == 0) return AQL_BLOCK_TCP;
    if (strcmp(block_name, "TCC") == 0) return AQL_BLOCK_TCC;
    return AQL_BLOCK_UNKNOWN;
}

// Generate START commands for performance counters
static aql_result_t generate_start_commands(const aql_arch_ops_t* ops,
                                          aql_cmd_buffer_t* cmd_buf,
                                          const aql_pmc_event_t* events,
                                          uint32_t event_count) {
    aql_result_t result;

    for (uint32_t i = 0; i < event_count; i++) {
        const aql_pmc_event_t* event = &events[i];
        uint32_t reg_addr;

        // Get the counter select register address
        result = ops->get_counter_register_addr(event->block_id,
                                              event->block_instance,
                                              AQL_REG_TYPE_SELECT,
                                              &reg_addr);
        if (result != AQL_SUCCESS) continue;

        // Write event ID to counter select register
        result = ops->build_write_config_reg(cmd_buf, reg_addr, event->event_id);
        if (result != AQL_SUCCESS) return result;
    }

    // Add wait idle to ensure configuration is complete
    return ops->build_wait_idle(cmd_buf);
}

// Generate STOP commands for performance counters
static aql_result_t generate_stop_commands(const aql_arch_ops_t* ops,
                                         aql_cmd_buffer_t* cmd_buf,
                                         const aql_pmc_event_t* events,
                                         uint32_t event_count) {
    aql_result_t result;

    for (uint32_t i = 0; i < event_count; i++) {
        const aql_pmc_event_t* event = &events[i];
        uint32_t reg_addr;

        // Get the counter select register address
        result = ops->get_counter_register_addr(event->block_id,
                                              event->block_instance,
                                              AQL_REG_TYPE_SELECT,
                                              &reg_addr);
        if (result != AQL_SUCCESS) continue;

        // Write 0 to counter select register to stop counting
        result = ops->build_write_config_reg(cmd_buf, reg_addr, 0);
        if (result != AQL_SUCCESS) return result;
    }

    // Add wait idle to ensure stop is complete
    return ops->build_wait_idle(cmd_buf);
}

// Generate READ commands for performance counters
static aql_result_t generate_read_commands(const aql_arch_ops_t* ops,
                                         aql_cmd_buffer_t* cmd_buf,
                                         const aql_pmc_event_t* events,
                                         uint32_t event_count,
                                         void* output_buffer) {
    aql_result_t result;
    uint64_t output_addr = (uint64_t)(uintptr_t)output_buffer;

    for (uint32_t i = 0; i < event_count; i++) {
        const aql_pmc_event_t* event = &events[i];
        uint32_t reg_addr;

        // Get the counter result register address (low 32 bits)
        result = ops->get_counter_register_addr(event->block_id,
                                              event->block_instance,
                                              AQL_REG_TYPE_RESULT,
                                              &reg_addr);
        if (result != AQL_SUCCESS) continue;

        // Copy counter result to output buffer
        uint64_t dest_addr = output_addr + (i * 8); // 8 bytes per counter (64-bit)
        result = ops->build_copy_reg_data(cmd_buf, reg_addr, (void*)dest_addr, 8, true);
        if (result != AQL_SUCCESS) return result;
    }

    // Add wait idle to ensure all reads are complete
    return ops->build_wait_idle(cmd_buf);
}

aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets,
                                   const aql_pmc_profile_t* profile) {
    if (!packets || !profile || !profile->events || profile->event_count == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    // Get architecture operations
    const aql_arch_ops_t* ops = aql_detect_architecture(profile->arch_name);
    if (!ops) {
        return AQL_ERROR_UNSUPPORTED_ARCH;
    }

    // Allocate command buffer for all three command sequences
    size_t total_cmd_size = profile->event_count * 64 * 3; // Estimate: 64 dwords per event * 3 sequences
#ifdef __KERNEL__
    uint32_t* cmd_buffer = (uint32_t*)kmalloc(total_cmd_size, GFP_KERNEL);
#else
    uint32_t* cmd_buffer = (uint32_t*)malloc(total_cmd_size);
#endif
    if (!cmd_buffer) {
        return AQL_ERROR_NO_MEMORY;
    }

    // Initialize packets structure
    memset(packets, 0, sizeof(*packets));
    packets->command_buffer = cmd_buffer;
    packets->command_buffer_size = total_cmd_size;
    packets->handle = (uint64_t)(uintptr_t)cmd_buffer; // Use buffer address as handle

    aql_result_t result;
    uint32_t* current_cmd_ptr = cmd_buffer;
    size_t remaining_size = total_cmd_size;

    // Generate START commands
    aql_cmd_buffer_t start_cmd_buf;
    result = aql_cmd_buffer_init(&start_cmd_buf, current_cmd_ptr, remaining_size / sizeof(uint32_t));
    if (result != AQL_SUCCESS) goto cleanup;

    result = generate_start_commands(ops, &start_cmd_buf, profile->events, profile->event_count);
    if (result != AQL_SUCCESS) goto cleanup;

    // Create START AQL packet
    result = aql_populate_packet_from_buffer(
        aql_cmd_buffer_data(&start_cmd_buf),
        aql_cmd_buffer_size_bytes(&start_cmd_buf),
        ops,
        &packets->start_packet
    );
    if (result != AQL_SUCCESS) goto cleanup;

    // Move to next command buffer section
    size_t start_cmd_size = aql_cmd_buffer_size_bytes(&start_cmd_buf);
    current_cmd_ptr += (start_cmd_size + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    remaining_size -= start_cmd_size;

    // Generate STOP commands
    aql_cmd_buffer_t stop_cmd_buf;
    result = aql_cmd_buffer_init(&stop_cmd_buf, current_cmd_ptr, remaining_size / sizeof(uint32_t));
    if (result != AQL_SUCCESS) goto cleanup;

    result = generate_stop_commands(ops, &stop_cmd_buf, profile->events, profile->event_count);
    if (result != AQL_SUCCESS) goto cleanup;

    // Create STOP AQL packet
    result = aql_populate_packet_from_buffer(
        aql_cmd_buffer_data(&stop_cmd_buf),
        aql_cmd_buffer_size_bytes(&stop_cmd_buf),
        ops,
        &packets->stop_packet
    );
    if (result != AQL_SUCCESS) goto cleanup;

    // Move to next command buffer section
    size_t stop_cmd_size = aql_cmd_buffer_size_bytes(&stop_cmd_buf);
    current_cmd_ptr += (stop_cmd_size + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    remaining_size -= stop_cmd_size;

    // Generate READ commands
    aql_cmd_buffer_t read_cmd_buf;
    result = aql_cmd_buffer_init(&read_cmd_buf, current_cmd_ptr, remaining_size / sizeof(uint32_t));
    if (result != AQL_SUCCESS) goto cleanup;

    result = generate_read_commands(ops, &read_cmd_buf, profile->events, profile->event_count, profile->output_buffer);
    if (result != AQL_SUCCESS) goto cleanup;

    // Create READ AQL packet
    result = aql_populate_packet_from_buffer(
        aql_cmd_buffer_data(&read_cmd_buf),
        aql_cmd_buffer_size_bytes(&read_cmd_buf),
        ops,
        &packets->read_packet
    );
    if (result != AQL_SUCCESS) goto cleanup;

    return AQL_SUCCESS;

cleanup:
#ifdef __KERNEL__
    kfree(cmd_buffer);
#else
    free(cmd_buffer);
#endif
    memset(packets, 0, sizeof(*packets));
    return result;
}

aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets) {
    if (!packets) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (packets->command_buffer) {
#ifdef __KERNEL__
        kfree(packets->command_buffer);
#else
        free(packets->command_buffer);
#endif
    }

    memset(packets, 0, sizeof(*packets));
    return AQL_SUCCESS;
}

aql_result_t aql_create_counter_event(aql_pmc_event_t* event,
                                     const char* block_name,
                                     uint32_t instance,
                                     uint32_t event_id) {
    if (!event || !block_name) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    aql_block_id_t block_id = get_block_id_from_name(block_name);
    if (block_id == AQL_BLOCK_UNKNOWN) {
        return AQL_ERROR_INVALID_BLOCK;
    }

    event->block_id = block_id;
    event->block_instance = instance;
    event->event_id = event_id;
    event->flags = 0;
    event->block_name = block_name;
    event->event_name = NULL; // Could be filled in later if needed

    return AQL_SUCCESS;
}