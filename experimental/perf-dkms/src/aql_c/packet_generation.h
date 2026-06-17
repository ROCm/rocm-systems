/**
 * @file packet_generation.h
 * @brief PM4 packet generation functions for counter operations
 *
 * This module provides functions to generate PM4 command packets for
 * starting, stopping, and reading performance counters on AMD GPUs.
 * Based on the Rust rocprofiler-oop implementation.
 */

#ifndef PACKET_GENERATION_H
#define PACKET_GENERATION_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <stddef.h>
#endif

#include "aql_structures.h"
#include "pm4_packets.h"

/**
 * @brief Counter information for packet generation
 *
 * Describes a single performance counter to be monitored, including which
 * hardware block it belongs to, which event to count, and which physical
 * counter register to use within that block.
 *
 * This structure is analogous to the counter selection in
 * projects/aqlprofile/src/pm4/pmc_builder.h:64-69
 */
typedef struct {
	hardware_ip_block_t block_id; /* Which hardware block (SQ, CPC, etc.) */
	uint32_t event_id; /* Event ID to monitor */
	uint32_t counter_index; /* Which counter in the block (0-7 for SQ) */
	const char *name; /* Optional counter name for debugging */
} counter_info_t;

/**
 * @brief Counter collection context for PM4 packet generation
 *
 * Aggregates all counters to be monitored together, along with the GPU
 * memory location where counter data will be written. Used by the
 * start/stop/read packet generation functions.
 *
 * This structure is analogous to the profile context in
 * projects/aqlprofile/src/core/aql_profile.cpp
 */
typedef struct {
	counter_info_t *counters; /* Array of counters to collect */
	size_t counter_count; /* Number of counters in array */
	uint64_t gpu_memory_addr; /* GPU address for counter data results */
	size_t memory_size; /* Size of allocated memory in bytes */
} counter_collection_t;

/**
 * Generate PM4 packet sequence to start performance counters
 *
 * Sequence:
 * 1. CS partial flush
 * 2. GRBM broadcast mode
 * 3. Disable perfmon initially
 * 4. Enable SQ control (for SQ counters)
 * 5. Configure counter select and control registers
 * 6. Enable compute perfcount
 * 7. Enable perfmon state
 * 8. Final CS partial flush
 *
 * @param buffer PM4 buffer to append commands to
 * @param arch Architecture information with control registers
 * @param collection Counter collection context
 * @return 0 on success, negative error code on failure
 */
int generate_start_packet(pm4_buffer_t *buffer, const arch_t *arch,
			  const counter_collection_t *collection);

/**
 * Generate PM4 packet sequence to read performance counters
 *
 * Sequence:
 * 1. Enable perfmon with sampling
 * 2. GRBM broadcast mode
 * 3. CS partial flush
 * 4. For each counter and topology location:
 *    - Set GRBM index for specific SE/SA/WGP
 *    - Copy counter data to GPU memory
 * 5. Cache coherency flush
 *
 * @param buffer PM4 buffer to append commands to
 * @param arch Architecture information with control registers
 * @param collection Counter collection context
 * @return 0 on success, negative error code on failure
 */
int generate_read_packet(pm4_buffer_t *buffer, const arch_t *arch,
			 const counter_collection_t *collection);

/**
 * Generate PM4 packet sequence to stop performance counters
 *
 * Sequence:
 * 1. GRBM broadcast mode
 * 2. Set perfmon state to stop
 * 3. CS partial flush
 *
 * @param buffer PM4 buffer to append commands to
 * @param arch Architecture information with control registers
 * @return 0 on success, negative error code on failure
 */
int generate_stop_packet(pm4_buffer_t *buffer, const arch_t *arch);

/**
 * Calculate total memory size needed for counter data collection
 *
 * Accounts for:
 * - Number of counters
 * - GPU topology dimensions (SE x SA x WGP)
 * - Counter data size (8 bytes per reading)
 *
 * @param arch Architecture information
 * @param collection Counter collection context
 * @return Total memory size in bytes, 0 on error
 */
size_t calculate_counter_memory_size(const arch_t *arch, const counter_collection_t *collection);

/**
 * Helper function: Generate CS partial flush packet
 * @param buffer PM4 buffer to append to
 * @param arch Architecture with event configuration
 * @return 0 on success, negative on error
 */
int generate_cs_partial_flush(pm4_buffer_t *buffer, const arch_t *arch);

/**
 * Helper function: Set GRBM to broadcast mode
 * @param buffer PM4 buffer to append to
 * @param arch Architecture with register information
 * @return 0 on success, negative on error
 */
int generate_grbm_broadcast(pm4_buffer_t *buffer, const arch_t *arch);

/**
 * Helper function: Enable/disable performance monitoring
 * @param buffer PM4 buffer to append to
 * @param arch Architecture with register information
 * @param enable_state Perfmon state (0=disable, 1=enable, 2=stop)
 * @param sample_enable Whether to enable sampling (boolean)
 * @return 0 on success, negative on error
 */
int generate_perfmon_enable(pm4_buffer_t *buffer, const arch_t *arch, uint8_t enable_state,
			    bool sample_enable);

/**
 * Helper function: Configure counter selection and control registers
 * @param buffer PM4 buffer to append to
 * @param arch Architecture information
 * @param counter Counter to configure
 * @return 0 on success, negative on error
 */
int generate_counter_config(pm4_buffer_t *buffer, const arch_t *arch,
			    const counter_info_t *counter);

/**
 * Validate counter collection configuration
 * @param arch Architecture information
 * @param collection Counter collection to validate
 * @return 0 if valid, negative error code otherwise
 */
int validate_counter_collection(const arch_t *arch, const counter_collection_t *collection);

#endif /* PACKET_GENERATION_H */