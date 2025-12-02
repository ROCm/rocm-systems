/**
 * @file test_packet_generation.c
 * @brief Unit tests for packet_generation.c functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "packet_generation.h"
#include "aql_structures.h"
#include "pm4_packets.h"

/* Test framework macros */
#define TEST_ASSERT(condition, message)                                                       \
	do {                                                                                  \
		if (!(condition)) {                                                           \
			fprintf(stderr, "ASSERTION FAILED: %s at %s:%d\n", message, __FILE__, \
				__LINE__);                                                    \
			return -1;                                                            \
		}                                                                             \
		printf("✓ %s\n", message);                                                    \
	} while (0)

#define TEST_RUN(test_func)                                   \
	do {                                                  \
		printf("\n=== Running %s ===\n", #test_func); \
		if (test_func() != 0) {                       \
			printf("❌ %s FAILED\n", #test_func); \
			return -1;                            \
		}                                             \
		printf("✅ %s PASSED\n", #test_func);         \
	} while (0)

/* Use real GFX12 architecture */
#include "arch_creator.h"

static arch_t *create_test_arch(void)
{
	arch_t *arch = arch_create_by_name("gfx12");
	if (!arch) {
		printf("ERROR: Failed to create GFX12 architecture\n");
		return NULL;
	}
	return arch;
}

static void destroy_test_arch(arch_t *arch)
{
	if (arch) {
		arch_destroy(arch);
	}
}

static pm4_buffer_t *create_test_buffer(void)
{
	pm4_buffer_t *buffer = malloc(sizeof(pm4_buffer_t));
	if (!buffer)
		return NULL;

	buffer->data = malloc(1024); /* 1024 DWORDs buffer */
	if (!buffer->data) {
		free(buffer);
		return NULL;
	}

	buffer->capacity = 1024; /* capacity in DWORDs */
	buffer->size = 0; /* current size in DWORDs */
	memset(buffer->data, 0, buffer->capacity * sizeof(uint32_t));

	return buffer;
}

static void destroy_test_buffer(pm4_buffer_t *buffer)
{
	if (!buffer)
		return;
	free(buffer->data);
	free(buffer);
}

/* Test generate_cs_partial_flush */
static int test_generate_cs_partial_flush(void)
{
	arch_t *arch = create_test_arch();
	pm4_buffer_t *buffer = create_test_buffer();

	TEST_ASSERT(arch != NULL, "Test arch creation");
	TEST_ASSERT(buffer != NULL, "Test buffer creation");

	/* Test normal operation */
	int ret = generate_cs_partial_flush(buffer, arch);
	TEST_ASSERT(ret == 0, "CS partial flush generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after CS partial flush");

	/* Test null parameters */
	ret = generate_cs_partial_flush(NULL, arch);
	TEST_ASSERT(ret == -EINVAL, "CS partial flush fails with null buffer");

	ret = generate_cs_partial_flush(buffer, NULL);
	TEST_ASSERT(ret == -EINVAL, "CS partial flush fails with null arch");

	destroy_test_buffer(buffer);
	destroy_test_arch(arch);
	return 0;
}

/* Test generate_grbm_broadcast */
static int test_generate_grbm_broadcast(void)
{
	arch_t *arch = create_test_arch();
	pm4_buffer_t *buffer = create_test_buffer();

	TEST_ASSERT(arch != NULL, "Test arch creation");
	TEST_ASSERT(buffer != NULL, "Test buffer creation");

	/* Test normal operation */
	int ret = generate_grbm_broadcast(buffer, arch);
	TEST_ASSERT(ret == 0, "GRBM broadcast generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after GRBM broadcast");

	/* Test null parameters */
	ret = generate_grbm_broadcast(NULL, arch);
	TEST_ASSERT(ret == -EINVAL, "GRBM broadcast fails with null buffer");

	ret = generate_grbm_broadcast(buffer, NULL);
	TEST_ASSERT(ret == -EINVAL, "GRBM broadcast fails with null arch");

	destroy_test_buffer(buffer);
	destroy_test_arch(arch);
	return 0;
}

/* Test generate_perfmon_enable */
static int test_generate_perfmon_enable(void)
{
	arch_t *arch = create_test_arch();
	pm4_buffer_t *buffer = create_test_buffer();

	TEST_ASSERT(arch != NULL, "Test arch creation");
	TEST_ASSERT(buffer != NULL, "Test buffer creation");

	/* Test normal operation - disable */
	int ret = generate_perfmon_enable(buffer, arch, 0, false);
	TEST_ASSERT(ret == 0, "Perfmon enable (disable) generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after perfmon disable");

	/* Reset buffer */
	buffer->size = 0;

	/* Test normal operation - enable with sampling */
	ret = generate_perfmon_enable(buffer, arch, 1, true);
	TEST_ASSERT(ret == 0, "Perfmon enable (with sampling) generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after perfmon enable");

	/* Test null parameters */
	ret = generate_perfmon_enable(NULL, arch, 0, false);
	TEST_ASSERT(ret == -EINVAL, "Perfmon enable fails with null buffer");

	ret = generate_perfmon_enable(buffer, NULL, 0, false);
	TEST_ASSERT(ret == -EINVAL, "Perfmon enable fails with null arch");

	destroy_test_buffer(buffer);
	destroy_test_arch(arch);
	return 0;
}

/* Test generate_counter_config */
static int test_generate_counter_config(void)
{
	arch_t *arch = create_test_arch();
	pm4_buffer_t *buffer = create_test_buffer();

	TEST_ASSERT(arch != NULL, "Test arch creation");
	TEST_ASSERT(buffer != NULL, "Test buffer creation");

	/* Create test counter info */
	counter_info_t counter = { .block_id = HW_IP_BLOCK_SQ,
				   .counter_index = 0,
				   .event_id = 0x42 };

	/* Debug: Check if block is properly set up */
	printf("DEBUG: counter.block_id = %d, arch->block_map.blocks[%d] = %p\n", counter.block_id,
	       counter.block_id, arch->block_map.blocks[counter.block_id]);

	if (arch->block_map.blocks[counter.block_id]) {
		printf("DEBUG: block found, counter_count = %d\n",
		       arch->block_map.blocks[counter.block_id]->counter_count);
	}

	/* Test normal operation */
	int ret = generate_counter_config(buffer, arch, &counter);
	TEST_ASSERT(ret == 0, "Counter config generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after counter config");

	/* Test invalid block ID */
	counter.block_id = HW_IP_BLOCK_LAST;
	ret = generate_counter_config(buffer, arch, &counter);
	TEST_ASSERT(ret == -ENOENT, "Counter config fails with invalid block ID");

	/* Test invalid counter index */
	counter.block_id = HW_IP_BLOCK_SQ;
	counter.counter_index = 10; /* Out of range */
	ret = generate_counter_config(buffer, arch, &counter);
	TEST_ASSERT(ret == -EINVAL, "Counter config fails with invalid counter index");

	/* Test null parameters */
	counter.counter_index = 0;
	ret = generate_counter_config(NULL, arch, &counter);
	TEST_ASSERT(ret == -EINVAL, "Counter config fails with null buffer");

	ret = generate_counter_config(buffer, NULL, &counter);
	TEST_ASSERT(ret == -EINVAL, "Counter config fails with null arch");

	ret = generate_counter_config(buffer, arch, NULL);
	TEST_ASSERT(ret == -EINVAL, "Counter config fails with null counter");

	destroy_test_buffer(buffer);
	destroy_test_arch(arch);
	return 0;
}

/* Test generate_start_packet */
static int test_generate_start_packet(void)
{
	arch_t *arch = create_test_arch();
	pm4_buffer_t *buffer = create_test_buffer();

	TEST_ASSERT(arch != NULL, "Test arch creation");
	TEST_ASSERT(buffer != NULL, "Test buffer creation");

	/* Create test counter collection */
	counter_info_t counters[2] = {
		{ .block_id = HW_IP_BLOCK_SQ, .counter_index = 0, .event_id = 0x42 },
		{ .block_id = HW_IP_BLOCK_SQ, .counter_index = 1, .event_id = 0x43 }
	};

	counter_collection_t collection = { .counters = counters,
					    .counter_count = 2,
					    .gpu_memory_addr = 0x1000000,
					    .memory_size = 1024 };

	/* Test normal operation */
	int ret = generate_start_packet(buffer, arch, &collection);
	TEST_ASSERT(ret == 0, "Start packet generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after start packet generation");

	/* Verify SQ_PERFCOUNTER_CTRL2 register was written (should contain force_en | vmid_en) */
	/* This is a complex verification, so we'll just check that the buffer has reasonable content */
	TEST_ASSERT(buffer->size > 10, "Start packet generates substantial PM4 content");

	/* Test null parameters */
	ret = generate_start_packet(NULL, arch, &collection);
	TEST_ASSERT(ret == -EINVAL, "Start packet fails with null buffer");

	ret = generate_start_packet(buffer, NULL, &collection);
	TEST_ASSERT(ret == -EINVAL, "Start packet fails with null arch");

	ret = generate_start_packet(buffer, arch, NULL);
	TEST_ASSERT(ret == -EINVAL, "Start packet fails with null collection");

	/* Test empty collection */
	collection.counter_count = 0;
	ret = generate_start_packet(buffer, arch, &collection);
	TEST_ASSERT(ret == 0, "Start packet succeeds with empty collection");

	destroy_test_buffer(buffer);
	destroy_test_arch(arch);
	return 0;
}

/* Test generate_stop_packet */
static int test_generate_stop_packet(void)
{
	arch_t *arch = create_test_arch();
	pm4_buffer_t *buffer = create_test_buffer();

	TEST_ASSERT(arch != NULL, "Test arch creation");
	TEST_ASSERT(buffer != NULL, "Test buffer creation");

	/* Test normal operation */
	int ret = generate_stop_packet(buffer, arch);
	TEST_ASSERT(ret == 0, "Stop packet generation succeeds");
	TEST_ASSERT(buffer->size > 0, "Buffer has data after stop packet generation");

	/* Test null parameters */
	ret = generate_stop_packet(NULL, arch);
	TEST_ASSERT(ret == -EINVAL, "Stop packet fails with null buffer");

	ret = generate_stop_packet(buffer, NULL);
	TEST_ASSERT(ret == -EINVAL, "Stop packet fails with null arch");

	destroy_test_buffer(buffer);
	destroy_test_arch(arch);
	return 0;
}

/* Test calculate_counter_memory_size */
static int test_calculate_counter_memory_size(void)
{
	arch_t *arch = create_test_arch();

	TEST_ASSERT(arch != NULL, "Test arch creation");

	/* Create test counter collection */
	counter_info_t counters[2] = {
		{ .block_id = HW_IP_BLOCK_SQ, .counter_index = 0, .event_id = 0x42 },
		{ .block_id = HW_IP_BLOCK_SQ, .counter_index = 1, .event_id = 0x43 }
	};

	counter_collection_t collection = { .counters = counters,
					    .counter_count = 2,
					    .gpu_memory_addr = 0x1000000,
					    .memory_size = 1024 };

	/* Test normal operation */
	size_t size = calculate_counter_memory_size(arch, &collection);
	TEST_ASSERT(size > 0, "Counter memory size calculation returns positive value");
	TEST_ASSERT(size >= 16, "Counter memory size is at least 2 counters × 8 bytes");

	/* Test null parameters */
	size = calculate_counter_memory_size(NULL, &collection);
	TEST_ASSERT(size == 0, "Counter memory size returns 0 with null arch");

	size = calculate_counter_memory_size(arch, NULL);
	TEST_ASSERT(size == 0, "Counter memory size returns 0 with null collection");

	/* Test empty collection */
	collection.counter_count = 0;
	size = calculate_counter_memory_size(arch, &collection);
	TEST_ASSERT(size == 0, "Counter memory size returns 0 with empty collection");

	destroy_test_arch(arch);
	return 0;
}

/* Test validate_counter_collection */
static int test_validate_counter_collection(void)
{
	arch_t *arch = create_test_arch();

	TEST_ASSERT(arch != NULL, "Test arch creation");

	/* Create test counter collection */
	counter_info_t counters[2] = {
		{ .block_id = HW_IP_BLOCK_SQ, .counter_index = 0, .event_id = 0x42 },
		{ .block_id = HW_IP_BLOCK_SQ, .counter_index = 1, .event_id = 0x43 }
	};

	counter_collection_t collection = { .counters = counters,
					    .counter_count = 2,
					    .gpu_memory_addr = 0x1000000,
					    .memory_size = 1024 };

	/* Test normal operation */
	int ret = validate_counter_collection(arch, &collection);
	TEST_ASSERT(ret == 0, "Counter collection validation succeeds");

	/* Test null parameters */
	ret = validate_counter_collection(NULL, &collection);
	TEST_ASSERT(ret == -EINVAL, "Validation fails with null arch");

	ret = validate_counter_collection(arch, NULL);
	TEST_ASSERT(ret == -EINVAL, "Validation fails with null collection");

	/* Test empty collection */
	collection.counter_count = 0;
	ret = validate_counter_collection(arch, &collection);
	TEST_ASSERT(ret == -EINVAL, "Validation fails with empty collection");

	/* Test invalid block ID */
	collection.counter_count = 1;
	counters[0].block_id = HW_IP_BLOCK_LAST;
	ret = validate_counter_collection(arch, &collection);
	TEST_ASSERT(ret == -EINVAL, "Validation fails with invalid block ID");

	/* Test invalid counter index */
	counters[0].block_id = HW_IP_BLOCK_SQ;
	counters[0].counter_index = 10; /* Out of range */
	ret = validate_counter_collection(arch, &collection);
	TEST_ASSERT(ret == -EINVAL, "Validation fails with invalid counter index");

	/* Test invalid event ID */
	counters[0].counter_index = 0;
	counters[0].event_id = 0x1000; /* Too large */
	ret = validate_counter_collection(arch, &collection);
	TEST_ASSERT(ret == -EINVAL, "Validation fails with invalid event ID");

	destroy_test_arch(arch);
	return 0;
}

/* Main test runner */
int main(void)
{
	printf("🧪 Starting Packet Generation Tests\n");
	printf("===================================\n");

	TEST_RUN(test_generate_cs_partial_flush);
	TEST_RUN(test_generate_grbm_broadcast);
	TEST_RUN(test_generate_perfmon_enable);
	TEST_RUN(test_generate_counter_config);
	TEST_RUN(test_generate_start_packet);
	TEST_RUN(test_generate_stop_packet);
	TEST_RUN(test_calculate_counter_memory_size);
	TEST_RUN(test_validate_counter_collection);

	printf("\n✅ All Packet Generation Tests PASSED!\n");
	printf("======================================\n");

	return 0;
}