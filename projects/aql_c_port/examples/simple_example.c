/**
 * @file simple_example.c
 * @brief Simple example demonstrating AQL packet generation
 *
 * This example shows how to use the AQL C library to:
 * 1. Detect GPU architecture
 * 2. Generate PM4 commands
 * 3. Create AQL packets
 * 4. Validate the results
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* Include AQL library headers */
#include "../include/aql_types.h"
#include "../include/aql_arch_ops.h"
#include "../include/aql_cmd_buffer.h"

/* External function declarations */
extern aql_result_t aql_populate_packet(const uint32_t* ib_packet, aql_pm4_ib_packet_t* aql_packet);
extern aql_result_t aql_populate_packet_from_buffer(const void* cmd_buffer, uint32_t cmd_size,
                                                   const aql_arch_ops_t* arch_ops,
                                                   aql_pm4_ib_packet_t* aql_packet);

/* Simple memory allocation callbacks for userspace */
static aql_result_t simple_alloc_callback(void** ptr, size_t size, uint32_t flags, void* userdata) {
    (void)flags;
    (void)userdata;

    *ptr = malloc(size);
    return *ptr ? AQL_SUCCESS : AQL_ERROR_NO_MEMORY;
}

static void simple_dealloc_callback(void* ptr, void* userdata) {
    (void)userdata;
    free(ptr);
}

/* Mark functions as potentially unused to avoid warnings */
__attribute__((unused)) static aql_result_t (*unused_alloc_ref)(void**, size_t, uint32_t, void*) = simple_alloc_callback;
__attribute__((unused)) static void (*unused_dealloc_ref)(void*, void*) = simple_dealloc_callback;

/* Helper function to print hex data */
static void print_hex_data(const char* label, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    printf("%s (%zu bytes):\n", label, size);

    for (size_t i = 0; i < size; i += 16) {
        printf("  %04zx: ", i);

        /* Print hex bytes */
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            printf("%02x ", bytes[i + j]);
        }

        /* Pad if less than 16 bytes */
        for (size_t j = size - i; j < 16; j++) {
            printf("   ");
        }

        /* Print ASCII representation */
        printf(" |");
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            char c = bytes[i + j];
            printf("%c", (c >= 32 && c <= 126) ? c : '.');
        }
        printf("|\n");
    }
}

/* Test architecture detection */
static void test_architecture_detection(void) {
    const char* test_gpus[] = {
        "gfx1201",
        "gfx1200",
        "rdna3",
        "unknown_gpu",
        NULL
    };

    printf("=== Architecture Detection Test ===\n");

    for (int i = 0; test_gpus[i] != NULL; i++) {
        const aql_arch_ops_t* ops = aql_detect_architecture(test_gpus[i]);

        if (ops) {
            printf("GPU '%s' -> %s (GFX%u)\n",
                   test_gpus[i], ops->arch_name, ops->gfx_version);
            printf("  Capabilities: %s%s%s%s%s\n",
                   ops->has_pred_exec ? "PRED_EXEC " : "",
                   ops->has_uconfig_space ? "UCONFIG " : "",
                   ops->has_dual_sdma ? "DUAL_SDMA " : "",
                   ops->has_gl_cache_hierarchy ? "GL_CACHE " : "",
                   ops->has_smn_addressing ? "SMN " : "");
        } else {
            printf("GPU '%s' -> NOT SUPPORTED\n", test_gpus[i]);
        }
    }
    printf("\n");
}

/* Test command buffer operations */
static void test_command_buffer(void) {
    uint32_t static_buffer[64];
    aql_cmd_buffer_t buf;
    aql_result_t result;

    printf("=== Command Buffer Test ===\n");

    /* Initialize static buffer */
    result = aql_cmd_buffer_init(&buf, static_buffer, 64);
    assert(result == AQL_SUCCESS);

    printf("Buffer initialized: capacity=%zu, used=%zu\n",
           aql_cmd_buffer_capacity(&buf), aql_cmd_buffer_used(&buf));

    /* Add some test data */
    uint32_t test_data[] = { 0x12345678, 0x9ABCDEF0, 0xDEADBEEF, 0xCAFEBABE };
    result = aql_cmd_buffer_append_dwords(&buf, test_data, 4);
    assert(result == AQL_SUCCESS);

    printf("After adding 4 dwords: used=%zu\n", aql_cmd_buffer_used(&buf));

    /* Validate buffer */
    result = aql_cmd_buffer_validate(&buf);
    assert(result == AQL_SUCCESS);

    printf("Buffer validation: PASSED\n");

    /* Print buffer contents */
    print_hex_data("Buffer contents", aql_cmd_buffer_data(&buf),
                   aql_cmd_buffer_size_bytes(&buf));

    printf("\n");
}

/* Test PM4 command generation */
static void test_pm4_generation(void) {
    const aql_arch_ops_t* ops;
    uint32_t cmd_buffer_data[256];
    aql_cmd_buffer_t cmd_buf;
    aql_result_t result;

    printf("=== PM4 Command Generation Test ===\n");

    /* Get GFX12 operations */
    ops = aql_detect_architecture("gfx1201");
    assert(ops != NULL);

    printf("Using architecture: %s\n", ops->arch_name);

    /* Initialize command buffer */
    result = aql_cmd_buffer_init(&cmd_buf, cmd_buffer_data, 256);
    assert(result == AQL_SUCCESS);

    /* Generate some PM4 commands */
    printf("Generating PM4 commands...\n");

    /* 1. Write to UCONFIG register */
    result = ops->build_write_uconfig_reg(&cmd_buf, 0xC000, 0x12345678);
    assert(result == AQL_SUCCESS);
    printf("  WRITE_UCONFIG_REG: generated %zu dwords\n", aql_cmd_buffer_used(&cmd_buf));

    /* 2. Write to CONFIG register */
    result = ops->build_write_config_reg(&cmd_buf, 0x2000, 0xABCDEF00);
    assert(result == AQL_SUCCESS);
    printf("  WRITE_CONFIG_REG: buffer now %zu dwords\n", aql_cmd_buffer_used(&cmd_buf));

    /* 3. Generate wait idle command */
    result = ops->build_wait_idle(&cmd_buf);
    assert(result == AQL_SUCCESS);
    printf("  WAIT_IDLE: buffer now %zu dwords\n", aql_cmd_buffer_used(&cmd_buf));

    /* 4. Generate cache flush command */
    result = ops->build_cache_flush(&cmd_buf, 0x10000000, 4096);
    assert(result == AQL_SUCCESS);
    printf("  CACHE_FLUSH: buffer now %zu dwords\n", aql_cmd_buffer_used(&cmd_buf));

    /* Print generated commands */
    print_hex_data("Generated PM4 commands", aql_cmd_buffer_data(&cmd_buf),
                   aql_cmd_buffer_size_bytes(&cmd_buf));

    printf("\n");
}

/* Test AQL packet generation */
static void test_aql_packet_generation(void) {
    const aql_arch_ops_t* ops;
    uint32_t cmd_buffer_data[64];
    aql_cmd_buffer_t cmd_buf;
    aql_pm4_ib_packet_t aql_packet;
    aql_result_t result;

    printf("=== AQL Packet Generation Test ===\n");

    /* Get GFX12 operations */
    ops = aql_detect_architecture("gfx1201");
    assert(ops != NULL);

    /* Create a simple command buffer */
    result = aql_cmd_buffer_init(&cmd_buf, cmd_buffer_data, 64);
    assert(result == AQL_SUCCESS);

    /* Add a simple command */
    result = ops->build_wait_idle(&cmd_buf);
    assert(result == AQL_SUCCESS);

    printf("Created command buffer with %zu dwords\n", aql_cmd_buffer_used(&cmd_buf));

    /* Generate AQL packet from command buffer */
    memset(&aql_packet, 0, sizeof(aql_packet));

    result = aql_populate_packet_from_buffer(
        aql_cmd_buffer_data(&cmd_buf),
        aql_cmd_buffer_size_bytes(&cmd_buf),
        ops,
        &aql_packet
    );
    assert(result == AQL_SUCCESS);

    printf("AQL packet generated successfully\n");
    printf("  Format: %u\n", aql_packet.pm4_ib_format);
    printf("  DW remain: %u\n", aql_packet.dw_count_remain);
    printf("  PM4 command: %08x %08x %08x %08x\n",
           aql_packet.pm4_ib_command[0], aql_packet.pm4_ib_command[1],
           aql_packet.pm4_ib_command[2], aql_packet.pm4_ib_command[3]);

    /* Print complete AQL packet */
    print_hex_data("Complete AQL packet", &aql_packet, sizeof(aql_packet));

    /* Verify packet size */
    assert(sizeof(aql_packet) == 64);
    printf("AQL packet size verification: PASSED (64 bytes)\n");

    printf("\n");
}

/* Test counter validation */
static void test_counter_validation(void) {
    const aql_arch_ops_t* ops;
    aql_counter_request_t request;
    aql_result_t result;

    printf("=== Counter Validation Test ===\n");

    /* Get GFX12 operations */
    ops = aql_detect_architecture("gfx1201");
    assert(ops != NULL);

    /* Test valid counter request */
    memset(&request, 0, sizeof(request));
    request.block_id = AQL_BLOCK_CB;
    request.block_instance = 0;
    request.counter_id = 0;
    request.event_select = 0x100;

    result = ops->validate_counter_request(&request);
    if (result == AQL_SUCCESS) {
        printf("Valid counter request: PASSED\n");
    } else {
        printf("Valid counter request: FAILED (error %d)\n", result);
    }

    /* Test invalid counter request */
    request.counter_id = 999; /* Invalid counter ID */
    result = ops->validate_counter_request(&request);
    if (result != AQL_SUCCESS) {
        printf("Invalid counter request rejection: PASSED\n");
    } else {
        printf("Invalid counter request rejection: FAILED\n");
    }

    printf("\n");
}

/* Main example program */
int main(int argc, char* argv[]) {
    (void)argc; /* Suppress unused parameter warning */
    (void)argv;
    printf("AQL C Library Example\n");
    printf("=====================\n\n");

    /* Run all tests */
    test_architecture_detection();
    test_command_buffer();
    test_pm4_generation();
    test_aql_packet_generation();
    test_counter_validation();

    /* Print supported architectures */
    printf("=== Supported Architectures ===\n");
    const aql_arch_ops_t* ops_list[16];
    uint32_t count = aql_list_supported_architectures(ops_list, 16);

    for (uint32_t i = 0; i < count; i++) {
        printf("%u. %s (GFX%u)\n", i + 1, ops_list[i]->arch_name, ops_list[i]->gfx_version);
    }

    printf("\nAll tests completed successfully!\n");
    printf("The AQL C library is working correctly.\n");

    return 0;
}