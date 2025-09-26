/**
 * @file packet_gen_tool.c
 * @brief Tool to generate start/stop/read packets for performance counters
 *
 * Usage: packet_gen_tool <gfx_name> <output_address> <counter1> [counter2] ...
 * Example: packet_gen_tool gfx12 0x1000000 SQ:0:0x42 SQ:1:0x43
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include "arch_creator.h"
#include "packet_generation.h"
#include "aql_structures.h"
#include "counter_registry.h"

#define MAX_COUNTERS 16
#define MAX_BUFFER_SIZE 4096

typedef struct {
    char *gfx_name;
    uint64_t output_address;
    counter_info_t counters[MAX_COUNTERS];
    size_t counter_count;
} tool_config_t;

static void print_usage(const char *program_name) {
    printf("Usage: %s <gfx_name> <output_address_hex> <counter1> [counter2] ...\n", program_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  gfx_name         : GPU architecture name (e.g., gfx12)\n");
    printf("  output_address   : Output memory address in hex (e.g., 0x1000000)\n");
    printf("  counterN         : Counter name (e.g., SQ_WAVES, GL2C_HIT) or\n");
    printf("                     BLOCK:INDEX:EVENT_ID format (e.g., SQ:0:0x42)\n");
    printf("\n");
    printf("Available counter names: SQ_WAVES, SQ_BUSY_CYCLES, GL2C_HIT, GL2C_MISS,\n");
    printf("                         TA_TA_BUSY, GRBM_COUNT, GRBM_GUI_ACTIVE, etc.\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s gfx12 0x1000000 SQ_WAVES GL2C_HIT\n", program_name);
    printf("  %s gfx12 0x1000000 SQ:0:0x42 SQ:1:0x43\n", program_name);
}

static int parse_block_name(const char *block_name) {
    if (strcmp(block_name, "ATC") == 0) return HW_IP_BLOCK_ATC;
    if (strcmp(block_name, "CPC") == 0) return HW_IP_BLOCK_CPC;
    if (strcmp(block_name, "CPF") == 0) return HW_IP_BLOCK_CPF;
    if (strcmp(block_name, "CPG") == 0) return HW_IP_BLOCK_CPG;
    if (strcmp(block_name, "DB") == 0) return HW_IP_BLOCK_DB;
    if (strcmp(block_name, "EA") == 0) return HW_IP_BLOCK_EA;
    if (strcmp(block_name, "GDS") == 0) return HW_IP_BLOCK_GDS;
    if (strcmp(block_name, "GRBM") == 0) return HW_IP_BLOCK_GRBM;
    if (strcmp(block_name, "GL2C") == 0) return HW_IP_BLOCK_GL2C;
    if (strcmp(block_name, "PA_SC") == 0) return HW_IP_BLOCK_PA_SC;
    if (strcmp(block_name, "PA_SU") == 0) return HW_IP_BLOCK_PA_SU;
    if (strcmp(block_name, "RMI") == 0) return HW_IP_BLOCK_RMI;
    if (strcmp(block_name, "SPI") == 0) return HW_IP_BLOCK_SPI;
    if (strcmp(block_name, "SQ") == 0) return HW_IP_BLOCK_SQ;
    if (strcmp(block_name, "SX") == 0) return HW_IP_BLOCK_SX;
    if (strcmp(block_name, "TA") == 0) return HW_IP_BLOCK_TA;
    if (strcmp(block_name, "TCP") == 0) return HW_IP_BLOCK_TCP;
    if (strcmp(block_name, "TD") == 0) return HW_IP_BLOCK_TD;
    if (strcmp(block_name, "TCA") == 0) return HW_IP_BLOCK_TCA;
    if (strcmp(block_name, "TCC") == 0) return HW_IP_BLOCK_TCC;
    if (strcmp(block_name, "UMC") == 0) return HW_IP_BLOCK_UMC;
    if (strcmp(block_name, "SDMA") == 0) return HW_IP_BLOCK_SDMA;
    return -1;
}

static int parse_counter_spec(const char *spec, counter_info_t *counter) {
    char *spec_copy = strdup(spec);
    if (!spec_copy) return -ENOMEM;

    char *block_str = strtok(spec_copy, ":");
    char *index_str = strtok(NULL, ":");
    char *event_str = strtok(NULL, ":");

    if (!block_str || !index_str || !event_str) {
        free(spec_copy);
        return -EINVAL;
    }

    int block_id = parse_block_name(block_str);
    if (block_id < 0) {
        free(spec_copy);
        return -EINVAL;
    }

    counter->block_id = block_id;
    counter->counter_index = strtoul(index_str, NULL, 10);
    counter->event_id = strtoul(event_str, NULL, 16);

    free(spec_copy);
    return 0;
}

static int parse_counter_name(const char *counter_name, const arch_t *arch, counter_info_t *counter) {
    if (!counter_name || !arch || !counter) return -EINVAL;

    /* Try to look up counter by name first */
    const counter_def_t *counter_def = lookup_counter_by_name(counter_name);
    if (counter_def) {
        /* Found by name - get the event ID from architecture */
        uint32_t event_id = lookup_event_id(counter_def, arch);
        if (event_id == 0 && counter_def->id != COUNTER_GRBM_COUNT) {
            printf("Warning: No event mapping found for counter '%s' in architecture\n", counter_name);
            return -ENOENT;
        }

        counter->block_id = counter_def->hw_block;
        counter->counter_index = 0;  /* Default to first available counter in the block */
        counter->event_id = event_id;
        return 0;
    }

    /* If not found by name, fall back to the old BLOCK:INDEX:EVENT format */
    return parse_counter_spec(counter_name, counter);
}

static int parse_arguments(int argc, char *argv[], tool_config_t *config) {
    if (argc < 4) {
        return -EINVAL;
    }

    config->gfx_name = argv[1];
    config->output_address = strtoull(argv[2], NULL, 16);
    config->counter_count = 0;

    /* Store the counter arguments for later parsing with architecture */
    for (int i = 3; i < argc && config->counter_count < MAX_COUNTERS; i++) {
        config->counter_count++;
    }

    return 0;
}

static int parse_counters_with_arch(int argc, char *argv[], tool_config_t *config, const arch_t *arch) {
    config->counter_count = 0;

    for (int i = 3; i < argc && config->counter_count < MAX_COUNTERS; i++) {
        if (parse_counter_name(argv[i], arch, &config->counters[config->counter_count]) == 0) {
            config->counter_count++;
        } else {
            fprintf(stderr, "Error: Invalid counter specification: %s\n", argv[i]);
            fprintf(stderr, "       Use counter name (e.g., SQ_WAVES) or BLOCK:INDEX:EVENT format (e.g., SQ:0:0x42)\n");
            return -EINVAL;
        }
    }

    if (config->counter_count == 0) {
        fprintf(stderr, "Error: No valid counters specified\n");
        return -EINVAL;
    }

    return 0;
}

static void print_pm4_buffer(const char *packet_type, pm4_buffer_t *buffer) {
    printf("=== %s PACKET ===\n", packet_type);
    printf("Size: %zu DWORDs (%zu bytes)\n", buffer->size, buffer->size * 4);

    for (size_t i = 0; i < buffer->size; i++) {
        printf("0x%08x", buffer->data[i]);
        if (i < buffer->size - 1) {
            printf(" ");
        }
        if ((i + 1) % 8 == 0) {
            printf("\n");
        }
    }
    if (buffer->size % 8 != 0) {
        printf("\n");
    }
    printf("\n");
}

static pm4_buffer_t* create_buffer(void) {
    pm4_buffer_t* buffer = malloc(sizeof(pm4_buffer_t));
    if (!buffer) return NULL;

    buffer->data = malloc(MAX_BUFFER_SIZE * sizeof(uint32_t));
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }

    buffer->capacity = MAX_BUFFER_SIZE;
    buffer->size = 0;
    memset(buffer->data, 0, buffer->capacity * sizeof(uint32_t));

    return buffer;
}

static void destroy_buffer(pm4_buffer_t* buffer) {
    if (!buffer) return;
    free(buffer->data);
    free(buffer);
}

int main(int argc, char *argv[]) {
    tool_config_t config;
    arch_t *arch = NULL;
    pm4_buffer_t *start_buffer = NULL, *read_buffer = NULL, *stop_buffer = NULL;
    int ret = 0;

    if (parse_arguments(argc, argv, &config) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    printf("Packet Generation Tool\n");
    printf("======================\n");
    printf("Architecture: %s\n", config.gfx_name);
    printf("Output Address: 0x%lx\n", config.output_address);

    /* Create architecture */
    arch = arch_create_by_name(config.gfx_name);
    if (!arch) {
        fprintf(stderr, "Error: Failed to create architecture '%s'\n", config.gfx_name);
        ret = 1;
        goto cleanup;
    }

    /* Parse counters with architecture available */
    if (parse_counters_with_arch(argc, argv, &config, arch) != 0) {
        ret = 1;
        goto cleanup;
    }

    printf("Counter Count: %zu\n", config.counter_count);
    for (size_t i = 0; i < config.counter_count; i++) {
        printf("  Counter %zu: Block %d, Index %d, Event 0x%x\n", i,
               config.counters[i].block_id,
               config.counters[i].counter_index,
               config.counters[i].event_id);
    }
    printf("\n");

    /* Create counter collection */
    counter_collection_t collection = {
        .counters = config.counters,
        .counter_count = config.counter_count,
        .gpu_memory_addr = config.output_address,
        .memory_size = calculate_counter_memory_size(arch, &(counter_collection_t){
            .counters = config.counters,
            .counter_count = config.counter_count
        })
    };

    printf("Required memory size: %zu bytes\n\n", collection.memory_size);

    /* Validate counter collection */
    ret = validate_counter_collection(arch, &collection);
    if (ret != 0) {
        fprintf(stderr, "Error: Counter collection validation failed: %s\n", strerror(-ret));
        ret = 1;
        goto cleanup;
    }

    /* Create buffers */
    start_buffer = create_buffer();
    read_buffer = create_buffer();
    stop_buffer = create_buffer();

    if (!start_buffer || !read_buffer || !stop_buffer) {
        fprintf(stderr, "Error: Failed to create PM4 buffers\n");
        ret = 1;
        goto cleanup;
    }

    /* Generate START packet */
    ret = generate_start_packet(start_buffer, arch, &collection);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to generate start packet: %s\n", strerror(-ret));
        ret = 1;
        goto cleanup;
    }

    /* Generate READ packet */
    ret = generate_read_packet(read_buffer, arch, &collection);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to generate read packet: %s\n", strerror(-ret));
        ret = 1;
        goto cleanup;
    }

    /* Generate STOP packet */
    ret = generate_stop_packet(stop_buffer, arch);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to generate stop packet: %s\n", strerror(-ret));
        ret = 1;
        goto cleanup;
    }

    /* Print results */
    print_pm4_buffer("START", start_buffer);
    print_pm4_buffer("READ", read_buffer);
    print_pm4_buffer("STOP", stop_buffer);

cleanup:
    if (start_buffer) destroy_buffer(start_buffer);
    if (read_buffer) destroy_buffer(read_buffer);
    if (stop_buffer) destroy_buffer(stop_buffer);
    if (arch) arch_destroy(arch);

    return ret;
}