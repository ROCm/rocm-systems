/**
 * @file aql_c_dumper.c
 * @brief AQL_C packet dumper for comparison testing
 *
 * This program creates AQL packets using the aql_c interface and dumps them
 * in a standardized format for comparison with aqlprofile_v2 output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <time.h>

// Include aql_c headers
#include "projects/aql_c_port/include/aql_types.h"
#include "projects/aql_c_port/include/aql_pmc_interface.h"
#include "projects/aql_c_port/include/aql_arch_ops.h"

// Output format types
typedef enum {
    OUTPUT_FORMAT_JSON,
    OUTPUT_FORMAT_BINARY,
    OUTPUT_FORMAT_TEXT
} output_format_t;

// Test configuration
typedef struct {
    char* arch_name;
    aql_pmc_event_t* events;
    uint32_t event_count;
    uint32_t max_events;
    output_format_t format;
    char* output_file;
    int verbose;
} test_config_t;

// Memory allocation callback for aql_c
static aql_result_t memory_alloc_callback(void** ptr, size_t size, uint32_t flags, void* userdata) {
    *ptr = malloc(size);
    if (*ptr == NULL) {
        return AQL_ERROR_NO_MEMORY;
    }

    // Clear memory for consistent comparison
    memset(*ptr, 0, size);

    if (((test_config_t*)userdata)->verbose) {
        printf("Allocated %zu bytes at %p (flags: 0x%x)\n", size, *ptr, flags);
    }

    return AQL_SUCCESS;
}

static void memory_dealloc_callback(void* ptr, void* userdata) {
    if (((test_config_t*)userdata)->verbose) {
        printf("Deallocating memory at %p\n", ptr);
    }
    free(ptr);
}

// Parse counter specification: "BLOCK:INSTANCE:EVENT"
static int parse_counter_spec(const char* spec, aql_pmc_event_t* event) {
    char* spec_copy = strdup(spec);
    char* block_name = strtok(spec_copy, ":");
    char* instance_str = strtok(NULL, ":");
    char* event_str = strtok(NULL, ":");

    if (!block_name || !instance_str || !event_str) {
        free(spec_copy);
        return -1;
    }

    // Convert block name to block ID
    aql_block_id_t block_id = AQL_BLOCK_UNKNOWN;
    if (strcmp(block_name, "CPC") == 0) block_id = AQL_BLOCK_CPC;
    else if (strcmp(block_name, "GRBM") == 0) block_id = AQL_BLOCK_GRBM;
    else if (strcmp(block_name, "SQ") == 0) block_id = AQL_BLOCK_SQ;
    else if (strcmp(block_name, "GL1A") == 0) block_id = AQL_BLOCK_GL1A;
    else if (strcmp(block_name, "GL1C") == 0) block_id = AQL_BLOCK_GL1C;
    else if (strcmp(block_name, "GL2A") == 0) block_id = AQL_BLOCK_GL2A;
    else if (strcmp(block_name, "GL2C") == 0) block_id = AQL_BLOCK_GL2C;
    else if (strcmp(block_name, "TCA") == 0) block_id = AQL_BLOCK_TCA;
    else if (strcmp(block_name, "TCC") == 0) block_id = AQL_BLOCK_TCC;
    else {
        fprintf(stderr, "Unknown block name: %s\n", block_name);
        free(spec_copy);
        return -1;
    }

    uint32_t instance = strtoul(instance_str, NULL, 10);
    uint32_t event_id = strtoul(event_str, NULL, 10);

    event->block_id = block_id;
    event->block_instance = instance;
    event->event_id = event_id;
    event->flags = 0;
    event->block_name = block_name; // This will be overwritten by aql_create_counter_event
    event->event_name = NULL;

    free(spec_copy);
    return 0;
}

// Dump packet data in hexadecimal format
static void dump_packet_hex(FILE* fp, const void* data, size_t size, const char* indent) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i += 16) {
        fprintf(fp, "%s", indent);
        for (size_t j = 0; j < 16 && i + j < size; j++) {
            fprintf(fp, "%02x", bytes[i + j]);
            if (j % 4 == 3) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
}

// Decode PM4 packet header
static void decode_pm4_packet(FILE* fp, const aql_pm4_ib_packet_t* packet, const char* packet_name) {
    fprintf(fp, "=== %s PACKET ===\n", packet_name);
    fprintf(fp, "Size: %zu bytes\n", sizeof(*packet));
    fprintf(fp, "Header: 0x%04x\n", packet->header);
    fprintf(fp, "PM4 IB Format: 0x%04x\n", packet->pm4_ib_format);
    fprintf(fp, "DW Count Remain: %u\n", packet->dw_count_remain);
    fprintf(fp, "Completion Signal: 0x%016lx\n", packet->completion_signal);

    fprintf(fp, "PM4 IB Command:\n");
    for (int i = 0; i < AQL_PM4_IB_COMMAND_DWORDS; i++) {
        fprintf(fp, "  [%d]: 0x%08x\n", i, packet->pm4_ib_command[i]);
    }

    fprintf(fp, "Hex Dump:\n");
    dump_packet_hex(fp, packet, sizeof(*packet), "  ");
    fprintf(fp, "\n");
}

// Output packets in text format
static void output_text_format(FILE* fp, const test_config_t* config, const aql_pmc_packets_t* packets) {
    time_t now = time(NULL);
    fprintf(fp, "AQL_C Packet Dump\n");
    fprintf(fp, "Generated: %s", ctime(&now));
    fprintf(fp, "Architecture: %s\n", config->arch_name);
    fprintf(fp, "Event Count: %u\n\n", config->event_count);

    fprintf(fp, "Events:\n");
    for (uint32_t i = 0; i < config->event_count; i++) {
        fprintf(fp, "  [%u] Block: %s, Instance: %u, Event: %u, Flags: 0x%08x\n",
                i, config->events[i].block_name ? config->events[i].block_name : "UNKNOWN",
                config->events[i].block_instance, config->events[i].event_id,
                config->events[i].flags);
    }
    fprintf(fp, "\n");

    decode_pm4_packet(fp, &packets->start_packet, "START");
    decode_pm4_packet(fp, &packets->stop_packet, "STOP");
    decode_pm4_packet(fp, &packets->read_packet, "READ");

    if (packets->command_buffer && packets->command_buffer_size > 0) {
        fprintf(fp, "=== COMMAND BUFFER ===\n");
        fprintf(fp, "Size: %zu bytes\n", packets->command_buffer_size);
        fprintf(fp, "Hex Dump:\n");
        dump_packet_hex(fp, packets->command_buffer, packets->command_buffer_size, "  ");
        fprintf(fp, "\n");
    }
}

// Output packets in JSON format
static void output_json_format(FILE* fp, const test_config_t* config, const aql_pmc_packets_t* packets) {
    fprintf(fp, "{\n");
    fprintf(fp, "  \"tool\": \"aql_c_dumper\",\n");
    fprintf(fp, "  \"timestamp\": %ld,\n", time(NULL));
    fprintf(fp, "  \"architecture\": \"%s\",\n", config->arch_name);
    fprintf(fp, "  \"event_count\": %u,\n", config->event_count);

    fprintf(fp, "  \"events\": [\n");
    for (uint32_t i = 0; i < config->event_count; i++) {
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"index\": %u,\n", i);
        fprintf(fp, "      \"block_name\": \"%s\",\n",
                config->events[i].block_name ? config->events[i].block_name : "UNKNOWN");
        fprintf(fp, "      \"block_instance\": %u,\n", config->events[i].block_instance);
        fprintf(fp, "      \"event_id\": %u,\n", config->events[i].event_id);
        fprintf(fp, "      \"flags\": \"0x%08x\"\n", config->events[i].flags);
        fprintf(fp, "    }%s\n", (i < config->event_count - 1) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    // Start packet
    fprintf(fp, "  \"start_packet\": {\n");
    fprintf(fp, "    \"size\": %zu,\n", sizeof(packets->start_packet));
    fprintf(fp, "    \"header\": \"0x%04x\",\n", packets->start_packet.header);
    fprintf(fp, "    \"pm4_ib_format\": \"0x%04x\",\n", packets->start_packet.pm4_ib_format);
    fprintf(fp, "    \"dw_count_remain\": %u,\n", packets->start_packet.dw_count_remain);
    fprintf(fp, "    \"completion_signal\": \"0x%016lx\",\n", packets->start_packet.completion_signal);
    fprintf(fp, "    \"pm4_ib_command\": [");
    for (int i = 0; i < AQL_PM4_IB_COMMAND_DWORDS; i++) {
        fprintf(fp, "\"0x%08x\"%s", packets->start_packet.pm4_ib_command[i],
                (i < AQL_PM4_IB_COMMAND_DWORDS - 1) ? ", " : "");
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  },\n");

    // Stop packet (similar structure)
    fprintf(fp, "  \"stop_packet\": {\n");
    fprintf(fp, "    \"size\": %zu,\n", sizeof(packets->stop_packet));
    fprintf(fp, "    \"header\": \"0x%04x\",\n", packets->stop_packet.header);
    fprintf(fp, "    \"pm4_ib_format\": \"0x%04x\",\n", packets->stop_packet.pm4_ib_format);
    fprintf(fp, "    \"dw_count_remain\": %u,\n", packets->stop_packet.dw_count_remain);
    fprintf(fp, "    \"completion_signal\": \"0x%016lx\",\n", packets->stop_packet.completion_signal);
    fprintf(fp, "    \"pm4_ib_command\": [");
    for (int i = 0; i < AQL_PM4_IB_COMMAND_DWORDS; i++) {
        fprintf(fp, "\"0x%08x\"%s", packets->stop_packet.pm4_ib_command[i],
                (i < AQL_PM4_IB_COMMAND_DWORDS - 1) ? ", " : "");
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  },\n");

    // Read packet
    fprintf(fp, "  \"read_packet\": {\n");
    fprintf(fp, "    \"size\": %zu,\n", sizeof(packets->read_packet));
    fprintf(fp, "    \"header\": \"0x%04x\",\n", packets->read_packet.header);
    fprintf(fp, "    \"pm4_ib_format\": \"0x%04x\",\n", packets->read_packet.pm4_ib_format);
    fprintf(fp, "    \"dw_count_remain\": %u,\n", packets->read_packet.dw_count_remain);
    fprintf(fp, "    \"completion_signal\": \"0x%016lx\",\n", packets->read_packet.completion_signal);
    fprintf(fp, "    \"pm4_ib_command\": [");
    for (int i = 0; i < AQL_PM4_IB_COMMAND_DWORDS; i++) {
        fprintf(fp, "\"0x%08x\"%s", packets->read_packet.pm4_ib_command[i],
                (i < AQL_PM4_IB_COMMAND_DWORDS - 1) ? ", " : "");
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  },\n");

    // Command buffer
    fprintf(fp, "  \"command_buffer\": {\n");
    fprintf(fp, "    \"size\": %zu\n", packets->command_buffer_size);
    if (packets->command_buffer && packets->command_buffer_size > 0) {
        fprintf(fp, "    ,\"present\": true\n");
    } else {
        fprintf(fp, "    ,\"present\": false\n");
    }
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");
}

// Output packets in binary format
static void output_binary_format(FILE* fp, const test_config_t* config, const aql_pmc_packets_t* packets) {
    // Binary format header
    uint32_t magic = 0x41514C43; // "AQLC"
    uint32_t version = 1;
    uint32_t arch_len = strlen(config->arch_name);

    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&arch_len, sizeof(arch_len), 1, fp);
    fwrite(config->arch_name, arch_len, 1, fp);
    fwrite(&config->event_count, sizeof(config->event_count), 1, fp);

    // Write events
    for (uint32_t i = 0; i < config->event_count; i++) {
        fwrite(&config->events[i], sizeof(aql_pmc_event_t), 1, fp);
    }

    // Write packets
    fwrite(&packets->start_packet, sizeof(packets->start_packet), 1, fp);
    fwrite(&packets->stop_packet, sizeof(packets->stop_packet), 1, fp);
    fwrite(&packets->read_packet, sizeof(packets->read_packet), 1, fp);

    // Write command buffer
    fwrite(&packets->command_buffer_size, sizeof(packets->command_buffer_size), 1, fp);
    if (packets->command_buffer && packets->command_buffer_size > 0) {
        fwrite(packets->command_buffer, packets->command_buffer_size, 1, fp);
    }
}

static void usage(const char* prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  --arch ARCH         GPU architecture (e.g., gfx942, gfx1101)\n");
    printf("  --counter SPEC      Counter specification: BLOCK:INSTANCE:EVENT\n");
    printf("                      Can be specified multiple times\n");
    printf("  --format FORMAT     Output format: json, binary, text (default: json)\n");
    printf("  --output FILE       Output file (default: stdout)\n");
    printf("  --verbose           Enable verbose output\n");
    printf("  --help              Show this help\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --arch gfx942 --counter CPC:0:123 --output test.json\n", prog_name);
    printf("  %s --arch gfx1101 --counter CPC:0:123 --counter GRBM:0:456 --format text\n", prog_name);
}

int main(int argc, char* argv[]) {
    test_config_t config = {0};
    config.max_events = 16;
    config.events = malloc(config.max_events * sizeof(aql_pmc_event_t));
    config.format = OUTPUT_FORMAT_JSON;

    static struct option long_options[] = {
        {"arch", required_argument, 0, 'a'},
        {"counter", required_argument, 0, 'c'},
        {"format", required_argument, 0, 'f'},
        {"output", required_argument, 0, 'o'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "a:c:f:o:vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'a':
                config.arch_name = strdup(optarg);
                break;
            case 'c':
                if (config.event_count >= config.max_events) {
                    fprintf(stderr, "Too many counters specified (max %u)\n", config.max_events);
                    return 1;
                }
                if (parse_counter_spec(optarg, &config.events[config.event_count]) != 0) {
                    fprintf(stderr, "Invalid counter specification: %s\n", optarg);
                    return 1;
                }
                config.event_count++;
                break;
            case 'f':
                if (strcmp(optarg, "json") == 0) config.format = OUTPUT_FORMAT_JSON;
                else if (strcmp(optarg, "binary") == 0) config.format = OUTPUT_FORMAT_BINARY;
                else if (strcmp(optarg, "text") == 0) config.format = OUTPUT_FORMAT_TEXT;
                else {
                    fprintf(stderr, "Unknown format: %s\n", optarg);
                    return 1;
                }
                break;
            case 'o':
                config.output_file = strdup(optarg);
                break;
            case 'v':
                config.verbose = 1;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (!config.arch_name) {
        fprintf(stderr, "Architecture must be specified\n");
        usage(argv[0]);
        return 1;
    }

    if (config.event_count == 0) {
        fprintf(stderr, "At least one counter must be specified\n");
        usage(argv[0]);
        return 1;
    }

    if (config.verbose) {
        printf("Architecture: %s\n", config.arch_name);
        printf("Event count: %u\n", config.event_count);
        for (uint32_t i = 0; i < config.event_count; i++) {
            printf("  Event %u: Block %u, Instance %u, Event %u\n",
                   i, config.events[i].block_id, config.events[i].block_instance, config.events[i].event_id);
        }
    }

    // Create profile configuration
    aql_pmc_profile_t profile = {
        .arch_name = config.arch_name,
        .events = config.events,
        .event_count = config.event_count,
        .output_buffer = malloc(4096), // 4KB buffer for results
        .output_buffer_size = 4096
    };

    // Create AQL packets
    aql_pmc_packets_t packets;
    aql_result_t result = aql_pmc_create_packets(&packets, &profile);
    if (result != AQL_SUCCESS) {
        fprintf(stderr, "Failed to create AQL packets: %d\n", result);
        return 1;
    }

    // Open output file
    FILE* fp = stdout;
    if (config.output_file) {
        fp = fopen(config.output_file, (config.format == OUTPUT_FORMAT_BINARY) ? "wb" : "w");
        if (!fp) {
            fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
            return 1;
        }
    }

    // Output packets in requested format
    switch (config.format) {
        case OUTPUT_FORMAT_JSON:
            output_json_format(fp, &config, &packets);
            break;
        case OUTPUT_FORMAT_BINARY:
            output_binary_format(fp, &config, &packets);
            break;
        case OUTPUT_FORMAT_TEXT:
            output_text_format(fp, &config, &packets);
            break;
    }

    if (fp != stdout) {
        fclose(fp);
    }

    // Cleanup
    aql_pmc_delete_packets(&packets);
    free(profile.output_buffer);
    free(config.events);
    free(config.arch_name);
    free(config.output_file);

    if (config.verbose) {
        printf("Packet dump completed successfully\n");
    }

    return 0;
}