/**
 * @file aqlprofile_v2_dumper.c
 * @brief AQLProfile v2 packet dumper for comparison testing
 *
 * This program creates AQL packets using the aqlprofile v2 interface and dumps them
 * in the same standardized format as aql_c_dumper for direct comparison.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <time.h>

// Include aqlprofile v2 headers
#include "projects/aqlprofile/src/core/include/aqlprofile-sdk/aql_profile_v2.h"
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

// Output format types
typedef enum {
    OUTPUT_FORMAT_JSON,
    OUTPUT_FORMAT_BINARY,
    OUTPUT_FORMAT_TEXT
} output_format_t;

// Test configuration
typedef struct {
    char* arch_name;
    aqlprofile_pmc_event_t* events;
    uint32_t event_count;
    uint32_t max_events;
    output_format_t format;
    char* output_file;
    int verbose;
} test_config_t;

// Global handles for cleanup
static aqlprofile_agent_handle_t g_agent_handle = {0};
static aqlprofile_handle_t g_profile_handle = {0};

// Memory allocation callback for aqlprofile v2
static hsa_status_t memory_alloc_callback(void** ptr, uint64_t size,
                                         aqlprofile_buffer_desc_flags_t flags,
                                         void* userdata) {
    *ptr = malloc(size);
    if (*ptr == NULL) {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    // Clear memory for consistent comparison
    memset(*ptr, 0, size);

    if (((test_config_t*)userdata)->verbose) {
        printf("Allocated %lu bytes at %p (flags: 0x%x)\n", size, *ptr, flags.raw);
    }

    return HSA_STATUS_SUCCESS;
}

static void memory_dealloc_callback(void* ptr, void* userdata) {
    if (((test_config_t*)userdata)->verbose) {
        printf("Deallocating memory at %p\n", ptr);
    }
    free(ptr);
}

static hsa_status_t memory_copy_callback(void* dst, const void* src, size_t size, void* userdata) {
    memcpy(dst, src, size);
    return HSA_STATUS_SUCCESS;
}

// Convert block name string to HSA block enum
static hsa_ven_amd_aqlprofile_block_name_t string_to_block_name(const char* block_name) {
    if (strcmp(block_name, "CPC") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_CPC;
    if (strcmp(block_name, "GRBM") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_GRBM;
    if (strcmp(block_name, "SQ") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_SQ;
    if (strcmp(block_name, "TA") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_TA;
    if (strcmp(block_name, "TD") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_TD;
    if (strcmp(block_name, "TCP") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCP;
    if (strcmp(block_name, "TCC") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCC;
    if (strcmp(block_name, "TCA") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCA;
    if (strcmp(block_name, "DB") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_DB;
    if (strcmp(block_name, "CB") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_CB;
    if (strcmp(block_name, "GDS") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_GDS;
    if (strcmp(block_name, "SPI") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_SPI;
    if (strcmp(block_name, "SX") == 0) return HSA_VEN_AMD_AQLPROFILE_BLOCKS_SX;
    // Add more blocks as needed
    return HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER; // Invalid
}

// Convert HSA block enum back to string
static const char* block_name_to_string(hsa_ven_amd_aqlprofile_block_name_t block_name) {
    switch (block_name) {
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_CPC: return "CPC";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_GRBM: return "GRBM";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_SQ: return "SQ";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_TA: return "TA";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_TD: return "TD";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCP: return "TCP";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCC: return "TCC";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCA: return "TCA";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_DB: return "DB";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_CB: return "CB";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_GDS: return "GDS";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_SPI: return "SPI";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_SX: return "SX";
        default: return "UNKNOWN";
    }
}

// Parse counter specification: "BLOCK:INSTANCE:EVENT"
static int parse_counter_spec(const char* spec, aqlprofile_pmc_event_t* event) {
    char* spec_copy = strdup(spec);
    char* block_name = strtok(spec_copy, ":");
    char* instance_str = strtok(NULL, ":");
    char* event_str = strtok(NULL, ":");

    if (!block_name || !instance_str || !event_str) {
        free(spec_copy);
        return -1;
    }

    hsa_ven_amd_aqlprofile_block_name_t block_id = string_to_block_name(block_name);
    if (block_id == HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER) {
        fprintf(stderr, "Unknown block name: %s\n", block_name);
        free(spec_copy);
        return -1;
    }

    uint32_t instance = strtoul(instance_str, NULL, 10);
    uint32_t event_id = strtoul(event_str, NULL, 10);

    event->block_index = instance;
    event->event_id = event_id;
    event->flags.raw = 0;
    event->block_name = block_id;

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
static void decode_pm4_packet(FILE* fp, const hsa_ext_amd_aql_pm4_packet_t* packet, const char* packet_name) {
    fprintf(fp, "=== %s PACKET ===\n", packet_name);
    fprintf(fp, "Size: %zu bytes\n", sizeof(*packet));
    fprintf(fp, "Header: 0x%04x\n", packet->header);
    fprintf(fp, "Type: 0x%04x\n", packet->type);
    fprintf(fp, "Source: 0x%08x\n", packet->source);
    fprintf(fp, "Format: 0x%08x\n", packet->format);
    fprintf(fp, "Control: 0x%08x\n", packet->control);
    fprintf(fp, "Address: 0x%016lx\n", packet->address);
    fprintf(fp, "Size: 0x%08x\n", packet->size);
    fprintf(fp, "Connection: 0x%08x\n", packet->connection);
    fprintf(fp, "Completion Signal: 0x%016lx\n", packet->completion_signal);

    fprintf(fp, "Hex Dump:\n");
    dump_packet_hex(fp, packet, sizeof(*packet), "  ");
    fprintf(fp, "\n");
}

// Output packets in text format
static void output_text_format(FILE* fp, const test_config_t* config, const aqlprofile_pmc_aql_packets_t* packets) {
    time_t now = time(NULL);
    fprintf(fp, "AQLProfile v2 Packet Dump\n");
    fprintf(fp, "Generated: %s", ctime(&now));
    fprintf(fp, "Architecture: %s\n", config->arch_name);
    fprintf(fp, "Event Count: %u\n\n", config->event_count);

    fprintf(fp, "Events:\n");
    for (uint32_t i = 0; i < config->event_count; i++) {
        fprintf(fp, "  [%u] Block: %s, Instance: %u, Event: %u, Flags: 0x%08x\n",
                i, block_name_to_string(config->events[i].block_name),
                config->events[i].block_index, config->events[i].event_id,
                config->events[i].flags.raw);
    }
    fprintf(fp, "\n");

    decode_pm4_packet(fp, &packets->start_packet, "START");
    decode_pm4_packet(fp, &packets->stop_packet, "STOP");
    decode_pm4_packet(fp, &packets->read_packet, "READ");
}

// Output packets in JSON format
static void output_json_format(FILE* fp, const test_config_t* config, const aqlprofile_pmc_aql_packets_t* packets) {
    fprintf(fp, "{\n");
    fprintf(fp, "  \"tool\": \"aqlprofile_v2_dumper\",\n");
    fprintf(fp, "  \"timestamp\": %ld,\n", time(NULL));
    fprintf(fp, "  \"architecture\": \"%s\",\n", config->arch_name);
    fprintf(fp, "  \"event_count\": %u,\n", config->event_count);

    fprintf(fp, "  \"events\": [\n");
    for (uint32_t i = 0; i < config->event_count; i++) {
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"index\": %u,\n", i);
        fprintf(fp, "      \"block_name\": \"%s\",\n", block_name_to_string(config->events[i].block_name));
        fprintf(fp, "      \"block_instance\": %u,\n", config->events[i].block_index);
        fprintf(fp, "      \"event_id\": %u,\n", config->events[i].event_id);
        fprintf(fp, "      \"flags\": \"0x%08x\"\n", config->events[i].flags.raw);
        fprintf(fp, "    }%s\n", (i < config->event_count - 1) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    // Start packet
    fprintf(fp, "  \"start_packet\": {\n");
    fprintf(fp, "    \"size\": %zu,\n", sizeof(packets->start_packet));
    fprintf(fp, "    \"header\": \"0x%04x\",\n", packets->start_packet.header);
    fprintf(fp, "    \"type\": \"0x%04x\",\n", packets->start_packet.type);
    fprintf(fp, "    \"source\": \"0x%08x\",\n", packets->start_packet.source);
    fprintf(fp, "    \"format\": \"0x%08x\",\n", packets->start_packet.format);
    fprintf(fp, "    \"control\": \"0x%08x\",\n", packets->start_packet.control);
    fprintf(fp, "    \"address\": \"0x%016lx\",\n", packets->start_packet.address);
    fprintf(fp, "    \"size_field\": \"0x%08x\",\n", packets->start_packet.size);
    fprintf(fp, "    \"connection\": \"0x%08x\",\n", packets->start_packet.connection);
    fprintf(fp, "    \"completion_signal\": \"0x%016lx\"\n", packets->start_packet.completion_signal);
    fprintf(fp, "  },\n");

    // Stop packet
    fprintf(fp, "  \"stop_packet\": {\n");
    fprintf(fp, "    \"size\": %zu,\n", sizeof(packets->stop_packet));
    fprintf(fp, "    \"header\": \"0x%04x\",\n", packets->stop_packet.header);
    fprintf(fp, "    \"type\": \"0x%04x\",\n", packets->stop_packet.type);
    fprintf(fp, "    \"source\": \"0x%08x\",\n", packets->stop_packet.source);
    fprintf(fp, "    \"format\": \"0x%08x\",\n", packets->stop_packet.format);
    fprintf(fp, "    \"control\": \"0x%08x\",\n", packets->stop_packet.control);
    fprintf(fp, "    \"address\": \"0x%016lx\",\n", packets->stop_packet.address);
    fprintf(fp, "    \"size_field\": \"0x%08x\",\n", packets->stop_packet.size);
    fprintf(fp, "    \"connection\": \"0x%08x\",\n", packets->stop_packet.connection);
    fprintf(fp, "    \"completion_signal\": \"0x%016lx\"\n", packets->stop_packet.completion_signal);
    fprintf(fp, "  },\n");

    // Read packet
    fprintf(fp, "  \"read_packet\": {\n");
    fprintf(fp, "    \"size\": %zu,\n", sizeof(packets->read_packet));
    fprintf(fp, "    \"header\": \"0x%04x\",\n", packets->read_packet.header);
    fprintf(fp, "    \"type\": \"0x%04x\",\n", packets->read_packet.type);
    fprintf(fp, "    \"source\": \"0x%08x\",\n", packets->read_packet.source);
    fprintf(fp, "    \"format\": \"0x%08x\",\n", packets->read_packet.format);
    fprintf(fp, "    \"control\": \"0x%08x\",\n", packets->read_packet.control);
    fprintf(fp, "    \"address\": \"0x%016lx\",\n", packets->read_packet.address);
    fprintf(fp, "    \"size_field\": \"0x%08x\",\n", packets->read_packet.size);
    fprintf(fp, "    \"connection\": \"0x%08x\",\n", packets->read_packet.connection);
    fprintf(fp, "    \"completion_signal\": \"0x%016lx\"\n", packets->read_packet.completion_signal);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"command_buffer\": {\n");
    fprintf(fp, "    \"note\": \"Command buffer content embedded in packets for AQLProfile v2\"\n");
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");
}

// Output packets in binary format
static void output_binary_format(FILE* fp, const test_config_t* config, const aqlprofile_pmc_aql_packets_t* packets) {
    // Binary format header
    uint32_t magic = 0x41514C56; // "AQLV" (for AQLProfile v2)
    uint32_t version = 1;
    uint32_t arch_len = strlen(config->arch_name);

    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&arch_len, sizeof(arch_len), 1, fp);
    fwrite(config->arch_name, arch_len, 1, fp);
    fwrite(&config->event_count, sizeof(config->event_count), 1, fp);

    // Write events
    for (uint32_t i = 0; i < config->event_count; i++) {
        fwrite(&config->events[i], sizeof(aqlprofile_pmc_event_t), 1, fp);
    }

    // Write packets
    fwrite(&packets->start_packet, sizeof(packets->start_packet), 1, fp);
    fwrite(&packets->stop_packet, sizeof(packets->stop_packet), 1, fp);
    fwrite(&packets->read_packet, sizeof(packets->read_packet), 1, fp);
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
    printf("  %s --arch gfx942 --counter CPC:0:123 --output test_ref.json\n", prog_name);
    printf("  %s --arch gfx1101 --counter CPC:0:123 --counter GRBM:0:456 --format text\n", prog_name);
}

// Cleanup function
static void cleanup() {
    if (g_profile_handle.handle != 0) {
        aqlprofile_pmc_delete_packets(g_profile_handle);
    }
}

int main(int argc, char* argv[]) {
    atexit(cleanup);

    test_config_t config = {0};
    config.max_events = 16;
    config.events = malloc(config.max_events * sizeof(aqlprofile_pmc_event_t));
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
            printf("  Event %u: Block %s, Instance %u, Event %u\n",
                   i, block_name_to_string(config.events[i].block_name),
                   config.events[i].block_index, config.events[i].event_id);
        }
    }

    // Register agent with AQLProfile v2
    aqlprofile_agent_info_v1_t agent_info = {
        .agent_gfxip = config.arch_name,
        .xcc_num = 1,     // Default values - adjust as needed
        .se_num = 4,
        .cu_num = 64,
        .shader_arrays_per_se = 1,
        .domain = 0,
        .location_id = 0x1000
    };

    hsa_status_t status = aqlprofile_register_agent_info(&g_agent_handle, &agent_info,
                                                        AQLPROFILE_AGENT_VERSION_V1);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to register agent: %d\n", status);
        return 1;
    }

    // Validate events
    for (uint32_t i = 0; i < config.event_count; i++) {
        bool valid = false;
        status = aqlprofile_validate_pmc_event(g_agent_handle, &config.events[i], &valid);
        if (status != HSA_STATUS_SUCCESS || !valid) {
            if (config.verbose) {
                printf("Warning: Event %u may not be valid\n", i);
            }
        }
    }

    // Create profile configuration
    aqlprofile_pmc_profile_t profile = {
        .agent = g_agent_handle,
        .events = config.events,
        .event_count = config.event_count
    };

    // Create AQL packets
    aqlprofile_pmc_aql_packets_t packets;
    status = aqlprofile_pmc_create_packets(&g_profile_handle, &packets, profile,
                                          memory_alloc_callback, memory_dealloc_callback,
                                          memory_copy_callback, &config);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create AQL packets: %d\n", status);
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
    free(config.events);
    free(config.arch_name);
    free(config.output_file);

    if (config.verbose) {
        printf("Packet dump completed successfully\n");
    }

    return 0;
}