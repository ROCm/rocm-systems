/**
 * @file aql_c_pmc_dumper.c
 * @brief Tool to dump AQL packets using aql_c PMC interface (v2-compatible)
 *
 * This tool uses the aql_c PMC interface to generate start/stop/read
 * AQL packets like the real AQLProfile v2 interface.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Include our AQL C library headers
#include "/root/aql_c/projects/aql_c_port/include/aql_pmc_interface.h"

struct counter_config {
    char name[64];
    char block[16];
    int event;
};

// Parse CSV file with counter configurations
int parse_counter_csv(const char* filename, struct counter_config* counters, int max_counters) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), file) && count < max_counters) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;

        char* name = strtok(line, ",");
        char* block = strtok(NULL, ",");
        char* event_str = strtok(NULL, ",\n");

        if (name && block && event_str) {
            strncpy(counters[count].name, name, sizeof(counters[count].name) - 1);
            strncpy(counters[count].block, block, sizeof(counters[count].block) - 1);
            counters[count].event = atoi(event_str);
            count++;
        }
    }

    fclose(file);
    return count;
}

// Print packet data in hex format
void print_packet_hex(const char* prefix, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    printf("%s: ", prefix);

    for (size_t i = 0; i < size; i++) {
        if (i > 0 && i % 16 == 0) printf("\n%s: ", prefix);
        printf("%02x ", bytes[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <counter_config.csv>\n", argv[0]);
        return 1;
    }

    // Parse counter configurations
    struct counter_config counters[32];
    int counter_count = parse_counter_csv(argv[1], counters, 32);
    if (counter_count < 0) {
        return 1;
    }

    printf("# AQL C PMC Interface Packet Dump (V2-Compatible)\n");
    printf("# Architecture: gfx942\n");
    printf("# Counters: %d\n", counter_count);
    printf("\n");

    // Allocate output buffer for counter results
    size_t output_buffer_size = counter_count * 8; // 8 bytes per counter
    void* output_buffer = malloc(output_buffer_size);
    if (!output_buffer) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        return 1;
    }

    // Process each counter individually (like AQLProfile v2 does)
    for (int i = 0; i < counter_count; i++) {
        printf("## Counter: %s\n", counters[i].name);
        printf("Block: %s, Event: %d\n", counters[i].block, counters[i].event);

        // Create PMC event
        aql_pmc_event_t event;
        aql_result_t result = aql_create_counter_event(&event, counters[i].block, 0, counters[i].event);
        if (result != AQL_SUCCESS) {
            fprintf(stderr, "Failed to create event for %s: %d\n", counters[i].name, result);
            continue;
        }

        // Create PMC profile
        aql_pmc_profile_t profile = {
            .arch_name = "gfx942",
            .events = &event,
            .event_count = 1,
            .output_buffer = output_buffer,
            .output_buffer_size = output_buffer_size
        };

        // Create AQL packets using PMC interface
        aql_pmc_packets_t packets;
        result = aql_pmc_create_packets(&packets, &profile);
        if (result == AQL_SUCCESS) {
            // Print all three packet types
            printf("START_PACKET:\n");
            print_packet_hex("AQL_PACKET", &packets.start_packet, sizeof(packets.start_packet));

            printf("STOP_PACKET:\n");
            print_packet_hex("AQL_PACKET", &packets.stop_packet, sizeof(packets.stop_packet));

            printf("READ_PACKET:\n");
            print_packet_hex("AQL_PACKET", &packets.read_packet, sizeof(packets.read_packet));

            // Clean up
            aql_pmc_delete_packets(&packets);
        } else {
            fprintf(stderr, "Failed to create packets for %s: %d\n", counters[i].name, result);
        }

        printf("\n");
    }

    free(output_buffer);
    return 0;
}