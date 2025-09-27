#include "aql_profiler.h"
#include <iostream>

// Simple example showing how to use the AQL profiler v2 interface

static hsa_agent_t find_gpu_agent() {
    hsa_agent_t gpu_agent = {0};

    auto agent_callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t device_type;
        CHECK_HSA(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type));

        if (device_type == HSA_DEVICE_TYPE_GPU) {
            *(static_cast<hsa_agent_t*>(data)) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_iterate_agents(agent_callback, &gpu_agent);
    return gpu_agent;
}

int main() {
    printf("Simple AQL Profiler Example\n");
    printf("===========================\n");

    // Initialize HSA
    CHECK_HSA(hsa_init());

    // Find GPU agent
    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle == 0) {
        printf("No GPU agent found\n");
        hsa_shut_down();
        return -1;
    }

    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));
    printf("Using GPU agent: %s\n", agent_name);

    // Create queue (not needed for packet creation but useful for actual submission)
    hsa_queue_t* queue = nullptr;
    CHECK_HSA(hsa_queue_create(gpu_agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &queue));

    // Define the specific events you provided: block_name=6 (SQ)
    aqlprofile_pmc_event_t events[] = {
        {.block_index = 0, .event_id = 35, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 37, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 40, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 54, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 36, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)}
    };
    uint32_t event_count = sizeof(events) / sizeof(events[0]);

    // Create AQL packets using simple interface
    aqlprofile_pmc_aql_packets_t packets;
    aqlprofile_handle_t handle;

    printf("Creating AQL profiling packets for %u events...\n", event_count);

    int result = aql_profiler_create_packets_simple(
        gpu_agent,
        queue,
        events,
        event_count,
        &packets,
        &handle
    );

    if (result != 0) {
        printf("Failed to create AQL packets\n");
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return -1;
    }

    printf("✓ Successfully created AQL profiling packets!\n");

    // Display packet information
    printf("\nPacket Details:\n");
    printf("  Start packet header: 0x%x\n", packets.start_packet.header);
    printf("  Stop packet header:  0x%x\n", packets.stop_packet.header);
    printf("  Read packet header:  0x%x\n", packets.read_packet.header);

    // Show PM4 command data (first few DWORDs)
    printf("\nStart packet PM4 command (first 4 DWORDs):\n");
    for (int i = 0; i < 4 && i < 16; i++) {
        printf("  [%d]: 0x%08x\n", i, packets.start_packet.pm4_command[i]);
    }

    printf("\nPackets are ready for submission to HSA queue.\n");
    printf("To use:\n");
    printf("1. Submit start_packet to begin profiling\n");
    printf("2. Run your workload\n");
    printf("3. Submit read_packet to copy counter values\n");
    printf("4. Submit stop_packet to end profiling\n");
    printf("5. Use aqlprofile_pmc_iterate_data() to get results\n");

    // Cleanup
    aql_profiler_cleanup_simple(handle);
    hsa_queue_destroy(queue);
    hsa_shut_down();

    printf("\n✓ Example completed successfully!\n");
    return 0;
}