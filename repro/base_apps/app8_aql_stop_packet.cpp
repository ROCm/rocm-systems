// HSA AQL Stop Packet Application
// Creates queue and submits AQL profiling start, read, and stop packets with debugger blocks around stop

#define __HIP_PLATFORM_AMD__

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include "aql_profiler.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <atomic>

// Simple logging macros
#define LOG_INFO(...) do { printf("[INFO] "); printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while(0)
#define LOG_ERROR(...) do { fprintf(stderr, "[ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while(0)
#define CHECK_HSA(status) do { if(status != HSA_STATUS_SUCCESS) { LOG_ERROR("HSA call failed at %s:%d with status %d", __FILE__, __LINE__, status); exit(1); } } while(0)

// Debugger blocking functions
namespace {
std::atomic<bool>& debugger_block_flag() {
    static std::atomic<bool> block = {true};
    return block;
}
}

extern "C" {
void app_debugger_block() {
    debugger_block_flag().exchange(true);
    fprintf(stderr, "AT BLOCK\n");
    fflush(stderr);
    while(debugger_block_flag().load() == true) {
        // Spin wait until app_debugger_continue is called
    }
}

void app_debugger_continue() {
    debugger_block_flag().exchange(false);
}
}

// Submit packet to queue
uint64_t submit_packet(hsa_queue_t* queue, const void* packet) {
    const uint32_t slot_size_b = 0x40;
    // Advance command queue
    const uint64_t write_idx = hsa_queue_add_write_index_scacq_screl(queue, 1);
    while((write_idx - hsa_queue_load_read_index_relaxed(queue)) >= queue->size) {
        sched_yield();
    }

    const uint32_t slot_idx = write_idx % queue->size;
    uint32_t* queue_slot = (uint32_t*)((uintptr_t)queue->base_address + slot_idx * slot_size_b);

    // Copy packet to queue
    memcpy(queue_slot, packet, slot_size_b);

    // Ring doorbell
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

    return write_idx;
}

int main(int argc, char** argv) {
    LOG_INFO("Starting HSA AQL stop packet application");

    // Initialize HSA
    LOG_INFO("Initializing HSA runtime");
    CHECK_HSA(hsa_init());

    // Find first GPU agent
    hsa_agent_t gpu_agent = {0};
    auto iterate_agents = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t device_type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if(device_type == HSA_DEVICE_TYPE_GPU) {
            *((hsa_agent_t*)data) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_iterate_agents(iterate_agents, &gpu_agent);
    if(gpu_agent.handle == 0) {
        LOG_ERROR("No GPU agent found");
        return 1;
    }

    // Get agent name
    char agent_name[64] = {0};
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));
    LOG_INFO("Found GPU agent: %s (handle: 0x%lx)", agent_name, gpu_agent.handle);

    // Create HSA queue
    LOG_INFO("Creating HSA queue");
    hsa_queue_t* queue;
    CHECK_HSA(hsa_queue_create(gpu_agent,
                              1024,
                              HSA_QUEUE_TYPE_SINGLE,
                              nullptr,
                              nullptr,
                              UINT32_MAX,
                              UINT32_MAX,
                              &queue));

    // Define the specific events: block_name=6 (SQ)
    aqlprofile_pmc_event_t events[] = {
        {.block_index = 0, .event_id = 35, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 37, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 40, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 54, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
        {.block_index = 0, .event_id = 36, .flags = {.raw = 0}, .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)}
    };
    uint32_t event_count = sizeof(events) / sizeof(events[0]);

    // Create AQL packets using simple interface
    LOG_INFO("Creating AQL profiling packets for %u events", event_count);
    aqlprofile_pmc_aql_packets_t packets;
    aqlprofile_handle_t handle;

    int result = aql_profiler_create_packets_simple(
        gpu_agent,
        queue,
        events,
        event_count,
        &packets,
        &handle
    );

    if (result != 0) {
        LOG_ERROR("Failed to create AQL packets");
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return 1;
    }

    LOG_INFO("Successfully created AQL profiling packets");

    // Submit start packet (no blocks)
    LOG_INFO("Submitting AQL start packet");
    uint64_t start_write_idx = submit_packet(queue, &packets.start_packet);
    LOG_INFO("AQL start packet submitted successfully at index %lu", start_write_idx);

    // Submit read packet (no blocks)
    LOG_INFO("Submitting AQL read packet");
    uint64_t read_write_idx = submit_packet(queue, &packets.read_packet);
    LOG_INFO("AQL read packet submitted successfully at index %lu", read_write_idx);

    // Submit stop packet with debugger blocks
    LOG_INFO("Submitting AQL stop packet");

    app_debugger_block();  // Block before stop packet submission
    uint64_t stop_write_idx = submit_packet(queue, &packets.stop_packet);
    app_debugger_block();  // Block after stop packet submission

    LOG_INFO("AQL stop packet submitted successfully at index %lu", stop_write_idx);

    // Display packet information
    LOG_INFO("Start packet header: 0x%x", packets.start_packet.header);
    LOG_INFO("Read packet header: 0x%x", packets.read_packet.header);
    LOG_INFO("Stop packet header: 0x%x", packets.stop_packet.header);

    LOG_INFO("Start packet PM4 command (first 4 DWORDs):");
    for (int i = 0; i < 4 && i < 16; i++) {
        LOG_INFO("  [%d]: 0x%08x", i, packets.start_packet.pm4_command[i]);
    }

    LOG_INFO("Read packet PM4 command (first 4 DWORDs):");
    for (int i = 0; i < 4 && i < 16; i++) {
        LOG_INFO("  [%d]: 0x%08x", i, packets.read_packet.pm4_command[i]);
    }

    LOG_INFO("Stop packet PM4 command (first 4 DWORDs):");
    for (int i = 0; i < 4 && i < 16; i++) {
        LOG_INFO("  [%d]: 0x%08x", i, packets.stop_packet.pm4_command[i]);
    }

    LOG_INFO("All AQL profiling packets have been submitted to the queue");
    LOG_INFO("Profiling session completed - counters stopped and data ready for collection");

    // Cleanup
    LOG_INFO("Cleaning up resources");
    aql_profiler_cleanup_simple(handle);
    hsa_queue_destroy(queue);
    hsa_shut_down();

    LOG_INFO("AQL stop packet application completed successfully");

    return 0;
}