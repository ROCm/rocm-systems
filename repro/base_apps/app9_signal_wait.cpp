// HSA Signal Wait Application
// Creates queue, signal, and barrier packets, then blocks around signal wait operation
// Based on app4_dual_barrier but focuses on measuring signal wait performance

#define __HIP_PLATFORM_AMD__

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

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

// Helper function to create packet header
uint16_t create_packet_header(hsa_packet_type_t type) {
    uint16_t header = type << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
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
    LOG_INFO("Starting HSA signal wait application");

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

    // Create a signal with initial value 1 (will be decremented by one barrier packet)
    LOG_INFO("Creating signal with initial value 1");
    hsa_signal_t completion_signal;
    CHECK_HSA(hsa_signal_create(1, 0, nullptr, &completion_signal));

    // Create single barrier packet
    LOG_INFO("Creating and submitting barrier packet");
    hsa_barrier_and_packet_t barrier_packet = {};
    barrier_packet.header = create_packet_header(HSA_PACKET_TYPE_BARRIER_AND);
    barrier_packet.completion_signal = completion_signal;
    submit_packet(queue, &barrier_packet);

    // Packet should decrement the signal from 1 -> 0
    LOG_INFO("Barrier packet submitted, signal should reach 0 when complete");

    // Wait for completion with blocks around the signal wait operation
    LOG_INFO("Waiting for completion (signal to reach 0)");

    app_debugger_block();  // Block before hsa_signal_wait_relaxed
    hsa_signal_value_t final_value = hsa_signal_wait_relaxed(completion_signal,
                                                            HSA_SIGNAL_CONDITION_EQ,
                                                            0,
                                                            5000000000000, // 5 second timeout
                                                            HSA_WAIT_STATE_ACTIVE);
    app_debugger_block();  // Block after hsa_signal_wait_relaxed

    if(final_value == 0) {
        LOG_INFO("Signal wait completed successfully - barrier packet finished!");
    } else {
        LOG_INFO("Signal wait timed out, final signal value: %ld", final_value);
    }

    // Cleanup
    LOG_INFO("Cleaning up resources");
    hsa_signal_destroy(completion_signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();

    LOG_INFO("Signal wait application completed successfully");

    return 0;
}