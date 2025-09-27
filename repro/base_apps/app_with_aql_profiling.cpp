#include "aql_profiler.h"
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <atomic>

// Example showing how to integrate AQL profiling with existing HSA applications

// Simple debugger blocking mechanism (from existing apps)
static std::atomic<bool> debugger_block_flag{false};

extern "C" {
void app_debugger_block() {
    debugger_block_flag.exchange(true);
    fprintf(stderr, "AT BLOCK\n");
    fflush(stderr);
    while(debugger_block_flag.load() == true) {
        // Spin wait until app_debugger_continue is called
    }
}

void app_debugger_continue() {
    debugger_block_flag.exchange(false);
}
}

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

    CHECK_HSA(hsa_iterate_agents(agent_callback, &gpu_agent));
    return gpu_agent;
}

static hsa_queue_t* create_queue(hsa_agent_t agent) {
    hsa_queue_t* queue = nullptr;
    CHECK_HSA(hsa_queue_create(agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &queue));
    return queue;
}

void demonstrate_signal_create_with_profiling() {
    printf("\n=== Signal Create with AQL Profiling ===\n");

    // Initialize HSA
    CHECK_HSA(hsa_init());

    // Find GPU agent and create queue
    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle == 0) {
        printf("No GPU agent found\n");
        hsa_shut_down();
        return;
    }

    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));
    printf("Using agent: %s\n", agent_name);

    hsa_queue_t* queue = create_queue(gpu_agent);

    // Create AQL profiler
    aql_profiler_context_t* profiler = aql_profiler_create(gpu_agent, queue);
    if (!profiler) {
        printf("Failed to create profiler\n");
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    // Add counters for monitoring
    aql_profiler_add_basic_counters(profiler, agent_name);

    // Add debugger block before starting profiling
    app_debugger_block();

    // Create profiling packets
    printf("Creating AQL profiling packets...\n");
    if (aql_profiler_create_packets(profiler) != 0) {
        printf("Failed to create profiling packets\n");
        aql_profiler_destroy(profiler);
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    // Main application work: create HSA signal
    printf("Creating HSA signal...\n");
    hsa_signal_t signal;
    CHECK_HSA(hsa_signal_create(1, 0, nullptr, &signal));

    // Add debugger block after signal creation
    app_debugger_block();

    // Set signal value (additional work)
    hsa_signal_store_relaxed(signal, 42);
    printf("Signal created and set to value: %ld\n", hsa_signal_load_relaxed(signal));

    // Collect profiling results
    printf("Collecting AQL profiling results...\n");
    if (aql_profiler_collect_results(profiler) != 0) {
        printf("Failed to collect profiling results\n");
    } else {
        // Print profiling results
        aql_profiler_print_results(profiler);
    }

    // Cleanup
    hsa_signal_destroy(signal);
    aql_profiler_destroy(profiler);
    hsa_queue_destroy(queue);
    hsa_shut_down();
}

void demonstrate_barrier_packet_with_profiling() {
    printf("\n=== Barrier Packet with AQL Profiling ===\n");

    CHECK_HSA(hsa_init());

    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle == 0) {
        printf("No GPU agent found\n");
        hsa_shut_down();
        return;
    }

    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));

    hsa_queue_t* queue = create_queue(gpu_agent);

    // Create profiler with memory counters for this test
    aql_profiler_context_t* profiler = aql_profiler_create(gpu_agent, queue);
    if (!profiler) {
        printf("Failed to create profiler\n");
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    aql_profiler_add_memory_counters(profiler, agent_name);

    // Create profiling packets
    if (aql_profiler_create_packets(profiler) == 0) {
        printf("AQL profiling packets created for barrier packet test\n");

        // Create signals for barrier packet
        hsa_signal_t completion_signal;
        CHECK_HSA(hsa_signal_create(1, 0, nullptr, &completion_signal));

        // Add debugger block before packet submission
        app_debugger_block();

        // Submit barrier packet
        printf("Submitting barrier packet...\n");
        uint64_t write_index = hsa_queue_add_write_index_relaxed(queue, 1);
        hsa_barrier_and_packet_t* barrier_packet =
            (hsa_barrier_and_packet_t*)(&queue->base_address[write_index % queue->size]);

        memset(barrier_packet, 0, sizeof(hsa_barrier_and_packet_t));
        barrier_packet->header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier_packet->completion_signal = completion_signal;

        // Ring doorbell
        hsa_signal_store_relaxed(queue->doorbell_signal, write_index);

        // Wait for completion
        while (hsa_signal_wait_relaxed(completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                      UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0) {
            // Wait for barrier to complete
        }

        printf("Barrier packet completed\n");

        // Add debugger block after packet completion
        app_debugger_block();

        // Collect results and show them
        if (aql_profiler_collect_results(profiler) == 0) {
            aql_profiler_print_results(profiler);
        }

        hsa_signal_destroy(completion_signal);
    }

    aql_profiler_destroy(profiler);
    hsa_queue_destroy(queue);
    hsa_shut_down();
}

int main(int argc, char* argv[]) {
    printf("HSA Application with AQL Profiling Integration\n");
    printf("==============================================\n");

    std::string test_type = "all";
    if (argc > 1) {
        test_type = argv[1];
    }

    try {
        if (test_type == "all" || test_type == "signal") {
            demonstrate_signal_create_with_profiling();
        }

        if (test_type == "all" || test_type == "barrier") {
            demonstrate_barrier_packet_with_profiling();
        }

        printf("\n=== Integration Demo Complete ===\n");

    } catch (const std::exception& e) {
        printf("Demo failed: %s\n", e.what());
        return 1;
    }

    return 0;
}