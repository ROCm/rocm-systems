#include "aql_profiler.h"
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

// Test workload functions
void test_simple_workload() {
    // Simulate some work
    volatile int sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }
}

void test_memory_workload() {
    // Simulate memory-intensive work
    const size_t size = 1024 * 1024;
    std::vector<int> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = i % 256;
    }

    // Access pattern to stress memory subsystem
    volatile int sum = 0;
    for (size_t i = 0; i < size; i += 64) {
        sum += data[i];
    }
}

// HSA initialization helpers
static hsa_agent_t find_gpu_agent() {
    hsa_agent_t gpu_agent = {0};

    auto agent_callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t device_type;
        CHECK_HSA(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type));

        if (device_type == HSA_DEVICE_TYPE_GPU) {
            *(static_cast<hsa_agent_t*>(data)) = agent;
            return HSA_STATUS_INFO_BREAK; // Found GPU, stop iteration
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

void run_basic_test() {
    printf("\n=== Running Basic AQL Profiler Test ===\n");

    // Initialize HSA
    CHECK_HSA(hsa_init());

    // Find GPU agent
    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle == 0) {
        printf("No GPU agent found, skipping test\n");
        hsa_shut_down();
        return;
    }

    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));
    printf("Using GPU agent: %s\n", agent_name);

    // Create queue
    hsa_queue_t* queue = create_queue(gpu_agent);

    // Create profiler
    aql_profiler_context_t* profiler = aql_profiler_create(gpu_agent, queue);
    if (!profiler) {
        printf("Failed to create AQL profiler\n");
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    // Add basic counters
    printf("Adding basic counters...\n");
    aql_profiler_add_basic_counters(profiler, agent_name);

    // Create profiling packets
    printf("Creating profiling packets...\n");
    if (aql_profiler_create_packets(profiler) != 0) {
        printf("Failed to create profiling packets\n");
        aql_profiler_destroy(profiler);
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    // Run test workload
    printf("Running test workload...\n");
    test_simple_workload();

    // Collect results
    printf("Collecting results...\n");
    if (aql_profiler_collect_results(profiler) != 0) {
        printf("Failed to collect results\n");
        aql_profiler_destroy(profiler);
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    // Print results
    aql_profiler_print_results(profiler);

    // Verify results
    aql_counter_result_t* results;
    uint32_t count;
    if (aql_profiler_get_results(profiler, &results, &count) == 0) {
        printf("Retrieved %u counter results\n", count);
        bool has_data = false;
        for (uint32_t i = 0; i < count; i++) {
            if (results[i].counter_value > 0) {
                has_data = true;
                break;
            }
        }

        if (has_data) {
            printf("✓ Basic test PASSED - Counter data collected\n");
        } else {
            printf("✗ Basic test FAILED - No counter data collected\n");
        }
    } else {
        printf("✗ Basic test FAILED - Could not retrieve results\n");
    }

    // Cleanup
    aql_profiler_destroy(profiler);
    hsa_queue_destroy(queue);
    hsa_shut_down();
}

void run_memory_test() {
    printf("\n=== Running Memory-Intensive Test ===\n");

    // Initialize HSA
    CHECK_HSA(hsa_init());

    // Find GPU agent
    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle == 0) {
        printf("No GPU agent found, skipping test\n");
        hsa_shut_down();
        return;
    }

    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));

    // Create queue
    hsa_queue_t* queue = create_queue(gpu_agent);

    // Create profiler
    aql_profiler_context_t* profiler = aql_profiler_create(gpu_agent, queue);
    if (!profiler) {
        printf("Failed to create AQL profiler\n");
        hsa_queue_destroy(queue);
        hsa_shut_down();
        return;
    }

    // Add memory counters
    printf("Adding memory counters...\n");
    aql_profiler_add_memory_counters(profiler, agent_name);

    // Create profiling packets
    printf("Creating profiling packets...\n");
    aql_profiler_create_packets(profiler);

    // Run memory workload
    printf("Running memory-intensive workload...\n");
    test_memory_workload();

    // Collect results
    printf("Collecting results...\n");
    aql_profiler_collect_results(profiler);

    // Print results
    aql_profiler_print_results(profiler);

    // Cleanup
    aql_profiler_destroy(profiler);
    hsa_queue_destroy(queue);
    hsa_shut_down();

    printf("✓ Memory test completed\n");
}

void run_stress_test() {
    printf("\n=== Running Stress Test ===\n");

    // Initialize HSA
    CHECK_HSA(hsa_init());

    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle == 0) {
        printf("No GPU agent found, skipping test\n");
        hsa_shut_down();
        return;
    }

    char agent_name[64];
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));

    hsa_queue_t* queue = create_queue(gpu_agent);

    // Test multiple profiling sessions
    const int num_sessions = 5;
    printf("Running %d profiling sessions...\n", num_sessions);

    for (int session = 0; session < num_sessions; session++) {
        printf("Session %d/%d...\n", session + 1, num_sessions);

        aql_profiler_context_t* profiler = aql_profiler_create(gpu_agent, queue);
        if (!profiler) {
            printf("Failed to create profiler for session %d\n", session);
            continue;
        }

        // Add different counter sets for variety
        if (session % 2 == 0) {
            aql_profiler_add_basic_counters(profiler, agent_name);
        } else {
            aql_profiler_add_sq_counters(profiler);
        }

        aql_profiler_create_packets(profiler);

        // Short workload
        test_simple_workload();

        aql_profiler_collect_results(profiler);

        aql_counter_result_t* results;
        uint32_t count;
        if (aql_profiler_get_results(profiler, &results, &count) == 0) {
            printf("  Session %d: %u counters collected\n", session + 1, count);
        }

        aql_profiler_destroy(profiler);

        // Small delay between sessions
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    hsa_queue_destroy(queue);
    hsa_shut_down();

    printf("✓ Stress test completed\n");
}

void run_error_handling_test() {
    printf("\n=== Running Error Handling Test ===\n");

    // Test null context handling
    printf("Testing null context handling...\n");
    if (aql_profiler_create_packets(nullptr) == -1) {
        printf("✓ Null context properly rejected\n");
    } else {
        printf("✗ Null context not properly handled\n");
    }

    // Test invalid operations
    printf("Testing invalid operation sequence...\n");

    CHECK_HSA(hsa_init());
    hsa_agent_t gpu_agent = find_gpu_agent();
    if (gpu_agent.handle != 0) {
        hsa_queue_t* queue = create_queue(gpu_agent);
        aql_profiler_context_t* profiler = aql_profiler_create(gpu_agent, queue);

        if (profiler) {
            // Try to collect without creating packets
            if (aql_profiler_collect_results(profiler) == -1) {
                printf("✓ Collect without packets properly rejected\n");
            }

            // Try to create packets twice
            aql_profiler_create_packets(profiler);
            if (aql_profiler_create_packets(profiler) == -1) {
                printf("✓ Double packet creation properly rejected\n");
            }

            aql_profiler_collect_results(profiler);
            aql_profiler_destroy(profiler);
        }

        hsa_queue_destroy(queue);
    }
    hsa_shut_down();

    printf("✓ Error handling test completed\n");
}

int main(int argc, char* argv[]) {
    printf("AQL Profiler Verification Tool\n");
    printf("==============================\n");

    bool run_all = true;
    std::string test_name;

    if (argc > 1) {
        test_name = argv[1];
        run_all = false;
    }

    try {
        if (run_all || test_name == "basic") {
            run_basic_test();
        }

        if (run_all || test_name == "memory") {
            run_memory_test();
        }

        if (run_all || test_name == "stress") {
            run_stress_test();
        }

        if (run_all || test_name == "error") {
            run_error_handling_test();
        }

        printf("\n=== All Tests Complete ===\n");

    } catch (const std::exception& e) {
        printf("Test failed with exception: %s\n", e.what());
        return 1;
    }

    return 0;
}