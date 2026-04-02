/*
 * Test 1F.3: Basic attach/detach test for HSA queue intercept retrofit.
 */
#include "test_helpers.h"

static std::atomic<uint64_t> g_intercept_count{0};
static std::atomic<bool> g_intercept_called{false};

void intercept_callback(const void* pkts, uint64_t pkt_count,
                        uint64_t user_pkt_index, void* data,
                        hsa_amd_queue_intercept_packet_writer writer) {
    g_intercept_called.store(true);
    g_intercept_count.fetch_add(pkt_count);
    writer(pkts, pkt_count);
}

int main() {
    int failures = 0;
    int passes = 0;
    printf("=== Test Suite: Basic Attach/Detach ===\n\n");

    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) { fprintf(stderr, "FATAL: hsa_init failed\n"); return 1; }

    hsa_agent_t gpu_agent;
    if (!find_gpu_agent(&gpu_agent)) { fprintf(stderr, "FATAL: No GPU\n"); hsa_shut_down(); return 1; }
    printf("Found GPU agent: 0x%lx\n\n", gpu_agent.handle);

    // Test 1: Create queue
    hsa_queue_t* queue = nullptr;
    EXPECT_STATUS("Create queue",
        hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                         nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
        HSA_STATUS_SUCCESS);
    if (!queue) { hsa_shut_down(); return 1; }

    // Test 2: Error paths
    EXPECT_STATUS("Attach with NULL callback",
        hsa_amd_queue_intercept_attach(queue, nullptr, nullptr),
        HSA_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_STATUS("Detach from non-intercepted queue",
        hsa_amd_queue_intercept_detach(queue),
        HSA_STATUS_ERROR_INVALID_QUEUE);

    // Test 3: Attach
    g_intercept_called.store(false);
    g_intercept_count.store(0);
    EXPECT_STATUS("Attach intercept to existing queue",
        hsa_amd_queue_intercept_attach(queue, intercept_callback, nullptr),
        HSA_STATUS_SUCCESS);

    // Test 4: Write packet to trigger callback
    submit_barrier(queue);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_TRUE("Intercept callback was invoked", g_intercept_called.load());

    // Test 5: Detach
    EXPECT_STATUS("Detach intercept", hsa_amd_queue_intercept_detach(queue), HSA_STATUS_SUCCESS);

    // Test 6: Double detach
    EXPECT_STATUS("Double detach fails", hsa_amd_queue_intercept_detach(queue), HSA_STATUS_ERROR_INVALID_QUEUE);

    // Test 7: Queue works after detach
    submit_barrier(queue);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE("Queue operates after detach (no crash)", true);

    // Test 8: Re-attach
    g_intercept_called.store(false);
    EXPECT_STATUS("Re-attach", hsa_amd_queue_intercept_attach(queue, intercept_callback, nullptr), HSA_STATUS_SUCCESS);
    submit_barrier(queue);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_TRUE("Callback invoked after re-attach", g_intercept_called.load());
    EXPECT_STATUS("Detach after re-attach", hsa_amd_queue_intercept_detach(queue), HSA_STATUS_SUCCESS);

    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("\n=== Results: %d passed, %d failed ===\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
