/*
 * Test 1F.1: HW-state separation tests for HSA queue intercept retrofit.
 */
#include "test_helpers.h"

static std::atomic<uint64_t> g_pkt_count{0};
static std::atomic<bool> g_called{false};

void handler(const void* pkts, uint64_t pkt_count, uint64_t user_pkt_index,
             void* data, hsa_amd_queue_intercept_packet_writer writer) {
    g_pkt_count.fetch_add(pkt_count);
    g_called.store(true);
    writer(pkts, pkt_count);
}

int main() {
    int failures = 0;
    int passes = 0;
    printf("=== Test Suite: HW-State Separation ===\n\n");

    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) { fprintf(stderr, "FATAL: hsa_init failed\n"); return 1; }

    hsa_agent_t gpu_agent;
    if (!find_gpu_agent(&gpu_agent)) { fprintf(stderr, "FATAL: No GPU\n"); hsa_shut_down(); return 1; }

    // Test 1: Base address and doorbell change/restore
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        void* orig_base = queue->base_address;
        hsa_signal_t orig_db = queue->doorbell_signal;
        printf("  Original  base=%p  doorbell=0x%lx\n", orig_base, orig_db.handle);

        EXPECT_STATUS("Attach", hsa_amd_queue_intercept_attach(queue, handler, nullptr), HSA_STATUS_SUCCESS);

        void* new_base = queue->base_address;
        hsa_signal_t new_db = queue->doorbell_signal;
        printf("  Attached  base=%p  doorbell=0x%lx\n", new_base, new_db.handle);

        EXPECT_TRUE("base_address changed after attach", new_base != orig_base);
        EXPECT_TRUE("doorbell changed after attach", new_db.handle != orig_db.handle);

        EXPECT_STATUS("Detach", hsa_amd_queue_intercept_detach(queue), HSA_STATUS_SUCCESS);

        void* rest_base = queue->base_address;
        hsa_signal_t rest_db = queue->doorbell_signal;
        printf("  Restored  base=%p  doorbell=0x%lx\n", rest_base, rest_db.handle);

        EXPECT_TRUE("base_address restored after detach", rest_base == orig_base);
        EXPECT_TRUE("doorbell restored after detach", rest_db.handle == orig_db.handle);

        hsa_queue_destroy(queue);
    }

    // Test 2: Packets route through intercept (verifies Submit uses hw_ring_buf_)
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue (submit test)",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        g_pkt_count.store(0);
        g_called.store(false);

        EXPECT_STATUS("Attach (submit test)", hsa_amd_queue_intercept_attach(queue, handler, nullptr), HSA_STATUS_SUCCESS);

        for (int i = 0; i < 5; i++) submit_barrier(queue);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        uint64_t count = g_pkt_count.load();
        printf("  Packets intercepted: %lu\n", (unsigned long)count);
        EXPECT_TRUE("Submit() routed packets through intercept (>0)", count > 0);
        EXPECT_TRUE("All 5 packets intercepted", count >= 5);

        EXPECT_STATUS("Detach (submit test)", hsa_amd_queue_intercept_detach(queue), HSA_STATUS_SUCCESS);
        hsa_queue_destroy(queue);
    }

    // Test 3: Verify no infinite recursion (base_address != proxy after swap)
    // If hw_ring_buf_ is not used, Submit() would read base_address which now
    // points to the proxy buffer, causing infinite recursion. If we get here
    // without hanging/crashing, hw_ring_buf_ is working correctly.
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue (recursion test)",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        g_pkt_count.store(0);
        EXPECT_STATUS("Attach (recursion test)", hsa_amd_queue_intercept_attach(queue, handler, nullptr), HSA_STATUS_SUCCESS);

        // Submit many packets rapidly - if there's infinite recursion, this will hang
        for (int i = 0; i < 20; i++) submit_barrier(queue);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        uint64_t count = g_pkt_count.load();
        printf("  Rapid submit count: %lu (expected >= 20)\n", (unsigned long)count);
        EXPECT_TRUE("No infinite recursion in Submit() (hw_ring_buf_ works)", count >= 20);

        EXPECT_STATUS("Detach (recursion test)", hsa_amd_queue_intercept_detach(queue), HSA_STATUS_SUCCESS);
        hsa_queue_destroy(queue);
    }

    hsa_shut_down();
    printf("\n=== Results: %d passed, %d failed ===\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
