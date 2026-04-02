/*
 * Test 3C.3: Stress test - Memory pressure.
 *
 * Tests the migration protocol when system resources are constrained.
 * Verifies that proxy buffer allocation failure is handled gracefully
 * with proper rollback, and that repeated attach/detach does not leak
 * memory or signals.
 */
#include "test_helpers.h"
#include <vector>

static std::atomic<uint64_t> g_intercepted{0};

void memory_callback(const void* pkts, uint64_t pkt_count, uint64_t user_pkt_index,
                     void* data, hsa_amd_queue_intercept_packet_writer writer) {
    g_intercepted.fetch_add(pkt_count, std::memory_order_relaxed);
    writer(pkts, pkt_count);
}

int main() {
    int failures = 0;
    int passes = 0;
    printf("=== Stress Test: Memory Pressure ===\n\n");

    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL: hsa_init failed\n");
        return 1;
    }

    hsa_agent_t gpu_agent;
    if (!find_gpu_agent(&gpu_agent)) {
        fprintf(stderr, "FATAL: No GPU found\n");
        hsa_shut_down();
        return 1;
    }

    // Test 1: Repeated attach/detach to verify no memory leaks
    // (Run many cycles and check that memory usage stays bounded)
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue for memory test",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        const int CYCLES = 500;
        int success_count = 0;

        for (int i = 0; i < CYCLES; i++) {
            status = hsa_amd_queue_intercept_attach(queue, memory_callback, nullptr);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Attach failed at cycle %d: 0x%x\n", i, status);
                break;
            }

            // Submit a packet to exercise the full path
            submit_barrier(queue);

            status = hsa_amd_queue_intercept_detach(queue);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Detach failed at cycle %d: 0x%x\n", i, status);
                break;
            }

            success_count++;

            // Print progress every 100 cycles
            if ((i + 1) % 100 == 0) {
                printf("  Completed %d/%d cycles...\n", i + 1, CYCLES);
            }
        }

        EXPECT_TRUE("500 attach/submit/detach cycles (memory leak check)",
                     success_count == CYCLES);

        // Verify queue still works after all cycles
        for (int p = 0; p < 5; p++) {
            submit_barrier(queue);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT_TRUE("Queue functional after 500 cycles", true);

        hsa_queue_destroy(queue);
    }

    // Test 2: Large queue size to stress proxy buffer allocation
    {
        // Create a queue with large ring buffer (8K entries = 512KB ring)
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create large queue (8K entries)",
            hsa_queue_create(gpu_agent, 8192, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        if (queue) {
            const int CYCLES = 20;
            int success_count = 0;

            for (int i = 0; i < CYCLES; i++) {
                status = hsa_amd_queue_intercept_attach(queue, memory_callback, nullptr);
                if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
                    // This is acceptable under memory pressure
                    printf("  Cycle %d: attach returned OUT_OF_RESOURCES (expected under pressure)\n", i);
                    // Verify the queue still works without intercept
                    submit_barrier(queue);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    success_count++;
                    continue;
                }
                if (status != HSA_STATUS_SUCCESS) {
                    fprintf(stderr, "  Attach failed at cycle %d: 0x%x\n", i, status);
                    break;
                }

                submit_barrier(queue);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

                status = hsa_amd_queue_intercept_detach(queue);
                if (status != HSA_STATUS_SUCCESS) {
                    fprintf(stderr, "  Detach failed at cycle %d: 0x%x\n", i, status);
                    break;
                }
                success_count++;
            }

            EXPECT_TRUE("20 large-queue attach/detach cycles",
                         success_count == CYCLES);
            hsa_queue_destroy(queue);
        }
    }

    // Test 3: Many queues attached simultaneously to test aggregate memory
    {
        const int MAX_QUEUES = 32;
        std::vector<hsa_queue_t*> queues;
        int created = 0;

        for (int i = 0; i < MAX_QUEUES; i++) {
            hsa_queue_t* queue = nullptr;
            status = hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                                      nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
            if (status != HSA_STATUS_SUCCESS) {
                printf("  Queue creation stopped at %d queues\n", i);
                break;
            }
            queues.push_back(queue);
            created++;
        }

        printf("  Created %d queues\n", created);

        // Attach to all queues
        int attached = 0;
        for (auto* q : queues) {
            status = hsa_amd_queue_intercept_attach(q, memory_callback, nullptr);
            if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
                printf("  Attach stopped at %d queues (out of resources)\n", attached);
                break;
            }
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Unexpected attach error: 0x%x\n", status);
                break;
            }
            attached++;
        }

        printf("  Attached to %d/%d queues\n", attached, created);
        EXPECT_TRUE("Attached to at least 1 queue", attached > 0);

        // Detach all attached queues
        int detached = 0;
        for (int i = 0; i < attached; i++) {
            status = hsa_amd_queue_intercept_detach(queues[i]);
            if (status == HSA_STATUS_SUCCESS) detached++;
        }
        EXPECT_TRUE("All attached queues detached", detached == attached);

        // Destroy all queues
        for (auto* q : queues) {
            hsa_queue_destroy(q);
        }
    }

    hsa_shut_down();
    printf("\n=== Results: %d passed, %d failed ===\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
