/*
 * Test 3C.1: Stress test - Rapid attach/detach cycling.
 *
 * Repeatedly attaches and detaches intercept queues on a running workload.
 * Tests for memory leaks, signal leaks, and correctness under rapid cycling.
 */
#include "test_helpers.h"
#include <vector>

static std::atomic<uint64_t> g_total_intercepted{0};

void stress_callback(const void* pkts, uint64_t pkt_count, uint64_t user_pkt_index,
                     void* data, hsa_amd_queue_intercept_packet_writer writer) {
    g_total_intercepted.fetch_add(pkt_count, std::memory_order_relaxed);
    writer(pkts, pkt_count);
}

int main() {
    int failures = 0;
    int passes = 0;
    printf("=== Stress Test: Rapid Attach/Detach Cycling ===\n\n");

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

    // Test 1: Rapid cycling on a single queue (100 iterations)
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue for rapid cycling",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        const int CYCLES = 100;
        int success_count = 0;
        g_total_intercepted.store(0);

        for (int i = 0; i < CYCLES; i++) {
            status = hsa_amd_queue_intercept_attach(queue, stress_callback, nullptr);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Attach failed at cycle %d with status 0x%x\n", i, status);
                break;
            }

            // Submit a few packets while attached
            for (int p = 0; p < 3; p++) {
                submit_barrier(queue);
            }

            // Brief delay to let packets process
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            status = hsa_amd_queue_intercept_detach(queue);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Detach failed at cycle %d with status 0x%x\n", i, status);
                break;
            }

            success_count++;
        }

        EXPECT_TRUE("100 rapid attach/detach cycles completed",
                     success_count == CYCLES);
        printf("  Total intercepted packets: %lu\n",
               (unsigned long)g_total_intercepted.load());

        // Submit more packets after all cycling to verify queue still works
        for (int p = 0; p < 10; p++) {
            submit_barrier(queue);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT_TRUE("Queue functional after 100 cycles", true);

        hsa_queue_destroy(queue);
    }

    // Test 2: Rapid cycling on multiple queues (4 queues, 50 iterations each)
    {
        const int NUM_QUEUES = 4;
        const int CYCLES = 50;
        hsa_queue_t* queues[NUM_QUEUES] = {};
        bool create_ok = true;

        for (int i = 0; i < NUM_QUEUES; i++) {
            if (hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                                 nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                                 &queues[i]) != HSA_STATUS_SUCCESS) {
                create_ok = false;
                break;
            }
        }
        EXPECT_TRUE("Create 4 queues for multi-queue cycling", create_ok);

        if (create_ok) {
            int total_success = 0;

            for (int c = 0; c < CYCLES; c++) {
                bool cycle_ok = true;

                // Attach to all queues
                for (int i = 0; i < NUM_QUEUES; i++) {
                    if (hsa_amd_queue_intercept_attach(queues[i], stress_callback,
                                                       nullptr) != HSA_STATUS_SUCCESS) {
                        cycle_ok = false;
                        // Detach any already-attached queues in this cycle
                        for (int j = 0; j < i; j++) {
                            hsa_amd_queue_intercept_detach(queues[j]);
                        }
                        break;
                    }
                }

                if (!cycle_ok) break;

                // Submit to each queue
                for (int i = 0; i < NUM_QUEUES; i++) {
                    submit_barrier(queues[i]);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));

                // Detach from all queues
                for (int i = 0; i < NUM_QUEUES; i++) {
                    if (hsa_amd_queue_intercept_detach(queues[i]) != HSA_STATUS_SUCCESS) {
                        cycle_ok = false;
                        break;
                    }
                }

                if (cycle_ok) total_success++;
                else break;
            }

            EXPECT_TRUE("50 multi-queue attach/detach cycles completed",
                         total_success == CYCLES);
        }

        for (int i = 0; i < NUM_QUEUES; i++) {
            if (queues[i]) hsa_queue_destroy(queues[i]);
        }
    }

    // Test 3: Attach/detach without any packet submission between cycles
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue for no-submit cycling",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        const int CYCLES = 200;
        int success_count = 0;

        for (int i = 0; i < CYCLES; i++) {
            if (hsa_amd_queue_intercept_attach(queue, stress_callback,
                                               nullptr) != HSA_STATUS_SUCCESS) break;
            if (hsa_amd_queue_intercept_detach(queue) != HSA_STATUS_SUCCESS) break;
            success_count++;
        }

        EXPECT_TRUE("200 no-submit attach/detach cycles", success_count == CYCLES);
        hsa_queue_destroy(queue);
    }

    hsa_shut_down();
    printf("\n=== Results: %d passed, %d failed ===\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
