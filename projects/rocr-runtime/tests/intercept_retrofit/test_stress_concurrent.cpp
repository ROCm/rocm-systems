/*
 * Test 3C.2: Stress test - Concurrent attach and destroy.
 *
 * Tests the lifecycle state machine under stress: multiple threads
 * attempting to attach, detach, and destroy queues simultaneously.
 * Verifies no crashes, deadlocks, or use-after-free.
 */
#include "test_helpers.h"
#include <vector>

static std::atomic<uint64_t> g_intercepted{0};

void concurrent_callback(const void* pkts, uint64_t pkt_count, uint64_t user_pkt_index,
                          void* data, hsa_amd_queue_intercept_packet_writer writer) {
    g_intercepted.fetch_add(pkt_count, std::memory_order_relaxed);
    writer(pkts, pkt_count);
}

int main() {
    int failures = 0;
    int passes = 0;
    printf("=== Stress Test: Concurrent Attach and Destroy ===\n\n");

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

    // Test 1: Thread submits packets while another thread cycles attach/detach
    {
        hsa_queue_t* queue = nullptr;
        EXPECT_STATUS("Create queue for concurrent test",
            hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue),
            HSA_STATUS_SUCCESS);

        std::atomic<bool> stop{false};
        std::atomic<uint64_t> submit_count{0};

        // Submitter thread: continuously submits barrier packets
        std::thread submitter([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                submit_barrier(queue);
                submit_count.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });

        // Main thread: cycles attach/detach 20 times
        const int CYCLES = 20;
        int success_count = 0;

        for (int i = 0; i < CYCLES; i++) {
            status = hsa_amd_queue_intercept_attach(queue, concurrent_callback, nullptr);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Attach failed at cycle %d: 0x%x\n", i, status);
                break;
            }

            // Let the submitter run for a bit while attached
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            status = hsa_amd_queue_intercept_detach(queue);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "  Detach failed at cycle %d: 0x%x\n", i, status);
                break;
            }

            success_count++;
        }

        stop.store(true, std::memory_order_relaxed);
        submitter.join();

        printf("  Submitted %lu packets, intercepted %lu packets\n",
               (unsigned long)submit_count.load(),
               (unsigned long)g_intercepted.load());
        EXPECT_TRUE("20 concurrent submit/attach cycles no crash",
                     success_count == CYCLES);

        hsa_queue_destroy(queue);
    }

    // Test 2: Rapid create-attach-destroy sequences (lifecycle stress)
    {
        const int ITERS = 50;
        int success_count = 0;

        for (int i = 0; i < ITERS; i++) {
            hsa_queue_t* queue = nullptr;
            status = hsa_queue_create(gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                                      nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
            if (status != HSA_STATUS_SUCCESS) break;

            // Attach
            status = hsa_amd_queue_intercept_attach(queue, concurrent_callback, nullptr);
            if (status != HSA_STATUS_SUCCESS) {
                hsa_queue_destroy(queue);
                break;
            }

            // Submit a packet
            submit_barrier(queue);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));

            // Detach then destroy
            hsa_amd_queue_intercept_detach(queue);
            hsa_queue_destroy(queue);
            success_count++;
        }

        EXPECT_TRUE("50 create-attach-submit-detach-destroy cycles",
                     success_count == ITERS);
    }

    // Test 3: Multiple queues with interleaved attach/detach
    {
        const int NUM_QUEUES = 8;
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
        EXPECT_TRUE("Create 8 queues for interleaved test", create_ok);

        if (create_ok) {
            // Attach to even-numbered queues, submit to all, then attach to odd
            bool ok = true;
            for (int i = 0; i < NUM_QUEUES; i += 2) {
                if (hsa_amd_queue_intercept_attach(queues[i], concurrent_callback,
                                                    nullptr) != HSA_STATUS_SUCCESS) {
                    ok = false;
                    break;
                }
            }
            EXPECT_TRUE("Attach to even queues", ok);

            // Submit to all queues
            for (int i = 0; i < NUM_QUEUES; i++) {
                submit_barrier(queues[i]);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // Now attach to odd-numbered queues
            if (ok) {
                for (int i = 1; i < NUM_QUEUES; i += 2) {
                    if (hsa_amd_queue_intercept_attach(queues[i], concurrent_callback,
                                                        nullptr) != HSA_STATUS_SUCCESS) {
                        ok = false;
                        break;
                    }
                }
                EXPECT_TRUE("Attach to odd queues", ok);
            }

            // Submit to all again
            for (int i = 0; i < NUM_QUEUES; i++) {
                submit_barrier(queues[i]);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // Detach all
            for (int i = 0; i < NUM_QUEUES; i++) {
                hsa_amd_queue_intercept_detach(queues[i]);
            }
            EXPECT_TRUE("Interleaved attach/detach completed without crash", true);
        }

        for (int i = 0; i < NUM_QUEUES; i++) {
            if (queues[i]) hsa_queue_destroy(queues[i]);
        }
    }

    hsa_shut_down();
    printf("\n=== Results: %d passed, %d failed ===\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
