/*
 * trace_injector.cpp - Shared library for late-attach queue intercept testing.
 *
 * This library is designed to be dlopen'd at runtime by a running HIP
 * application. It attaches an intercept handler to an existing HSA queue,
 * counts intercepted packet submissions, and detaches on stop.
 *
 * Exported C API:
 *   trace_injector_start(hsa_queue_t** queues, int num_queues)
 *   trace_injector_stop(int* trace_count)
 */

#include <cstdio>
#include <cstring>
#include <atomic>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// hsa_ext_amd.h provides:
//   hsa_amd_queue_intercept_handler typedef
//   hsa_amd_queue_intercept_packet_writer typedef
//   hsa_amd_queue_intercept_attach()
//   hsa_amd_queue_intercept_detach()

// Global state
static std::atomic<uint64_t> g_packet_count{0};
static std::atomic<uint64_t> g_dispatch_count{0};
static std::atomic<bool> g_active{false};

// Queue handles we've attached to
static constexpr int MAX_QUEUES = 16;
static hsa_queue_t* g_attached_queues[MAX_QUEUES] = {};
static int g_num_attached = 0;

// Intercept callback: count packets and forward them.
// This function signature exactly matches hsa_amd_queue_intercept_handler.
static void intercept_handler(const void* pkts, uint64_t pkt_count,
                               uint64_t user_pkt_index, void* data,
                               hsa_amd_queue_intercept_packet_writer writer) {
    g_packet_count.fetch_add(pkt_count, std::memory_order_relaxed);

    // Count kernel dispatch packets by inspecting AQL header type field
    const uint8_t* pkt_bytes = static_cast<const uint8_t*>(pkts);
    for (uint64_t i = 0; i < pkt_count; i++) {
        // AQL packet header: bits[7:0] = type
        uint8_t pkt_type = pkt_bytes[i * 64] & 0xFF;
        // HSA_PACKET_TYPE_KERNEL_DISPATCH = 2
        if (pkt_type == 2) {
            g_dispatch_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Forward all packets to the hardware queue
    writer(pkts, pkt_count);
}

extern "C" {

__attribute__((visibility("default")))
int trace_injector_start(hsa_queue_t** queues, int num_queues) {
    if (g_active.load()) {
        fprintf(stderr, "[trace_injector] Already active\n");
        return -1;
    }

    fprintf(stdout, "[trace_injector] Starting: attaching to %d queue(s)\n", num_queues);
    g_packet_count.store(0);
    g_dispatch_count.store(0);
    g_num_attached = 0;

    for (int i = 0; i < num_queues && i < MAX_QUEUES; i++) {
        hsa_status_t status = hsa_amd_queue_intercept_attach(
            queues[i], intercept_handler, nullptr);

        if (status != HSA_STATUS_SUCCESS) {
            const char* str = nullptr;
            hsa_status_string(status, &str);
            fprintf(stderr, "[trace_injector] Failed to attach to queue %d: %s (0x%x)\n",
                    i, str ? str : "unknown", status);
            // Detach any already-attached queues
            for (int j = 0; j < g_num_attached; j++) {
                hsa_amd_queue_intercept_detach(g_attached_queues[j]);
            }
            g_num_attached = 0;
            return -1;
        }

        g_attached_queues[g_num_attached++] = queues[i];
        fprintf(stdout, "[trace_injector] Attached to queue %d (handle=%p)\n",
                i, (void*)queues[i]);
    }

    g_active.store(true);
    fprintf(stdout, "[trace_injector] Active: intercepting %d queue(s)\n", g_num_attached);
    return 0;
}

__attribute__((visibility("default")))
int trace_injector_stop(int* trace_count) {
    if (!g_active.load()) {
        fprintf(stderr, "[trace_injector] Not active\n");
        if (trace_count) *trace_count = 0;
        return -1;
    }

    fprintf(stdout, "[trace_injector] Stopping: detaching from %d queue(s)\n", g_num_attached);

    int errors = 0;
    for (int i = 0; i < g_num_attached; i++) {
        hsa_status_t status = hsa_amd_queue_intercept_detach(g_attached_queues[i]);
        if (status != HSA_STATUS_SUCCESS) {
            const char* str = nullptr;
            hsa_status_string(status, &str);
            fprintf(stderr, "[trace_injector] Failed to detach queue %d: %s (0x%x)\n",
                    i, str ? str : "unknown", status);
            errors++;
        } else {
            fprintf(stdout, "[trace_injector] Detached queue %d\n", i);
        }
        g_attached_queues[i] = nullptr;
    }

    g_active.store(false);

    uint64_t total_packets = g_packet_count.load();
    uint64_t total_dispatches = g_dispatch_count.load();

    fprintf(stdout, "[trace_injector] Summary: %lu total packets, %lu kernel dispatches\n",
            total_packets, total_dispatches);

    if (trace_count) *trace_count = static_cast<int>(total_packets);
    g_num_attached = 0;

    return errors;
}

}  // extern "C"
