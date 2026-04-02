/*
 * Shared helpers for HSA queue intercept retrofit tests.
 */
#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <chrono>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_api_trace.h"

#define EXPECT_STATUS(msg, status, expected) do { \
    hsa_status_t s = (status); \
    if (s != (expected)) { \
        const char* str = nullptr; \
        hsa_status_string(s, &str); \
        fprintf(stderr, "FAIL: %s: expected 0x%x, got %s (0x%x)\n", \
                msg, expected, str ? str : "unknown", s); \
        failures++; \
    } else { \
        fprintf(stdout, "PASS: %s\n", msg); \
        passes++; \
    } \
} while(0)

#define EXPECT_TRUE(msg, cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        failures++; \
    } else { \
        fprintf(stdout, "PASS: %s\n", msg); \
        passes++; \
    } \
} while(0)

// Find the first GPU agent
inline hsa_status_t find_gpu_agent_cb(hsa_agent_t agent, void* data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t*)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

inline bool find_gpu_agent(hsa_agent_t* agent) {
    agent->handle = 0;
    hsa_status_t status = hsa_iterate_agents(find_gpu_agent_cb, agent);
    return (status == HSA_STATUS_INFO_BREAK && agent->handle != 0);
}

// Submit a barrier packet to a queue (simple, no signal wait)
inline void submit_barrier(hsa_queue_t* queue) {
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    uint64_t mask = queue->size - 1;
    hsa_barrier_and_packet_t* barrier =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
                                    (write_idx & mask) * 64);
    memset(barrier, 0, sizeof(*barrier));
    barrier->header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier->header |= (1 << HSA_PACKET_HEADER_BARRIER);
    hsa_signal_store_screlease(queue->doorbell_signal, write_idx);
}

#endif // TEST_HELPERS_H
