#include "test_helpers.h"
#include <dlfcn.h>

static std::atomic<uint64_t> g_count{0};
static std::atomic<bool> g_called{false};

void cb(const void* pkts, uint64_t pkt_count, uint64_t idx, void* data,
        hsa_amd_queue_intercept_packet_writer writer) {
    fprintf(stderr, "CALLBACK: %lu packets\n", (unsigned long)pkt_count);
    g_called.store(true);
    g_count.fetch_add(pkt_count);
    writer(pkts, pkt_count);
}

int main() {
    int failures=0, passes=0;
    hsa_init();
    hsa_agent_t gpu; find_gpu_agent(&gpu);

    hsa_queue_t* q = nullptr;
    hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q);

    fprintf(stderr, "Before attach: base=%p doorbell=0x%lx\n", q->base_address, q->doorbell_signal.handle);

    hsa_status_t s = hsa_amd_queue_intercept_attach(q, cb, nullptr);
    fprintf(stderr, "Attach returned: 0x%x\n", s);
    fprintf(stderr, "After attach: base=%p doorbell=0x%lx\n", q->base_address, q->doorbell_signal.handle);
    fprintf(stderr, "Queue size: %u\n", q->size);

    // Write a barrier
    uint64_t wi = hsa_queue_add_write_index_relaxed(q, 1);
    fprintf(stderr, "write_index after add: %lu\n", (unsigned long)wi);
    uint64_t mask = q->size - 1;
    hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)((char*)q->base_address + (wi & mask) * 64);
    memset(pkt, 0, sizeof(*pkt));
    pkt->header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    pkt->header |= (1 << HSA_PACKET_HEADER_BARRIER);

    fprintf(stderr, "Ringing doorbell with value %lu...\n", (unsigned long)wi);
    hsa_signal_store_screlease(q->doorbell_signal, wi);
    fprintf(stderr, "Doorbell rung. Waiting...\n");

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    fprintf(stderr, "g_called=%d g_count=%lu\n", (int)g_called.load(), (unsigned long)g_count.load());

    hsa_amd_queue_intercept_detach(q);
    hsa_queue_destroy(q);
    hsa_shut_down();
    return 0;
}
