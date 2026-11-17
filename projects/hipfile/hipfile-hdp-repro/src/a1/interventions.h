// Interventions applied before/after a write into VRAM, selected by comma-separated tokens.
// Shared by barprobe.cpp (HIP-only) and the instrumented hipfileread-cpu-visibility test.
#pragma once
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#ifdef A1_WITH_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#ifdef __HIPCC__
__global__ void a1_empty_kernel() {}
#endif

namespace a1 {

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline std::vector<std::string> split(const char *s) {
    std::vector<std::string> out;
    if (!s) return out;
    std::string cur;
    for (const char *p = s; *p; p++) {
        if (*p == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur += *p;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

struct Ctx {
    hipStream_t sblk{}, snb{};
    hipEvent_t  edef{}, esys{}, edev{}, enofence{};
    void       *other{}, *other2{};
    uint32_t   *hsmall{};
    volatile uint32_t *hdpmem{}, *hdpreg{};
#ifdef A1_WITH_HSA
    hsa_agent_t gpu{}, cpu{};
    hsa_signal_t sig{};
    hsa_queue_t *q{};
    bool hsa_ok{false};
#endif
};

#ifdef A1_WITH_HSA
inline hsa_status_t a1_agent_cb(hsa_agent_t a, void *data) {
    auto *c = static_cast<Ctx *>(data);
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU && c->gpu.handle == 0) c->gpu = a;
    if (t == HSA_DEVICE_TYPE_CPU && c->cpu.handle == 0) c->cpu = a;
    return HSA_STATUS_SUCCESS;
}
#endif

inline void init(Ctx &c, bool verbose) {
    (void)hipStreamCreate(&c.sblk);
    (void)hipStreamCreateWithFlags(&c.snb, hipStreamNonBlocking);
    (void)hipEventCreateWithFlags(&c.edef, hipEventDefault);
    (void)hipEventCreateWithFlags(&c.esys, hipEventReleaseToSystem);
    (void)hipEventCreateWithFlags(&c.edev, hipEventReleaseToDevice);
    (void)hipEventCreateWithFlags(&c.enofence, hipEventDisableSystemFence);
    (void)hipMalloc(&c.other, 4096);
    (void)hipMalloc(&c.other2, 4096);
    (void)hipHostMalloc(reinterpret_cast<void **>(&c.hsmall), 4096, hipHostMallocDefault);
    (void)hipMemset(c.other, 0, 4096);
    (void)hipMemset(c.other2, 0, 4096);
    (void)hipDeviceSynchronize();
    unsigned int *mem{}, *reg{};
    hipError_t e1 = hipDeviceGetAttribute(reinterpret_cast<int *>(&mem), hipDeviceAttributeHdpMemFlushCntl, 0);
    hipError_t e2 = hipDeviceGetAttribute(reinterpret_cast<int *>(&reg), hipDeviceAttributeHdpRegFlushCntl, 0);
    c.hdpmem = mem;
    c.hdpreg = reg;
#ifdef A1_WITH_HSA
    if (hsa_init() == HSA_STATUS_SUCCESS) {
        hsa_iterate_agents(a1_agent_cb, &c);
        hsa_amd_hdp_flush_t f{};
        hsa_status_t s = hsa_agent_get_info(c.gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH), &f);
        if (verbose)
            fprintf(stderr, "# HSA HDP_FLUSH status=%d MEM=%p REG=%p; HIP attr (err %d/%d) MEM=%p REG=%p\n", s,
                    (void *)f.HDP_MEM_FLUSH_CNTL, (void *)f.HDP_REG_FLUSH_CNTL, e1, e2, (void *)mem, (void *)reg);
        if (!c.hdpmem) c.hdpmem = f.HDP_MEM_FLUSH_CNTL;
        if (!c.hdpreg) c.hdpreg = f.HDP_REG_FLUSH_CNTL;
        hsa_signal_create(1, 0, nullptr, &c.sig);
        if (std::getenv("A1_AQL")) {
            hsa_status_t qs = hsa_queue_create(c.gpu, 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                                               UINT32_MAX, &c.q);
            if (verbose) fprintf(stderr, "# own AQL queue status=%d\n", qs);
        }
        c.hsa_ok = true;
    }
#else
    if (verbose)
        fprintf(stderr, "# HIP attr (err %d/%d) HDP MEM=%p REG=%p\n", e1, e2, (void *)mem, (void *)reg);
#endif
}

#ifdef A1_WITH_HSA
// SDMA copy on a specific engine through ROCr directly (bypasses ROCclr)
inline void hsa_sdma_copy(Ctx &c, void *dst, const void *src, size_t size, bool force_sdma = true) {
    hsa_signal_store_screlease(c.sig, 1);
    hsa_status_t s = force_sdma
        ? hsa_amd_memory_async_copy_on_engine(dst, c.gpu, src, c.gpu, size, 0, nullptr, c.sig,
                                              HSA_AMD_SDMA_ENGINE_0, true)
        : hsa_amd_memory_async_copy(dst, c.gpu, src, c.gpu, size, 0, nullptr, c.sig);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "hsa copy failed %d\n", s); std::exit(2); }
    hsa_signal_wait_scacquire(c.sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
}
// Submit one BARRIER_AND packet on our own AQL queue (created when A1_AQL is set) with the given
// acquire/release fence scopes, and wait for its completion signal.
inline void aql_barrier(Ctx &c, hsa_fence_scope_t acq, hsa_fence_scope_t rel) {
    if (!c.q) { fprintf(stderr, "set A1_AQL=1 for aql tokens\n"); std::exit(2); }
    hsa_signal_store_screlease(c.sig, 1);
    const uint64_t idx = hsa_queue_add_write_index_screlease(c.q, 1);
    auto *pkt = static_cast<hsa_barrier_and_packet_t *>(c.q->base_address) + (idx % c.q->size);
    std::memset(reinterpret_cast<char *>(pkt) + 4, 0, sizeof(*pkt) - 4);
    pkt->completion_signal = c.sig;
    const uint16_t header = uint16_t((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                     (1 << HSA_PACKET_HEADER_BARRIER) |
                                     (acq << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                                     (rel << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE));
    __atomic_store_n(reinterpret_cast<uint32_t *>(pkt), uint32_t(header), __ATOMIC_RELEASE);
    hsa_signal_store_screlease(c.q->doorbell_signal, int64_t(idx));
    hsa_signal_wait_scacquire(c.sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
}
#endif

// Mirrors hipDeviceFlushGPUDirectRDMAWrites (clr hip_device_runtime.cpp:543-545): write 1, read back.
inline void hdp_write(volatile uint32_t *r, uint32_t v) {
    if (!r) { fprintf(stderr, "no HDP register pointer\n"); std::exit(2); }
    *r = v;  // write only: no read-back (upstream amdgpu a5cb344033c7)
}

inline void spin_us(long us) {
    const uint64_t end = now_ns() + uint64_t(us) * 1000;
    while (now_ns() < end) {}
}

// Returns false for unknown tokens
inline bool apply(Ctx &c, const std::string &tok) {
    const std::string_view t{tok};
    auto arg = [&](std::string_view p) { return std::atol(tok.c_str() + p.size()); };
    if (t == "none") {}
    else if (t == "sync0") (void)hipStreamSynchronize(0);
    else if (t == "syncblk") (void)hipStreamSynchronize(c.sblk);
    else if (t == "syncnb") (void)hipStreamSynchronize(c.snb);
    else if (t == "devsync") (void)hipDeviceSynchronize();
    else if (t == "evt") { (void)hipEventRecord(c.edef, 0); (void)hipEventSynchronize(c.edef); }
    else if (t == "evtsys") { (void)hipEventRecord(c.esys, 0); (void)hipEventSynchronize(c.esys); }
    else if (t == "evtdev") { (void)hipEventRecord(c.edev, 0); (void)hipEventSynchronize(c.edev); }
    else if (t == "evtnofence") { (void)hipEventRecord(c.enofence, 0); (void)hipEventSynchronize(c.enofence); }
    else if (t == "waitevt") { (void)hipEventRecord(c.edef, 0); (void)hipStreamWaitEvent(c.snb, c.edef, 0); (void)hipStreamSynchronize(c.snb); }
    else if (t == "waitevtnosync") { (void)hipEventRecord(c.edef, 0); (void)hipStreamWaitEvent(c.snb, c.edef, 0); }
    else if (t == "hdpmem") hdp_write(c.hdpmem, 1);
    else if (t == "hdpmem0") hdp_write(c.hdpmem, 0);
    else if (t == "hdpreg") hdp_write(c.hdpreg, 1);
    else if (t == "hdpmemnoread") { *c.hdpmem = 1; }
#ifdef __HIPCC__
    else if (t == "knull") a1_empty_kernel<<<1, 1, 0, 0>>>();
    else if (t == "knb") a1_empty_kernel<<<1, 1, 0, c.snb>>>();
    else if (t == "kblk") a1_empty_kernel<<<1, 1, 0, c.sblk>>>();
#endif
    else if (t == "memset") (void)hipMemset(c.other, 0, 4);
    else if (t == "memsetasync") (void)hipMemsetAsync(c.other, 0, 4, 0);
    else if (t == "memsetasyncnb") (void)hipMemsetAsync(c.other, 0, 4, c.snb);
    else if (t == "h2d") (void)hipMemcpy(c.other, c.hsmall, 4, hipMemcpyHostToDevice);
    else if (t == "d2h") (void)hipMemcpy(c.hsmall, c.other, 4, hipMemcpyDeviceToHost);
    else if (t == "d2d") (void)hipMemcpy(c.other2, c.other, 4, hipMemcpyDeviceToDevice);
    else if (t == "h2dasync") (void)hipMemcpyAsync(c.other, c.hsmall, 4, hipMemcpyHostToDevice, 0);
    else if (t == "d2dasync") (void)hipMemcpyAsync(c.other2, c.other, 4, hipMemcpyDeviceToDevice, 0);
    else if (t == "squery") (void)hipStreamQuery(0);
    else if (t == "squerynb") (void)hipStreamQuery(c.snb);
#ifdef A1_WITH_HSA
    else if (t == "hsasdma") hsa_sdma_copy(c, c.other2, c.other, 4, true);      // D2D on SDMA engine 0
    else if (t == "hsasdmah2d") hsa_sdma_copy(c, c.other, c.hsmall, 4, true);  // H2D on SDMA engine 0
    else if (t == "hsacopy") hsa_sdma_copy(c, c.other2, c.other, 4, false);    // ROCr picks engine
    else if (t == "aqlnone") aql_barrier(c, HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_NONE);
    else if (t == "aqlacqsys") aql_barrier(c, HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_NONE);
    else if (t == "aqlrelsys") aql_barrier(c, HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_SYSTEM);
    else if (t == "aqlbothsys") aql_barrier(c, HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_SYSTEM);
    else if (t == "aqlacqagent") aql_barrier(c, HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_NONE);
    else if (t == "aqlrelagent") aql_barrier(c, HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_AGENT);
#endif
    else if (t.starts_with("usleep:")) std::this_thread::sleep_for(std::chrono::microseconds(arg("usleep:")));
    else if (t.starts_with("spin:")) spin_us(arg("spin:"));
    else if (t == "cpubarwrite") { *static_cast<volatile uint32_t *>(c.other) = 0; } // CPU write to unrelated VRAM via BAR
    else if (t == "cpubarread") { (void)*static_cast<volatile uint32_t *>(c.other); }
    else return false;
    return true;
}

inline void apply_all(Ctx &c, const std::vector<std::string> &toks) {
    for (auto &t : toks)
        if (!apply(c, t)) { fprintf(stderr, "unknown token %s\n", t.c_str()); std::exit(2); }
}

} // namespace a1
