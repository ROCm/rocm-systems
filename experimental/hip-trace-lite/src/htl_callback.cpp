// htl_callback.cpp
#include "htl_callback.hpp"
#include "htl_prof_protocol.hpp"

#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>

namespace htl {

Writer*               g_writer = nullptr;
std::atomic<bool>     g_capture_hip_ops{true};
std::atomic<bool>     g_capture_hip_api{false};

namespace {
inline uint32_t gettid_cached() {
    static thread_local uint32_t tid =
        static_cast<uint32_t>(::syscall(SYS_gettid));
    return tid;
}
}  // namespace

// CLR contract:
//
// HIP_OPS path (rocclr/platform/activity.cpp):
//   - Probe: report(HIP_OPS, op, nullptr) — return 0 to enable, non-zero to skip.
//   - Submit sentinel: report(HIP_OPS, 0x1, sentinel_ptr) — informational, return value ignored.
//   - Record: report(HIP_OPS, op, &activity_record_t) — process and return.
//
// HIP_API path (hipamd/src/hip_prof_api.h api_callbacks_spawner_t):
//   - Per-call probe: report(HIP_API, op, &trace_data_) — return 0 to OPT IN, in
//     which case CLR calls trace_data_.phase_enter (must be set by callee, else
//     CLR dereferences a possibly-uninitialised pointer and crashes). We do NOT
//     yet support providing phase_enter/phase_exit hooks, so we MUST always
//     return non-zero for HIP_API to opt out of the spawner protocol.
extern "C" int htl_tracer_callback(uint32_t domain, uint32_t op, void* data) {
    // HIP_API: opt out unconditionally. Setting phase_enter/phase_exit hooks
    // is a v2 feature; returning 0 here without setting them crashes CLR.
    if (domain == ACTIVITY_DOMAIN_HIP_API) return 1;

    // Only HIP_OPS handled below.
    if (domain != ACTIVITY_DOMAIN_HIP_OPS) return 1;

    // OPS enablement probe.
    if (data == nullptr) {
        return g_capture_hip_ops.load(std::memory_order_relaxed) ? 0 : 1;
    }

    // OPS commit-record sentinel: CLR passes data == (void*)0x1 to inform
    // tracers that an op was submitted. Real records carry a real pointer.
    // See projects/clr/rocclr/platform/activity.cpp:22.
    if (reinterpret_cast<uintptr_t>(data) == kCommitRecordSentinelValue) return 0;

    if (g_writer == nullptr) return 0;
    if (!g_capture_hip_ops.load(std::memory_order_relaxed)) return 0;

    const auto* rec = static_cast<const activity_record_prefix_t*>(data);

    slot_t s{};
    s.rec.domain         = static_cast<uint8_t>(domain & 0xff);
    s.rec.op             = static_cast<uint8_t>(op & 0xff);
    s.rec.flags          = 0;
    s.rec.reserved0      = 0;
    s.rec.correlation_id = rec->correlation_id;
    s.rec.begin_ns       = rec->begin_ns;
    s.rec.end_ns         = rec->end_ns;
    s.rec.process_id     = static_cast<uint32_t>(::getpid());
    s.rec.thread_id      = gettid_cached();
    s.rec.device_id      = rec->device_id;
    s.rec.queue_id       = static_cast<uint32_t>(rec->queue_id & 0xffffffffu);

    if (op == HIP_OP_ID_DISPATCH) {
        s.rec.bytes = 0;
        if (rec->kernel_name) {
            std::strncpy(s.name, rec->kernel_name, sizeof(s.name) - 1);
            s.name[sizeof(s.name) - 1] = '\0';
        }
    } else if (op == HIP_OP_ID_COPY) {
        s.rec.bytes = rec->bytes;
    } else {
        s.rec.bytes = 0;
    }

    g_writer->enqueue(s);
    return 0;
}

}  // namespace htl
