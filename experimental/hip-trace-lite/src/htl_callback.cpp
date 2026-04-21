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

extern "C" int htl_tracer_callback(uint32_t domain, uint32_t op, void* data) {
    // Enablement probe: data == nullptr.
    if (data == nullptr) {
        if (domain == ACTIVITY_DOMAIN_HIP_OPS && g_capture_hip_ops.load(std::memory_order_relaxed)) return 1;
        if (domain == ACTIVITY_DOMAIN_HIP_API && g_capture_hip_api.load(std::memory_order_relaxed)) return 1;
        return 0;
    }

    // CLR also fires the callback for HIP_OPS with op == 0x1 and data != nullptr
    // as a "submitted" counter (kCommitRecordSentinel). We treat it as a no-op.
    if (domain == ACTIVITY_DOMAIN_HIP_OPS && op == kEnablementProbeOp) return 0;

    if (g_writer == nullptr) return 0;

    if (domain != ACTIVITY_DOMAIN_HIP_OPS && domain != ACTIVITY_DOMAIN_HIP_API)
        return 0;
    if (domain == ACTIVITY_DOMAIN_HIP_API &&
        !g_capture_hip_api.load(std::memory_order_relaxed)) return 0;
    if (domain == ACTIVITY_DOMAIN_HIP_OPS &&
        !g_capture_hip_ops.load(std::memory_order_relaxed)) return 0;

    const auto* rec = static_cast<const activity_record_prefix_t*>(data);

    slot_t s{};
    s.rec.domain         = static_cast<uint8_t>(domain & 0xff);
    s.rec.op             = static_cast<uint8_t>(op & 0xff);
    s.rec.flags          = 0;
    s.rec.reserved0      = 0;
    s.rec.correlation_id = rec->correlation_id;
    s.rec.begin_ns       = rec->begin_ns;
    s.rec.end_ns         = rec->end_ns;

    // The upstream activity_record_t uses a tagged union for
    //   {device_id, queue_id} (HIP_OPS) vs {process_id, thread_id} (HIP_API).
    // Read only the union arm matching the domain.
    if (domain == ACTIVITY_DOMAIN_HIP_OPS) {
        s.rec.process_id = static_cast<uint32_t>(::getpid());
        s.rec.thread_id  = gettid_cached();
        s.rec.device_id  = rec->device_id;
        s.rec.queue_id   = static_cast<uint32_t>(rec->queue_id & 0xffffffffu);
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
    } else {  // ACTIVITY_DOMAIN_HIP_API
        s.rec.process_id = rec->process_id;
        s.rec.thread_id  = rec->thread_id ? rec->thread_id : gettid_cached();
        s.rec.device_id  = -1;
        s.rec.queue_id   = 0;
        s.rec.bytes      = 0;
    }

    g_writer->enqueue(s);
    return 0;
}

}  // namespace htl
