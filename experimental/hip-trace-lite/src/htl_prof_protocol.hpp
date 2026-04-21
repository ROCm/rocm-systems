// htl_prof_protocol.hpp — inline minimal copy of the activity_record_t layout
// we consume from CLR via hipRegisterTracerCallback. No dependency on the
// roctracer source tree. Field layout matches projects/roctracer/inc/ext/prof_protocol.h.
#pragma once

#include <cstdint>

namespace htl {

enum activity_domain_t : uint32_t {
    ACTIVITY_DOMAIN_HSA_API   = 0,
    ACTIVITY_DOMAIN_HSA_OPS   = 1,
    ACTIVITY_DOMAIN_HIP_OPS   = 2,
    ACTIVITY_DOMAIN_HIP_API   = 3,
    ACTIVITY_DOMAIN_EXT_API   = 5,
    ACTIVITY_DOMAIN_ROCTX     = 6,
};

// Op IDs CLR reports for ACTIVITY_DOMAIN_HIP_OPS.
enum hip_op_id_t : uint32_t {
    HIP_OP_ID_DISPATCH = 0,
    HIP_OP_ID_COPY     = 1,
    HIP_OP_ID_BARRIER  = 2,
};

// CLR's CommitRecord sentinel: when CLR commits a record, it passes
// data = (void*)0x1 with the actual op. A real record is passed with
// data = pointer to a real activity_record_t.  See
// projects/clr/rocclr/platform/activity.cpp:22 (kCommitRecordSentinel).
inline constexpr uintptr_t kCommitRecordSentinelValue = 0x1;

// Layout of activity_record_t prefix that we read. Any fields beyond
// kernel_name we ignore. DO NOT add fields without verifying against
// projects/roctracer/inc/ext/prof_protocol.h.
struct activity_record_prefix_t {
    uint32_t domain;
    uint32_t kind;
    uint32_t op;
    uint64_t correlation_id;
    uint64_t begin_ns;
    uint64_t end_ns;
    union {
        struct {
            int      device_id;
            uint64_t queue_id;
        };
        struct {
            uint32_t process_id;
            uint32_t thread_id;
        };
    };
    union {
        uint64_t bytes;
        const char* kernel_name;
    };
};

// Callback signature CLR calls.
using tracer_callback_fn_t = int (*)(uint32_t domain, uint32_t op, void* data);

}  // namespace htl
