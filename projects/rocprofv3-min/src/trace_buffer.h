// Per-domain in-memory trace row buffers, flushed to CSV on shutdown.
#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofv3_min {

struct HipApiRow {
    const char* domain;          // "HIP_API" or "HIP_COMPILER_API"
    const char* function_name;   // pointer to static literal
    uint32_t    process_id;
    uint32_t    thread_id;
    uint64_t    correlation_id;
    uint64_t    start_ns;
    uint64_t    end_ns;
};

struct KernelRow {
    const char* kind;            // "KERNEL_DISPATCH"
    uint64_t    agent_id;
    uint64_t    queue_id;
    uint32_t    thread_id;
    uint64_t    dispatch_id;
    uint64_t    kernel_id;       // kernel_object handle
    std::string kernel_name;     // typically empty (resolution stubbed)
    uint64_t    correlation_id;
    uint64_t    start_ns;
    uint64_t    end_ns;
    uint32_t    private_segment_size;
    uint32_t    group_segment_size;
    uint16_t    workgroup_size_x;
    uint16_t    workgroup_size_y;
    uint16_t    workgroup_size_z;
    uint32_t    grid_size_x;
    uint32_t    grid_size_y;
    uint32_t    grid_size_z;
};

struct CopyRow {
    const char* kind;            // "MEMORY_COPY"
    const char* direction;       // "ASYNC" / "ASYNC_ON_ENGINE"
    uint64_t    source_agent_id;
    uint64_t    destination_agent_id;
    uint64_t    correlation_id;
    uint64_t    start_ns;
    uint64_t    end_ns;
    uint64_t    bytes;
};

class TraceBuffers {
public:
    static TraceBuffers& instance();

    void push_hip(HipApiRow&& row);
    void push_kernel(KernelRow&& row);
    void push_copy(CopyRow&& row);

    void flush(const std::string& out_dir, uint32_t pid);

    uint64_t next_correlation_id() { return ++correlation_; }

private:
    std::mutex hip_mu_;
    std::mutex kern_mu_;
    std::mutex copy_mu_;
    std::vector<HipApiRow> hip_rows_;
    std::vector<KernelRow> kern_rows_;
    std::vector<CopyRow>   copy_rows_;
    std::atomic<uint64_t>  correlation_{0};
};

} // namespace rocprofv3_min
