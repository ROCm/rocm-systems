// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "generatePerfUserEvents.hpp"
#include "lib/common/logging.hpp"

#include <fcntl.h>
#include <linux/user_events.h>
#include <sys/ioctl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <cstring>

namespace rocprofiler
{
namespace tool
{
namespace
{
// Generic user event info structure
struct user_event_info
{
    int               fd                                         = -1;
    uint32_t          write_index                                = 0;
    uint32_t          enable_storage __attribute__((aligned(4))) = 0;
    std::atomic<bool> initialized{false};
};

// Event structure matching the kernel dispatch schema
struct kernel_dispatch_event
{
    uint64_t dispatch_id;
    uint64_t correlation_id;
    uint64_t kernel_id;
    uint64_t start_ts;
    uint64_t end_ts;
    uint32_t agent_id;
    uint32_t queue_id;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint16_t wg_x;
    uint16_t wg_y;
    uint16_t wg_z;
} __attribute__((packed));

// Event structure matching the HSA API schema
struct hsa_api_event
{
    uint32_t kind;
    uint32_t operation;
    uint64_t correlation_id;
    uint64_t start_ts;
    uint64_t end_ts;
    uint32_t thread_id;
} __attribute__((packed));

// Event structure matching the HIP API schema
struct hip_api_event
{
    uint32_t kind;
    uint32_t operation;
    uint64_t correlation_id;
    uint64_t start_ts;
    uint64_t end_ts;
    uint32_t thread_id;
} __attribute__((packed));

// Global state for the registered events
static user_event_info g_kernel_dispatch_event;
static user_event_info g_hsa_api_event;
static user_event_info g_hip_api_event;

// Helper function to register a user event
static bool
register_user_event(user_event_info& info, const char* event_def, const char* event_name)
{
    if(info.initialized.load())
    {
        return true;  // Already initialized
    }

    // Open the user_events_data file
    info.fd = open("/sys/kernel/tracing/user_events_data", O_RDWR);
    if(info.fd < 0)
    {
        ROCP_WARNING
            << "Failed to open /sys/kernel/tracing/user_events_data for event '" << event_name
            << "': " << strerror(errno)
            << ". This feature requires Linux kernel 5.18+ with CONFIG_USER_EVENTS enabled.";
        return false;
    }

    // Register the event
    struct user_reg reg = {};
    reg.size            = sizeof(reg);
    reg.name_args       = reinterpret_cast<uint64_t>(event_def);
    reg.enable_bit      = 0;
    reg.enable_size     = sizeof(info.enable_storage);
    reg.enable_addr     = reinterpret_cast<uint64_t>(&info.enable_storage);

    if(ioctl(info.fd, DIAG_IOCSREG, &reg) < 0)
    {
        ROCP_WARNING << "Failed to register perf user_event '" << event_name
                     << "': " << strerror(errno);
        close(info.fd);
        info.fd = -1;
        return false;
    }

    info.write_index = reg.write_index;
    info.initialized.store(true);

    ROCP_INFO << "Successfully registered perf user_event '" << event_name << "' "
              << "(write_index=" << info.write_index << "). "
              << "To capture events, run: perf record -e user_events:" << event_name << " -a "
              << "<command>";

    return true;
}

}  // namespace

bool
init_perf_user_events()
{
    // Define event schemas
    const char* kernel_dispatch_def =
        "rocprof_kernel_dispatch u64 dispatch_id; u64 correlation_id; "
        "u64 kernel_id; u64 start_ts; u64 end_ts; u32 agent_id; "
        "u32 queue_id; u32 grid_x; u32 grid_y; u32 grid_z; "
        "u16 wg_x; u16 wg_y; u16 wg_z";

    const char* hsa_api_def = "rocprof_hsa_api u32 kind; u32 operation; "
                              "u64 correlation; u64 start_ts; u64 end_ts; u32 thread_id";

    const char* hip_api_def = "rocprof_hip_api u32 kind; u32 operation; "
                              "u64 correlation; u64 start_ts; u64 end_ts; u32 thread_id";

    // Register all events
    bool kernel_ok = register_user_event(
        g_kernel_dispatch_event, kernel_dispatch_def, "rocprof_kernel_dispatch");
    bool hsa_ok = register_user_event(g_hsa_api_event, hsa_api_def, "rocprof_hsa_api");
    bool hip_ok = register_user_event(g_hip_api_event, hip_api_def, "rocprof_hip_api");

    // Return true if at least one event registered successfully
    return kernel_ok || hsa_ok || hip_ok;
}

void
cleanup_perf_user_events()
{
    // Cleanup kernel dispatch event
    if(g_kernel_dispatch_event.fd >= 0)
    {
        close(g_kernel_dispatch_event.fd);
        g_kernel_dispatch_event.fd = -1;
    }
    g_kernel_dispatch_event.initialized.store(false);
    g_kernel_dispatch_event.enable_storage = 0;

    // Cleanup HSA API event
    if(g_hsa_api_event.fd >= 0)
    {
        close(g_hsa_api_event.fd);
        g_hsa_api_event.fd = -1;
    }
    g_hsa_api_event.initialized.store(false);
    g_hsa_api_event.enable_storage = 0;

    // Cleanup HIP API event
    if(g_hip_api_event.fd >= 0)
    {
        close(g_hip_api_event.fd);
        g_hip_api_event.fd = -1;
    }
    g_hip_api_event.initialized.store(false);
    g_hip_api_event.enable_storage = 0;
}

void
write_perf_user_events(
    const output_config&                                               cfg,
    const metadata&                                                    tool_metadata,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>& kernel_dispatch_gen)
{
    (void) cfg;
    (void) tool_metadata;

    if(!g_kernel_dispatch_event.initialized.load() || g_kernel_dispatch_event.fd < 0)
    {
        ROCP_WARNING << "Perf user_event 'rocprof_kernel_dispatch' not initialized.";
        return;
    }

    // Check if tracepoint is enabled (someone is listening)
    // The kernel sets enable_storage to non-zero when tracepoint is enabled
    if(g_kernel_dispatch_event.enable_storage == 0)
    {
        ROCP_WARNING << "Perf user_event 'rocprof_kernel_dispatch' is registered but not enabled. "
                     << "No events will be written. "
                     << "To capture events, start perf before running rocprofv3: "
                     << "perf record -e user_events:rocprof_kernel_dispatch -a &";
        return;
    }

    ROCP_INFO << "Tracepoint enabled, writing kernel dispatch events...";

    // Iterate through kernel dispatch records and emit events
    uint64_t event_count   = 0;
    uint64_t total_records = 0;
    for(auto ditr : kernel_dispatch_gen)
    {
        for(const auto& record : kernel_dispatch_gen.get(ditr))
        {
            total_records++;

            // Populate the event structure
            struct kernel_dispatch_event evt = {};

            evt.dispatch_id    = record.dispatch_info.dispatch_id;
            evt.correlation_id = record.correlation_id.internal;
            evt.kernel_id      = record.dispatch_info.kernel_id;
            evt.start_ts       = record.start_timestamp;
            evt.end_ts         = record.end_timestamp;
            evt.agent_id       = record.dispatch_info.agent_id.handle;
            evt.queue_id       = record.dispatch_info.queue_id.handle;
            evt.grid_x         = record.dispatch_info.grid_size.x;
            evt.grid_y         = record.dispatch_info.grid_size.y;
            evt.grid_z         = record.dispatch_info.grid_size.z;
            evt.wg_x           = static_cast<uint16_t>(record.dispatch_info.workgroup_size.x);
            evt.wg_y           = static_cast<uint16_t>(record.dispatch_info.workgroup_size.y);
            evt.wg_z           = static_cast<uint16_t>(record.dispatch_info.workgroup_size.z);

            // Write event using writev
            struct iovec iov[2] = {
                {&g_kernel_dispatch_event.write_index, sizeof(g_kernel_dispatch_event.write_index)},
                {&evt, sizeof(evt)}};

            ssize_t ret = writev(g_kernel_dispatch_event.fd, iov, 2);
            if(ret < 0)
            {
                // Only log first failure to avoid spam
                if(event_count == 0 && errno != EBADF)
                {
                    ROCP_WARNING << "Failed to write perf user_event: " << strerror(errno);
                }
            }
            else
            {
                event_count++;
            }
        }
    }

    ROCP_INFO << "Wrote " << event_count << " of " << total_records
              << " kernel dispatch events to perf user_events";
}

void
write_hsa_api_events(const output_config&                                          cfg,
                     const metadata&                                               tool_metadata,
                     const generator<rocprofiler_buffer_tracing_hsa_api_record_t>& hsa_api_gen)
{
    (void) cfg;
    (void) tool_metadata;

    if(!g_hsa_api_event.initialized.load() || g_hsa_api_event.fd < 0)
    {
        ROCP_WARNING << "Perf user_event 'rocprof_hsa_api' not initialized.";
        return;
    }

    // Check if tracepoint is enabled (someone is listening)
    if(g_hsa_api_event.enable_storage == 0)
    {
        ROCP_WARNING << "Perf user_event 'rocprof_hsa_api' is registered but not enabled. "
                     << "No events will be written. "
                     << "To capture events, start perf before running rocprofv3: "
                     << "perf record -e user_events:rocprof_hsa_api -a &";
        return;
    }

    ROCP_INFO << "Tracepoint enabled, writing HSA API events...";

    // Iterate through HSA API records and emit events
    uint64_t event_count   = 0;
    uint64_t total_records = 0;
    for(auto ditr : hsa_api_gen)
    {
        for(const auto& record : hsa_api_gen.get(ditr))
        {
            total_records++;

            // Populate the event structure
            struct hsa_api_event evt = {};

            evt.kind           = record.kind;
            evt.operation      = record.operation;
            evt.correlation_id = record.correlation_id.internal;
            evt.start_ts       = record.start_timestamp;
            evt.end_ts         = record.end_timestamp;
            evt.thread_id      = record.thread_id;

            // Write event using writev
            struct iovec iov[2] = {
                {&g_hsa_api_event.write_index, sizeof(g_hsa_api_event.write_index)},
                {&evt, sizeof(evt)}};

            ssize_t ret = writev(g_hsa_api_event.fd, iov, 2);
            if(ret < 0)
            {
                // Only log first failure to avoid spam
                if(event_count == 0 && errno != EBADF)
                {
                    ROCP_WARNING << "Failed to write perf user_event: " << strerror(errno);
                }
            }
            else
            {
                event_count++;
            }
        }
    }

    ROCP_INFO << "Wrote " << event_count << " of " << total_records
              << " HSA API events to perf user_events";
}

void
write_hip_api_events(const output_config&                                       cfg,
                     const metadata&                                            tool_metadata,
                     const generator<tool_buffer_tracing_hip_api_ext_record_t>& hip_api_gen)
{
    (void) cfg;
    (void) tool_metadata;

    if(!g_hip_api_event.initialized.load() || g_hip_api_event.fd < 0)
    {
        ROCP_WARNING << "Perf user_event 'rocprof_hip_api' not initialized.";
        return;
    }

    // Check if tracepoint is enabled (someone is listening)
    if(g_hip_api_event.enable_storage == 0)
    {
        ROCP_WARNING << "Perf user_event 'rocprof_hip_api' is registered but not enabled. "
                     << "No events will be written. "
                     << "To capture events, start perf before running rocprofv3: "
                     << "perf record -e user_events:rocprof_hip_api -a &";
        return;
    }

    ROCP_INFO << "Tracepoint enabled, writing HIP API events...";

    // Iterate through HIP API records and emit events
    uint64_t event_count   = 0;
    uint64_t total_records = 0;
    for(auto ditr : hip_api_gen)
    {
        for(const auto& record : hip_api_gen.get(ditr))
        {
            total_records++;

            // Populate the event structure
            struct hip_api_event evt = {};

            evt.kind           = record.kind;
            evt.operation      = record.operation;
            evt.correlation_id = record.correlation_id.internal;
            evt.start_ts       = record.start_timestamp;
            evt.end_ts         = record.end_timestamp;
            evt.thread_id      = record.thread_id;

            // Write event using writev
            struct iovec iov[2] = {
                {&g_hip_api_event.write_index, sizeof(g_hip_api_event.write_index)},
                {&evt, sizeof(evt)}};

            ssize_t ret = writev(g_hip_api_event.fd, iov, 2);
            if(ret < 0)
            {
                // Only log first failure to avoid spam
                if(event_count == 0 && errno != EBADF)
                {
                    ROCP_WARNING << "Failed to write perf user_event: " << strerror(errno);
                }
            }
            else
            {
                event_count++;
            }
        }
    }

    ROCP_INFO << "Wrote " << event_count << " of " << total_records
              << " HIP API events to perf user_events";
}

}  // namespace tool
}  // namespace rocprofiler
