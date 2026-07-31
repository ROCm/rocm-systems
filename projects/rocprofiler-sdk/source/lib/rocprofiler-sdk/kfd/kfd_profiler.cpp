// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"

// NOTE: this vendored header carries the *active* (v5) dispatch-log profiler
// ABI: AMDKFD_IOC_PROFILER at 0x28 with the REGISTER_BUFFER / OPEN_STREAM
// union. It deliberately conflicts with the older lib/rocprofiler-sdk/details/
// kfd_ioctl.h (VERSION_NUM 1, ioctl 0x86, no dlog ops). The two must never be
// included in the same translation unit. This file includes ONLY the dlog UAPI.
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <fmt/core.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace rocprofiler
{
namespace kfd
{
namespace
{
// Minimum profiler ABI. The reader issues the unified-profiler and stream-fd
// ioctl encodings vendored in kfd_dlog_uapi.h, so anything older than that
// header's ABI would be sent incompatible ioctls.
constexpr uint32_t kMinProfilerAbiVersion = KFD_IOC_PROFILER_VERSION_NUM;

// Translation-unit-owned state. Not exposed as bare globals; only the accessor
// functions below read it. Reset by shutdown_kfd_profiler().
struct profiler_state
{
    std::atomic<bool>            available   = {false};
    uint32_t                     abi_version = 0;
    std::unordered_set<uint32_t> supported_gpu_ids;
};

// Singleton via common::static_object (matches kfd.cpp get_node_map / tool.cpp):
// placement-new into a static buffer, ordered teardown via register_static_dtor,
// no heap leak and no unspecified static-destructor ordering at library unload.
profiler_state&
state()
{
    static auto*& _v = common::static_object<profiler_state>::construct();
    return *_v;
}

// Populate state().supported_gpu_ids by walking the KFD topology and testing
// for the dispatch_log_format sysfs node. Presence of that file is the
// definitive per-GPU support check (CPU nodes and unsupported GPU archs do not
// have it). gpu_id == 0 denotes a CPU-only node and is skipped.
void
discover_dispatch_log_gpus()
{
    auto& st = state();
    st.supported_gpu_ids.clear();

    static constexpr const char* nodes_path = "/sys/class/kfd/kfd/topology/nodes";

    DIR* dir = opendir(nodes_path);
    if(dir == nullptr)
    {
        ROCP_INFO << "KFD dispatch-log: topology sysfs unavailable, using HSA timestamps";
        return;
    }

    struct dirent* entry = nullptr;
    while((entry = readdir(dir)) != nullptr)
    {
        if(entry->d_name[0] == '.') continue;

        char path[PATH_MAX];

        // Read gpu_id; 0 == CPU-only node, skip.
        snprintf(path, sizeof(path), "%s/%s/gpu_id", nodes_path, entry->d_name);
        uint32_t gpu_id = 0;
        if(FILE* f = fopen(path, "r"); f != nullptr)
        {
            if(fscanf(f, "%u", &gpu_id) != 1) gpu_id = 0;
            fclose(f);
        }
        if(gpu_id == 0) continue;

        // Presence of dispatch_log_format is the definitive support check.
        snprintf(path, sizeof(path), "%s/%s/dispatch_log_format", nodes_path, entry->d_name);
        if(access(path, F_OK) == 0)
        {
            st.supported_gpu_ids.insert(gpu_id);
            ROCP_INFO << fmt::format("KFD dispatch-log: gpu_id={} supported", gpu_id);
        }
    }

    closedir(dir);
}
}  // namespace

void
init_kfd_profiler()
{
    auto& st = state();

    // Idempotent: if a previous call already succeeded, keep the result.
    if(st.available) return;

    // Respect user opt-out before touching /dev/kfd.
    if(!common::get_env("ROCPROFILER_KFD_DISPATCH_LOG", true))
    {
        ROCP_INFO << "KFD dispatch-log: disabled by ROCPROFILER_KFD_DISPATCH_LOG=0";
        return;
    }

    // Private probe fd, used only for the VERSION ioctl below; the reader opens
    // its own.
    int probe_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if(probe_fd < 0)
    {
        ROCP_INFO << "KFD dispatch-log: /dev/kfd unavailable, using HSA timestamps";
        return;
    }

    // Probe the profiler ABI version.
    struct kfd_ioctl_profiler_args args = {};
    args.op                             = KFD_IOC_PROFILER_VERSION;
    const int probe_rc                  = ioctl(probe_fd, AMDKFD_IOC_PROFILER, &args);
    ::close(probe_fd);
    if(probe_rc != 0)
    {
        ROCP_INFO << "KFD dispatch-log: AMDKFD_IOC_PROFILER not supported, using HSA timestamps";
        return;
    }

    st.abi_version = args.version;
    if(st.abi_version < kMinProfilerAbiVersion)
    {
        ROCP_INFO << fmt::format(
            "KFD dispatch-log: profiler ABI version {} < {}, using HSA timestamps",
            st.abi_version,
            kMinProfilerAbiVersion);
        return;
    }

    // ABI probe passed; discover which GPUs expose dispatch-log.
    discover_dispatch_log_gpus();
    if(st.supported_gpu_ids.empty())
    {
        ROCP_INFO << "KFD dispatch-log: no GPU exposes dispatch_log_format, using HSA timestamps";
        return;
    }

    // Publish availability only once the reader thread and its fork handler exist:
    // the destroy_queue path uses it to decide whether to take the DoorbellMap lock.
    if(!start_kfd_reader()) return;

    st.available = true;
    ROCP_INFO << fmt::format("KFD dispatch-log: available (ABI version {}, {} supported GPU(s))",
                             st.abi_version,
                             st.supported_gpu_ids.size());
}

void
shutdown_kfd_profiler()
{
    // Stop the reader thread before tearing down the KFD session/state.
    stop_kfd_reader();

    auto& st       = state();
    st.available   = false;
    st.abi_version = 0;
    st.supported_gpu_ids.clear();
}

void
disable_kfd_dispatch_log()
{
    // Reached from the atfork child handler: state() is always already
    // constructed (the handler is registered only after init succeeded), so this
    // is a single lock-free atomic store and nothing else.
    state().available = false;
}

bool
kfd_dispatch_log_available()
{
    return state().available;
}

bool
gpu_supports_dispatch_log(uint32_t gpu_id)
{
    const auto& ids = state().supported_gpu_ids;
    return ids.find(gpu_id) != ids.end();
}
}  // namespace kfd
}  // namespace rocprofiler
