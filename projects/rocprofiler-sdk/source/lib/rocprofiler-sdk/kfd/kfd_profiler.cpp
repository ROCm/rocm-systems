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

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"

// NOTE: this vendored header carries the *active* (profiler ABI v6, stream ABI
// v3) dispatch-log UAPI: AMDKFD_IOC_PROFILER at 0x28 with the single OPEN_STREAM
// op. It deliberately conflicts with the older lib/rocprofiler-sdk/details/
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
// The reader issues the unified-profiler ioctls, which need this exact ABI. All
// dispatch-log ABI is unshipped and replaced in place, so a different profiler ABI
// -- older OR newer -- may re-lay-out the fixed-offset args this TU reads, and
// must not be silently accepted. Matches the stream ABI's exact-equality check.
constexpr uint32_t kProfilerAbiVersion = KFD_IOC_PROFILER_VERSION_NUM;

// Translation-unit-owned state. Not exposed as bare globals; only the accessor
// functions below read it. Reset by shutdown_kfd_profiler().
struct profiler_state
{
    // probe_ok is the platform capability; available additionally means the reader
    // thread and its fork handler exist. They are distinct because the reader is
    // started lazily, only once a consumer asks for a session.
    std::atomic<bool> probe_ok    = {false};
    std::atomic<bool> available   = {false};
    uint32_t          abi_version = 0;
    // Populated ONCE in init_kfd_profiler(), which runs inside interposition_init()
    // before the HSA API table is installed -- so before any queue creation or
    // context start can read it. Immutable and read-only thereafter, which is why
    // the lock-free reads (gpu_supports_dispatch_log, arm_dispatch_log_sessions)
    // need no mutex: the single write happens-before every read.
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

// a non-constructing peek. With the feature off, the SDK must construct
// no profiler_state at all -- so the query/no-op entry points, and the atfork
// child handler (which must not construct), go through this instead of state().
profiler_state*
state_or_null()
{
    return common::static_object<profiler_state>::get();
}

constexpr const char* kTopologyNodesPath = "/sys/class/kfd/kfd/topology/nodes";
}  // namespace

// Presence of dispatch_log_stream_format is the definitive per-GPU support check
// for the stream reader (CPU nodes and unsupported GPU archs do not have it).
// The legacy GFX9-only dispatch_log_format is deliberately NOT consulted: this
// reader speaks stream ABI v3 and must test the stream-specific contract. gpu_id
// == 0 denotes a CPU-only node and is skipped.
std::unordered_set<uint32_t>
discover_stream_dispatch_log_gpus(const char* nodes_path)
{
    std::unordered_set<uint32_t> gpu_ids;

    DIR* dir = opendir(nodes_path);
    if(dir == nullptr)
    {
        ROCP_INFO << "KFD dispatch-log: topology sysfs unavailable, using HSA timestamps";
        return gpu_ids;
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

        // Presence of dispatch_log_stream_format is the definitive support check.
        snprintf(path, sizeof(path), "%s/%s/dispatch_log_stream_format", nodes_path, entry->d_name);
        if(access(path, F_OK) == 0)
        {
            gpu_ids.insert(gpu_id);
            ROCP_INFO << fmt::format("KFD dispatch-log: gpu_id={} supported", gpu_id);
        }
    }

    closedir(dir);
    return gpu_ids;
}

void
init_kfd_profiler()
{
    auto& st = state();

    // Idempotent: if a previous call already succeeded, keep the result.
    if(st.available) return;

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
    if(st.abi_version != kProfilerAbiVersion)
    {
        ROCP_INFO << fmt::format(
            "KFD dispatch-log: profiler ABI version {} != required {}, using HSA timestamps",
            st.abi_version,
            kProfilerAbiVersion);
        return;
    }

    // ABI probe passed; discover which GPUs expose the stream dispatch-log.
    st.supported_gpu_ids = discover_stream_dispatch_log_gpus(kTopologyNodesPath);
    if(st.supported_gpu_ids.empty())
    {
        ROCP_INFO
            << "KFD dispatch-log: no GPU exposes dispatch_log_stream_format, using HSA timestamps";
        return;
    }

    st.probe_ok = true;
    ROCP_INFO << fmt::format("KFD dispatch-log: supported (ABI version {}, {} supported GPU(s))",
                             st.abi_version,
                             st.supported_gpu_ids.size());

    // Neither the reader thread nor the ring is created here: installing the HSA
    // table says nothing about whether anyone wants kernel traces, and the SDK
    // must create no internal thread when no tool consumes them. Both happen on
    // the first session request -- from a kernel-dispatch context starting, or
    // failing that from the first intercepted dispatch.
}

void
shutdown_kfd_profiler()
{
    // Feature never active in this process -> nothing was constructed. The reader is
    // only ever started after init_kfd_profiler() constructs this state (arming and
    // the dispatch path both run after HSA install), so a null state means a null
    // reader too. Return before stop_kfd_reader(), whose stop_reader() would
    // otherwise construct the reader_state singleton just to tear it down.
    if(state_or_null() == nullptr) return;

    stop_kfd_reader();

    if(auto* st = state_or_null())
    {
        st->available   = false;
        st->probe_ok    = false;
        st->abi_version = 0;
    }
}

void
disable_kfd_dispatch_log()
{
    // Reached from the atfork child handler, so it MUST NOT construct state; it
    // peeks and no-ops if this process never created a profiler.
    if(auto* st = state_or_null())
    {
        st->available = false;
        st->probe_ok  = false;
    }
}

bool
kfd_dispatch_log_available()
{
    auto* st = state_or_null();
    return st != nullptr && st->available.load();
}

bool
kfd_dispatch_log_supported()
{
    auto* st = state_or_null();
    return st != nullptr && st->probe_ok.load();
}

bool
kfd_profiler_state_constructed()
{
    return state_or_null() != nullptr;
}

void
note_kfd_reader_started()
{
    state().available = true;
}

bool
gpu_supports_dispatch_log(uint32_t gpu_id)
{
    const auto& ids = state().supported_gpu_ids;
    return ids.find(gpu_id) != ids.end();
}
}  // namespace kfd
}  // namespace rocprofiler
