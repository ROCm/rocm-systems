// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
constexpr uint64_t kPasses = 4;

rocprofiler_context_id_t g_replay_ctx{0};
std::atomic<int>         g_replayed{0};
std::atomic<int>         g_config_exits{0};
std::atomic<int>         g_passes_seen{0};
std::atomic<int>         g_pass_options_seen{0};
std::atomic<int>         g_bad_options{0};

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

bool
placement_is(const char* name)
{
    const char* placement = std::getenv("KR_SNAPSHOT");
    return placement != nullptr && std::strcmp(placement, name) == 0;
}

// KR_SNAPSHOT=host-pinned, device or device-and-host-pinned limits where the SDK keeps the
// snapshot, and KR_SNAPSHOT=unknown-flag sets a flag the SDK does not know, so the dispatch runs
// once.
uint64_t
snapshot_flags()
{
    if(placement_is("host-pinned")) return ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PINNED;
    if(placement_is("device")) return ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DEVICE;
    if(placement_is("device-and-host-pinned"))
        return ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DEVICE |
               ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PINNED;
    if(placement_is("unknown-flag")) return uint64_t{1} << 63;
    return ROCPROFILER_KERNEL_REPLAY_OPTION_NONE;
}

uint64_t
expected_passes()
{
    return placement_is("unknown-flag") ? 0 : kPasses;
}

// KR_REPLAYS: how many times the application dispatches the replayed kernel (default 1)
int
expected_replays()
{
    const char* replays = std::getenv("KR_REPLAYS");
    return replays != nullptr ? std::atoi(replays) : 1;
}

// An older SDK's record ends before replay_options.
bool
has_replay_options(const rocprofiler_callback_tracing_kernel_replay_data_t* p)
{
    return p->size >= offsetof(rocprofiler_callback_tracing_kernel_replay_data_t, replay_options) +
                          sizeof(p->replay_options);
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(p->dispatch_info.workgroup_size.x != kReplayBlockX) return;

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        p->replay_pass_count = replay_pass_count;
        if(has_replay_options(p))
        {
            // The SDK provides options holding its defaults; change them through the pointer.
            if(p->replay_options == nullptr ||
               p->replay_options->flags != ROCPROFILER_KERNEL_REPLAY_OPTION_NONE)
                g_bad_options.fetch_add(1);
            else
                p->replay_options->flags = snapshot_flags();
        }
        g_replayed.fetch_add(1);
        return;
    }

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        if(has_replay_options(p) && p->replay_options != nullptr) g_bad_options.fetch_add(1);
        g_config_exits.fetch_add(1);
        return;
    }

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        fprintf(stderr,
                "[basic] pass %lu / %lu dispatch_id=%lu\n",
                static_cast<unsigned long>(p->current_pass),
                static_cast<unsigned long>(p->total_passes),
                static_cast<unsigned long>(p->dispatch_info.dispatch_id));
        g_passes_seen.fetch_add(1);
        if(has_replay_options(p) && p->replay_options != nullptr) g_pass_options_seen.fetch_add(1);
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    KR_CHECK(rocprofiler_create_context(&g_replay_ctx));
    KR_CHECK(
        rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                       nullptr,
                                                       0,
                                                       kernel_replay_cb,
                                                       nullptr));
    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    const int replays = expected_replays();
    fprintf(stderr,
            "[basic] replayed=%d config_exits=%d passes_seen=%d expected_replays=%d "
            "expected_passes=%lu pass_options_seen=%d bad_options=%d\n",
            g_replayed.load(),
            g_config_exits.load(),
            g_passes_seen.load(),
            replays,
            static_cast<unsigned long>(expected_passes()),
            g_pass_options_seen.load(),
            g_bad_options.load());
    // An abort would make ctest report a failure even when SKIP_REGULAR_EXPRESSION matches, and
    // _Exit skips the flush of the application's buffered output.
    if(g_replayed.load() != replays || g_config_exits.load() != replays ||
       g_passes_seen.load() != replays * static_cast<int>(expected_passes()) ||
       g_pass_options_seen.load() != 0 || g_bad_options.load() != 0)
    {
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-basic";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
