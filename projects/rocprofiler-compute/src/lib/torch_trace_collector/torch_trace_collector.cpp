// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "torch_trace_collector.h"

#include "args_capture.h"
#include "wire_format.h"

#include <ATen/record_function.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace
{
using namespace torch_trace_collector::detail;

constexpr const char* kRecordFnBackend = "torch";

constexpr std::string_view kLauncherTidKindName{"ROCPROF_COMPUTE_LAUNCHER_TID"};
const c10::DebugInfoKind   kLauncherTidKind{&kLauncherTidKindName};

struct LauncherTidInfo : public c10::DebugInfoBase
{
    std::uint64_t launcher_tid = 0;
};

thread_local int g_launcher_tid_depth = 0;

std::string launcher_tid_for_marker()
{
    auto* base = c10::ThreadLocalDebugInfo::get(kLauncherTidKind);
    if (base == nullptr)
    {
        return kUnavailable;
    }
    auto* info = static_cast<LauncherTidInfo*>(base);
    return std::to_string(info->launcher_tid);
}

struct RoctxObserverContext : public at::ObserverContext
{
    bool pushed = false;
};

const char* record_scope_name(at::RecordScope scope)
{
    switch (scope)
    {
    case at::RecordScope::FUNCTION:
        return "FUNCTION";
    case at::RecordScope::BACKWARD_FUNCTION:
        return "BACKWARD_FUNCTION";
    case at::RecordScope::TORCHSCRIPT_FUNCTION:
        return "TORCHSCRIPT_FUNCTION";
    case at::RecordScope::KERNEL_FUNCTION_DTYPE:
        return "KERNEL_FUNCTION_DTYPE";
    case at::RecordScope::CUSTOM_CLASS:
        return "CUSTOM_CLASS";
    case at::RecordScope::BUILD_FEATURE:
        return "BUILD_FEATURE";
    case at::RecordScope::LITE_INTERPRETER:
        return "LITE_INTERPRETER";
    case at::RecordScope::USER_SCOPE:
        return "USER_SCOPE";
    case at::RecordScope::STATIC_RUNTIME_OP:
        return "STATIC_RUNTIME_OP";
    case at::RecordScope::STATIC_RUNTIME_MODEL:
        return "STATIC_RUNTIME_MODEL";
    default:
        return kUnavailable;
    }
}

std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& record_fn)
{
    auto observer_ctx = std::make_unique<RoctxObserverContext>();
    try
    {
        const char* name = record_fn.name();
        if (name == nullptr || name[0] == '\0')
        {
            name = "<anonymous>";
        }

        const std::int64_t seqNrValue = record_fn.seqNr();
        const std::string  seqNr      = (seqNrValue < 0) ? std::string{kUnavailable}
                                                         : std::to_string(seqNrValue);
        const std::string  tid        = std::to_string(at::RecordFunction::currentThreadId());
        const std::string  ftid       = std::to_string(record_fn.forwardThreadId());
        const std::string  ltid       = launcher_tid_for_marker();
        const std::string  wire       = build_range_name(name,
                                                         kUnavailable,
                                                         seqNr,
                                                         tid,
                                                         ftid,
                                                         ltid,
                                                         record_scope_name(record_fn.scope()),
                                                         capture_record_function_args(record_fn),
                                                         kRecordFnBackend);
        observer_ctx->pushed          = (roctxRangePushA(wire.c_str()) >= 0);
    }
    catch (...)
    {
        observer_ctx->pushed = false;
    }
    return observer_ctx;
}

void end_cb(const at::RecordFunction& /*record_fn*/, at::ObserverContext* obs_ctx)
{
    if (obs_ctx == nullptr)
    {
        return;
    }
    auto* observer_ctx = static_cast<RoctxObserverContext*>(obs_ctx);
    if (observer_ctx->pushed)
    {
        roctxRangePop();
    }
}

at::CallbackHandle g_handle{};

bool install()
{
    if (g_handle > 0)
    {
        return true;
    }
    g_handle = at::addGlobalCallback(at::RecordFunctionCallback(start_cb, end_cb).needsInputs(true));
    return g_handle > 0;
}

}  // namespace

extern "C" int torch_trace_collector_install(void)
{
    return install() ? 0 : 1;
}

extern "C" int torch_trace_collector_push_launcher_tid(uint64_t launcher_tid)
{
    try
    {
        auto info          = std::make_shared<LauncherTidInfo>();
        info->launcher_tid = launcher_tid;
        c10::ThreadLocalDebugInfo::_push(kLauncherTidKind, info);
        ++g_launcher_tid_depth;
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" int torch_trace_collector_pop_launcher_tid(void)
{
    if (g_launcher_tid_depth <= 0)
    {
        return 1;
    }
    try
    {
        c10::ThreadLocalDebugInfo::_pop(kLauncherTidKind);
        --g_launcher_tid_depth;
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}
