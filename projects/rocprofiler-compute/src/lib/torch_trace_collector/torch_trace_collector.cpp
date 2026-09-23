// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "torch_trace_collector.h"

#include "argument_capture.h"
#include "torch_abi/runtime.h"
#include "wire_format.h"

#include <ATen/record_function.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace
{

constexpr const char* kRecordFnBackend         = "torch";
constexpr std::size_t kMarkerMetadataAllowance = 512;
constexpr std::size_t kInlineMarkerSize = torch_trace_collector::detail::kMaxEncodedArgumentsSize +
                                          kMarkerMetadataAllowance;
using torch_trace_collector::detail::kUnavailable;

std::atomic_flag g_callback_failure_warning = ATOMIC_FLAG_INIT;

constexpr std::string_view kLauncherTidKindName{"ROCPROF_COMPUTE_LAUNCHER_TID"};
const c10::DebugInfoKind   kLauncherTidKind{&kLauncherTidKindName};

struct LauncherTidInfo final : c10::DebugInfoBase
{
    std::uint64_t launcher_tid = 0;
};

thread_local std::size_t g_launcher_tid_depth = 0;

void warn_callback_failure_once() noexcept
{
    if (!g_callback_failure_warning.test_and_set(std::memory_order_relaxed))
    {
        std::fputs("rocprof-compute: PyTorch RecordFunction callback failed; "
                   "some operator markers may be missing.\n",
                   stderr);
    }
}

struct RoctxObserverContext final : at::ObserverContext
{
};

std::optional<std::uint64_t> launcher_tid_for_marker()
{
    const auto* info = static_cast<const LauncherTidInfo*>(
        c10::ThreadLocalDebugInfo::get(kLauncherTidKind));
    if (info == nullptr)
    {
        return std::nullopt;
    }
    return info->launcher_tid;
}

std::string_view capture_args(
    const at::RecordFunction&                                                  record_function,
    std::array<char, torch_trace_collector::detail::kMaxEncodedArgumentsSize>& buffer) noexcept
{
    const std::size_t required = torch_trace_collector::detail::capture_args(record_function,
                                                                             buffer.data(),
                                                                             buffer.size());
    if (required > buffer.size())
    {
        return kUnavailable;
    }
    return {buffer.data(), required - 1};
}

std::string_view scope_name(at::RecordScope scope)
{
    static constexpr auto names = std::to_array<std::string_view>({
        "FUNCTION",
        "BACKWARD_FUNCTION",
        "TORCHSCRIPT_FUNCTION",
        "KERNEL_FUNCTION_DTYPE",
        "CUSTOM_CLASS",
        "BUILD_FEATURE",
        "LITE_INTERPRETER",
        "USER_SCOPE",
        "STATIC_RUNTIME_OP",
        "STATIC_RUNTIME_MODEL",
    });
    static_assert(names.size() == torch_abi::kScopeCount);

    const std::size_t index = static_cast<std::size_t>(scope);
    if (index < names.size())
    {
        return names[index];
    }
    return kUnavailable;
}

bool push_range(const at::RecordFunction& record_function, std::string_view name, std::string_view arguments)
{
    std::array<char, kInlineMarkerSize> marker_buffer;
    const auto                          launcher_tid = launcher_tid_for_marker();
    const auto format = [&record_function, name, arguments, launcher_tid](std::span<char> destination)
    {
        const torch_trace_collector::detail::RangeNameFields fields{
            .name               = name,
            .context            = kUnavailable,
            .sequence_number    = record_function.seqNr(),
            .thread_id          = at::RecordFunction::currentThreadId(),
            .forward_thread_id  = record_function.forwardThreadId(),
            .launcher_thread_id = launcher_tid,
            .scope              = scope_name(record_function.scope()),
            .arguments          = arguments,
            .backend            = kRecordFnBackend,
        };
        return torch_trace_collector::detail::format_range_name(destination, fields);
    };

    const std::size_t required = format(marker_buffer);
    if (required <= marker_buffer.size())
    {
        return roctxRangePushA(marker_buffer.data()) >= 0;
    }

    std::string marker(required, '\0');
    format(std::span<char>{marker.data(), marker.size()});
    marker.resize(required - 1);
    return roctxRangePushA(marker.c_str()) >= 0;
}

std::unique_ptr<at::ObserverContext> start_callback(const at::RecordFunction& record_function)
{
    try
    {
        auto        context = std::make_unique<RoctxObserverContext>();
        const char* name    = record_function.name();
        if (name == nullptr || name[0] == '\0')
        {
            name = "<anonymous>";
        }
        std::array<char, torch_trace_collector::detail::kMaxEncodedArgumentsSize> argument_buffer;
        if (!push_range(record_function, name, capture_args(record_function, argument_buffer)))
        {
            return nullptr;
        }
        return context;
    }
    catch (...)
    {
        warn_callback_failure_once();
        return nullptr;
    }
}

// PyTorch's callback ABI requires a mutable ObserverContext pointer.
// cppcheck-suppress constParameterCallback
void end_callback(const at::RecordFunction&, at::ObserverContext* context)
{
    if (context != nullptr)
    {
        roctxRangePop();
    }
}

std::mutex         g_install_mutex;
at::CallbackHandle g_handle = at::INVALID_CALLBACK_HANDLE;
std::atomic<bool>  g_installed{false};

bool install()
{
    const std::lock_guard<std::mutex> lock{g_install_mutex};
    if (g_handle != at::INVALID_CALLBACK_HANDLE)
    {
        return true;
    }
    if (!torch_abi::runtime_is_supported())
    {
        return false;
    }
    g_handle = at::addGlobalCallback(
        at::RecordFunctionCallback(start_callback, end_callback).needsInputs(true));
    if (g_handle == at::INVALID_CALLBACK_HANDLE)
    {
        return false;
    }
    g_installed.store(true, std::memory_order_release);
    return true;
}

bool push_launcher_tid(std::uint64_t launcher_tid)
{
    if (!g_installed.load(std::memory_order_acquire))
    {
        return false;
    }
    auto info          = std::make_shared<LauncherTidInfo>();
    info->launcher_tid = launcher_tid;
    c10::ThreadLocalDebugInfo::_push(kLauncherTidKind, std::move(info));
    ++g_launcher_tid_depth;
    return true;
}

bool pop_launcher_tid()
{
    if (!g_installed.load(std::memory_order_acquire) || g_launcher_tid_depth == 0)
    {
        return false;
    }
    c10::ThreadLocalDebugInfo::_pop(kLauncherTidKind);
    --g_launcher_tid_depth;
    return true;
}

}  // namespace

extern "C" std::uint32_t torch_trace_collector_abi_revision(void)
{
    return TORCH_TRACE_COLLECTOR_ABI_REVISION;
}

extern "C" int torch_trace_collector_install(void)
{
    try
    {
        return install() ? 0 : 1;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" int torch_trace_collector_push_launcher_tid(std::uint64_t launcher_tid)
{
    try
    {
        return push_launcher_tid(launcher_tid) ? 0 : 1;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" int torch_trace_collector_pop_launcher_tid(void)
{
    try
    {
        return pop_launcher_tid() ? 0 : 1;
    }
    catch (...)
    {
        return 1;
    }
}
