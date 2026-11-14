#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace
{
struct test_state
{
    rocprofiler_context_id_t     context        = {};
    rocprofiler_buffer_id_t      buffer         = {};
    std::mutex                   mutex          = {};
    std::unordered_set<uint64_t> dispatches     = {};
    std::atomic<uint64_t>        callbacks      = {0};
    std::atomic<bool>            delayed        = {false};
    std::atomic<bool>            reentry        = {false};
    std::atomic<bool>            bad            = {false};
    std::atomic<int>             kernel_reentry = {-1};
    std::atomic<int>             buffer_reentry = {-1};
};

test_state&
state()
{
    static auto value = test_state{};
    return value;
}

bool
check(rocprofiler_status_t status, const char* operation)
{
    if(status == ROCPROFILER_STATUS_SUCCESS) return true;
    std::fprintf(
        stderr, "[flush-contract] FAIL %s: %s\n", operation, rocprofiler_get_status_string(status));
    return false;
}

void
kernel_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.operation != ROCPROFILER_KERNEL_DISPATCH_COMPLETE) return;
    auto&      current = state();
    const auto index   = current.callbacks.fetch_add(1);
    if(current.delayed.load() && index == 42)
        std::this_thread::sleep_for(std::chrono::milliseconds{150});
    if(current.reentry.load() && current.kernel_reentry.load() == -1)
        current.kernel_reentry.store(static_cast<int>(rocprofiler_flush_buffer(current.buffer)));
}

void
buffer_callback(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t** headers,
                size_t                        count,
                void*,
                uint64_t dropped)
{
    auto& current = state();
    if(dropped != 0) current.bad.store(true);
    if(current.reentry.load() && current.buffer_reentry.load() == -1)
        current.buffer_reentry.store(static_cast<int>(rocprofiler_flush_buffer(current.buffer)));
    auto lock = std::lock_guard<std::mutex>{current.mutex};
    for(size_t index = 0; index < count; ++index)
    {
        const auto* header = headers[index];
        if(!header || header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
           header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            current.bad.store(true);
            continue;
        }
        const auto* record =
            static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                header->payload);
        if(!current.dispatches.emplace(record->dispatch_info.dispatch_id).second)
            current.bad.store(true);
    }
}

int
initialize(rocprofiler_client_finalize_t, void*)
{
    auto& current = state();
    if(!check(rocprofiler_create_context(&current.context), "create_context")) return -1;
    if(!check(rocprofiler_create_buffer(current.context,
                                        1024 * 1024,
                                        1024 * 1024,
                                        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                        buffer_callback,
                                        nullptr,
                                        &current.buffer),
              "create_buffer"))
        return -1;
    if(!check(
           rocprofiler_configure_buffer_tracing_service(current.context,
                                                        ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                        nullptr,
                                                        0,
                                                        current.buffer),
           "configure_buffer"))
        return -1;
    const auto operation = rocprofiler_tracing_operation_t{ROCPROFILER_KERNEL_DISPATCH_COMPLETE};
    return check(rocprofiler_configure_callback_tracing_service(
                     current.context,
                     ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                     &operation,
                     1,
                     kernel_callback,
                     nullptr),
                 "configure_callback")
               ? 0
               : -1;
}

void
finalize(void*)
{}

int
verify(uint64_t expected)
{
    auto&      current = state();
    auto       lock    = std::lock_guard<std::mutex>{current.mutex};
    const auto busy    = static_cast<int>(ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
    if(current.bad.load() || current.dispatches.size() != expected ||
       (current.reentry.load() &&
        (current.kernel_reentry.load() != busy || current.buffer_reentry.load() != busy)))
    {
        std::fprintf(stderr,
                     "[flush-contract] FAIL expected=%lu delivered=%zu bad=%d reentry=%d/%d\n",
                     static_cast<unsigned long>(expected),
                     current.dispatches.size(),
                     static_cast<int>(current.bad.load()),
                     current.kernel_reentry.load(),
                     current.buffer_reentry.load());
        return 1;
    }
    return 0;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int
flush_contract_begin(int delayed, int reentry)
{
    auto& current = state();
    {
        auto lock = std::lock_guard<std::mutex>{current.mutex};
        current.dispatches.clear();
    }
    current.callbacks.store(0);
    current.delayed.store(delayed != 0);
    current.reentry.store(reentry != 0);
    current.bad.store(false);
    current.kernel_reentry.store(-1);
    current.buffer_reentry.store(-1);
    return check(rocprofiler_start_context(current.context), "start_context") ? 0 : 1;
}

// A flush issued from another application thread while the window is still running. Several of
// these at once is what puts more than one requester in the completion fence's queue at a time,
// which is the state the monitor's queue-swap and its shutdown drain have to agree about. Records
// delivered early are not a failure: verify() checks the cumulative set, and the buffer callback
// flags any dispatch id it sees twice.
extern "C" __attribute__((visibility("default"))) int
flush_contract_poke()
{
    auto& current = state();
    auto  status  = rocprofiler_flush_buffer(current.buffer);
    // Another flush already draining this buffer is expected under concurrency.
    if(status == ROCPROFILER_STATUS_SUCCESS || status == ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        return 0;
    check(status, "concurrent flush_buffer");
    current.bad.store(true);
    return 1;
}

extern "C" __attribute__((visibility("default"))) int
flush_contract_end(int stop_first, uint64_t expected)
{
    auto& current = state();
    if(stop_first != 0 && !check(rocprofiler_stop_context(current.context), "stop_context"))
        return 1;
    const auto status = rocprofiler_flush_buffer(current.buffer);
    if(stop_first == 0 && !check(rocprofiler_stop_context(current.context), "stop_context"))
        return 1;
    return check(status, "flush_buffer") ? verify(expected) : 1;
}

extern "C" __attribute__((visibility("default"))) int
flush_contract_late(uint64_t expected)
{
    return check(rocprofiler_flush_buffer(state().buffer), "late_flush") ? verify(expected) : 1;
}

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* client)
{
    client->name              = "kernel-flush-contract";
    static auto configuration = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), initialize, finalize, nullptr};
    return &configuration;
}
