// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdint>
#include <stack>

namespace rocprofsys::rocprofiler_sdk
{

template <typename SdkBackend>
struct stream_stack_service
{
    static void push(SdkBackend::stream_id_t sid) { get_stack().push(sid); }
    static SdkBackend::stream_id_t top() { return get_stack().top(); }
    static void                    pop() { get_stack().pop(); }

    // NOLINTNEXTLINE (readability-function-size)
    static int request_stream_correlation_id(
        SdkBackend::thread_id_t /*thread_id*/, SdkBackend::context_id_t /*context_id*/,
        SdkBackend::external_correlation_request_kind_t /*kind*/,
        SdkBackend::tracing_operation_t /*operation*/, std::uint64_t /*internal_corr_id*/,
        SdkBackend::user_data_t* external_corr_id, void* /*user_data*/)
    {
        auto* info            = new kernel_dispatch_stream_correlation_t{};
        info->handle          = top();
        external_corr_id->ptr = info;
        return 0;
    }

    template <typename RecordT>
    static SdkBackend::stream_id_t get_stream_id(RecordT* record)
    {
        typename SdkBackend::stream_id_t stream_id{};
        if(record->correlation_id.external.ptr == nullptr)
        {
            return stream_id;
        }

        auto* info = static_cast<kernel_dispatch_stream_correlation_t*>(
            record->correlation_id.external.ptr);
        stream_id                             = info->handle;
        auto region_id                        = info->region_id;
        record->correlation_id.external.value = region_id;
        delete info;
        record->correlation_id.external.ptr = nullptr;

        return stream_id;
    }

private:
    struct kernel_dispatch_stream_correlation_t
    {
        std::uint64_t           region_id = 0;
        SdkBackend::stream_id_t handle    = {};
    };

    static auto& get_stack()
    {
        static thread_local std::stack<typename SdkBackend::stream_id_t> s_stack{
            std::deque<typename SdkBackend::stream_id_t>{
                typename SdkBackend::stream_id_t{} }
        };
        return s_stack;
    }
};

}  // namespace rocprofsys::rocprofiler_sdk
