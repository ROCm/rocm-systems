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

#include "lib/rocprofiler-sdk/external_correlation.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>

#include <unistd.h>

namespace rocprofiler
{
namespace external_correlation
{
namespace
{
#define ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(CODE)                                 \
    template <>                                                                                    \
    struct external_correlation_id_request_kind_string<                                            \
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_##CODE>                                           \
    {                                                                                              \
        static constexpr auto value =                                                              \
            std::pair<const char*, size_t>{#CODE, std::string_view{#CODE}.length()};               \
    };

template <size_t Idx>
struct external_correlation_id_request_kind_string;

ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(NONE)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(HSA_CORE_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(HSA_AMD_EXT_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(HSA_IMAGE_EXT_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(HSA_FINALIZE_EXT_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(HIP_RUNTIME_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(HIP_COMPILER_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(MARKER_CORE_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(MARKER_CONTROL_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(MARKER_NAME_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(MEMORY_COPY)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(MEMORY_ALLOCATION)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(KERNEL_DISPATCH)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(SCRATCH_MEMORY)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(RCCL_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(OMPT)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(ROCDECODE_API)
ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING(ROCJPEG_API)

#undef ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KIND_STRING

template <size_t Idx, size_t... Tail>
std::pair<const char*, size_t>
get_kind_name(rocprofiler_external_correlation_id_request_kind_t kind,
              std::index_sequence<Idx, Tail...>)
{
    if(kind == Idx) return external_correlation_id_request_kind_string<Idx>::value;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0) return get_kind_name(kind, std::index_sequence<Tail...>{});
    return {nullptr, 0};
}

auto
get_default_tid()
{
    static auto _v = common::get_tid();
    return _v;
}

constexpr auto empty_user_data = rocprofiler_user_data_t{.value = 0};

auto&
get_default_data_impl()
{
    static auto _v = std::atomic<uint64_t>{0};
    return _v;
}

auto
get_default_data()
{
    return rocprofiler_user_data_t{.value =
                                       get_default_data_impl().load(std::memory_order_relaxed)};
}

auto f_default_tid = get_default_tid();  // make sure it is initialized
}  // namespace

rocprofiler_user_data_t
external_correlation::get(rocprofiler_thread_id_t tid) const
{
    return data.rlock(
        [](const external_correlation_map_t& _data, rocprofiler_thread_id_t tid_v) {
            if(_data.count(tid_v) == 0) return get_default_data();
            const auto& itr = _data.at(tid_v);
            return itr.rlock([](const external_correlation_stack_t& data_stack) {
                if(data_stack.empty()) return get_default_data();
                return data_stack.back();
            });
        },
        tid);
}

rocprofiler_user_data_t
external_correlation::get(rocprofiler_thread_id_t tid,
                          const context::context* ctx,
                          request_kind_t          kind,
                          uint32_t                op,
                          uint64_t                internal_corr_id) const
{
    if(requires_request(kind))
    {
        auto opt_data = invoke_callback(tid, ctx, kind, op, internal_corr_id);
        if(opt_data) return *opt_data;
    }

    return get(tid);
}

rocprofiler_user_data_t&
external_correlation::update(rocprofiler_user_data_t& value,
                             rocprofiler_thread_id_t  thr_id,
                             request_kind_t           kind) const
{
    // if requires request is true, do not update, otherwise, get the latest pushed external
    // correlation id
    return (requires_request(kind)) ? value : (value = get(thr_id));
}

void
external_correlation::push(rocprofiler_thread_id_t tid, rocprofiler_user_data_t user_data)
{
    static auto default_tid = get_default_tid();

    // ensure that data contains key for provided thread id
    while(!data.ulock(
        [](const external_correlation_map_t& _data, rocprofiler_thread_id_t tid_v) {
            return (_data.find(tid_v) != _data.end());
        },
        [](external_correlation_map_t& _data, rocprofiler_thread_id_t tid_v) {
            _data.emplace(tid_v, external_correlation_stack_t{});
            return true;
        },
        tid))
    {}

    // since we know from above that there will be a key for the tid, we start with a read
    // lock and then once we have have the mapped data for the key, we leverage the enabling
    // of the wlock const overload to remove the constness and use a write lock. If we were to use a
    // write lock at the top lovel, then we would unnecessarily block other threads from writing to
    // the stack of another thread
    data.rlock(
        [](const external_correlation_map_t& _data,
           rocprofiler_thread_id_t           tid_v,
           rocprofiler_user_data_t           user_data_v) {
            const auto& itr = _data.at(tid_v);
            itr.wlock([](external_correlation_stack_t& data_stack,
                         rocprofiler_user_data_t       value) { data_stack.emplace_back(value); },
                      user_data_v);
            // child threads inherit the current value on default thread
            if(tid_v == default_tid)
                get_default_data_impl().store(user_data_v.value, std::memory_order_relaxed);
        },
        tid,
        user_data);
}

rocprofiler_user_data_t
external_correlation::pop(rocprofiler_thread_id_t tid)
{
    static auto default_tid = get_default_tid();

    return data.rlock(
        [](const external_correlation_map_t& _data, rocprofiler_thread_id_t tid_v) {
            if(_data.count(tid_v) == 0) return empty_user_data;
            const auto& itr = _data.at(tid_v);
            return itr.wlock([tid_v](external_correlation_stack_t& data_stack) {
                if(data_stack.empty()) return empty_user_data;
                auto ret = data_stack.back();
                data_stack.pop_back();
                // child threads inherit the current value on default thread
                if(tid_v == default_tid)
                {
                    uint64_t value = (!data_stack.empty()) ? data_stack.back().value : 0;
                    get_default_data_impl().store(value, std::memory_order_relaxed);
                }
                return ret;
            });
        },
        tid);
}

rocprofiler_status_t
external_correlation::configure_request(request_cb_t                       callback_v,
                                        void*                              callback_data_v,
                                        const std::vector<request_kind_t>& kinds_v)
{
    if(!callback_v)
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    else if(callback || callback_data || request.any())
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    callback      = callback_v;
    callback_data = callback_data_v;

    if(kinds_v.empty())
    {
        request.flip();
    }
    else
    {
        for(auto itr : kinds_v)
        {
            if(itr <= ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_NONE ||
               itr >= ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST)
                return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;

            request.set(itr - 1, true);
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

bool
external_correlation::requires_request(request_kind_t kind) const
{
    return request.test(kind - 1);
}

std::optional<rocprofiler_user_data_t>
external_correlation::invoke_callback(rocprofiler_thread_id_t thr_id,
                                      const context::context* ctx,
                                      request_kind_t          kind,
                                      uint32_t                op,
                                      uint64_t                internal_corr_id) const
{
    auto value  = rocprofiler_user_data_t{.value = 0};
    auto ctx_id = rocprofiler_context_id_t{ctx->context_idx};

    if(callback(thr_id, ctx_id, kind, op, internal_corr_id, &value, callback_data) == 0)
        return value;

    return std::nullopt;
}
}  // namespace external_correlation
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_configure_external_correlation_id_request_service(
    rocprofiler_context_id_t                                  context_id,
    const rocprofiler_external_correlation_id_request_kind_t* kinds,
    size_t                                                    kinds_count,
    rocprofiler_external_correlation_id_request_cb_t          callback,
    void*                                                     callback_args)
{
    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    auto kinds_v = std::vector<rocprofiler_external_correlation_id_request_kind_t>{};
    if(kinds)
    {
        kinds_v.reserve(kinds_count);

        if(kinds_count < 1) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        for(size_t i = 0; i < kinds_count; ++i)
            kinds_v.emplace_back(kinds[i]);
    }

    return ctx->correlation_tracer.external_correlator.configure_request(
        callback, callback_args, kinds_v);
}

rocprofiler_status_t
rocprofiler_query_external_correlation_id_request_kind_name(
    rocprofiler_external_correlation_id_request_kind_t kind,
    const char**                                       name,
    uint64_t*                                          name_len)
{
    auto&& val = rocprofiler::external_correlation::get_kind_name(
        kind, std::make_index_sequence<ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST>{});

    if(name) *name = val.first;
    if(name_len) *name_len = val.second;

    return (val.first) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
}

rocprofiler_status_t
rocprofiler_push_external_correlation_id(rocprofiler_context_id_t context,
                                         rocprofiler_thread_id_t  tid,
                                         rocprofiler_user_data_t  external_correlation_id)
{
    // assumption is that thread ids are monotonically increasing from the pid
    static uint64_t pid_v = getpid();
    if(tid < pid_v) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    ctx->correlation_tracer.external_correlator.push(tid, external_correlation_id);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_pop_external_correlation_id(rocprofiler_context_id_t context,
                                        rocprofiler_thread_id_t  tid,
                                        rocprofiler_user_data_t* external_correlation_id)
{
    // assumption is that thread ids are monotonically increasing from the pid
    static uint64_t pid_v = getpid();
    if(tid < pid_v) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    auto former = ctx->correlation_tracer.external_correlator.pop(tid);
    if(external_correlation_id) *external_correlation_id = former;
    return ROCPROFILER_STATUS_SUCCESS;
}
}
