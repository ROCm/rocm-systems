// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/external_correlation.hpp"

#include <unistd.h>

namespace rocprofiler
{
namespace external_correlation
{
namespace
{
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

    return data.wlock(
        [](external_correlation_map_t& _data, rocprofiler_thread_id_t tid_v) {
            if(_data.count(tid_v) == 0) return empty_user_data;
            auto& itr = _data.at(tid_v);
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
}  // namespace external_correlation
}  // namespace rocprofiler

extern "C" {
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
