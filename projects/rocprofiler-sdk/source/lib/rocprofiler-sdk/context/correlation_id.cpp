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

#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace context
{
namespace
{
auto*&
get_correlation_id_map()
{
    using data_type  = common::container::stable_vector<std::unique_ptr<correlation_id>>;
    static auto*& _v = common::static_object<common::Synchronized<data_type>>::construct();
    return _v;
}

auto&
get_latest_correlation_id_impl()
{
    static thread_local auto _v = common::container::small_vector<correlation_id*, 16>{};
    return _v;
}

uint64_t
get_unique_internal_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return ++_v;
}
}  // namespace

uint32_t
correlation_id::add_ref_count()
{
    auto _ret = m_ref_count.fetch_add(1);

    ROCP_CI_LOG_IF(WARNING, _ret == 0)
        << fmt::format("correlation id {} already retired", internal);

    return _ret;
}

uint32_t
correlation_id::sub_ref_count()
{
    if(m_ref_count == 0)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "attempt to decrement correlation id {} reference count but reference count is zero",
            internal);
        return 0;
    }

    auto _ret = m_ref_count.fetch_sub(1);

    if(registration::get_fini_status() > 0) return 0;

    ROCP_CI_LOG_IF(WARNING, _ret == 0) << fmt::format("correlation id underflow on {}", internal);

    if(_ret == 1)
    {
        auto ctxs = get_active_contexts([](const context* ctx) {
            return (ctx->buffered_tracer &&
                    (ctx->buffered_tracer->domains(
                        ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT)));
        });

        auto record = rocprofiler_buffer_tracing_correlation_id_retirement_record_t{
            .size      = sizeof(rocprofiler_buffer_tracing_correlation_id_retirement_record_t),
            .kind      = ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
            .timestamp = common::timestamp_ns(),
            .internal_correlation_id = internal};

        if(!ctxs.empty())
        {
            for(const auto* itr : ctxs)
            {
                auto* _buffer = buffer::get_buffer(itr->buffered_tracer->buffer_data.at(
                    ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT));

                auto success = CHECK_NOTNULL(_buffer)->emplace(
                    ROCPROFILER_BUFFER_CATEGORY_TRACING,
                    ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
                    record);

                ROCP_CI_LOG_IF(WARNING, !success)
                    << fmt::format("failed to emplace correlation id retirement for {}", internal);
            }
        }
    }

    return _ret;
}

uint32_t
correlation_id::add_kern_count()
{
    return m_kern_count.fetch_add(1);
}

uint32_t
correlation_id::sub_kern_count()
{
    return m_kern_count.fetch_sub(1);
}

correlation_id*
correlation_tracing_service::construct(uint32_t _init_ref_count)
{
    ROCP_FATAL_IF(_init_ref_count == 0) << "must have reference count > 0";

    auto  _internal_id = get_unique_internal_id();
    auto* corr_id_map  = get_correlation_id_map();
    if(!corr_id_map) return nullptr;
    auto& ret = corr_id_map->wlock([](auto& data) -> auto& { return data.emplace_back(); });

    ret = std::make_unique<correlation_id>(_init_ref_count, common::get_tid(), _internal_id);

    if(auto* prev_api_corr_id = get_latest_correlation_id())
        ret->ancestor = prev_api_corr_id->internal;

    get_latest_correlation_id_impl().emplace_back(ret.get());

    return ret.get();
}

correlation_id*
get_latest_correlation_id()
{
    return (get_latest_correlation_id_impl().empty()) ? nullptr
                                                      : get_latest_correlation_id_impl().back();
}

const correlation_id*
pop_latest_correlation_id(correlation_id* val)
{
    if(!val)
    {
        ROCP_CI_LOG(ERROR) << "passed nullptr to correlation id";
        return nullptr;
    }

    auto& stack = get_latest_correlation_id_impl();
    if(stack.empty())
    {
        ROCP_CI_LOG(ERROR) << "empty thread-local correlation id stack";
        return nullptr;
    }

    ROCP_CI_LOG_IF(ERROR, get_latest_correlation_id_impl().back() != val)
        << "pop_latest_correlation_id is happening out of order for " << val->internal
        << ". top of stack is " << get_latest_correlation_id_impl().back()->internal;

    stack.pop_back();

    return (stack.empty()) ? nullptr : stack.back();
}

correlation_id*
push_correlation_id(correlation_id* val)
{
    if(!val)
    {
        ROCP_ERROR << "passed nullptr to correlation id";
        return nullptr;
    }

    val->thread_idx = common::get_tid();
    get_latest_correlation_id_impl().emplace_back(val);

    return val;
}

void
dump_correlation_stack(const char* s)
{
    auto& stack = get_latest_correlation_id_impl();
    auto  info  = std::stringstream{};
    info << s << ": tid: " << common::get_tid() << " :";
    for(const auto* itr : stack)
    {
        info << " " << itr->internal;
        ;
    }
    info << "\n";
    printf("%s", info.str().c_str());
}

void
correlation_id_finalize()
{
    if(!get_correlation_id_map()) return;

    get_correlation_id_map()->rlock([](const auto& data) {
        uint64_t ndangling = 0;
        for(const auto& itr : data)
        {
            if(itr && itr->get_ref_count() > 0)
            {
                ++ndangling;
                ROCP_WARNING << "retiring dangling correlation ID " << itr->internal
                             << " from thread " << itr->thread_idx
                             << " :: remaining reference count: " << itr->get_ref_count();
                while(itr && itr->get_ref_count() > 0 && itr->sub_ref_count() > 1)
                {}
            }
        }
        ROCP_CI_LOG_IF(INFO, ndangling > 0) << "retired dangling correlation IDs: " << ndangling;
    });
}
}  // namespace context
}  // namespace rocprofiler
