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
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <memory>
#include <mutex>
#include <vector>

namespace rocprofiler
{
namespace context
{
namespace
{
struct correlation_id_storage
{
    static constexpr size_t block_size = (1UL << 6);  // 64
    std::mutex              mutex      = {};
    bool                    closed     = false;
    common::container::stable_vector<std::unique_ptr<correlation_id>, block_size> ids = {};
};

struct correlation_id_registry
{
    bool                                                 closed  = false;
    std::vector<std::unique_ptr<correlation_id_storage>> threads = {};
};

auto*&
get_correlation_id_registry()
{
    static auto*& _v =
        common::static_object<common::Synchronized<correlation_id_registry>>::construct();
    return _v;
}

correlation_id_storage*
get_correlation_id_storage()
{
    // The registry, not TLS, owns IDs: asynchronous completions can outlive their creator.
    static thread_local correlation_id_storage* storage = nullptr;
    if(storage) return storage;

    auto* registry = get_correlation_id_registry();
    if(!registry) return nullptr;
    auto next = std::make_unique<correlation_id_storage>();
    storage   = registry->wlock([&next](auto& data) -> correlation_id_storage* {
        if(data.closed || registration::get_fini_status() != 0) return nullptr;
        auto* result = next.get();
        data.threads.emplace_back(std::move(next));
        return result;
    });
    return storage;
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
        // Below-zero is a real bug pre-finalization, but once finalize starts the retirement sweep
        // races in-flight decrements (benign teardown), so only treat it as CI-fatal before then.
        ROCP_CI_LOG_IF(WARNING, registration::get_fini_status() == 0) << fmt::format(
            "attempt to decrement correlation id {} reference count but reference count is zero",
            internal);
        return 0;
    }

    auto _ret = m_ref_count.fetch_sub(1);

    if(registration::get_fini_status() > 0) return 0;

    // Same finalization race as the zero-check above: a concurrent decrement can slip between that
    // check and this fetch_sub during the retirement sweep, so only flag underflow before finalize.
    ROCP_CI_LOG_IF(WARNING, _ret == 0 && registration::get_fini_status() == 0)
        << fmt::format("correlation id underflow on {}", internal);

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

correlation_id*
correlation_tracing_service::construct(uint32_t _init_ref_count)
{
    // During finalization, we cannot create correlation IDs.
    // This can happen when HSA async threads call APIs after finalization starts.
    // This is not an error condition - just return nullptr and let the caller handle it.
    if(registration::get_fini_status() != 0)
    {
        ROCP_TRACE << "correlation_tracing_service::construct called during finalization"
                   << " (fini_status=" << registration::get_fini_status() << "), returning nullptr";
        return nullptr;
    }

    ROCP_FATAL_IF(_init_ref_count == 0) << "must have reference count > 0";

    auto* storage = get_correlation_id_storage();
    if(!storage) return nullptr;
    auto lock = std::unique_lock<std::mutex>{storage->mutex};
    if(storage->closed || registration::get_fini_status() != 0) return nullptr;

    auto id = std::make_unique<correlation_id>(
        _init_ref_count, common::get_tid(), get_unique_internal_id());
    if(auto* prev_api_corr_id = get_latest_correlation_id())
        id->ancestor = prev_api_corr_id->internal;

    auto& ret = storage->ids.emplace_back(std::move(id));
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
    auto* registry = get_correlation_id_registry();
    if(!registry) return;

    auto storages = registry->wlock([](auto& data) {
        auto result = std::vector<correlation_id_storage*>{};
        result.reserve(data.threads.size());
        for(const auto& storage : data.threads)
            result.emplace_back(storage.get());
        data.closed = true;
        return result;
    });

    // Seal each single-writer container and wait for any active insertion to finish.
    // Retirement can invoke buffer consumers, so release all allocation locks first.
    for(auto* storage : storages)
    {
        auto lock       = std::unique_lock<std::mutex>{storage->mutex};
        storage->closed = true;
    }

    uint64_t ndangling = 0;
    for(const auto* storage : storages)
    {
        for(const auto& itr : storage->ids)
        {
            if(itr && itr->get_ref_count() > 0)
            {
                // "Leaked" = a signal-less dispatch whose firmware completion record was
                // lost (ring overrun, slot quarantine, or teardown), so no record is
                // emitted and the id is intentionally not retired here: its kernel may
                // still be running, and force-retiring would release state the GPU can
                // still reach.
                if(kfd::signal_less_id_is_leaked(itr->internal)) continue;

                ++ndangling;
                ROCP_WARNING << "retiring dangling correlation ID " << itr->internal
                             << " from thread " << itr->thread_idx
                             << " :: remaining reference count: " << itr->get_ref_count();
                while(itr && itr->get_ref_count() > 0 && itr->sub_ref_count() > 1)
                {}
            }
        }
    }

    if(ndangling > 0)
    {
        auto retirement_ctxs = get_active_contexts([](const context* ctx) {
            return (ctx->buffered_tracer &&
                    (ctx->buffered_tracer->domains(
                        ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT)));
        });

        if(!retirement_ctxs.empty())
        {
            ROCP_CI_LOG_IF(INFO, ndangling > 0)
                << fmt::format("retired dangling correlation IDs: {}", ndangling);
        }
        else
        {
            ROCP_INFO << fmt::format(
                "retired dangling correlation IDs: {} (no retirement consumer active)", ndangling);
        }
    }
}
}  // namespace context
}  // namespace rocprofiler
