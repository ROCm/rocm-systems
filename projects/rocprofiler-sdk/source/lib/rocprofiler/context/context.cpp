// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <rocprofiler/fwd.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/rocprofiler/context/context.hpp"

#include <glog/logging.h>

#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>

namespace rocprofiler
{
namespace context
{
namespace
{
auto&
get_contexts_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

constexpr auto invalid_client_idx = std::numeric_limits<uint32_t>::max();

auto&
get_client_index()
{
    static auto _v = invalid_client_idx;
    return _v;
}
}  // namespace

uint64_t
correlation_tracing_service::get_unique_record_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v++;
}

using reserve_size_t = common::container::reserve_size;

unique_context_vec_t&
get_registered_contexts()
{
    static auto _v = unique_context_vec_t{reserve_size_t{unique_context_vec_t::chunk_size}};
    return _v;
}

active_context_vec_t&
get_active_contexts()
{
    static auto* _v    = new active_context_vec_t{reserve_size_t{active_context_vec_t::chunk_size}};
    static auto  _once = std::once_flag{};
    std::call_once(_once, std::atexit, []() {
        for(auto& itr : *_v)
        {
            itr.store(nullptr);
        }
    });
    return *_v;
}

// set the client index needs to be called before allocate_context()
void
push_client(uint32_t value)
{
    LOG_ASSERT(get_client_index() == invalid_client_idx)
        << " rocprofiler client index is currently " << get_client_index()
        << "... which means that a new client is initializing before the last client finished "
           "initializing. This is an internal error, please file a bug report with a reproducer";
    get_client_index() = value;
}

// remove the client index
void
pop_client(uint32_t value)
{
    LOG_ASSERT(get_client_index() == value)
        << " rocprofiler client index is currently not " << value
        << "... which means that a new client was initialized before this client finished "
           "initializing. This is an internal error, please file a bug report with a reproducer";
    get_client_index() = invalid_client_idx;
}

std::optional<rocprofiler_context_id_t>
allocate_context()
{
    // ... allocate any internal space needed to handle another context ...
    auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

    // initial context identifier number
    auto _idx = get_registered_contexts().size();

    // make space in registered
    get_registered_contexts().emplace_back(nullptr);

    // create an entry in the registered
    auto& _cfg_v = get_registered_contexts().back();
    _cfg_v       = std::make_unique<context>();
    auto* _cfg   = _cfg_v.get();
    // ...

    if(!_cfg) return std::nullopt;

    _cfg->size        = sizeof(context);
    _cfg->context_idx = _idx;
    _cfg->client_idx  = get_client_index();

    LOG_ASSERT(_cfg->client_idx != invalid_client_idx)
        << " rocprofiler internal error: a context was allocated without an associated tool client "
           "identifier";

    return rocprofiler_context_id_t{_idx};
}

rocprofiler_status_t
validate_context(const context* cfg)
{
    // if(cfg->buffer == nullptr) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    // if(cfg->filter == nullptr) return ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND;

    return (cfg) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
}

rocprofiler_status_t
start_context(rocprofiler_context_id_t context_id)
{
    if(context_id.handle >= get_registered_contexts().size())
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
    }

    context* cfg = get_registered_contexts().at(context_id.handle).get();

    if(!cfg)
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
    }

    if(validate_context(cfg) != ROCPROFILER_STATUS_SUCCESS)
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
    }

    uint64_t rocp_tot_contexts = get_registered_contexts().size();
    auto     idx               = rocp_tot_contexts;
    {
        // hold a lock here so prevent multiple threads from finding the same nullptr slot
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};
        // try to find a nullptr slot first
        for(size_t i = 0; i < get_active_contexts().size(); ++i)
        {
            auto* itr = get_active_contexts().at(i).load(std::memory_order_relaxed);
            if(itr == nullptr)
            {
                idx = i;
                break;
            }
            else if(context_id.handle == itr->context_idx)
            {
                return ROCPROFILER_STATUS_SUCCESS;
            }
        }
        // if no nullptr slot was found, then create one while lock is held
        if(idx == rocp_tot_contexts)
        {
            idx = get_active_contexts().size();
            get_active_contexts().emplace_back();
        }
    }

    // atomic swap the pointer into the "active" array used internally
    context* _expected = nullptr;
    bool     success   = get_active_contexts().at(idx).compare_exchange_strong(
        _expected, get_registered_contexts().at(context_id.handle).get());

    if(!success) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
stop_context(rocprofiler_context_id_t idx)
{
    // atomically assign the context pointer to NULL so that it is skipped in future
    // callbacks
    for(auto& itr : get_active_contexts())
    {
        auto* _expected = itr.load(std::memory_order_relaxed);
        if(_expected && _expected->context_idx == idx.handle)
        {
            bool success = itr.compare_exchange_strong(_expected, nullptr);

            if(success) return ROCPROFILER_STATUS_SUCCESS;
        }
    }

    return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;  // compare exchange failed
}
}  // namespace context
}  // namespace rocprofiler
