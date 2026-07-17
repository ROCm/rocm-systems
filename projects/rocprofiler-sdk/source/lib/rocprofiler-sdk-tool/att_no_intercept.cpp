// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "att_no_intercept_impl.hpp"

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>

#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define CHECK_PTR(_x) ROCP_FATAL_IF(_x == nullptr) << "Called after finalize()";

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
namespace
{
using agent_state_map_t = std::unordered_map<uint64_t, std::shared_ptr<agent_state_t>>;

std::mutex manager_mutex{};

// Never used after global destructor
std::unordered_map<uint64_t, std::shared_ptr<agent_state_t>> code_objects{};

agent_state_map_t*&
agent_states()
{
    static auto* _v = new agent_state_map_t{};
    CHECK_PTR(_v);
    return _v;
}

shader_data_forwarder_t&
shader_data_forwarder()
{
    static auto _v = shader_data_forwarder_t{};
    return _v;
}

std::unordered_set<size_t>&
kernel_filter_range()
{
    static auto _v = std::unordered_set<size_t>{};
    return _v;
}

void
check_status(rocprofiler_status_t status, std::string_view msg)
{
    ROCP_FATAL_IF(status != ROCPROFILER_STATUS_SUCCESS)
        << msg << " failed with error code " << status << ": "
        << rocprofiler_get_status_string(status);
}

void
add_range_locked(agent_state_t& state,
                 uint64_t       code_object_id,
                 uint64_t       begin,
                 uint64_t       size,
                 bool           targeted)
{
    if(begin == 0) return;
    auto end = uint64_t{0};
    if(size > 0 && begin <= std::numeric_limits<uint64_t>::max() - size)
        end = begin + size;
    else if(size > 0)
        end = std::numeric_limits<uint64_t>::max();
    else
        end = begin + 1;
    state.kernel_ranges_by_code_object[code_object_id].emplace_back(
        kernel_symbol_range_t{begin, end, targeted});
}

void
add_exact_locked(agent_state_t& state, uint64_t code_object_id, uint64_t address, bool targeted)
{
    if(address == 0) return;
    state.kernel_iterations_by_entry[entry_key_t{code_object_id, address}] =
        targeted ? std::make_shared<std::atomic<size_t>>(0) : nullptr;
}

uint64_t
get_symbol_size_locked(const code_object_record_t& code_object, uint64_t kernel_address)
{
    auto elf_vaddr = kernel_address - static_cast<uint64_t>(code_object.load_delta);
    if(auto itr = code_object.symbol_sizes_by_vaddr.find(elf_vaddr);
       itr != code_object.symbol_sizes_by_vaddr.end())
        return itr->second;

    return 0;
}

void
register_kernel_symbol_locked(agent_state_t& state, const kernel_symbol_t& symbol, bool targeted)
{
    auto code_object_itr = state.code_objects.find(symbol.code_object_id);
    auto symbol_size     = uint64_t{0};
    if(code_object_itr != state.code_objects.end())
        symbol_size = get_symbol_size_locked(code_object_itr->second, symbol.kernel_address.handle);

    add_exact_locked(state, 0, symbol.kernel_address.handle, targeted);
    add_range_locked(state, 0, symbol.kernel_address.handle, symbol_size, targeted);

    if(code_object_itr == state.code_objects.end()) return;

    const auto& code_object = code_object_itr->second;
    auto elf_vaddr = symbol.kernel_address.handle - static_cast<uint64_t>(code_object.load_delta);
    add_exact_locked(state, symbol.code_object_id, elf_vaddr, targeted);
    add_range_locked(state, symbol.code_object_id, elf_vaddr, symbol_size, targeted);
}

void
start_agent_context(agent_state_t& state)
{
    auto lock = std::unique_lock{state.mutex};
    if(state.started || state.finalized) return;

    state.started = true;
    ROCP_INFO << fmt::format("starting ATT no-intercept context for agent {}", state.id.handle);
    check_status(rocprofiler_start_context(state.context), "ATT no-intercept context start");
}

void
stop_agent_context(agent_state_t& state)
{
    auto stop_context = false;
    {
        auto lock       = std::unique_lock{state.mutex};
        stop_context    = state.started;
        state.started   = false;
        state.finalized = true;
    }

    if(!stop_context) return;

    auto status = rocprofiler_stop_context(state.context);
    if(status != ROCPROFILER_STATUS_SUCCESS && status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
    {
        check_status(status, "ATT no-intercept context stop");
    }
}
}  // namespace

bool
is_supported()
{
    return backend_supported();
}

void
configure(shader_data_forwarder_t forwarder, std::unordered_set<size_t> filter_range)
{
    ROCP_FATAL_IF(!backend_supported())
        << "--att-no-intercept was requested, but this rocprofv3 build was configured with "
           "ROCPROFILER_DISABLE_TRACE_DECODER=ON and does not include "
           "rocprof-trace-decoder quick-scan support";

    ROCP_FATAL_IF(forwarder == nullptr)
        << "ATT no-intercept setup requires a shader-data forwarding callback";

    auto lock               = std::unique_lock{manager_mutex};
    shader_data_forwarder() = forwarder;
    kernel_filter_range()   = std::move(filter_range);
}

agent_trace_config_t
configure_agent(rocprofiler_agent_id_t id, uint64_t consecutive_kernels)
{
    auto lock = std::unique_lock{manager_mutex};
    if(auto itr = agent_states()->find(id.handle); itr != agent_states()->end())
        return agent_trace_config_t{itr->second->context, itr->second->userdata};

    auto state                 = std::make_shared<agent_state_t>();
    state->id                  = id;
    state->consecutive_kernels = consecutive_kernels;
    state->userdata.ptr        = state.get();

    check_status(rocprofiler_create_context(&state->context), "ATT no-intercept context creation");
    backend_create(*state);

    auto trace_config = agent_trace_config_t{state->context, state->userdata};
    agent_states()->emplace(id.handle, std::move(state));
    ROCP_INFO << fmt::format("configured ATT no-intercept context for agent {}", id.handle);
    return trace_config;
}

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t                userdata)
{
    auto* state = static_cast<agent_state_t*>(userdata.ptr);
    if(state == nullptr || shader_data_forwarder() == nullptr) return;

    backend_shader_data(*state, shader_data, shader_data_forwarder(), kernel_filter_range());
}

void
code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& data)
{
    std::shared_ptr<agent_state_t> state = nullptr;
    {
        auto lock = std::unique_lock{manager_mutex};
        auto itr  = agent_states()->find(data.agent_id.handle);
        if(itr == agent_states()->end()) return;

        state                             = itr->second;
        code_objects[data.code_object_id] = state;
    }

    {
        auto lock                                = std::unique_lock{state->mutex};
        state->code_objects[data.code_object_id] = code_object_record_t{data.load_delta};
    }

    backend_code_object_load(*state, data);

    start_agent_context(*state);
}

void
kernel_symbol_load(const kernel_symbol_t& data, bool is_targeted)
{
    std::shared_ptr<agent_state_t> state = nullptr;
    {
        auto lock = std::unique_lock{manager_mutex};
        if(auto itr = code_objects.find(data.code_object_id); itr != code_objects.end())
        {
            state = itr->second;
        }
        else
        {
            ROCP_WARNING << "Unknown kernel symbol: " << data.kernel_id;
            return;
        }
    }

    auto lock = std::unique_lock{state->mutex};
    register_kernel_symbol_locked(*state, data, is_targeted);
}

void
finalize()
{
    auto lock = std::unique_lock{manager_mutex};

    for(auto& itr : *agent_states())
    {
        auto& state = *itr.second;
        stop_agent_context(state);
        backend_destroy(state);
    }

    delete agent_states();
    agent_states() = nullptr;
}

#if !defined(ROCPROFILER_ATT_QUICK_SCAN_ENABLED) || ROCPROFILER_ATT_QUICK_SCAN_ENABLED == 0
bool
backend_supported()
{
    return false;
}

void
backend_create(agent_state_t&)
{}

void
backend_destroy(agent_state_t&)
{}

void
backend_code_object_load(agent_state_t&,
                         const rocprofiler_callback_tracing_code_object_load_data_t&)
{}

void
backend_shader_data(agent_state_t&,
                    rocprofiler_thread_trace_shader_data_t,
                    shader_data_forwarder_t,
                    const std::unordered_set<size_t>&)
{}
#endif
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
