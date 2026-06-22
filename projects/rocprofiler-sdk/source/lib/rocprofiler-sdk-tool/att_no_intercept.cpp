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
#include <utility>
#include <iostream>

//#define LOG_TRACE ROCP_TRACE
//#define LOG_INFO ROCP_INFO
#define LOG_TRACE std::cout << std::endl
#define LOG_INFO std::cout << std::endl

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
namespace
{
using agent_state_map_t    = std::unordered_map<uint64_t, std::unique_ptr<agent_state>>;
using pending_symbol_map_t = std::unordered_map<uint64_t, std::vector<kernel_symbol_record>>;

std::mutex&
manager_mutex()
{
    static auto* _v = new std::mutex{};
    return *_v;
}

agent_state_map_t&
agent_states()
{
    static auto* _v = new agent_state_map_t{};
    return *_v;
}

std::unordered_map<uint64_t, agent_state*>&
code_object_agents()
{
    static auto* _v = new std::unordered_map<uint64_t, agent_state*>{};
    return *_v;
}

pending_symbol_map_t&
pending_symbols()
{
    static auto* _v = new pending_symbol_map_t{};
    return *_v;
}

shader_data_forwarder_t&
shader_data_forwarder()
{
    static auto _v = shader_data_forwarder_t{};
    return _v;
}

kernel_target_filter_t&
kernel_target_filter()
{
    static auto _v = kernel_target_filter_t{};
    return _v;
}

void
check_status(rocprofiler_status_t status, std::string_view msg)
{
    ROCP_FATAL_IF(status != ROCPROFILER_STATUS_SUCCESS)
        << msg << " failed with error code " << status << ": "
        << rocprofiler_get_status_string(status);
}

std::optional<uint64_t>
subtract_signed_delta(uint64_t address, int64_t delta)
{
    if(delta >= 0)
    {
        auto unsigned_delta = static_cast<uint64_t>(delta);
        if(address < unsigned_delta) return std::nullopt;
        return address - unsigned_delta;
    }

    auto unsigned_delta = static_cast<uint64_t>(-(delta + 1)) + 1;
    if(address > std::numeric_limits<uint64_t>::max() - unsigned_delta) return std::nullopt;
    return address + unsigned_delta;
}

void
add_range_locked(agent_state&            state,
                 uint64_t                code_object_id,
                 uint64_t                begin,
                 uint64_t                size,
                 rocprofiler_kernel_id_t kernel_id)
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
        kernel_symbol_range{kernel_id, code_object_id, begin, end});
}

void
add_exact_locked(agent_state&            state,
                 uint64_t                code_object_id,
                 uint64_t                address,
                 rocprofiler_kernel_id_t kernel_id)
{
    if(address == 0) return;
    state.kernels_by_entry[entry_key{code_object_id, address}] = kernel_id;
}

uint64_t
get_symbol_size_locked(const code_object_record& code_object, const kernel_symbol_record& symbol)
{
    if(auto elf_vaddr = subtract_signed_delta(symbol.kernel_address, code_object.load_delta))
    {
        if(auto itr = code_object.symbol_sizes_by_vaddr.find(*elf_vaddr);
           itr != code_object.symbol_sizes_by_vaddr.end())
            return itr->second;
    }

    if(auto itr = code_object.symbol_sizes_by_name.find(symbol.kernel_name);
       itr != code_object.symbol_sizes_by_name.end())
        return itr->second;

    return 0;
}

void
rebuild_kernel_ranges_locked(agent_state& state)
{
    state.kernels_by_entry.clear();
    state.kernel_ranges_by_code_object.clear();

    for(const auto& symbol : state.kernel_symbols)
    {
        auto code_object_itr = state.code_objects.find(symbol.code_object_id);
        auto symbol_size     = uint64_t{0};
        if(code_object_itr != state.code_objects.end())
            symbol_size = get_symbol_size_locked(code_object_itr->second, symbol);

        add_exact_locked(state, 0, symbol.kernel_address, symbol.kernel_id);
        add_range_locked(state, 0, symbol.kernel_address, symbol_size, symbol.kernel_id);

        if(code_object_itr == state.code_objects.end()) continue;

        const auto& code_object = code_object_itr->second;
        if(auto elf_vaddr = subtract_signed_delta(symbol.kernel_address, code_object.load_delta))
        {
            add_exact_locked(state, symbol.code_object_id, *elf_vaddr, symbol.kernel_id);
            add_range_locked(
                state, symbol.code_object_id, *elf_vaddr, symbol_size, symbol.kernel_id);
        }
    }
}

void
register_kernel_symbol_locked(agent_state& state, const kernel_symbol_record& symbol)
{
    state.kernel_symbols.emplace_back(symbol);
    rebuild_kernel_ranges_locked(state);
}

kernel_symbol_record
make_symbol_record(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t& data)
{
    return kernel_symbol_record{data.kernel_id,
                                data.code_object_id,
                                (data.kernel_name != nullptr) ? data.kernel_name : "",
                                data.kernel_address.handle,
                                data.kernel_code_entry_byte_offset};
}

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t                userdata)
{
    auto* state = static_cast<agent_state*>(userdata.ptr);
    if(state == nullptr || state->backend == nullptr || shader_data_forwarder() == nullptr) return;

    backend_shader_data(*state, shader_data, shader_data_forwarder(), kernel_target_filter());
}

void
start_agent_context(agent_state& state)
{
    auto start_context = false;
    {
        auto lock = std::unique_lock{state.mutex};
        if(!state.started && !state.finalized)
        {
            state.started = true;
            start_context = true;
        }
    }

    if(start_context)
    {
        LOG_INFO << fmt::format(
            "starting ATT no-intercept context for agent {} ({})", state.id.handle, state.name);
        check_status(rocprofiler_start_context(state.context), "ATT no-intercept context start");
    }
}

void
stop_agent_context(agent_state& state)
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
configure(std::vector<agent_config> agents,
          shader_data_forwarder_t   forwarder,
          kernel_target_filter_t    filter)
{
    ROCP_FATAL_IF(!backend_supported())
        << "--att-no-intercept was requested, but this rocprofv3 build was configured with "
           "ROCPROFILER_DISABLE_ATT_QUICK_SCAN=ON and does not include "
           "rocprof-trace-decoder quick-scan support";

    ROCP_FATAL_IF(forwarder == nullptr)
        << "ATT no-intercept setup requires a shader-data forwarding callback";

    auto lock               = std::unique_lock{manager_mutex()};
    shader_data_forwarder() = forwarder;
    kernel_target_filter()  = filter;

    for(auto& agent : agents)
    {
        if(agent_states().count(agent.id.handle) != 0) continue;

        auto state                 = std::make_unique<agent_state>();
        state->id                  = agent.id;
        state->gpu_index           = agent.gpu_index;
        state->name                = std::move(agent.name);
        state->parameters          = std::move(agent.parameters);
        state->consecutive_kernels = agent.consecutive_kernels;
        state->userdata.ptr        = state.get();

        check_status(rocprofiler_create_context(&state->context),
                     "ATT no-intercept context creation");

        state->backend = backend_create(*state);
        ROCP_FATAL_IF(state->backend == nullptr)
            << "failed to create ATT no-intercept decoder state for agent " << state->id.handle;

        check_status(rocprofiler_configure_device_thread_trace_service(state->context,
                                                                       state->id,
                                                                       state->parameters.data(),
                                                                       state->parameters.size(),
                                                                       shader_data_callback,
                                                                       state->userdata),
                     "ATT no-intercept thread-trace service configure");

        auto* state_ptr = state.get();
        agent_states().emplace(agent.id.handle, std::move(state));
        LOG_INFO << fmt::format("configured ATT no-intercept context for agent {} ({})",
                                 state_ptr->id.handle,
                                 state_ptr->name);
    }
}

void
code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& data)
{
    std::vector<kernel_symbol_record> symbols = {};
    agent_state*                      state   = nullptr;
    {
        auto lock = std::unique_lock{manager_mutex()};
        auto itr  = agent_states().find(data.agent_id.handle);
        if(itr == agent_states().end()) return;

        state                                     = itr->second.get();
        code_object_agents()[data.code_object_id] = state;

        auto pending_itr = pending_symbols().find(data.code_object_id);
        if(pending_itr != pending_symbols().end())
        {
            symbols = std::move(pending_itr->second);
            pending_symbols().erase(pending_itr);
        }
    }

    {
        auto lock                                = std::unique_lock{state->mutex};
        state->code_objects[data.code_object_id] = code_object_record{
            data.code_object_id, data.load_delta, data.load_base, data.load_size};
    }

    backend_code_object_load(*state, data);

    {
        auto lock = std::unique_lock{state->mutex};
        for(const auto& symbol : symbols)
            state->kernel_symbols.emplace_back(symbol);
        rebuild_kernel_ranges_locked(*state);
    }

    start_agent_context(*state);
}

void
kernel_symbol_load(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t& data)
{
    auto         symbol = make_symbol_record(data);
    agent_state* state  = nullptr;
    {
        auto lock = std::unique_lock{manager_mutex()};
        if(auto itr = code_object_agents().find(data.code_object_id);
           itr != code_object_agents().end())
        {
            state = itr->second;
        }
        else
        {
            pending_symbols()[data.code_object_id].emplace_back(symbol);
            return;
        }
    }

    auto lock = std::unique_lock{state->mutex};
    register_kernel_symbol_locked(*state, symbol);
}

void
finalize()
{
    std::vector<agent_state*> states = {};
    {
        auto lock = std::unique_lock{manager_mutex()};
        states.reserve(agent_states().size());
        for(auto& itr : agent_states())
            states.emplace_back(itr.second.get());
    }

    for(auto* state : states)
    {
        if(state == nullptr) continue;
        if(state->backend == nullptr) continue;

        stop_agent_context(*state);

        LOG_INFO << fmt::format(
            "ATT no-intercept agent {} ({}) stats: chunks={}, bytes={}, dispatches={}, "
            "targets={}, cuts={}, cross_chunk_skips={}, unknown_dispatches={}, scan_failures={}, "
            "cut_failures={}, buffer_full={}",
            state->id.handle,
            state->name,
            state->stats.chunks_scanned.load(),
            state->stats.bytes_scanned.load(),
            state->stats.dispatches_seen.load(),
            state->stats.target_dispatches.load(),
            state->stats.cuts_built.load(),
            state->stats.cross_chunk_skips.load(),
            state->stats.unknown_dispatches.load(),
            state->stats.quick_scan_failures.load(),
            state->stats.cut_build_failures.load(),
            state->stats.buffer_full_events.load());

        backend_destroy(*state);
    }
}

#if !defined(ROCPROFILER_ATT_QUICK_SCAN_ENABLED) || ROCPROFILER_ATT_QUICK_SCAN_ENABLED == 0
bool
backend_supported()
{
    return false;
}

void*
backend_create(agent_state&)
{
    return nullptr;
}

void
backend_destroy(agent_state&)
{}

void
backend_code_object_load(agent_state&, const rocprofiler_callback_tracing_code_object_load_data_t&)
{}

void
backend_shader_data(agent_state&,
                    rocprofiler_thread_trace_shader_data_t,
                    shader_data_forwarder_t,
                    kernel_target_filter_t)
{}
#endif
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
