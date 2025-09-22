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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "lib/rocprofiler-sdk/external_correlation.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace rocprofiler
{
namespace context
{
using external_cid_cb_t = uint64_t (*)(rocprofiler_callback_tracing_kind_t, uint32_t, uint64_t);

constexpr auto null_user_data = rocprofiler_user_data_t{.value = 0};

struct callback_tracing_service
{
    struct callback_data
    {
        rocprofiler_callback_tracing_cb_t callback = nullptr;
        void*                             data     = nullptr;
    };

    using domain_t         = rocprofiler_callback_tracing_kind_t;
    using callback_array_t = std::array<callback_data, domain_info<domain_t>::last>;

    domain_context<domain_t> domains       = {};
    callback_array_t         callback_data = {};
};

struct buffer_tracing_service
{
    using domain_t       = rocprofiler_buffer_tracing_kind_t;
    using buffer_array_t = std::array<rocprofiler_buffer_id_t, domain_info<domain_t>::last>;

    domain_context<domain_t> domains     = {};
    buffer_array_t           buffer_data = {};
};

struct dispatch_counter_collection_service
{
    // Contains a vector of counter collection instances associated with this context.
    // Each instance is assocated with an agent and a counter collection profile.
    // Contains callback information along with other data needed to collect/process
    // counters.
    std::vector<std::shared_ptr<counters::counter_callback_info>> callbacks{};
    // A flag to state wether or not the counter set is currently enabled. This is primarily
    // to protect against multithreaded calls to enable a context (and enabling already enabled
    // counters).
    common::Synchronized<bool> enabled{false};
};

struct device_counting_service
{
    std::unordered_set<uint64_t>                            conf_agents;
    std::vector<rocprofiler::counters::agent_callback_data> agent_data;

    enum class state
    {
        DISABLED,
        LOCKED,
        ENABLED,
        EXIT
    };
    std::atomic<state> status{state::DISABLED};

    common::Synchronized<bool> enabled{false};
};

struct pc_sampling_service
{
    // Contains a map with pairs (rocprofiler_agent_id_t, PCSAgentSession*).
    // The PCSAgentSession encapsulates the information about the configured PC sampling session
    // used on the agent with `rocprofiler_agent_id_t`.
    std::unordered_map<rocprofiler_agent_id_t,
                       std::unique_ptr<rocprofiler::pc_sampling::PCSAgentSession>>
        agent_sessions;
};

struct context
{
    // size is used to ensure that we never read past the end of the version
    size_t                                    size               = 0;
    uint64_t                                  context_idx        = 0;  // context id
    uint32_t                                  client_idx         = 0;  // tool id
    correlation_tracing_service               correlation_tracer = {};
    std::unique_ptr<callback_tracing_service> callback_tracer    = {};
    std::unique_ptr<buffer_tracing_service>   buffered_tracer    = {};
    // Only one of counter collection/agent counter collection can exists in the ctx.
    std::unique_ptr<dispatch_counter_collection_service> counter_collection        = {};
    std::unique_ptr<device_counting_service>             device_counter_collection = {};
    std::unique_ptr<pc_sampling_service>                 pc_sampler                = {};

    std::unique_ptr<thread_trace::DispatchThreadTracer> dispatch_thread_trace = {};
    std::unique_ptr<thread_trace::DeviceThreadTracer>   device_thread_trace   = {};

    template <typename KindT>
    bool is_tracing(KindT _kind) const;

    template <typename... Args>
    bool is_tracing_one_of(Args... _args) const
    {
        return (is_tracing(_args) || ...);
    }
};

// set the client index needs to be called before allocate_context()
void push_client(uint32_t);

// remove the client index
void pop_client(uint32_t);

/// @brief creates a context struct and returns a handle for locating the context struct
///
std::optional<rocprofiler_context_id_t>
allocate_context();

/// \brief rocprofiler validates context, checks for conflicts, etc. Ensures that
///  the contexturation is valid *in isolation*, e.g. it may check that the user
///  set the compat_version field and that required context fields, such as buffer
///  are set. This function will be called before \ref start_context
///  but is provided to help the user validate one or more contexts without starting
///  them
///
/// \param [in] cfg contexturation to validate
rocprofiler_status_t
validate_context(const context* cfg);

/// \brief rocprofiler activates contexturation and provides a context identifier
/// \param [in] id the context identifier to start.
rocprofiler_status_t
start_context(rocprofiler_context_id_t id);

/// \brief remove context from active array
rocprofiler_status_t
stop_context(rocprofiler_context_id_t id);

using context_array_t = common::container::small_vector<const context*>;

context*
get_mutable_registered_context(rocprofiler_context_id_t id);

const context*
get_registered_context(rocprofiler_context_id_t id);

using context_filter_t = bool (*)(const context*);

inline bool
default_context_filter(const context* val);

context_array_t&
get_registered_contexts(context_array_t& data, context_filter_t filter = default_context_filter);

context_array_t
get_registered_contexts(context_filter_t filter = default_context_filter);

context_array_t&
get_registered_contexts(context_array_t& data, context_filter_t filter);

context_array_t
get_registered_contexts(context_filter_t filter);

context_array_t&
get_active_contexts(context_array_t& data, context_filter_t filter = default_context_filter);

context_array_t
get_active_contexts(context_filter_t filter = default_context_filter);

const context*
get_active_context(rocprofiler_context_id_t id);

/// \brief disable the contexturation.
rocprofiler_status_t
stop_client_contexts(rocprofiler_client_id_t id);

void
deactivate_client_contexts(rocprofiler_client_id_t id);

// should only be called if the client failed to initialize
void
deregister_client_contexts(rocprofiler_client_id_t id);

inline bool
default_context_filter(const context* val)
{
    return (val != nullptr);
}
}  // namespace context
}  // namespace rocprofiler
