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

#pragma once

#include <rocprofiler/fwd.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/rocprofiler/context/domain.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace rocprofiler
{
namespace context
{
using external_cid_cb_t = uint64_t (*)(rocprofiler_service_callback_tracing_kind_t,
                                       uint32_t,
                                       uint64_t);

/// permits tools opportunity to modify the correlation id based on the domain, op, and
/// the rocprofiler generated correlation id
struct correlation_tracing_service
{
    uint64_t          id                   = 0;
    uint64_t          external_id          = 0;
    external_cid_cb_t external_id_callback = nullptr;

    static uint64_t get_unique_record_id();
};

struct callback_tracing_service
{
    struct callback_data
    {
        rocprofiler_callback_tracing_cb_t callback = nullptr;
        void*                             data     = nullptr;
    };

    using domain_t         = rocprofiler_service_callback_tracing_kind_t;
    using callback_array_t = std::array<callback_data, domain_info<domain_t>::last>;

    domain_context<domain_t> domains       = {};
    callback_array_t         callback_data = {};
};

struct buffer_tracing_service
{
    using domain_t       = rocprofiler_service_buffer_tracing_kind_t;
    using buffer_array_t = std::array<rocprofiler_buffer_id_t, domain_info<domain_t>::last>;

    domain_context<domain_t> domains     = {};
    buffer_array_t           buffer_data = {};
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

/// \brief disable the contexturation.
rocprofiler_status_t stop_context(rocprofiler_context_id_t);

using unique_context_vec_t = common::container::stable_vector<std::unique_ptr<context>, 8>;
using active_context_vec_t = common::container::stable_vector<std::atomic<context*>, 8>;

unique_context_vec_t&
get_registered_contexts();

active_context_vec_t&
get_active_contexts();
}  // namespace context
}  // namespace rocprofiler
