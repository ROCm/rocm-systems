// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/types.hpp"

#include <gmock/gmock.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk::testing
{

// Self-contained mock types — no SDK or production includes.

enum class mock_status : int
{
    success        = 0,
    error          = 1,
    hsa_not_loaded = 2,
};

enum class mock_flag : int
{
    none = 0,
};

struct mock_context_id
{
    std::uint64_t handle{};
};
struct mock_agent_id
{
    std::uint64_t handle{};
};
struct mock_buffer_id
{
    std::uint64_t handle{};
};
struct mock_counter_id
{
    std::uint64_t handle{};
};
struct mock_config_id
{
    std::uint64_t handle{};
};
struct mock_counter_record
{
    std::uint64_t id{};
    double        counter_value{};
};
struct mock_user_data
{};

// Types that exist purely to satisfy
// policies::rocprofiler_sdk::domain_service_backend's static interface. The GPU
// perf-counter device<Backend> never exercises this surface (it uses the
// device-counting-service API below), so these are trivial stand-ins.
struct mock_record_header
{
    std::uint32_t category{};
    std::uint32_t kind{};
};
struct mock_callback_thread_id
{
    std::uint64_t handle{};
};
struct mock_callback_tracing_record
{
    std::uint32_t kind{};
    std::uint32_t operation{};
};
using mock_tracing_operation_t     = std::uint32_t;
using mock_buffer_tracing_kind_t   = std::uint32_t;
using mock_callback_tracing_kind_t = std::uint32_t;
using mock_buffer_policy_t         = std::uint32_t;
using mock_on_records_cb_t         = void (*)();
using mock_on_record_cb_t          = void (*)();

struct mock_tracing_name_entry
{
    std::string_view              name;
    std::vector<std::string_view> operations;
    std::size_t                   value{};
};

// Satisfies both std::ranges::range (via begin()/end()) and the `.at(kind, op)`
// lookup that domain_service_backend requires of the tracing-name tables.
struct mock_tracing_name_table
{
    std::vector<mock_tracing_name_entry> entries;

    [[nodiscard]] auto             begin() const { return entries.begin(); }
    [[nodiscard]] auto             end() const { return entries.end(); }
    [[nodiscard]] std::string_view at(std::uint32_t, std::uint32_t) const { return {}; }
};

// Stateless lambdas used as callbacks are convertible to these function-pointer types.
using mock_available_counters_cb_t      = mock_status (*)(mock_agent_id, mock_counter_id*,
                                                     std::size_t, void*);
using mock_device_counting_agent_cb_t   = void (*)(mock_context_id, mock_config_id);
using mock_device_counting_service_cb_t = void (*)(mock_context_id, mock_agent_id,
                                                   mock_device_counting_agent_cb_t,
                                                   void*);

// The GMock-verifiable object. Tests set EXPECT_CALL expectations directly on an
// instance of this class. It is never used as the `Backend` template parameter
// itself (see `mock_backend` below) because domain_service_backend requires several
// methods to be callable as unqualified static calls (`Backend::start_context(...)`),
// which is incompatible with GMock's instance-based MOCK_METHOD dispatch.
class mock_backend_impl
{
public:
    using status_t                     = mock_status;
    using context_id_t                 = mock_context_id;
    using agent_id_t                   = mock_agent_id;
    using buffer_id_t                  = mock_buffer_id;
    using counter_id_t                 = mock_counter_id;
    using counter_config_id_t          = mock_config_id;
    using counter_record_t             = mock_counter_record;
    using counter_metadata_t           = counter_metadata;
    using counter_flag_t               = mock_flag;
    using user_data_t                  = mock_user_data;
    using available_counters_cb_t      = mock_available_counters_cb_t;
    using device_counting_agent_cb_t   = mock_device_counting_agent_cb_t;
    using device_counting_service_cb_t = mock_device_counting_service_cb_t;

    static constexpr counter_flag_t flag_none             = mock_flag::none;
    static constexpr status_t       status_success        = mock_status::success;
    static constexpr status_t       status_error          = mock_status::error;
    static constexpr status_t       status_hsa_not_loaded = mock_status::hsa_not_loaded;

    static agent_id_t make_agent_id(std::uint64_t handle) { return agent_id_t{ handle }; }

    MOCK_METHOD(void, create_context, (context_id_t * context));
    MOCK_METHOD(void, start_context, (context_id_t context));
    MOCK_METHOD(void, stop_context, (context_id_t context));

    MOCK_METHOD(status_t, sample_device_counting_service,
                (context_id_t ctx, user_data_t user_data, counter_flag_t flags,
                 counter_record_t* output_records, size_t* record_count));

    MOCK_METHOD(status_t, query_record_counter_id,
                (counter_record_t record, counter_id_t* counter_id));

    MOCK_METHOD((std::vector<counter_metadata>), query_counter_details,
                (counter_id_t counter_id));

    MOCK_METHOD(status_t, iterate_agent_supported_counters,
                (agent_id_t agent_id, available_counters_cb_t callback, void* user_data));

    MOCK_METHOD(status_t, create_counter_config,
                (agent_id_t agent_id, counter_id_t* counters_list, size_t counters_count,
                 counter_config_id_t* config_id));

    MOCK_METHOD(status_t, configure_device_counting_service,
                (context_id_t ctx, buffer_id_t buf, agent_id_t agent,
                 device_counting_service_cb_t callback, void* user_data));
};

// The type actually used as the `Backend` / `backend_t` template parameter. It holds
// no per-instance state: every call is forwarded to whichever `mock_backend_impl`
// instance is currently bound (via `bind()`), so it can satisfy
// policies::rocprofiler_sdk::domain_service_backend's requirement that members like
// `create_context`/`start_context` be reachable via an unqualified static call
// (`Backend::start_context(ctx)`), while still routing through a GMock object that
// tests can set EXPECT_CALL expectations on.
class mock_backend
{
public:
    using status_t                     = mock_backend_impl::status_t;
    using context_id_t                 = mock_backend_impl::context_id_t;
    using agent_id_t                   = mock_backend_impl::agent_id_t;
    using buffer_id_t                  = mock_backend_impl::buffer_id_t;
    using counter_id_t                 = mock_backend_impl::counter_id_t;
    using counter_config_id_t          = mock_backend_impl::counter_config_id_t;
    using counter_record_t             = mock_backend_impl::counter_record_t;
    using counter_metadata_t           = mock_backend_impl::counter_metadata_t;
    using counter_flag_t               = mock_backend_impl::counter_flag_t;
    using user_data_t                  = mock_backend_impl::user_data_t;
    using available_counters_cb_t      = mock_backend_impl::available_counters_cb_t;
    using device_counting_agent_cb_t   = mock_backend_impl::device_counting_agent_cb_t;
    using device_counting_service_cb_t = mock_backend_impl::device_counting_service_cb_t;

    // domain_service_backend-only surface (unused by device<Backend>'s logic).
    using record_header_t           = mock_record_header;
    using callback_thread_id_t      = mock_callback_thread_id;
    using callback_tracing_record_t = mock_callback_tracing_record;
    using tracing_operation_t       = mock_tracing_operation_t;
    using buffer_tracing_kind_t     = mock_buffer_tracing_kind_t;
    using callback_tracing_kind_t   = mock_callback_tracing_kind_t;
    using buffer_policy_t           = mock_buffer_policy_t;
    using on_records_cb_t           = mock_on_records_cb_t;
    using on_record_cb_t            = mock_on_record_cb_t;

    static constexpr counter_flag_t flag_none      = mock_backend_impl::flag_none;
    static constexpr status_t       status_success = mock_backend_impl::status_success;
    static constexpr status_t       status_error   = mock_backend_impl::status_error;
    static constexpr status_t       status_hsa_not_loaded =
        mock_backend_impl::status_hsa_not_loaded;
    static constexpr std::uint32_t   compile_time_version   = 0;
    static constexpr buffer_policy_t BUFFER_POLICY_LOSSLESS = 0;

    static agent_id_t make_agent_id(std::uint64_t handle)
    {
        return mock_backend_impl::make_agent_id(handle);
    }

    // Binds the impl instance that static calls below forward to. Tests must call
    // this in SetUp() before constructing a device<mock_backend>/provider, and
    // unbind() in TearDown().
    static void bind(std::shared_ptr<mock_backend_impl> impl)
    {
        s_active = std::move(impl);
    }
    static void unbind() { s_active.reset(); }

    // --- backend_contract surface (forwarded to the bound impl for verification) ---
    static void create_context(context_id_t* ctx) { active().create_context(ctx); }
    static void start_context(context_id_t ctx) { active().start_context(ctx); }
    static void stop_context(context_id_t ctx) { active().stop_context(ctx); }

    // NOLINTBEGIN(readability-function-size)
    static status_t sample_device_counting_service(context_id_t      ctx,
                                                   user_data_t       user_data,
                                                   counter_flag_t    flags,
                                                   counter_record_t* out, size_t* count)
    {
        return active().sample_device_counting_service(ctx, user_data, flags, out, count);
    }
    // NOLINTEND

    static status_t query_record_counter_id(counter_record_t record,
                                            counter_id_t*    counter_id)
    {
        return active().query_record_counter_id(record, counter_id);
    }

    static std::vector<counter_metadata> query_counter_details(counter_id_t counter_id)
    {
        return active().query_counter_details(counter_id);
    }

    static status_t iterate_agent_supported_counters(agent_id_t              agent_id,
                                                     available_counters_cb_t callback,
                                                     void*                   user_data)
    {
        return active().iterate_agent_supported_counters(agent_id, callback, user_data);
    }

    static status_t create_counter_config(agent_id_t agent_id, counter_id_t* counters,
                                          size_t               counters_count,
                                          counter_config_id_t* config_id)
    {
        return active().create_counter_config(agent_id, counters, counters_count,
                                              config_id);
    }

    // NOLINTBEGIN(readability-function-size)
    static status_t configure_device_counting_service(
        context_id_t ctx, buffer_id_t buf, agent_id_t agent,
        device_counting_service_cb_t callback, void* user_data)
    {
        return active().configure_device_counting_service(ctx, buf, agent, callback,
                                                          user_data);
    }
    // NOLINTEND

    // --- domain_service_backend-only surface: never exercised, trivial no-ops ---
    static void create_buffer(context_id_t, std::size_t, std::size_t, buffer_policy_t,
                              on_records_cb_t, void*, buffer_id_t*)
    {}
    static void configure_buffer_tracing_service(context_id_t, buffer_tracing_kind_t,
                                                 tracing_operation_t*, std::size_t,
                                                 buffer_id_t)
    {}
    static void create_callback_thread(callback_thread_id_t*) {}
    static void assign_callback_thread(buffer_id_t, callback_thread_id_t) {}
    static void flush_buffer(buffer_id_t) {}
    static void destroy_buffer(buffer_id_t) {}
    static void configure_callback_tracing_service(context_id_t, callback_tracing_kind_t,
                                                   tracing_operation_t*, std::size_t,
                                                   on_record_cb_t, void*)
    {}
    static const mock_tracing_name_table& get_buffer_tracing_names()
    {
        static const mock_tracing_name_table table{};
        return table;
    }
    static const mock_tracing_name_table& get_callback_tracing_names()
    {
        static const mock_tracing_name_table table{};
        return table;
    }

private:
    static mock_backend_impl& active()
    {
        assert(s_active != nullptr && "mock_backend: call bind() before use");
        return *s_active;
    }

    static inline std::shared_ptr<mock_backend_impl> s_active{};
};

struct mock_backend_factory
{
    using backend_t = mock_backend;

    static void set_mock(std::shared_ptr<mock_backend_impl> impl)
    {
        mock_backend::bind(std::move(impl));
    }

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<backend_t>();
    }

    static void reset() { mock_backend::unbind(); }
};

}  // namespace rocprofsys::backends::rocprofiler_sdk::testing
