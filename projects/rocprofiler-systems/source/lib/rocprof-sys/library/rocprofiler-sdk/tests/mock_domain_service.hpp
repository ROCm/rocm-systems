// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::domains::test_support
{

// Shared stand-in for SdkBackend, satisfying policies::domain_service::backend for
// every test that instantiates callback_domain<>, buffered_domain<>, registry<>,
// domain_service<>, or a single kfd_* on_record(s)/on_configure() pair. Carries every
// kfd_* domain's record type and BUFFER_TRACING_*/CALLBACK_TRACING_* id, since
// registry<>/domain_service<> pull in the full domains::registry<> regardless of which
// single domain a given test exercises.
struct context_id_t
{
    std::uint64_t handle                                 = 0;
    auto          operator<=>(const context_id_t&) const = default;
};

struct buffer_id_t
{
    std::uint64_t handle                                = 0;
    auto          operator<=>(const buffer_id_t&) const = default;
};

struct callback_thread_id_t
{
    std::uint64_t handle                                         = 0;
    auto          operator<=>(const callback_thread_id_t&) const = default;
};

struct record_header_t
{
    void* payload = nullptr;
};

struct user_data_t
{
    std::uint64_t value = 0;
};

using callback_phase_t = int;

struct correlation_id_t
{
    std::uint64_t internal = 0;
};

struct callback_tracing_record_t
{
    std::uint64_t    kind      = 0;
    callback_phase_t phase     = 0;
    std::uint32_t    operation = 0;
    std::uint64_t    thread_id = 0;
    correlation_id_t correlation_id{};
};

using tracing_operation_t     = std::size_t;
using buffer_tracing_kind_t   = std::size_t;
using callback_tracing_kind_t = std::size_t;
using buffer_policy_t         = int;
using on_records_cb_t         = void (*)(context_id_t, buffer_id_t, record_header_t**,
                                 std::size_t, void*, std::uint64_t);
using on_record_cb_t          = void (*)(callback_tracing_record_t, user_data_t*, void*);

struct agent_id_t
{
    std::uint64_t handle = 0;
};

struct address_t
{
    std::uint64_t value = 0;
};

struct kfd_event_dropped_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    std::uint64_t timestamp = 0;
    std::uint64_t count     = 0;
};
struct kfd_event_page_fault_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    address_t     address{};
    std::uint64_t timestamp = 0;
};
struct kfd_event_page_migrate_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    src_agent{};
    agent_id_t    dst_agent{};
    agent_id_t    prefetch_agent{};
    agent_id_t    preferred_agent{};
    address_t     start_address{};
    address_t     end_address{};
    std::uint64_t timestamp  = 0;
    std::int32_t  error_code = 0;
};
struct kfd_event_queue_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    std::uint64_t timestamp = 0;
};
struct kfd_event_unmap_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    std::uint64_t timestamp = 0;
    address_t     start_address{};
    address_t     end_address{};
};
struct kfd_page_fault_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    address_t     address{};
    std::uint64_t start_timestamp = 0;
    std::uint64_t end_timestamp   = 0;
};
struct kfd_page_migrate_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    src_agent{};
    agent_id_t    dst_agent{};
    agent_id_t    prefetch_agent{};
    agent_id_t    preferred_agent{};
    address_t     start_address{};
    address_t     end_address{};
    std::int32_t  error_code      = 0;
    std::uint64_t start_timestamp = 0;
    std::uint64_t end_timestamp   = 0;
};
struct kfd_queue_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    std::uint64_t start_timestamp = 0;
    std::uint64_t end_timestamp   = 0;
};

// Satisfies policies::domain_service::backend's requirement that
// get_{buffer,callback}_tracing_names() return a std::ranges::range of entries exposing
// name/operations/value. Kept as a plain (non-gmock) value: on_kfd_*<...> calls
// get_buffer_tracing_names().at(...) unconditionally, so it must work without a test
// having to set up an expectation for it.
struct tracing_names_t
{
    struct entry_t
    {
        std::string_view              name;
        std::vector<std::string_view> operations;
        std::size_t                   value = 0;
    };

    std::vector<entry_t> entries;

    [[nodiscard]] auto begin() const { return entries.begin(); }
    [[nodiscard]] auto end() const { return entries.end(); }

    [[nodiscard]] std::string_view at(std::size_t /*kind*/,
                                      std::uint32_t /*operation*/) const
    {
        return "operation";
    }
};

// Every policies::domain_service::backend member that tests actually verify calls
// into, as a GMock method -- a test EXPECT_CALLs only the handful its scenario
// exercises; StrictMock fails it if anything else is touched.
// get_{buffer,callback}_tracing_names are deliberately NOT here; see tracing_names_t
// above.
struct gmock_sdk_backend
{
    MOCK_METHOD(void, create_context, (context_id_t * context));
    MOCK_METHOD(void, start_context, (context_id_t context));
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, create_buffer,
                (context_id_t context, std::size_t buffer_size,
                 std::size_t buffer_watermark, buffer_policy_t policy,
                 on_records_cb_t callback, void* callback_data, buffer_id_t* buffer_out));
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, configure_buffer_tracing_service,
                (context_id_t context, buffer_tracing_kind_t kind,
                 tracing_operation_t* operations, std::size_t num_operations,
                 buffer_id_t buffer));
    MOCK_METHOD(void, create_callback_thread, (callback_thread_id_t * thread));
    MOCK_METHOD(void, assign_callback_thread,
                (buffer_id_t buffer, callback_thread_id_t thread));
    MOCK_METHOD(void, flush_buffer, (buffer_id_t buffer));
    MOCK_METHOD(int, destroy_buffer, (buffer_id_t buffer));
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, configure_callback_tracing_service,
                (context_id_t context, callback_tracing_kind_t kind,
                 tracing_operation_t* operations, std::size_t num_operations,
                 on_record_cb_t on_record, void* callback_data));
};

inline std::unique_ptr<::testing::StrictMock<gmock_sdk_backend>> g_mock;

// Settable by domain_service<> tests to drive filter_supported_domains() with a
// specific set of supported buffered/callback domains; left empty (the default) by
// tests -- e.g. registry<>/callback_domain<>/buffered_domain<> tests -- that never
// call get_{buffer,callback}_tracing_names().
inline tracing_names_t g_buffer_table;
inline tracing_names_t g_callback_table;

// SdkBackend stand-in: every lifecycle member forwards to g_mock, so tests drive
// behavior entirely through EXPECT_CALL instead of hand-written shim bodies. Tests
// that never invoke a given member (e.g. the kfd_* on_record(s)/on_configure() tests,
// which call the free function directly instead of going through buffered_domain<>)
// simply never touch g_mock and can leave it unset.
struct mock_sdk
{
    using context_id_t              = test_support::context_id_t;
    using buffer_id_t               = test_support::buffer_id_t;
    using callback_thread_id_t      = test_support::callback_thread_id_t;
    using record_header_t           = test_support::record_header_t;
    using user_data_t               = test_support::user_data_t;
    using callback_tracing_record_t = test_support::callback_tracing_record_t;
    using callback_phase_t          = test_support::callback_phase_t;
    using tracing_operation_t       = test_support::tracing_operation_t;
    using buffer_tracing_kind_t     = test_support::buffer_tracing_kind_t;
    using callback_tracing_kind_t   = test_support::callback_tracing_kind_t;
    using buffer_policy_t           = test_support::buffer_policy_t;
    using on_records_cb_t           = test_support::on_records_cb_t;
    using on_record_cb_t            = test_support::on_record_cb_t;
    using tracing_names_t           = test_support::tracing_names_t;
    using agent_id_t                = test_support::agent_id_t;
    using timestamp_t               = std::uint64_t;
    using correlation_id_t          = test_support::correlation_id_t;
    // Satisfies policies::domain_service::backend's requirement that
    // iterate_callback_tracing_kind_operation_args() accept a callback of this shape;
    // the real backend<Wrapper> forwards this type straight from rocprofiler-sdk.
    using callback_tracing_operation_args_cb_t = int (*)(std::uint64_t, std::int32_t,
                                                         std::uint32_t, const void* const,
                                                         std::int32_t, const char*,
                                                         const char*, const char*,
                                                         std::int32_t, void*);

    // NOLINTBEGIN(readability-identifier-naming)
    static constexpr std::size_t      compile_time_version                    = 90909;
    static constexpr buffer_policy_t  BUFFER_POLICY_LOSSLESS                  = 1;
    static constexpr std::size_t      BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS = 20;
    static constexpr std::size_t      BUFFER_TRACING_KFD_EVENT_PAGE_FAULT     = 21;
    static constexpr std::size_t      BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE   = 22;
    static constexpr std::size_t      BUFFER_TRACING_KFD_EVENT_QUEUE          = 23;
    static constexpr std::size_t      BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU = 24;
    static constexpr std::size_t      BUFFER_TRACING_KFD_PAGE_FAULT           = 25;
    static constexpr std::size_t      BUFFER_TRACING_KFD_PAGE_MIGRATE         = 26;
    static constexpr std::size_t      BUFFER_TRACING_KFD_QUEUE                = 27;
    static constexpr std::size_t      CALLBACK_TRACING_CODE_OBJECT            = 1;
    static constexpr std::size_t      CALLBACK_TRACING_HSA_CORE_API           = 2;
    static constexpr std::size_t      CALLBACK_TRACING_HSA_AMD_EXT_API        = 3;
    static constexpr std::size_t      CALLBACK_TRACING_HSA_IMAGE_EXT_API      = 4;
    static constexpr std::size_t      CALLBACK_TRACING_HSA_FINALIZE_EXT_API   = 5;
    static constexpr std::size_t      CALLBACK_TRACING_HIP_RUNTIME_API        = 6;
    static constexpr std::size_t      CALLBACK_TRACING_HIP_COMPILER_API       = 7;
    static constexpr std::size_t      CALLBACK_TRACING_ROCJPEG_API            = 8;
    static constexpr std::size_t      CALLBACK_TRACING_ROCDECODE_API          = 9;
    static constexpr std::size_t      CALLBACK_TRACING_ROCSHMEM_API           = 10;
    static constexpr std::size_t      CALLBACK_TRACING_HIPFILE_API            = 11;
    static constexpr callback_phase_t CALLBACK_PHASE_ENTER                    = 0;
    static constexpr callback_phase_t CALLBACK_PHASE_EXIT                     = 1;
    static constexpr callback_phase_t CALLBACK_PHASE_NONE                     = 2;
    // NOLINTEND(readability-identifier-naming)

    using kfd_event_dropped_record      = test_support::kfd_event_dropped_record;
    using kfd_event_page_fault_record   = test_support::kfd_event_page_fault_record;
    using kfd_event_page_migrate_record = test_support::kfd_event_page_migrate_record;
    using kfd_event_queue_record        = test_support::kfd_event_queue_record;
    using kfd_event_unmap_record        = test_support::kfd_event_unmap_record;
    using kfd_page_fault_record         = test_support::kfd_page_fault_record;
    using kfd_page_migrate_record       = test_support::kfd_page_migrate_record;
    using kfd_queue_record              = test_support::kfd_queue_record;

    static void create_context(context_id_t* context) { g_mock->create_context(context); }
    static void start_context(context_id_t context) { g_mock->start_context(context); }

    // NOLINTNEXTLINE(readability-function-size)
    static void create_buffer(context_id_t context, std::size_t buffer_size,
                              std::size_t buffer_watermark, buffer_policy_t policy,
                              on_records_cb_t callback, void* callback_data,
                              buffer_id_t* buffer_out)
    {
        g_mock->create_buffer(context, buffer_size, buffer_watermark, policy, callback,
                              callback_data, buffer_out);
    }

    // NOLINTNEXTLINE(readability-function-size)
    static void configure_buffer_tracing_service(context_id_t          context,
                                                 buffer_tracing_kind_t kind,
                                                 tracing_operation_t*  operations,
                                                 std::size_t           num_operations,
                                                 buffer_id_t           buffer)
    {
        g_mock->configure_buffer_tracing_service(context, kind, operations,
                                                 num_operations, buffer);
    }

    static void create_callback_thread(callback_thread_id_t* thread)
    {
        g_mock->create_callback_thread(thread);
    }

    static void assign_callback_thread(buffer_id_t buffer, callback_thread_id_t thread)
    {
        g_mock->assign_callback_thread(buffer, thread);
    }

    static void flush_buffer(buffer_id_t buffer) { g_mock->flush_buffer(buffer); }

    static int destroy_buffer(buffer_id_t buffer)
    {
        return g_mock->destroy_buffer(buffer);
    }

    // NOLINTNEXTLINE(readability-function-size)
    static void configure_callback_tracing_service(context_id_t            context,
                                                   callback_tracing_kind_t kind,
                                                   tracing_operation_t*    operations,
                                                   std::size_t             num_operations,
                                                   on_record_cb_t          on_record,
                                                   void*                   callback_data)
    {
        g_mock->configure_callback_tracing_service(
            context, kind, operations, num_operations, on_record, callback_data);
    }

    static tracing_names_t get_buffer_tracing_names() { return g_buffer_table; }
    static tracing_names_t get_callback_tracing_names() { return g_callback_table; }

    static timestamp_t get_timestamp() { return 0; }

    static std::uint64_t get_parent_stack_id(const correlation_id_t& /*correlation_id*/)
    {
        return 0;
    }

    static void iterate_callback_tracing_kind_operation_args(
        const callback_tracing_record_t& /*record*/,
        callback_tracing_operation_args_cb_t /*callback*/, std::int32_t /*max_deref*/,
        void* /*data*/)
    {}
};

// Stand-in for the agent/trace_cache::info shapes every on_kfd_*<...> touches through
// Externals. Every field beyond type/device_type_index exists solely so agent_t
// satisfies policies::agent_policy (required transitively by
// policies::domain_service::externals's agent_manager_policy check); tests never read
// them.
struct agent_t
{
    int           type                 = 0;
    std::uint64_t handle               = 0;
    std::uint64_t device_id            = 0;
    std::uint32_t node_id              = 0;
    std::int32_t  logical_node_id      = 0;
    std::int32_t  logical_node_type_id = 0;
    std::string   name;
    std::string   model_name;
    std::string   vendor_name;
    std::string   product_name;
    std::size_t   device_type_index = 0;
    std::string   agent_info;
    std::uint32_t location_id = 0;
    std::uint32_t domain      = 0;
    bool          hip_visible = true;
};

struct pmc_info_data_t
{
    int           type             = 0;
    std::size_t   agent_type_index = 0;
    std::string   target_arch;
    std::size_t   event_code  = 0;
    std::size_t   instance_id = 0;
    std::string   name;
    std::string   symbol;
    std::string   description;
    std::string   long_description;
    std::string   component;
    std::string   units;
    std::string   value_type;
    std::string   block;
    std::string   expression;
    std::uint32_t is_constant = 0;
    std::uint32_t is_derived  = 0;
    std::string   extdata;
};

// Every production on_configure() body calls exactly these members
// unconditionally-or-conditionally; mocked so tests can verify they ran correctly
// instead of just not crashing.
//
// The members below add_pmc_info are only touched by on_tracing_api_enter/exit
// (library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp) and its
// externals_with_tracing wrapper further down this file -- no on_configure() test
// exercises them, so extending this struct is safe for every existing StrictMock user.
struct gmock_externals
{
    MOCK_METHOD(void, add_string, (std::string_view value));
    MOCK_METHOD(std::vector<std::shared_ptr<agent_t>>, get_agents_by_type, (int type));
    MOCK_METHOD(void, add_pmc_info, (const pmc_info_data_t& info));

    MOCK_METHOD(bool, is_active, ());
    MOCK_METHOD(bool, get_use_timemory, ());
    MOCK_METHOD(void, tracing_push_timemory, (std::string_view name));
    MOCK_METHOD(void, tracing_pop_timemory, (std::string_view name));
    MOCK_METHOD(bool, check_backtrace_operations,
                (std::uint64_t kind, std::uint32_t operation));
    MOCK_METHOD(std::optional<int>, get_backtrace_data, (bool are_operations_available));
    MOCK_METHOD(void, metadata_add_string, (std::string_view value));
    MOCK_METHOD(void, metadata_add_thread_info,
                (std::int32_t parent_process_id, std::int32_t process_id,
                 std::uint64_t thread_id));
    MOCK_METHOD(std::int32_t, get_pid, ());
    MOCK_METHOD(std::int32_t, get_ppid, ());
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, region_sample_buffer_storage_store,
                (std::uint64_t thread_id, std::string region_name,
                 std::uint64_t correlation_id, std::uint64_t parent_stack_id,
                 std::uint64_t begin_timestamp, std::uint64_t end_timestamp,
                 std::string args_str, std::string category));
};

inline std::unique_ptr<::testing::StrictMock<gmock_externals>> g_externals_mock;

// Externals mirrors the real ExternalDeps policy surface used by every on_kfd_* and
// on_kfd_*_configure, plus domain_service<>/registry<>. The on_records-only members
// (add_thread_info/add_track/buffer_storage_store) stay plain no-ops -- only
// add_string/get_agents_by_type/add_pmc_info, which on_configure() exercises, are
// mocked. Category name/description constants for every kfd_* domain are carried
// here since policies::domain_service::externals requires the full set regardless of
// which single domain a given test exercises.
struct externals
{
    using agent_t      = test_support::agent_t;
    using pmc_info_t   = pmc_info_data_t;
    using agent_type_t = int;

    struct thread_info_t
    {
        std::int32_t  parent_process_id = 0;
        std::int32_t  process_id        = 0;
        std::uint64_t thread_id         = 0;
        std::uint32_t start             = 0;
        std::uint32_t end               = 0;
        std::string   extdata;
    };

    struct track_t
    {
        std::string   track_name;
        std::uint64_t thread_id = 0;
        std::string   extdata;
    };

    struct kfd_sample_t
    {
        std::uint64_t               thread_id = 0;
        std::string                 name;
        std::uint64_t               start_timestamp = 0;
        std::uint64_t               end_timestamp   = 0;
        std::string                 args_str;
        std::string                 category;
        std::string                 track_name;
        std::string                 event_metadata;
        std::uint32_t               device_id   = 0;
        std::uint8_t                device_type = 0;
        std::string                 pmc_info_name;
        double                      value = 0.0;
        std::optional<std::int64_t> system_tid;
    };

    // Satisfies policies::agent_manager_policy (required transitively by
    // policies::domain_service::externals); only get_agents_by_type and the
    // single-argument get_agent_by_handle are ever exercised by these tests, so the
    // rest are unreachable stubs.
    struct agent_manager_t
    {
        agent_manager_t() = default;
        explicit agent_manager_t(const std::vector<std::shared_ptr<agent_t>>& /*agents*/)
        {}

        void insert_agent(agent_t& /*agent*/) {}

        [[nodiscard]] std::vector<std::shared_ptr<agent_t>> get_agents_by_type(
            int type) const
        {
            return g_externals_mock->get_agents_by_type(type);
        }

        [[nodiscard]] const agent_t& get_agent_by_type_index(std::size_t /*type_index*/,
                                                             int /*type*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] const agent_t& get_agent_by_id(std::size_t /*device_id*/,
                                                     int /*type*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] const agent_t& get_agent_by_handle(std::size_t /*handle*/,
                                                         int /*type*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        // Opt-in sentinel handle a test can pass to exercise the not-found path, since
        // production get_agent_by_handle() implementations throw std::out_of_range when
        // the agent isn't registered.
        static constexpr std::uint64_t k_unknown_agent_handle = 0xDEAD;

        [[nodiscard]] const agent_t& get_agent_by_handle(std::uint64_t handle) const
        {
            if(handle == k_unknown_agent_handle)
            {
                throw std::out_of_range(
                    fmt::format("Agent not found for device handle: {}", handle));
            }
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] std::vector<std::shared_ptr<agent_t>> get_agents() const
        {
            return {};
        }

        [[nodiscard]] std::size_t get_gpu_agents_count() const { return 0; }
        [[nodiscard]] std::size_t get_cpu_agents_count() const { return 0; }
    };

    static constexpr int k_agent_type_gpu = 1;
    static constexpr int k_agent_type_cpu = 0;

    static agent_manager_t& get_agent_manager()
    {
        static agent_manager_t s_manager;
        return s_manager;
    }

    static void add_string(std::string_view value)
    {
        g_externals_mock->add_string(value);
    }
    static void add_thread_info(const thread_info_t& /*info*/) {}
    static void add_track(const track_t& /*info*/) {}
    static void add_pmc_info(const pmc_info_t& info)
    {
        g_externals_mock->add_pmc_info(info);
    }
    static void buffer_storage_store(kfd_sample_t&& /*sample*/) {}

    // ─── Members required by domains::callback::hip::{runtime,compiler}_api ────────
    struct rocm_hip_api_category
    {};

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_hip_api_category_name = "rocm_hip_api";

    // ─── Members required by domains::callback::hsa::{core,amd_ext,image_ext,
    // finalize_ext}_api ─────────────────────────────────────────────────────────
    struct rocm_hsa_api_category
    {};

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_hsa_api_category_name = "rocm_hsa_api";

    // ─── Members required by domains::callback::{rocjpeg,rocdecode,rocshmem,
    // hipfile}_api ──────────────────────────────────────────────────────────────
    struct rocm_rocjpeg_api_category
    {};

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rocjpeg_api_category_name = "rocm_rocjpeg_api";

    struct rocm_rocdecode_api_category
    {};

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rocdecode_api_category_name =
        "rocm_rocdecode_api";

    struct rocm_rocshmem_api_category
    {};

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rocshmem_api_category_name =
        "rocm_rocshmem_api";

    struct rocm_hipfile_api_category
    {};

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_hipfile_api_category_name = "rocm_hipfile_api";

    struct region_sample
    {
        std::uint64_t    thread_id = 0;
        std::string_view name;
        std::uint64_t    correlation_id  = 0;
        std::uint64_t    parent_stack_id = 0;
        std::uint64_t    start_timestamp = 0;
        std::uint64_t    end_timestamp   = 0;
        std::string_view call_stack;
        std::string_view args_str;
        std::string_view category;
    };

    struct backtrace_json_t
    {
        // dump() must mirror the SDK's std::string-returning json dump(), which
        // implicitly converts to std::string_view at the call site.
        // NOLINTNEXTLINE(modernize-use-string-view)
        [[nodiscard]] std::string dump() const { return {}; }
    };

    static bool is_active() { return true; }
    static bool get_use_timemory() { return false; }

    template <typename CategoryT>
    static void tracing_push_timemory(CategoryT, std::string_view /*name*/)
    {}

    template <typename CategoryT>
    static void tracing_pop_timemory(CategoryT, std::string_view /*name*/)
    {}

    static void metadata_add_string(std::string_view /*value*/) {}
    static void metadata_add_thread_info(const thread_info_t& /*info*/) {}

    static void buffer_storage_store(region_sample&& /*sample*/) {}

    static bool check_backtrace_operations(std::uint64_t /*kind*/,
                                           std::uint32_t /*operation*/)
    {
        return false;
    }

    static std::optional<int> get_backtrace_data(bool /*are_operations_available*/)
    {
        return std::nullopt;
    }

    static backtrace_json_t get_backtrace_json(const std::optional<int>& /*bt_data*/)
    {
        return {};
    }

    static std::int32_t get_pid() { return 0; }
    static std::int32_t get_ppid() { return 0; }

    static constexpr std::string_view k_pmc_value_type_absolute = "ABS";

    static constexpr std::string_view k_kfd_event_dropped_events_category_name =
        "rocm_kfd_event_dropped_events";
    static constexpr std::string_view k_kfd_event_dropped_events_category_description =
        "KFD Dropped Events";
    static constexpr std::string_view k_kfd_event_queue_category_name =
        "rocm_kfd_event_queue";
    static constexpr std::string_view k_kfd_event_queue_category_description =
        "KFD Event Queue";
    static constexpr std::string_view k_kfd_event_unmap_from_gpu_category_name =
        "rocm_kfd_event_unmap_from_gpu";
    static constexpr std::string_view k_kfd_event_unmap_from_gpu_category_description =
        "KFD Unmap from GPU";
    static constexpr std::string_view k_kfd_page_fault_category_name =
        "rocm_kfd_page_fault";
    static constexpr std::string_view k_kfd_page_fault_category_description =
        "KFD Page Fault Events";
    static constexpr std::string_view k_kfd_page_migrate_category_name =
        "rocm_kfd_page_migrate";
    static constexpr std::string_view k_kfd_page_migrate_category_description =
        "KFD Page Migrate Events";
    static constexpr std::string_view k_kfd_event_page_fault_category_name =
        "rocm_kfd_event_page_fault";
    static constexpr std::string_view k_kfd_event_page_fault_category_description =
        "KFD Event Page Fault Events";
    static constexpr std::string_view k_kfd_event_page_migrate_category_name =
        "rocm_kfd_event_page_migrate";
    static constexpr std::string_view k_kfd_event_page_migrate_category_description =
        "KFD Event Page Migrate Events";
    static constexpr std::string_view k_kfd_queue_category_name = "rocm_kfd_queue";
    static constexpr std::string_view k_kfd_queue_category_description =
        "KFD Queue Events";
};

// Every SdkBackend member on_tracing_api_enter/exit
// (library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp) touches; mocked so
// tests can control return values and assert exactly what gets called, instead of the
// fixed get_timestamp()==0/get_parent_stack_id()==0 behavior mock_sdk hard-codes for
// every other callback-domain test.
struct gmock_tracing_backend
{
    MOCK_METHOD(std::uint64_t, get_timestamp, ());
    MOCK_METHOD(tracing_names_t, get_callback_tracing_names, ());
    MOCK_METHOD(std::uint64_t, get_parent_stack_id, (const correlation_id_t&) );
    // Forwards the real callback function pointer and user-data blob through the mock
    // so a test can WillOnce(Invoke(...)) it to actually call back into
    // production's iterate_args_callback (library/rocprofiler-sdk/callback/
    // common_tracing_callbacks.hpp) -- otherwise that function has zero coverage,
    // since the default (no WillOnce) StrictMock action never invokes it.
    MOCK_METHOD(void, iterate_args,
                (std::uint64_t kind, std::uint32_t operation,
                 mock_sdk::callback_tracing_operation_args_cb_t callback, void* data));
};

inline std::unique_ptr<::testing::StrictMock<gmock_tracing_backend>>
    g_tracing_backend_mock;

// SdkBackend stand-in for on_tracing_api_enter/exit tests: inherits mock_sdk's
// boilerplate (context_id_t, kfd_*, ...) so it still satisfies
// policies::domain_service::backend, then hides (re-declares) exactly the four
// members those two functions touch, routing them through g_tracing_backend_mock.
struct mock_sdk_with_tracing : mock_sdk
{
    static std::uint64_t get_timestamp()
    {
        return g_tracing_backend_mock->get_timestamp();
    }

    static tracing_names_t get_callback_tracing_names()
    {
        return g_tracing_backend_mock->get_callback_tracing_names();
    }

    static std::uint64_t get_parent_stack_id(const correlation_id_t& correlation_id)
    {
        return g_tracing_backend_mock->get_parent_stack_id(correlation_id);
    }

    static void iterate_callback_tracing_kind_operation_args(
        const callback_tracing_record_t&     record,
        callback_tracing_operation_args_cb_t callback, std::int32_t /*max_deref*/,
        void*                                data)
    {
        g_tracing_backend_mock->iterate_args(record.kind, record.operation, callback,
                                             data);
    }
};

// Externals stand-in for on_tracing_api_enter/exit tests: inherits externals'
// boilerplate (agent_manager_t, kfd_sample_t, category name constants, ...) so it
// still satisfies policies::domain_service::externals, then hides exactly the members
// on_tracing_api_enter/exit touch, routing them through g_externals_mock (the same
// StrictMock<gmock_externals> instance on_configure() tests already use).
struct externals_with_tracing : externals
{
    // externals also overloads buffer_storage_store() for kfd_sample_t; pull that
    // overload back in since declaring the region_sample overload below would
    // otherwise hide it, breaking policies::domain_service::externals for
    // Externals::buffer_storage_store(std::move(kfd_sample_t{})).
    using externals::buffer_storage_store;

    static bool is_active() { return g_externals_mock->is_active(); }
    static bool get_use_timemory() { return g_externals_mock->get_use_timemory(); }

    template <typename CategoryT>
    static void tracing_push_timemory(CategoryT, std::string_view name)
    {
        g_externals_mock->tracing_push_timemory(name);
    }

    template <typename CategoryT>
    static void tracing_pop_timemory(CategoryT, std::string_view name)
    {
        g_externals_mock->tracing_pop_timemory(name);
    }

    static bool check_backtrace_operations(std::uint64_t kind, std::uint32_t operation)
    {
        return g_externals_mock->check_backtrace_operations(kind, operation);
    }

    static std::optional<int> get_backtrace_data(bool are_operations_available)
    {
        return g_externals_mock->get_backtrace_data(are_operations_available);
    }

    static void metadata_add_string(std::string_view value)
    {
        g_externals_mock->metadata_add_string(value);
    }

    static void metadata_add_thread_info(const thread_info_t& info)
    {
        g_externals_mock->metadata_add_thread_info(info.parent_process_id,
                                                   info.process_id, info.thread_id);
    }

    static std::int32_t get_pid() { return g_externals_mock->get_pid(); }
    static std::int32_t get_ppid() { return g_externals_mock->get_ppid(); }

    static void buffer_storage_store(region_sample&& sample)
    {
        g_externals_mock->region_sample_buffer_storage_store(
            sample.thread_id, std::string{ sample.name }, sample.correlation_id,
            sample.parent_stack_id, sample.start_timestamp, sample.end_timestamp,
            std::string{ sample.args_str }, std::string{ sample.category });
    }
};

// Drives a hip/hsa callback domain's k_domain.on_record() through one ENTER phase and
// one EXIT phase (against mock_sdk_with_tracing/externals_with_tracing), and asserts
// the given category name is exactly what reaches every SdkBackend/Externals call that
// surfaces it as a string: metadata_add_string and the category field of
// region_sample_buffer_storage_store. (tracing_push_timemory/tracing_pop_timemory are
// NOT checked against the category name here: their std::string_view argument is the
// per-call *operation* name from get_callback_tracing_names() -- always "operation" in
// this mock -- the category itself is conveyed only through their first argument's
// *type*, Category<Externals>::type, which externals_with_tracing's templated
// overload discards.) This is a regression guard for domain files: every domain wires
// its own per-file Category trait (e.g. hip::runtime_api_category,
// hsa::core_api_category), which is easy to copy-paste from the wrong sibling (an
// hsa::* file left wired to rocm_hip_api_category, or vice versa) without any other
// test catching it.
template <typename Domain>
inline void
expect_domain_uses_category(const Domain& domain, std::string_view expected_category_name)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::StrictMock;

    g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
    g_externals_mock       = std::make_unique<StrictMock<gmock_externals>>();

    mock_sdk_with_tracing::user_data_t user_data{};

    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(true));
    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp())
        .WillOnce(Return(std::uint64_t{ 1 }));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_push_timemory("operation"));

    auto enter_record  = mock_sdk_with_tracing::callback_tracing_record_t{};
    enter_record.phase = mock_sdk_with_tracing::CALLBACK_PHASE_ENTER;
    domain.on_record(enter_record, &user_data, nullptr);

    g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
    g_externals_mock       = std::make_unique<StrictMock<gmock_externals>>();

    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp())
        .WillOnce(Return(std::uint64_t{ 2 }));
    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_pop_timemory("operation"));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(expected_category_name));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_))
        .WillOnce(Return(std::uint64_t{ 0 }));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(
                    _, _, _, _, _, _, _, std::string{ expected_category_name }));

    auto exit_record  = mock_sdk_with_tracing::callback_tracing_record_t{};
    exit_record.phase = mock_sdk_with_tracing::CALLBACK_PHASE_EXIT;
    domain.on_record(exit_record, &user_data, nullptr);

    g_tracing_backend_mock.reset();
    g_externals_mock.reset();

    // Every domain wires tracing_callback_dispatcher with only OnEnter/OnExit (no
    // OnNone), so CALLBACK_PHASE_NONE must reach neither SdkBackend nor Externals.
    // StrictMock<...> with zero EXPECT_CALLs set fails the test if it does.
    g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
    g_externals_mock       = std::make_unique<StrictMock<gmock_externals>>();

    auto none_record  = mock_sdk_with_tracing::callback_tracing_record_t{};
    none_record.phase = mock_sdk_with_tracing::CALLBACK_PHASE_NONE;
    domain.on_record(none_record, &user_data, nullptr);

    g_tracing_backend_mock.reset();
    g_externals_mock.reset();
}

}  // namespace rocprofsys::domains::test_support
