// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/sdk/tracing-config.hpp"
#include "core/tests/mock_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace rocprofsys::rocprofiler_sdk::testing
{

namespace gtest = ::testing;

using ::rocprofsys::mock::rocprofiler_sdk::g_mock_wrapper;
using ::rocprofsys::mock::rocprofiler_sdk::gmock_wrapper;
using mock_backend = ::rocprofsys::mock::rocprofiler_sdk::wrapper;

// ─── Backend policy ───────────────────────────────────────────────────────────
//
// tagged_backend is a self-contained mock (no real SDK headers): it inherits
// mock_backend, whose get_version()/get_callback_tracing_names()/
// get_buffer_tracing_names() all forward to g_mock_wrapper.
//
// Each enumerator maps to a distinct tagged_backend<Tag> type (see
// cached_backend_methods below), so tests that verify first-call caching behaviour
// don't interfere with one another.
enum backend_tag : int
{
    version_fields                              = 1,
    version_formatted                           = 2,
    version_caching                             = 3,
    ops_throw_get_operations_callback           = 60,
    ops_throw_get_operations_buffer             = 61,
    ops_throw_get_backtrace_callback            = 62,
    ops_throw_get_backtrace_buffer              = 63,
    ops_throw_message                           = 64,
    domain_settings_test                        = 80,
    operation_settings_test                     = 81,
    callback_backtrace_operations               = 84,
    buffered_backtrace_operations               = 85,
    callback_operations                         = 86,
    buffered_operations                         = 87,
    buffered_domains_memory_copy                = 88,
    buffered_domains_aliases                    = 89,
    buffered_domains_kfd_events                 = 91,
    buffered_domains_kfd_individual             = 92,
    buffered_domains_allocation                 = 93,
    buffered_domains_unified_memory             = 94,
    buffered_domains_generic_lookup             = 95,
    buffered_domains_invalid                    = 96,
    buffered_domains_page_migration             = 97,
    callback_domains_aliases                    = 98,
    callback_domains_generic_lookup             = 99,
    callback_domains_implicit_flags             = 100,
    callback_domains_invalid                    = 101,
    callback_domains_rocshmem_hipfile_supported = 103,
    buffered_domains_callback_only_skipped      = 104,
    callback_domains_buffer_only_no_warning     = 105,
    buffered_domains_callback_only_no_warning   = 106,
    operation_settings_marker_core_api_skip     = 107,
    buffered_domains_kfd_below_boundary_version = 108,
    buffered_domains_kfd_at_boundary_version    = 109,
};

// tracing_config no longer caches get_version()/get_callback_tracing_names()/
// get_buffer_tracing_names() itself (that moved to backend<Wrapper> in
// production, see backends/rocprofiler_sdk/backend.hpp). To keep this mock's
// observable behavior consistent (each is hit at most once per distinct
// Tag/test), cached_backend_methods reproduces that same per-instantiation
// caching here — parameterized by Tag (not just Base) so each tagged_backend<Tag>
// gets its own static locals, matching backend<Wrapper>'s per-Wrapper-type cache.
template <int Tag, typename Base>
struct cached_backend_methods : Base
{
    // backend.hpp (production) names these with a `_t` suffix; mock_wrapper.hpp's
    // Base does not — adapt here rather than touching either established file.
    using callback_tracing_kind_t = Base::callback_tracing_kind;
    using buffer_tracing_kind_t   = Base::buffer_tracing_kind;

    static auto get_version(std::uint32_t* major, std::uint32_t* minor,
                            std::uint32_t* patch)
    {
        static const auto cached = [] {
            std::uint32_t maj    = 0;
            std::uint32_t min    = 0;
            std::uint32_t pat    = 0;
            auto          status = Base::get_version(&maj, &min, &pat);
            return std::tuple{ status, maj, min, pat };
        }();
        const auto& [status, maj, min, pat] = cached;
        *major                              = maj;
        *minor                              = min;
        *patch                              = pat;
        return status;
    }

    static const auto& get_callback_tracing_names()
    {
        static const auto names = Base::get_callback_tracing_names();
        return names;
    }

    static const auto& get_buffer_tracing_names()
    {
        static const auto names = Base::get_buffer_tracing_names();
        return names;
    }
};

template <int Tag>
struct tagged_backend : cached_backend_methods<Tag, mock_backend>
{};

// buffered_domains_page_migration simulates a pre-1.0 SDK: below the KFD gate
// (>= 10000), so get_buffered_domains() falls back to the legacy
// BUFFER_TRACING_PAGE_MIGRATION path instead of the granular KFD_* domains.
template <>
struct tagged_backend<buffered_domains_page_migration>
: cached_backend_methods<buffered_domains_page_migration, mock_backend>
{
    static constexpr std::uint32_t compile_time_version = 500;
};

// The KFD gate is s_sdk_version_1_2_2 (10202): SDK versions below 1.2.2 have a
// fatal bug parsing KFD events with undefined node IDs. 10201 (1.2.1) is one
// patch below the boundary, so KFD domains must stay unsupported.
template <>
struct tagged_backend<buffered_domains_kfd_below_boundary_version>
: cached_backend_methods<buffered_domains_kfd_below_boundary_version, mock_backend>
{
    static constexpr std::uint32_t compile_time_version = 10201U;
};

// 10202 (1.2.2) is the exact KFD gate boundary: KFD domains must be supported
// here, proving the gate is inclusive (">=", not ">").
template <>
struct tagged_backend<buffered_domains_kfd_at_boundary_version>
: cached_backend_methods<buffered_domains_kfd_at_boundary_version, mock_backend>
{
    static constexpr std::uint32_t compile_time_version = 10202U;
};

// ROCSHMEM_API (>= 1.3.4) and HIPFILE_API (>= 1.3.5) are gated by compile-time
// `if constexpr` checks in get_callback_domains(). Overriding compile_time_version
// to 10305 makes both blocks compile in, so the test below can exercise them.
template <>
struct tagged_backend<callback_domains_rocshmem_hipfile_supported>
: cached_backend_methods<callback_domains_rocshmem_hipfile_supported, mock_backend>
{
    static constexpr std::uint32_t compile_time_version = 10305U;
};

// ─── Shared fake name tables ───────────────────────────────────────────────────
//
// Every test that calls domain_choices()/operation_settings()/get_buffered_domains()/
// get_callback_domains() needs Wrapper::get_buffer_tracing_names()/
// get_callback_tracing_names() (both GMock methods on mock_backend) to return a
// populated table, since tracing_config validates/resolves domain names against it.

// Names mirror the real SDK's tracing-kind name tables, which are UPPER_SNAKE_CASE
// (e.g. "MEMORY_COPY", not "memory_copy"): operation_settings()'s per-domain
// operation-filter env var names are built directly from this raw name (not
// lowercased) — only the ROCPROFSYS_ROCM_DOMAINS choice list is lowercased.
mock_backend::buffer_name_info_t
make_buffer_name_info()
{
    auto table = mock_backend::buffer_name_info_t{};
    table.emplace(mock_backend::BUFFER_TRACING_HSA_CORE_API, "HSA_CORE_API");
    table.emplace(mock_backend::BUFFER_TRACING_HSA_AMD_EXT_API, "HSA_AMD_EXT_API");
    table.emplace(mock_backend::BUFFER_TRACING_HSA_IMAGE_EXT_API, "HSA_IMAGE_EXT_API");
    table.emplace(mock_backend::BUFFER_TRACING_HSA_FINALIZE_EXT_API,
                  "HSA_FINALIZE_EXT_API");
    table.emplace(mock_backend::BUFFER_TRACING_HIP_COMPILER_API, "HIP_COMPILER_API");
    table.emplace(mock_backend::BUFFER_TRACING_HIP_RUNTIME_API, "HIP_RUNTIME_API");
    table.emplace(mock_backend::BUFFER_TRACING_MARKER_CORE_API, "MARKER_CORE_API");
    table.emplace(mock_backend::BUFFER_TRACING_KERNEL_DISPATCH, "KERNEL_DISPATCH");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, "MEMORY_COPY");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, 0, "HOST_TO_DEVICE");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, 1, "DEVICE_TO_HOST");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, 2, "DEVICE_TO_DEVICE");
    table.emplace(mock_backend::BUFFER_TRACING_SCRATCH_MEMORY, "SCRATCH_MEMORY");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_ALLOCATION, "MEMORY_ALLOCATION");
    table.emplace(mock_backend::BUFFER_TRACING_PAGE_MIGRATION, "PAGE_MIGRATION");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_PAGE_FAULT, "KFD_PAGE_FAULT");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_PAGE_MIGRATE, "KFD_PAGE_MIGRATE");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_QUEUE, "KFD_QUEUE");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_QUEUE, "KFD_EVENT_QUEUE");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                  "KFD_EVENT_UNMAP_FROM_GPU");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                  "KFD_EVENT_DROPPED_EVENTS");
    return table;
}

mock_backend::callback_name_info_t
make_callback_name_info()
{
    auto table = mock_backend::callback_name_info_t{};
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_CORE_API, "HSA_CORE_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_AMD_EXT_API, "HSA_AMD_EXT_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_IMAGE_EXT_API, "HSA_IMAGE_EXT_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
                  "HSA_FINALIZE_EXT_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API, "HIP_RUNTIME_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_COMPILER_API, "HIP_COMPILER_API");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API, "MARKER_CORE_API");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API,
                  mock_backend::MARKER_CORE_API_ID_roctxMarkA, "roctxMarkA");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API,
                  mock_backend::MARKER_CORE_API_ID_roctxRangePushA, "roctxRangePushA");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API,
                  mock_backend::MARKER_CORE_API_ID_roctxRangePop, "roctxRangePop");
    table.emplace(mock_backend::CALLBACK_TRACING_CODE_OBJECT, "CODE_OBJECT");
    table.emplace(mock_backend::CALLBACK_TRACING_RCCL_API, "RCCL_API");
    table.emplace(mock_backend::CALLBACK_TRACING_OMPT, "OMPT");
    table.emplace(mock_backend::CALLBACK_TRACING_ROCSHMEM_API, "ROCSHMEM_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HIPFILE_API, "HIPFILE_API");
    return table;
}

// ─── Externals mock ───────────────────────────────────────────────────────────
//
// get_callback_domains() / get_buffered_domains() read through the Externals
// policy. Mocking it lets tests control the RCCLP/OMPT/unified-memory-profiling
// flags without ever touching rocprofsys::settings::instance(). ROCm domain
// validity is checked via sut::domain_choices() directly (no longer through
// Externals), so no settings registry needs to be mocked here.

class gmock_sdk_externals
{
public:
    MOCK_METHOD(bool, get_use_rcclp, ());
    MOCK_METHOD(bool, get_use_ompt, ());
    MOCK_METHOD(bool, get_use_unified_memory_profiling, ());
    MOCK_METHOD(std::string, get_rocm_domains, ());
    MOCK_METHOD(std::optional<std::string>, get_setting_value, (std::string_view));
    MOCK_METHOD(void, set_state, (std::uint32_t));
};

inline std::unique_ptr<gtest::StrictMock<gmock_sdk_externals>> g_mock_externals;

struct mock_sdk_externals
{
    static bool get_use_rcclp() { return g_mock_externals->get_use_rcclp(); }
    static bool get_use_ompt() { return g_mock_externals->get_use_ompt(); }
    static bool get_use_unified_memory_profiling()
    {
        return g_mock_externals->get_use_unified_memory_profiling();
    }
    static std::string get_rocm_domains() { return g_mock_externals->get_rocm_domains(); }
    static std::optional<std::string> get_setting_value(std::string_view s)
    {
        return g_mock_externals->get_setting_value(s);
    }

    // Mirrors ::rocprofsys::state::process's static interface (State/Finalized/set)
    // instead of a bespoke method, so the DI seam matches the real dependency shape.
    struct ProcessState
    {
        using State                      = std::uint32_t;
        constexpr static State Finalized = 3;
        static void            set(State state) { g_mock_externals->set_state(state); }
    };
};

// ─── Fixtures ─────────────────────────────────────────────────────────────────

class tracing_config_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock_wrapper = std::make_unique<gmock_wrapper>(); }
    void TearDown() override { g_mock_wrapper.reset(); }
};

// Fixture for functions that read through the Externals policy
// (get_callback_domains, get_buffered_domains).
class tracing_config_domains_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_wrapper   = std::make_unique<gmock_wrapper>();
        g_mock_externals = std::make_unique<gtest::StrictMock<gmock_sdk_externals>>();
    }
    void TearDown() override
    {
        g_mock_wrapper.reset();
        g_mock_externals.reset();
    }
};

// ─── get_version ─────────────────────────────────────────────────────────────

TEST_F(tracing_config_test, get_version_populates_major_minor_patch)
{
    using sut = tracing_config<tagged_backend<version_fields>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 2;
            *pat = 3;
            return 0;
        });

    auto ver = sut::get_version();
    EXPECT_EQ(ver.major, 1u);
    EXPECT_EQ(ver.minor, 2u);
    EXPECT_EQ(ver.patch, 3u);
}

TEST_F(tracing_config_test, get_version_formatted_equals_major_10000_minor_100_patch)
{
    using sut = tracing_config<tagged_backend<version_formatted>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 1;
            *pat = 0;
            return 0;
        });

    EXPECT_EQ(sut::get_version().formatted(), (1u * 10000u) + (1u * 100u) + 0u);
}

TEST_F(tracing_config_test, get_version_caches_result_calling_backend_exactly_once)
{
    using sut = tracing_config<tagged_backend<version_caching>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 2;
            *min = 0;
            *pat = 0;
            return 0;
        });

    // get_version() returns a fresh version_info by value each call (no cache of
    // its own in tracing_config) — the ".Times(1)" above is what proves
    // caching, via SdkBackend, not referential identity of the return value.
    auto ver1 = sut::get_version();
    auto ver2 = sut::get_version();
    auto ver3 = sut::get_version();

    EXPECT_EQ(ver1.formatted(), 20000u);
    EXPECT_EQ(ver2.formatted(), 20000u);
    EXPECT_EQ(ver3.formatted(), 20000u);
}

// ─── throw when domain has no operations ──────────────────────────────────────
//
// Design A removed the "operation_settings() must run first" ordering
// dependency entirely (see tracing-config.hpp's operation_env_names_for_kind):
// get_operations()/get_backtrace_operations() now derive the env-var names
// on demand from the tracing-name table instead of reading a map populated as
// a side effect elsewhere. The only remaining throw condition is a kind whose
// domain has no named operations — modeled here with an empty tracing-name
// table, which makes any queried kind resolve to a default (empty-operations)
// entry. Each test uses its own Tag: get_callback_tracing_names()/
// get_buffer_tracing_names() are cached per-Tag (cached_backend_methods), so a
// shared Tag across tests would let one test's empty table leak into another.

class tracing_config_throw_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_wrapper   = std::make_unique<gmock_wrapper>();
        g_mock_externals = std::make_unique<gtest::StrictMock<gmock_sdk_externals>>();
    }
    void TearDown() override
    {
        g_mock_wrapper.reset();
        g_mock_externals.reset();
    }
};

TEST_F(tracing_config_throw_test,
       get_operations_callback_throws_when_domain_has_no_operations)
{
    using backend_t = tagged_backend<ops_throw_get_operations_callback>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(mock_backend::callback_name_info_t{}));
    EXPECT_CALL(*g_mock_externals, set_state).Times(1);

    EXPECT_THROW(sut::get_operations(backend_t::CALLBACK_TRACING_HIP_RUNTIME_API),
                 std::runtime_error);
}

TEST_F(tracing_config_throw_test,
       get_operations_buffer_throws_when_domain_has_no_operations)
{
    using backend_t = tagged_backend<ops_throw_get_operations_buffer>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(mock_backend::buffer_name_info_t{}));
    EXPECT_CALL(*g_mock_externals, set_state).Times(1);

    EXPECT_THROW(sut::get_operations(backend_t::BUFFER_TRACING_KERNEL_DISPATCH),
                 std::runtime_error);
}

TEST_F(tracing_config_throw_test,
       get_backtrace_callback_throws_when_domain_has_no_operations)
{
    using backend_t = tagged_backend<ops_throw_get_backtrace_callback>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(mock_backend::callback_name_info_t{}));
    EXPECT_CALL(*g_mock_externals, set_state).Times(1);

    EXPECT_THROW(
        sut::get_backtrace_operations(backend_t::CALLBACK_TRACING_HIP_RUNTIME_API),
        std::runtime_error);
}

TEST_F(tracing_config_throw_test,
       get_backtrace_buffer_throws_when_domain_has_no_operations)
{
    using backend_t = tagged_backend<ops_throw_get_backtrace_buffer>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(mock_backend::buffer_name_info_t{}));
    EXPECT_CALL(*g_mock_externals, set_state).Times(1);

    EXPECT_THROW(sut::get_backtrace_operations(backend_t::BUFFER_TRACING_KERNEL_DISPATCH),
                 std::runtime_error);
}

TEST_F(tracing_config_throw_test, get_operations_error_message_contains_kind_value)
{
    using backend_t = tagged_backend<ops_throw_message>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(mock_backend::callback_name_info_t{}));
    EXPECT_CALL(*g_mock_externals, set_state).Times(1);

    try
    {
        sut::get_operations(backend_t::CALLBACK_TRACING_HIP_RUNTIME_API);
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& ex)
    {
        EXPECT_THAT(ex.what(), gtest::HasSubstr("callback tracing kind"));
    }
}

// ─── domain_choices / domain_defaults ─────────────────────────────────────────
//
// domain_choices()/domain_defaults()/operation_settings() are pure data-gathering
// functions: no Externals interaction, so these tests only need g_mock_wrapper
// (reusing tracing_config_test). Each test uses its own Tag since
// operation_settings() populates a function-local-static-scoped-per-(Wrapper,Externals)
// lookup map, and a shared Tag across tests would make later tests see already-populated
// state.

TEST_F(tracing_config_test, domain_choices_returns_expected_choices)
{
    using sut = tracing_config<tagged_backend<domain_settings_test>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    const auto choices = sut::get_domain_choices();

    EXPECT_THAT(choices,
                gtest::IsSupersetOf({ "hip_api", "hsa_api", "marker_api", "roctx" }));
    EXPECT_THAT(choices, gtest::Not(gtest::Contains(std::string{ "code_object" })));
    EXPECT_THAT(choices, gtest::Not(gtest::Contains(std::string{ "none" })));
}

TEST_F(tracing_config_test, domain_defaults_returns_expected_defaults)
{
    using sut = tracing_config<tagged_backend<domain_settings_test>, mock_sdk_externals>;

    EXPECT_EQ(sut::get_domain_defaults(),
              "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory");
}

// ─── operation_settings ───────────────────────────────────────────────────────

TEST_F(tracing_config_test, operation_settings_registers_marker_api_domain_alias)
{
    using sut =
        tracing_config<tagged_backend<operation_settings_test>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    const auto specs = sut::get_operation_settings();

    auto itr = std::ranges::find_if(specs, [](const auto& spec) {
        return spec.env_names.operations_include_env_name ==
               "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS";
    });
    ASSERT_NE(itr, specs.end());

    EXPECT_EQ(itr->env_names.operations_exclude_env_name,
              "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_EXCLUDE");
    EXPECT_EQ(itr->env_names.operations_annotate_backtrace_env_name,
              "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_ANNOTATE_BACKTRACE");
    EXPECT_THAT(itr->operation_choices, gtest::Not(gtest::IsEmpty()));
}

// The buffered marker_core_api entry keeps its raw SDK name (MARKER_CORE_API);
// only the callback path renames it to MARKER_API. Without skipping
// marker_core_api in get_domains_to_skip_for_operation_options(), the buffered
// entry would register a second, unused ROCPROFSYS_ROCM_MARKER_CORE_API_OPERATIONS
// setting alongside the correct MARKER_API one.
TEST_F(tracing_config_test, operation_settings_does_not_register_marker_core_api_domain)
{
    using sut = tracing_config<tagged_backend<operation_settings_marker_core_api_skip>,
                               mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    const auto specs = sut::get_operation_settings();

    const auto marker_api_count = std::ranges::count_if(specs, [](const auto& spec) {
        return spec.env_names.operations_include_env_name ==
               "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS";
    });
    EXPECT_EQ(marker_api_count, 1);

    EXPECT_TRUE(std::ranges::none_of(specs, [](const auto& spec) {
        return spec.env_names.operations_include_env_name ==
               "ROCPROFSYS_ROCM_MARKER_CORE_API_OPERATIONS";
    }));
}

// ─── get_callback_domains ─────────────────────────────────────────────────────
//
// get_callback_domains() calls Wrapper::get_callback_tracing_names() directly and
// also calls domain_choices() (to validate ROCm domain names), which calls both
// get_buffer_tracing_names() and get_callback_tracing_names() again;
// cached_backend_methods memoizes both per-Tag, so each mock still only fires once
// per test despite the two call sites.

TEST_F(tracing_config_domains_test, get_callback_domains_aliases_expand_to_exact_domains)
{
    using backend_t = tagged_backend<callback_domains_aliases>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "hsa_api,hip_api,marker_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_callback_domains(),
        gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_HSA_CORE_API,
                                    backend_t::CALLBACK_TRACING_HSA_AMD_EXT_API,
                                    backend_t::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                                    backend_t::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
                                    backend_t::CALLBACK_TRACING_HIP_RUNTIME_API,
                                    backend_t::CALLBACK_TRACING_HIP_COMPILER_API,
                                    backend_t::CALLBACK_TRACING_MARKER_CORE_API));
}

TEST_F(tracing_config_domains_test,
       get_callback_domains_supported_callback_info_name_returns_domain)
{
    using backend_t = tagged_backend<callback_domains_generic_lookup>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "rccl_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_callback_domains(),
                gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_RCCL_API));
}

TEST_F(tracing_config_domains_test,
       get_callback_domains_rccl_and_ompt_flags_return_both_domains)
{
    using backend_t = tagged_backend<callback_domains_implicit_flags>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(true));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(true));

    EXPECT_THAT(sut::get_callback_domains(),
                gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_RCCL_API,
                                            backend_t::CALLBACK_TRACING_OMPT));
}

TEST_F(tracing_config_domains_test, get_callback_domains_invalid_domain)
{
    using backend_t = tagged_backend<callback_domains_invalid>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "invalid_domain" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));

    try
    {
        static_cast<void>(sut::get_callback_domains());
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(),
                     "unsupported ROCPROFSYS_ROCM_DOMAINS value: invalid_domain");
    }
}

// kernel_dispatch is a buffered-only domain (present in make_buffer_name_info(),
// absent from make_callback_name_info()). It must be silently skipped without
// logging the "not supported by the loaded rocprofiler-sdk headers" warning,
// which would otherwise misreport a valid, buffered-only domain as unsupported.
TEST_F(tracing_config_domains_test,
       get_callback_domains_skips_buffer_only_domain_without_warning)
{
    using backend_t = tagged_backend<callback_domains_buffer_only_no_warning>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "kernel_dispatch,hip_runtime_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));

    gtest::internal::CaptureStdout();
    gtest::internal::CaptureStderr();
    const auto domains = sut::get_callback_domains();
    const auto output =
        gtest::internal::GetCapturedStdout() + gtest::internal::GetCapturedStderr();

    EXPECT_THAT(domains,
                gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_HIP_RUNTIME_API));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("not supported")));
}

TEST_F(tracing_config_domains_test,
       get_callback_domains_rocshmem_and_hipfile_supported_when_version_at_or_above_gate)
{
    using backend_t = tagged_backend<callback_domains_rocshmem_hipfile_supported>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "rocshmem_api,hipfile_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_callback_domains(),
                gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_ROCSHMEM_API,
                                            backend_t::CALLBACK_TRACING_HIPFILE_API));
}

// ─── get_buffered_domains ─────────────────────────────────────────────────────
//
// get_buffered_domains() calls Wrapper::get_buffer_tracing_names() directly and
// also calls domain_choices() (to validate ROCm domain names), which calls both
// get_buffer_tracing_names() and get_callback_tracing_names() again;
// cached_backend_methods memoizes both per-Tag, so each mock still only fires once
// per test despite the two call sites.

TEST_F(tracing_config_domains_test, get_buffered_domains_memory_copy_returns_memory_copy)
{
    using backend_t = tagged_backend<buffered_domains_memory_copy>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "memory_copy" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_MEMORY_COPY));
}

TEST_F(tracing_config_domains_test, get_buffered_domains_aliases_expand_to_exact_domains)
{
    using backend_t = tagged_backend<buffered_domains_aliases>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "hsa_api,hip_api,marker_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_HSA_CORE_API,
                                    backend_t::BUFFER_TRACING_HSA_AMD_EXT_API,
                                    backend_t::BUFFER_TRACING_HSA_IMAGE_EXT_API,
                                    backend_t::BUFFER_TRACING_HSA_FINALIZE_EXT_API,
                                    backend_t::BUFFER_TRACING_HIP_COMPILER_API,
                                    backend_t::BUFFER_TRACING_HIP_RUNTIME_API,
                                    backend_t::BUFFER_TRACING_MARKER_CORE_API));
}

TEST_F(tracing_config_domains_test,
       get_buffered_domains_kfd_events_returns_all_kfd_domains)
{
    using backend_t = tagged_backend<buffered_domains_kfd_events>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "kfd_events" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                    backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                    backend_t::BUFFER_TRACING_KFD_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS));
}

TEST_F(tracing_config_domains_test,
       get_buffered_domains_individual_kfd_names_return_all_kfd_domains)
{
    using backend_t = tagged_backend<buffered_domains_kfd_individual>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(
            std::string{ "kfd_page_fault, kfd_page_migrate, kfd_queue, kfd_event_queue, "
                         "kfd_event_unmap_from_gpu, kfd_event_dropped_events" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                    backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                    backend_t::BUFFER_TRACING_KFD_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS));
}

TEST_F(tracing_config_domains_test,
       get_buffered_domains_memory_allocation_returns_memory_allocation)
{
    using backend_t = tagged_backend<buffered_domains_allocation>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "memory_allocation" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_MEMORY_ALLOCATION));
}

TEST_F(tracing_config_domains_test,
       get_buffered_domains_unified_memory_enables_page_fault_and_migrate)
{
    using backend_t = tagged_backend<buffered_domains_unified_memory>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(true));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                            backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE));
}

TEST_F(tracing_config_domains_test,
       get_buffered_domains_supported_buffer_info_name_returns_domain)
{
    using backend_t = tagged_backend<buffered_domains_generic_lookup>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "scratch_memory" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_SCRATCH_MEMORY));
}

// rccl_api is a callback-only domain (present in make_callback_name_info(), absent
// from make_buffer_name_info()). domain_choices() accepts it as valid, but
// get_buffered_domain_map() has no entry for it, so it must be silently skipped
// rather than throwing std::out_of_range.
TEST_F(tracing_config_domains_test, get_buffered_domains_skips_callback_only_domain)
{
    using backend_t = tagged_backend<buffered_domains_callback_only_skipped>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "memory_copy,rccl_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_MEMORY_COPY));
}

// rccl_api is a callback-only domain. It must be silently skipped without
// logging the "has no buffered-tracing equivalent" warning, which would
// otherwise misreport a valid, callback-only domain as unsupported.
TEST_F(tracing_config_domains_test,
       get_buffered_domains_skips_callback_only_domain_without_warning)
{
    using backend_t = tagged_backend<buffered_domains_callback_only_no_warning>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "memory_copy,rccl_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    gtest::internal::CaptureStdout();
    gtest::internal::CaptureStderr();
    const auto domains = sut::get_buffered_domains();
    const auto output =
        gtest::internal::GetCapturedStdout() + gtest::internal::GetCapturedStderr();

    EXPECT_THAT(domains,
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_MEMORY_COPY));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("has no buffered-tracing "
                                                            "equivalent")));
}

TEST_F(tracing_config_domains_test, get_buffered_domains_invalid_domain_throws)
{
    using backend_t = tagged_backend<buffered_domains_invalid>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "invalid_domain" }));

    try
    {
        static_cast<void>(sut::get_buffered_domains());
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(),
                     "unsupported ROCPROFSYS_ROCM_DOMAINS value: invalid_domain");
    }
}

TEST_F(tracing_config_domains_test,
       get_buffered_domains_legacy_page_migration_returns_page_migration)
{
    using sut = tracing_config<tagged_backend<buffered_domains_page_migration>,
                               mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "page_migration" }));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(
                    tagged_backend<
                        buffered_domains_page_migration>::BUFFER_TRACING_PAGE_MIGRATION));
}

// SDK 1.2.1 (10201) is one patch below the KFD gate (s_sdk_version_1_2_2 = 10202):
// kfd_page_fault is a known, valid domain name (present in the raw name table),
// but get_supported_buffer_domains() excludes it below the gate, so it must
// resolve to nothing rather than to BUFFER_TRACING_KFD_PAGE_FAULT.
TEST_F(tracing_config_domains_test,
       get_buffered_domains_kfd_domain_unsupported_below_version_gate)
{
    using backend_t = tagged_backend<buffered_domains_kfd_below_boundary_version>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "kfd_page_fault" }));

    EXPECT_THAT(sut::get_buffered_domains(), gtest::IsEmpty());
}

// SDK 1.2.2 (10202) is exactly the KFD gate boundary: the "kfd_events" alias
// becomes a valid domain choice and all KFD_* kinds become supported, proving
// the gate is inclusive.
TEST_F(tracing_config_domains_test,
       get_buffered_domains_kfd_events_supported_at_version_gate)
{
    using backend_t = tagged_backend<buffered_domains_kfd_at_boundary_version>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "kfd_events" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                    backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                    backend_t::BUFFER_TRACING_KFD_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS));
}

// ─── get_operations ───────────────────────────────────────────────────────────

TEST_F(tracing_config_domains_test,
       get_operations_callback_applies_include_and_exclude_settings)
{
    using backend_t = tagged_backend<callback_operations>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    sut::get_operation_settings();

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq("ROCPROFSYS_ROCM_MARKER_API_OPERATIONS")))
        .Times(1)
        .WillOnce(
            gtest::Return(std::string{ "roctxMarkA|roctxRangePushA|roctxRangePop" }));
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::Eq(
                                       "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_EXCLUDE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "roctxRangePushA" }));

    EXPECT_THAT(
        sut::get_operations(backend_t::CALLBACK_TRACING_MARKER_CORE_API),
        gtest::ElementsAre(
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxMarkA),
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxRangePop)));
}

TEST_F(tracing_config_domains_test,
       get_operations_buffered_applies_include_and_exclude_settings)
{
    using backend_t = tagged_backend<buffered_operations>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    // Memory-copy operation ids: match the order populated by make_buffer_name_info()
    // ("HOST_TO_DEVICE"=0, "DEVICE_TO_HOST"=1, "DEVICE_TO_DEVICE"=2).
    constexpr std::int32_t k_memory_copy_host_to_device = 0;
    constexpr std::int32_t k_memory_copy_device_to_host = 1;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    sut::get_operation_settings();

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq("ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS")))
        .Times(1)
        .WillOnce(gtest::Return(
            std::string{ "HOST_TO_DEVICE|DEVICE_TO_HOST|DEVICE_TO_DEVICE" }));
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::Eq(
                                       "ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS_EXCLUDE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "DEVICE_TO_DEVICE" }));

    EXPECT_THAT(
        sut::get_operations(backend_t::BUFFER_TRACING_MEMORY_COPY),
        gtest::ElementsAre(k_memory_copy_host_to_device, k_memory_copy_device_to_host));
}

// ─── get_backtrace_operations ─────────────────────────────────────────────────

TEST_F(tracing_config_domains_test,
       get_backtrace_operations_callback_returns_operations_matching_setting)
{
    using backend_t = tagged_backend<callback_backtrace_operations>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    sut::get_operation_settings();

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq(
                    "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_ANNOTATE_BACKTRACE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "roctxMarkA|roctxRangePop" }));

    EXPECT_THAT(
        sut::get_backtrace_operations(backend_t::CALLBACK_TRACING_MARKER_CORE_API),
        gtest::UnorderedElementsAre(
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxMarkA),
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxRangePop)));
}

TEST_F(tracing_config_domains_test,
       get_backtrace_operations_buffered_returns_operations_matching_setting)
{
    using backend_t = tagged_backend<buffered_backtrace_operations>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    // Memory-copy operation ids: match the order populated by make_buffer_name_info()
    // ("HOST_TO_DEVICE"=0, "DEVICE_TO_HOST"=1, "DEVICE_TO_DEVICE"=2).
    constexpr std::int32_t k_memory_copy_host_to_device = 0;
    constexpr std::int32_t k_memory_copy_device_to_host = 1;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    sut::get_operation_settings();

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq(
                    "ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS_ANNOTATE_BACKTRACE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "HOST_TO_DEVICE|DEVICE_TO_HOST" }));

    EXPECT_THAT(sut::get_backtrace_operations(backend_t::BUFFER_TRACING_MEMORY_COPY),
                gtest::UnorderedElementsAre(k_memory_copy_host_to_device,
                                            k_memory_copy_device_to_host));
}

}  // namespace rocprofsys::rocprofiler_sdk::testing
