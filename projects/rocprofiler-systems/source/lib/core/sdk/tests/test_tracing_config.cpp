// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/sdk/tracing-config.hpp"
#include "core/tests/mock_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

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
    ops_throw_message                           = 64,
    domain_settings_test                        = 80,
    operation_settings_test                     = 81,
    buffered_domains_memory_copy                = 88,
    buffered_domains_kfd_events                 = 91,
    buffered_domains_kfd_individual             = 92,
    buffered_domains_allocation                 = 93,
    buffered_domains_unified_memory             = 94,
    buffered_domains_generic_lookup             = 95,
    callback_domains_aliases                    = 98,
    callback_domains_generic_lookup             = 99,
    callback_domains_implicit_flags             = 100,
    callback_domains_invalid                    = 101,
    callback_domains_rocshmem_hipfile_supported = 103,
    callback_domains_buffer_only_no_warning     = 105,
    operation_settings_marker_core_api_skip     = 107,
    buffered_domains_kfd_below_boundary_version = 108,
    buffered_domains_kfd_at_boundary_version    = 109,
    domain_requests_buffered                    = 110,
    domain_requests_callback_filtered           = 111,
    domain_requests_callback_unfiltered         = 112,
    domain_requests_marker_skipped              = 113,
    domain_requests_legacy_page_migration       = 114,
    domain_requests_before_hip_stream           = 115,
    domain_requests_marker_skipped_aliases      = 116,
    domain_requests_buffered_operations         = 117,
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
            auto const    status = Base::get_version(&maj, &min, &pat);
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

// 500 (0.5.0) predates both the 0.6.0 domains and the 1.0.0 KFD split, so
// page_migration is the only buffered domain the SDK offers beyond the basics.
template <>
struct tagged_backend<domain_requests_legacy_page_migration>
: cached_backend_methods<domain_requests_legacy_page_migration, mock_backend>
{
    static constexpr std::uint32_t compile_time_version = 500U;
};

// 600 (0.6.0) is one minor below the 0.7.0 gate for the derived hip_stream domain.
template <>
struct tagged_backend<domain_requests_before_hip_stream>
: cached_backend_methods<domain_requests_before_hip_stream, mock_backend>
{
    static constexpr std::uint32_t compile_time_version = 600U;
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
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
                  "KFD_EVENT_PAGE_FAULT");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
                  "KFD_EVENT_PAGE_MIGRATE");
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

// ─── get_domains helpers ──────────────────────────────────────────────────────

struct domain_flags
{
    bool rcclp          = false;
    bool ompt           = false;
    bool unified_memory = false;
};

// Every get_domains() test goes through the same Externals reads; the flags a test
// does not care about stay off and the operation settings default to "unset".
// Expectations declared after this call take precedence over these defaults.
void
expect_domains_environment(const std::string& domains, const domain_flags& flags = {})
{
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillRepeatedly(gtest::Return(domains));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp)
        .WillRepeatedly(gtest::Return(flags.rcclp));
    EXPECT_CALL(*g_mock_externals, get_use_ompt)
        .WillRepeatedly(gtest::Return(flags.ompt));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillRepeatedly(gtest::Return(flags.unified_memory));
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::_))
        .WillRepeatedly(gtest::Return(std::string{}));
}

void
expect_name_tables(const mock_backend::callback_name_info_t& callback_names)
{
    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(callback_names));
}

template <typename Requests>
std::vector<std::string>
domain_names(const Requests& requests)
{
    auto names = std::vector<std::string>{};
    for(const auto& request : requests)
    {
        names.push_back(request.name);
    }
    return names;
}

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

    auto const ver = sut::get_version();
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
    auto const ver1 = sut::get_version();
    auto const ver2 = sut::get_version();
    auto const ver3 = sut::get_version();

    EXPECT_EQ(ver1.formatted(), 20000u);
    EXPECT_EQ(ver2.formatted(), 20000u);
    EXPECT_EQ(ver3.formatted(), 20000u);
}

// ─── get_domains throws ───────────────────────────────────────────────────────
//
// The only throw condition left once the domain list is valid is an operation
// filter that was never registered. Each test uses its own Tag: the tracing-name
// tables are cached per-Tag (cached_backend_methods), so a shared Tag would let one
// test's table leak into another.

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
       get_domains_throws_when_operation_setting_is_unregistered)
{
    using backend_t = tagged_backend<ops_throw_message>;
    using sut       = tracing_config<backend_t, mock_sdk_externals>;

    auto callback_names = mock_backend::callback_name_info_t{};
    callback_names.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API,
                           "HIP_RUNTIME_API");
    callback_names.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API, 0,
                           "hipMalloc");
    expect_name_tables(callback_names);
    expect_domains_environment("hip_runtime_api");
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::_))
        .WillRepeatedly(gtest::Return(std::nullopt));
    EXPECT_CALL(*g_mock_externals, set_state).Times(1);

    try
    {
        static_cast<void>(sut::get_domains());
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& ex)
    {
        EXPECT_THAT(ex.what(), gtest::HasSubstr("no registered setting"));
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

    auto const itr = std::ranges::find_if(specs, [](const auto& spec) {
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

// ─── get_domains ──────────────────────────────────────────────────────────────

TEST_F(tracing_config_domains_test, get_domains_expands_aliases_to_their_domains)
{
    using sut =
        tracing_config<tagged_backend<callback_domains_aliases>, mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("hsa_api,hip_api");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("hsa_core_api", "hsa_amd_ext_api",
                                            "hsa_image_ext_api", "hsa_finalize_ext_api",
                                            "hip_runtime_api", "hip_compiler_api"));
}

TEST_F(tracing_config_domains_test, get_domains_skips_marker_api_and_roctx)
{
    using sut = tracing_config<tagged_backend<domain_requests_marker_skipped_aliases>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("marker_api,roctx");

    EXPECT_THAT(domain_names(sut::get_domains()), gtest::IsEmpty());
}

TEST_F(tracing_config_domains_test, get_domains_returns_domain_named_by_the_sdk)
{
    using sut = tracing_config<tagged_backend<callback_domains_generic_lookup>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("rccl_api");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("rccl_api"));
}

TEST_F(tracing_config_domains_test, get_domains_adds_rccl_and_ompt_when_enabled)
{
    using sut = tracing_config<tagged_backend<callback_domains_implicit_flags>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("", domain_flags{ .rcclp = true, .ompt = true });

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("rccl_api", "ompt"));
}

TEST_F(tracing_config_domains_test,
       get_domains_returns_buffered_and_callback_domains_together)
{
    using sut = tracing_config<tagged_backend<callback_domains_buffer_only_no_warning>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("kernel_dispatch,hip_runtime_api");

    EXPECT_THAT(
        domain_names(sut::get_domains()),
        gtest::UnorderedElementsAre("kernel_dispatch", "hip_stream", "hip_runtime_api"));
}

TEST_F(tracing_config_domains_test, get_domains_returns_rocshmem_and_hipfile)
{
    using sut =
        tracing_config<tagged_backend<callback_domains_rocshmem_hipfile_supported>,
                       mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("rocshmem_api,hipfile_api");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("rocshmem_api", "hipfile_api"));
}

TEST_F(tracing_config_domains_test, get_domains_adds_derived_hip_stream)
{
    using sut =
        tracing_config<tagged_backend<buffered_domains_memory_copy>, mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("memory_copy");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("memory_copy", "hip_stream"));
}

TEST_F(tracing_config_domains_test, get_domains_omits_hip_stream_before_sdk_0_7_0)
{
    using sut = tracing_config<tagged_backend<domain_requests_before_hip_stream>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("kernel_dispatch");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("kernel_dispatch"));
}

TEST_F(tracing_config_domains_test, get_domains_expands_kfd_events)
{
    using sut =
        tracing_config<tagged_backend<buffered_domains_kfd_events>, mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("kfd_events");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre(
                    "kfd_page_fault", "kfd_page_migrate", "kfd_queue",
                    "kfd_event_page_fault", "kfd_event_page_migrate", "kfd_event_queue",
                    "kfd_event_unmap_from_gpu", "kfd_event_dropped_events"));
}

TEST_F(tracing_config_domains_test, get_domains_returns_individual_kfd_domains)
{
    using sut = tracing_config<tagged_backend<buffered_domains_kfd_individual>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("kfd_page_fault,kfd_queue");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("kfd_page_fault", "kfd_queue"));
}

TEST_F(tracing_config_domains_test, get_domains_returns_memory_allocation)
{
    using sut =
        tracing_config<tagged_backend<buffered_domains_allocation>, mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("memory_allocation");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("memory_allocation"));
}

TEST_F(tracing_config_domains_test, get_domains_returns_scratch_memory)
{
    using sut = tracing_config<tagged_backend<buffered_domains_generic_lookup>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("scratch_memory");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("scratch_memory"));
}

TEST_F(tracing_config_domains_test,
       get_domains_unified_memory_enables_page_fault_and_migrate)
{
    using sut = tracing_config<tagged_backend<buffered_domains_unified_memory>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("", domain_flags{ .unified_memory = true });

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("kfd_page_fault", "kfd_page_migrate"));
}

TEST_F(tracing_config_domains_test, get_domains_skips_legacy_page_migration)
{
    using sut = tracing_config<tagged_backend<domain_requests_legacy_page_migration>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("page_migration");

    EXPECT_THAT(domain_names(sut::get_domains()), gtest::IsEmpty());
}

TEST_F(tracing_config_domains_test, get_domains_expands_kfd_events_at_sdk_1_2_2)
{
    using sut = tracing_config<tagged_backend<buffered_domains_kfd_at_boundary_version>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("kfd_events");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre(
                    "kfd_page_fault", "kfd_page_migrate", "kfd_queue",
                    "kfd_event_page_fault", "kfd_event_page_migrate", "kfd_event_queue",
                    "kfd_event_unmap_from_gpu", "kfd_event_dropped_events"));
}

TEST_F(tracing_config_domains_test, get_domains_deduplicates_domains_named_twice)
{
    using sut =
        tracing_config<tagged_backend<domain_requests_buffered>, mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("hip_api,hip_runtime_api");

    EXPECT_THAT(domain_names(sut::get_domains()),
                gtest::UnorderedElementsAre("hip_runtime_api", "hip_compiler_api"));
}

TEST_F(tracing_config_domains_test, get_domains_rejects_kfd_events_before_sdk_1_2_2)
{
    using sut =
        tracing_config<tagged_backend<buffered_domains_kfd_below_boundary_version>,
                       mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("kfd_events");

    EXPECT_THROW(static_cast<void>(sut::get_domains()), std::runtime_error);
}

TEST_F(tracing_config_domains_test, get_domains_rejects_unknown_domain)
{
    using sut =
        tracing_config<tagged_backend<callback_domains_invalid>, mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("invalid_domain");

    try
    {
        static_cast<void>(sut::get_domains());
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(),
                     "unsupported ROCPROFSYS_ROCM_DOMAINS value: invalid_domain");
    }
}

mock_backend::callback_name_info_t
make_hip_runtime_name_info()
{
    auto table = mock_backend::callback_name_info_t{};
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API, "HIP_RUNTIME_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API, 0, "hipMalloc");
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API, 1, "hipFree");
    return table;
}

TEST_F(tracing_config_domains_test,
       get_domains_carries_filtered_operations_and_backtrace_operations)
{
    using sut = tracing_config<tagged_backend<domain_requests_callback_filtered>,
                               mock_sdk_externals>;

    expect_name_tables(make_hip_runtime_name_info());
    expect_domains_environment("hip_runtime_api");
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::Eq(
                                       "ROCPROFSYS_ROCM_HIP_RUNTIME_API_OPERATIONS")))
        .WillOnce(gtest::Return(std::string{ "hipMalloc" }));
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq(
                    "ROCPROFSYS_ROCM_HIP_RUNTIME_API_OPERATIONS_ANNOTATE_BACKTRACE")))
        .WillOnce(gtest::Return(std::string{ "hipFree" }));

    const auto requests = sut::get_domains();

    ASSERT_EQ(requests.size(), 1U);
    const auto& request = requests.front();
    EXPECT_EQ(request.name, "hip_runtime_api");
    EXPECT_THAT(request.operations, gtest::Optional(gtest::ElementsAre("hipMalloc")));
    EXPECT_THAT(request.backtrace_operations, gtest::UnorderedElementsAre(1));
}

TEST_F(tracing_config_domains_test,
       get_domains_excludes_operations_matching_exclude_setting)
{
    using sut = tracing_config<tagged_backend<domain_requests_callback_unfiltered>,
                               mock_sdk_externals>;

    expect_name_tables(make_hip_runtime_name_info());
    expect_domains_environment("hip_runtime_api");
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(
                    gtest::Eq("ROCPROFSYS_ROCM_HIP_RUNTIME_API_OPERATIONS_EXCLUDE")))
        .WillOnce(gtest::Return(std::string{ "hipFree" }));

    const auto requests = sut::get_domains();

    ASSERT_EQ(requests.size(), 1U);
    EXPECT_THAT(requests.front().operations,
                gtest::Optional(gtest::ElementsAre("hipMalloc")));
}

TEST_F(tracing_config_domains_test, get_domains_requests_all_operations_without_filters)
{
    using sut = tracing_config<tagged_backend<domain_requests_marker_skipped>,
                               mock_sdk_externals>;

    expect_name_tables(make_hip_runtime_name_info());
    expect_domains_environment("hip_runtime_api");

    const auto requests = sut::get_domains();

    ASSERT_EQ(requests.size(), 1U);
    EXPECT_EQ(requests.front().operations, std::nullopt);
    EXPECT_THAT(requests.front().backtrace_operations, gtest::IsEmpty());
}

TEST_F(tracing_config_domains_test,
       get_domains_applies_operation_filter_to_buffered_domains)
{
    using sut = tracing_config<tagged_backend<domain_requests_buffered_operations>,
                               mock_sdk_externals>;

    expect_name_tables(make_callback_name_info());
    expect_domains_environment("scratch_memory,memory_copy");
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq("ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS")))
        .WillOnce(gtest::Return(std::string{ "HOST_TO_DEVICE" }));

    const auto requests = sut::get_domains();

    const auto memory_copy =
        std::ranges::find(requests, "memory_copy", &sut::domain_request::name);
    ASSERT_NE(memory_copy, requests.end());
    EXPECT_THAT(memory_copy->operations,
                gtest::Optional(gtest::ElementsAre("HOST_TO_DEVICE")));
}

}  // namespace rocprofsys::rocprofiler_sdk::testing
