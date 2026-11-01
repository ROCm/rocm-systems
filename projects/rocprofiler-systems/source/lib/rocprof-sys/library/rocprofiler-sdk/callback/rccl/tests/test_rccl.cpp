// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rccl/rccl.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <cstdint>
#include <optional>
#include <string>

namespace rocprofsys::domains::callback
{
namespace
{

using test_support::expect_domain_uses_category;
using test_support::externals;
using test_support::externals_with_tracing;
using test_support::mock_sdk;
using test_support::mock_sdk_with_tracing;
using test_support::pmc_info_data_t;
using test_support::track_data_t;

using ::testing::AllOf;
using ::testing::DoubleEq;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Return;
using ::testing::StrictMock;

using sdk      = mock_sdk_with_tracing;
using ext      = externals_with_tracing;
using resolver = rccl::device_resolver<sdk>;

constexpr std::uint64_t k_timestamp = 555;

// Stand-in for ncclCommCuDevice: the "communicator" is an int holding the device id.
sdk::nccl_result_t
fake_comm_cu_device(sdk::nccl_comm_t comm, int* device)
{
    *device = *static_cast<int*>(comm);
    return sdk::NCCL_SUCCESS;
}

constexpr sdk::nccl_result_t k_nccl_failure = 1;

sdk::nccl_result_t
failing_comm_cu_device(sdk::nccl_comm_t, int*)
{
    return k_nccl_failure;
}

void*
as_symbol(sdk::nccl_result_t (*fn)(sdk::nccl_comm_t, int*))
{
    return reinterpret_cast<void*>(fn);  // NOLINT: dlsym returns object pointers
}

class rccl_callbacks_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_support::g_externals_mock =
            std::make_unique<StrictMock<test_support::gmock_externals>>();
        test_support::g_metadata_registry_mock =
            std::make_unique<StrictMock<test_support::gmock_metadata_registry>>();
        test_support::g_buffer_storage_mock =
            std::make_unique<StrictMock<test_support::gmock_buffer_storage>>();
    }

    void TearDown() override
    {
        // Static resolver state must not leak into other tests.
        ext::dlsym_result   = nullptr;
        ext::dlerror_result = nullptr;
        resolver::configure_comm_cu_device_function<ext>();

        test_support::g_externals_mock.reset();
        test_support::g_metadata_registry_mock.reset();
        test_support::g_buffer_storage_mock.reset();
    }

    void configure_with(sdk::nccl_result_t (*fn)(sdk::nccl_comm_t, int*))
    {
        ext::dlsym_result = as_symbol(fn);
        resolver::configure_comm_cu_device_function<ext>();
    }

    // Drives on_rccl_exit with is_active()==false so only the RCCL-specific part runs.
    void exit_with(sdk::rccl_api_id_t op, sdk::rccl_api_data* payload)
    {
        EXPECT_CALL(*test_support::g_externals_mock, is_active()).WillOnce(Return(false));
        sdk::user_data_t               user_data{};
        sdk::callback_tracing_record_t record{};
        record.kind      = sdk::CALLBACK_TRACING_RCCL_API;
        record.operation = static_cast<std::uint32_t>(op);
        record.payload   = payload;
        rccl::on_rccl_exit<sdk, ext>(record, &user_data, nullptr, k_timestamp);
    }
};

auto
pmc_named(const std::string& name)
{
    return AllOf(Field(&pmc_info_data_t::name, Eq(name)),
                 Field(&pmc_info_data_t::type, Eq(ext::k_agent_type_gpu)),
                 Field(&pmc_info_data_t::units, Eq(std::string{ "bytes" })),
                 Field(&pmc_info_data_t::value_type,
                       Eq(std::string{ ext::k_pmc_value_type_absolute })));
}

}  // namespace

TEST(rccl_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = rccl::k_rccl_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "rccl_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_RCCL_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_FALSE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group, std::nullopt);
}

// Regression guard: rccl must push/pop timemory and stamp buffer-storage records with
// "rocm_rccl_api", not with the "comm_data" category used for the transfer-bytes PMC.
TEST(rccl_test, uses_rocm_rccl_api_category)
{
    constexpr const auto& k_domain =
        rccl::k_rccl_api<mock_sdk_with_tracing, externals_with_tracing>;

    expect_domain_uses_category(k_domain, "rocm_rccl_api");
}

TEST_F(rccl_callbacks_test, register_gpu_pmc_registers_send_and_recv_for_the_device)
{
    auto& registry = *test_support::g_metadata_registry_mock;
    EXPECT_CALL(registry,
                add_pmc_info(AllOf(pmc_named("RCCL Comm Send GPU 3"),
                                   Field(&pmc_info_data_t::agent_type_index, Eq(3U)))))
        .Times(1);
    EXPECT_CALL(registry,
                add_pmc_info(AllOf(pmc_named("RCCL Comm Recv GPU 3"),
                                   Field(&pmc_info_data_t::agent_type_index, Eq(3U)))))
        .Times(1);

    rccl::register_gpu_pmc<ext>(3);
}

// Same registration through the plain (non-tracing) externals, which is the pairing the
// production k_rccl_api<mock_sdk, externals> descriptor instantiates.
TEST_F(rccl_callbacks_test, register_gpu_pmc_works_with_plain_externals)
{
    auto& registry = *test_support::g_metadata_registry_mock;
    EXPECT_CALL(registry, add_pmc_info(Field(&pmc_info_data_t::name,
                                             Eq(std::string{ "RCCL Comm Send GPU 4" }))))
        .Times(1);
    EXPECT_CALL(registry, add_pmc_info(Field(&pmc_info_data_t::name,
                                             Eq(std::string{ "RCCL Comm Recv GPU 4" }))))
        .Times(1);

    rccl::register_gpu_pmc<externals>(4);
}

TEST_F(rccl_callbacks_test, configure_registers_category_and_both_tracks)
{
    auto& registry = *test_support::g_metadata_registry_mock;
    EXPECT_CALL(registry, add_string(Eq(ext::comm_data_name))).Times(1);
    EXPECT_CALL(registry,
                ensure_track(Eq(track_data_t{ std::string{ ext::rccl_send_track_name },
                                              std::nullopt, "{}" })))
        .Times(1);
    EXPECT_CALL(registry,
                ensure_track(Eq(track_data_t{ std::string{ ext::rccl_recv_track_name },
                                              std::nullopt, "{}" })))
        .Times(1);

    rccl::on_rccl_configure<sdk, ext>();
}

TEST_F(rccl_callbacks_test,
       configure_with_unresolved_symbol_and_error_text_keeps_default_device)
{
    ext::dlerror_result = "symbol lookup failed";
    EXPECT_CALL(*test_support::g_metadata_registry_mock,
                add_string(Eq(ext::comm_data_name)));
    EXPECT_CALL(*test_support::g_metadata_registry_mock, ensure_track(::testing::_))
        .Times(2);

    rccl::on_rccl_configure<sdk, ext>();

    int device = 9;
    EXPECT_EQ(resolver::resolve_device_id(&device), 0U);
}

TEST_F(rccl_callbacks_test, resolver_uses_resolved_function_result)
{
    configure_with(fake_comm_cu_device);
    int device = 9;

    EXPECT_EQ(resolver::resolve_device_id(&device), 9U);
    EXPECT_EQ(resolver::resolve_device_id(nullptr), 0U);
}

TEST_F(rccl_callbacks_test, resolver_defaults_to_zero_when_function_reports_failure)
{
    configure_with(failing_comm_cu_device);
    int device = 9;

    EXPECT_EQ(resolver::resolve_device_id(&device), 0U);
}

TEST_F(rccl_callbacks_test, exit_without_payload_stores_nothing)
{
    exit_with(sdk::RCCL_API_ID_ncclSend, nullptr);
}

TEST_F(rccl_callbacks_test, exit_with_unhandled_operation_stores_nothing)
{
    sdk::rccl_api_data payload{};
    exit_with(sdk::RCCL_API_ID_ncclSend + 100, &payload);
}

TEST_F(rccl_callbacks_test, exit_with_zero_size_transfer_stores_nothing)
{
    int                comm = 20;
    sdk::rccl_api_data payload{};
    payload.args.ncclSend = { &comm, sdk::NCCL_FLOAT32, 0 };
    configure_with(fake_comm_cu_device);

    exit_with(sdk::RCCL_API_ID_ncclSend, &payload);
}

TEST_F(rccl_callbacks_test, first_send_registers_pmc_and_stores_cumulative_sample)
{
    int                comm = 21;
    sdk::rccl_api_data payload{};
    payload.args.ncclSend = { &comm, sdk::NCCL_FLOAT32, 10 };
    configure_with(fake_comm_cu_device);

    auto& registry = *test_support::g_metadata_registry_mock;
    EXPECT_CALL(registry, add_pmc_info(pmc_named("RCCL Comm Send GPU 21"))).Times(1);
    EXPECT_CALL(registry, add_pmc_info(pmc_named("RCCL Comm Recv GPU 21"))).Times(1);
    EXPECT_CALL(*test_support::g_buffer_storage_mock,
                store_pmc_event(Eq(std::size_t{ ext::comm_data_enum_value }),
                                Eq(std::string{ "RCCL Comm Send" }), Eq(k_timestamp),
                                Eq(std::string{ R"({"transfer_bytes":40})" }), Eq(21U),
                                Eq(std::uint8_t{ ext::k_agent_type_gpu }),
                                Eq(std::string{ "RCCL Comm Send GPU 21" }),
                                DoubleEq(40.0)))
        .Times(1);

    exit_with(sdk::RCCL_API_ID_ncclSend, &payload);
}

TEST_F(rccl_callbacks_test, second_recv_accumulates_without_re_registering_pmc)
{
    int                comm = 22;
    sdk::rccl_api_data send_payload{};
    sdk::rccl_api_data recv_payload{};
    send_payload.args.ncclSend = { &comm, sdk::NCCL_INT8, 6 };
    recv_payload.args.ncclRecv = { &comm, sdk::NCCL_FLOAT64, 2 };
    configure_with(fake_comm_cu_device);

    auto& registry = *test_support::g_metadata_registry_mock;
    EXPECT_CALL(registry, add_pmc_info(::testing::_)).Times(2);
    auto& storage = *test_support::g_buffer_storage_mock;
    EXPECT_CALL(storage,
                store_pmc_event(::testing::_, Eq(std::string{ "RCCL Comm Send" }),
                                ::testing::_, ::testing::_, Eq(22U), ::testing::_,
                                ::testing::_, DoubleEq(6.0)))
        .Times(1);
    EXPECT_CALL(storage, store_pmc_event(
                             Eq(std::size_t{ ext::comm_data_enum_value }),
                             Eq(std::string{ "RCCL Comm Recv" }), Eq(k_timestamp),
                             Eq(std::string{ R"({"transfer_bytes":16})" }), Eq(22U),
                             Eq(std::uint8_t{ ext::k_agent_type_gpu }),
                             Eq(std::string{ "RCCL Comm Recv GPU 22" }), DoubleEq(22.0)))
        .Times(1);

    exit_with(sdk::RCCL_API_ID_ncclSend, &send_payload);
    exit_with(sdk::RCCL_API_ID_ncclRecv, &recv_payload);
}

}  // namespace rocprofsys::domains::callback
