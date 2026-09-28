// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rccl.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

#include <optional>

namespace rocprofsys::domains::callback
{
namespace
{

using test_support::expect_domain_uses_category;
using test_support::externals;
using test_support::externals_with_tracing;
using test_support::mock_sdk;
using test_support::mock_sdk_with_tracing;

}  // namespace

TEST(rccl_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_rccl<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "rccl");
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
        k_rccl<mock_sdk_with_tracing, externals_with_tracing>;

    expect_domain_uses_category(k_domain, "rocm_rccl_api");
}

TEST(rccl_extract_event_info_test, returns_default_when_payload_is_null)
{
    auto record      = mock_sdk::callback_tracing_record_t{};
    record.payload   = nullptr;
    record.operation = mock_sdk::RCCL_API_ID_ncclSend;

    const auto info = detail::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.size, 0U);
    EXPECT_EQ(info.comm, nullptr);
}

TEST(rccl_extract_event_info_test, ncclSend_is_classified_as_send_using_count)
{
    constexpr std::size_t k_count = 7;
    int                   fake_comm{};

    auto payload                   = mock_sdk::rccl_api_data{};
    payload.args.ncclSend.comm     = &fake_comm;
    payload.args.ncclSend.datatype = mock_sdk::NCCL_FLOAT32;
    payload.args.ncclSend.count    = k_count;

    auto record      = mock_sdk::callback_tracing_record_t{};
    record.operation = mock_sdk::RCCL_API_ID_ncclSend;
    record.payload   = &payload;

    const auto info = detail::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.type, detail::event_type::send);
    EXPECT_EQ(info.comm, &fake_comm);
    EXPECT_EQ(info.size, k_count * mock_sdk::rccl_type_size(mock_sdk::NCCL_FLOAT32));
}

TEST(rccl_extract_event_info_test, ncclRecv_is_classified_as_recv_using_count)
{
    constexpr std::size_t k_count = 3;
    int                   fake_comm{};

    auto payload                   = mock_sdk::rccl_api_data{};
    payload.args.ncclRecv.comm     = &fake_comm;
    payload.args.ncclRecv.datatype = mock_sdk::NCCL_INT8;
    payload.args.ncclRecv.count    = k_count;

    auto record      = mock_sdk::callback_tracing_record_t{};
    record.operation = mock_sdk::RCCL_API_ID_ncclRecv;
    record.payload   = &payload;

    const auto info = detail::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.type, detail::event_type::recv);
    EXPECT_EQ(info.size, k_count * mock_sdk::rccl_type_size(mock_sdk::NCCL_INT8));
}

TEST(rccl_extract_event_info_test, ncclAllGather_reads_sendcount)
{
    constexpr std::size_t k_sendcount = 5;

    auto payload                         = mock_sdk::rccl_api_data{};
    payload.args.ncclAllGather.datatype  = mock_sdk::NCCL_UINT64;
    payload.args.ncclAllGather.sendcount = k_sendcount;

    auto record      = mock_sdk::callback_tracing_record_t{};
    record.operation = mock_sdk::RCCL_API_ID_ncclAllGather;
    record.payload   = &payload;

    const auto info = detail::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.size, k_sendcount * mock_sdk::rccl_type_size(mock_sdk::NCCL_UINT64));
}

// Regression guard: on_rccl_exit's ncclReduceScatter case previously read
// payload.args.recvcount (not a member of the payload union), silently miscomputing the
// transfer size for every reduce-scatter call.
TEST(rccl_extract_event_info_test, ncclReduceScatter_reads_recvcount_and_is_send)
{
    constexpr std::size_t k_recvcount = 11;

    auto payload                             = mock_sdk::rccl_api_data{};
    payload.args.ncclReduceScatter.datatype  = mock_sdk::NCCL_FLOAT16;
    payload.args.ncclReduceScatter.recvcount = k_recvcount;

    auto record      = mock_sdk::callback_tracing_record_t{};
    record.operation = mock_sdk::RCCL_API_ID_ncclReduceScatter;
    record.payload   = &payload;

    const auto info = detail::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.type, detail::event_type::send);
    EXPECT_EQ(info.size, k_recvcount * mock_sdk::rccl_type_size(mock_sdk::NCCL_FLOAT16));
}

TEST(rccl_device_id_resolver_test, resolve_device_id_defaults_to_zero_for_null_comm)
{
    const auto device_id =
        detail2::device_id_resolver<mock_sdk>::resolve_device_id(nullptr);

    EXPECT_EQ(device_id, 0U);
}

}  // namespace rocprofsys::domains::callback
