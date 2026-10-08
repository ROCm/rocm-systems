// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rccl/event_info.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::domains::callback
{
namespace
{

using test_support::mock_sdk;

}  // namespace

TEST(rccl_extract_event_info_test, returns_default_when_payload_is_null)
{
    auto record      = mock_sdk::callback_tracing_record_t{};
    record.payload   = nullptr;
    record.operation = mock_sdk::RCCL_API_ID_ncclSend;

    const auto info = rccl::extract_event_info<mock_sdk>(record);

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

    const auto info = rccl::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.type, rccl::event_type::send);
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

    const auto info = rccl::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.type, rccl::event_type::recv);
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

    const auto info = rccl::extract_event_info<mock_sdk>(record);

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

    const auto info = rccl::extract_event_info<mock_sdk>(record);

    EXPECT_EQ(info.type, rccl::event_type::send);
    EXPECT_EQ(info.size, k_recvcount * mock_sdk::rccl_type_size(mock_sdk::NCCL_FLOAT16));
}

}  // namespace rocprofsys::domains::callback
