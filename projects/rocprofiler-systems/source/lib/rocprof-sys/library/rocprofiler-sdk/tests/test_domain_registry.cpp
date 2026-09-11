// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/domain_registry.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace rocprofsys::domains
{
namespace
{

// registry<SdkBackend, Externals> instantiates every buffered/callback domain
// definition in library/rocprofiler-sdk/{buffered,callback}/*.hpp, so it needs a fake
// that satisfies the union of everything those headers touch on SdkBackend and
// Externals -- not just what a single domain needs. test_support::mock_sdk/externals
// already carry every kfd_* domain's record type, id, and category name, so registry<>
// needs no domain-specific additions of its own.
using test_support::externals;
using test_support::mock_sdk;

using sut_t = registry<mock_sdk, externals>;

TEST(domain_registry_test, find_descriptor_finds_buffered_domain_case_insensitively)
{
    const domain_descriptor* descriptor = sut_t::find_descriptor("KfD_QuEuE");

    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->name, "kfd_queue");
    EXPECT_EQ(descriptor->id, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_EQ(descriptor->mode, collection_mode::buffered);
}

TEST(domain_registry_test, find_descriptor_finds_callback_domain_case_insensitively)
{
    const domain_descriptor* descriptor = sut_t::find_descriptor("CODE_OBJECT");

    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->name, "code_object");
    EXPECT_EQ(descriptor->id, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
    EXPECT_EQ(descriptor->mode, collection_mode::callback);
}

TEST(domain_registry_test, find_descriptor_returns_nullptr_for_unknown_name)
{
    EXPECT_EQ(sut_t::find_descriptor("not_a_real_domain"), nullptr);
}

TEST(domain_registry_test, get_buffered_returns_definition_matching_domain_id)
{
    const auto& definition = sut_t::get_buffered(mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);

    EXPECT_EQ(definition.meta.name, "kfd_page_fault");
    EXPECT_EQ(definition.meta.id, mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);
}

TEST(domain_registry_test, get_buffered_throws_runtime_error_for_unknown_domain_id)
{
    constexpr domain_id_t k_unknown_id = 9999;

    EXPECT_THROW(
        { static_cast<void>(sut_t::get_buffered(k_unknown_id)); }, std::runtime_error);
}

TEST(domain_registry_test, get_callback_returns_definition_matching_domain_id)
{
    const auto& definition = sut_t::get_callback(mock_sdk::CALLBACK_TRACING_CODE_OBJECT);

    EXPECT_EQ(definition.meta.name, "code_object");
    EXPECT_EQ(definition.meta.id, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
}

TEST(domain_registry_test, get_callback_throws_runtime_error_for_unknown_domain_id)
{
    constexpr domain_id_t k_unknown_id = 9999;

    EXPECT_THROW(
        { static_cast<void>(sut_t::get_callback(k_unknown_id)); }, std::runtime_error);
}

}  // namespace
}  // namespace rocprofsys::domains
