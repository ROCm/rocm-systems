// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/static_object.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace
{
constexpr auto k_over_aligned_object_alignment = std::size_t{ 64 };
constexpr auto k_expected_value                = std::uint64_t{ 42 };

struct alignas(k_over_aligned_object_alignment) over_aligned_object
{
    std::uint64_t value = 0;
};

struct over_aligned_context
{};
}  // namespace

TEST(static_object, aligns_backing_storage_to_object)
{
    using object_type =
        rocprofsys::common::static_object<over_aligned_object, over_aligned_context>;

    auto* object = object_type::construct(k_expected_value);

    ASSERT_NE(object, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(object) % alignof(over_aligned_object),
              std::uintptr_t{ 0 });
    EXPECT_EQ(object->value, k_expected_value);
}
