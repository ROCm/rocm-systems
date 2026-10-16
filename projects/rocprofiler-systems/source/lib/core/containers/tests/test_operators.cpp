// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/containers/operators.hpp"

#include <gtest/gtest.h>

namespace
{
using namespace rocprofsys::container;

struct point
: less_than_comparable1<point>
, equality_comparable1<point>
, additive2<point, int>
{
    constexpr point() = default;
    constexpr point(int x_val, int y_val)
    : x{ x_val }
    , y{ y_val }
    {}

    constexpr point& operator+=(const point& rhs)
    {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
    constexpr point& operator-=(const point& rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }
    constexpr point& operator+=(int rhs)
    {
        x += rhs;
        y += rhs;
        return *this;
    }
    constexpr point& operator-=(int rhs)
    {
        x -= rhs;
        y -= rhs;
        return *this;
    }

    friend constexpr bool operator==(const point& lhs, const point& rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
    friend constexpr bool operator<(const point& lhs, const point& rhs)
    {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
    }

    int x = 0;
    int y = 0;
};

constexpr point k_origin{ 0, 0 };
constexpr point k_unit{ 1, 1 };

static_assert(k_origin < k_unit, "less_than_comparable1::operator< base case");
static_assert(k_unit > k_origin, "less_than_comparable1 synthesizes operator>");
static_assert(k_unit >= k_origin, "less_than_comparable1 synthesizes operator>=");
static_assert(k_origin <= k_unit, "less_than_comparable1 synthesizes operator<=");

constexpr bool k_origin_equals_unit = k_origin == k_unit;
static_assert(!k_origin_equals_unit, "");
static_assert(k_origin != k_unit, "equality_comparable1 synthesizes operator!=");

constexpr point k_unit_plus_scalar = k_unit + 2;
static_assert(k_unit_plus_scalar.x == 3,
              "additive2 synthesizes operator+ with scalar rhs");

constexpr point k_scalar_plus_unit = 2 + k_unit;
static_assert(k_scalar_plus_unit.x == 3, "addable2 is commutative");

constexpr point k_unit_minus_scalar = k_unit - 1;
static_assert(k_unit_minus_scalar.x == 0, "additive2 synthesizes operator-");

}  // namespace

TEST(Operators, less_than_comparable_runtime)
{
    const point pt_lhs{ 0, 0 };
    const point pt_rhs{ 1, 1 };
    EXPECT_LT(pt_lhs, pt_rhs);
    EXPECT_GT(pt_rhs, pt_lhs);
    EXPECT_LE(pt_lhs, pt_rhs);
    EXPECT_GE(pt_rhs, pt_lhs);
    EXPECT_NE(pt_lhs, pt_rhs);
}

TEST(Operators, additive2_runtime)
{
    const point pt_val{ 1, 1 };
    EXPECT_EQ((pt_val + 2).x, 3);
    EXPECT_EQ((2 + pt_val).x, 3);
    EXPECT_EQ((pt_val - 1).x, 0);
}

TEST(Operators, incrementable_postfix)
{
    struct counter : incrementable<counter>
    {
        constexpr counter& operator++()
        {
            ++value;
            return *this;
        }
        int value = 0;
    };

    counter       ctr;
    const counter prev = ctr++;
    EXPECT_EQ(prev.value, 0);
    EXPECT_EQ(ctr.value, 1);
}

TEST(Operators, decrementable_postfix)
{
    struct counter : decrementable<counter>
    {
        constexpr counter& operator--()
        {
            --value;
            return *this;
        }
        int value = 0;
    };

    counter       ctr;
    const counter prev = ctr--;
    EXPECT_EQ(prev.value, 0);
    EXPECT_EQ(ctr.value, -1);
}
