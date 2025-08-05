// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/mpl.hpp"

#include <gtest/gtest.h>

#include <tuple>

TEST(common, mpl)
{
    namespace mpl = ::rocprofiler::common::mpl;
    struct Foo;
    using test_type_list_t = mpl::type_list<int, long, int, Foo, Foo*>;
    using test_tuple_t     = std::tuple<int, long, int, Foo, Foo*>;

    EXPECT_EQ(mpl::size_of<test_type_list_t>::value, 5);
    EXPECT_EQ(mpl::size_of<test_tuple_t>::value, 5);

    {
        constexpr bool _foo_p_is_one_of  = mpl::is_one_of<Foo*, test_type_list_t>::value;
        constexpr bool _int_is_one_of    = mpl::is_one_of<int, test_type_list_t>::value;
        constexpr bool _double_is_one_of = mpl::is_one_of<double, test_type_list_t>::value;

        EXPECT_TRUE(_foo_p_is_one_of);
        EXPECT_TRUE(_int_is_one_of);
        EXPECT_FALSE(_double_is_one_of);

        constexpr auto _foo_p_index_of = mpl::index_of<Foo*, test_type_list_t>::value;
        constexpr auto _int_index_of   = mpl::index_of<int, test_type_list_t>::value;

        EXPECT_EQ(_foo_p_index_of, 4);
        EXPECT_EQ(_int_index_of, 0);
    }

    {
        constexpr bool _foo_p_is_one_of  = mpl::is_one_of<Foo*, test_tuple_t>::value;
        constexpr bool _int_is_one_of    = mpl::is_one_of<int, test_tuple_t>::value;
        constexpr bool _double_is_one_of = mpl::is_one_of<double, test_tuple_t>::value;

        EXPECT_TRUE(_foo_p_is_one_of);
        EXPECT_TRUE(_int_is_one_of);
        EXPECT_FALSE(_double_is_one_of);

        constexpr auto _foo_p_index_of = mpl::index_of<Foo*, test_tuple_t>::value;
        constexpr auto _int_index_of   = mpl::index_of<int, test_tuple_t>::value;

        EXPECT_EQ(_foo_p_index_of, 4);
        EXPECT_EQ(_int_index_of, 0);
    }
}
