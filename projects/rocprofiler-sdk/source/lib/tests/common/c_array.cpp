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

#include <rocprofiler-sdk/cxx/container/c_array.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

namespace
{
template <typename Tp>
void
compare_serialization_of_vector_and_c_array(std::vector<Tp>& vec)
{
    std::stringstream ss_vec;
    {
        cereal::BinaryOutputArchive oarchive(ss_vec);
        save(oarchive, vec);
    }

    std::stringstream ss_arr;
    {
        cereal::BinaryOutputArchive oarchive(ss_arr);
        auto arr = rocprofiler::sdk::container::make_c_array(vec.data(), vec.size());
        save(oarchive, arr);
    }

    ASSERT_EQ(ss_vec.str(), ss_arr.str());
}

template <typename Tp>
void
compare_serialization_of_vector_and_c_array_of_pointers(std::vector<Tp>& vec)
{
    std::vector<Tp*> ptrs;
    ptrs.reserve(vec.size());
    for(auto& val : vec)
        ptrs.emplace_back(&val);

    std::stringstream ss_vec;
    {
        cereal::BinaryOutputArchive oarchive(ss_vec);
        save(oarchive, vec);
    }

    std::stringstream ss_arr;
    {
        cereal::BinaryOutputArchive oarchive(ss_arr);
        auto arr = rocprofiler::sdk::container::make_c_array(ptrs.data(), ptrs.size());
        save(oarchive, arr);
    }

    ASSERT_EQ(ss_vec.str(), ss_arr.str());
}

}  // namespace

TEST(common, c_array)
{
    std::vector<int> vec{1, 2, 3, 4, 5};
    auto             arr = rocprofiler::sdk::container::make_c_array(vec.data(), vec.size());

    // Validate size
    ASSERT_EQ(arr.size(), vec.size());

    // Test operator[] and at()
    for(size_t i = 0; i < arr.size(); ++i)
    {
        EXPECT_EQ(arr[i], vec[i]);
        EXPECT_EQ(arr.at(i), vec.at(i));
    }

    // Bounds checking on at()
    EXPECT_THROW(arr.at(arr.size()), std::out_of_range);

    // Test slice
    auto sub_arr = arr.slice(1, 4);  // should contain 2, 3, 4
    ASSERT_EQ(sub_arr.size(), 3);
    EXPECT_EQ(sub_arr[0], 2);
    EXPECT_EQ(sub_arr[2], 4);

    // Test pop_front()
    arr.pop_front();  // drops 1
    ASSERT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 2);

    // Test pop_back()
    arr.pop_back();  // drops 5
    ASSERT_EQ(arr.size(), 3);
    EXPECT_EQ(arr[2], 4);

    // Test iterators
    int expected[] = {2, 3, 4};
    int idx        = 0;
    for(auto v : arr)
    {
        EXPECT_EQ(v, expected[idx++]);
    }

    // Const iterator
    const auto& const_arr = arr;
    idx                   = 0;
    for(auto v : const_arr)
    {
        EXPECT_EQ(v, expected[idx++]);
    }

    // Test implicit cast to pointer
    int* ptr = arr;
    EXPECT_EQ(ptr[0], 2);
    EXPECT_EQ(ptr[2], 4);

    // Test Serialization.
    compare_serialization_of_vector_and_c_array(vec);
    compare_serialization_of_vector_and_c_array_of_pointers(vec);
}
