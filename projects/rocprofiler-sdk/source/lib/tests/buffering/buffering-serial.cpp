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

#include "buffering.hpp"
#include "lib/common/container/record_header_buffer.hpp"

#include <gtest/gtest.h>
#include <pthread.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <typeinfo>

namespace
{
namespace test = ::rocprofiler::test;

using uint_raw_array_t       = test::raw_array<uint64_t, 32>;
using flt_raw_array_t        = test::raw_array<double, 64>;
using record_header_buffer_t = rocprofiler::common::container::record_header_buffer;

// generates an array with random data
template <typename Tp, size_t N>
auto
generate_array(Tp _low = 0UL, Tp _high = 1000UL)
{
    auto _v = test::raw_array<Tp, N>{};
    test::generate(_v, _low, _high);
    return _v;
}

// pulls out a raw array of the given type and puts back into a vector
template <typename Tp>
void
extract_header(std::vector<Tp>& _arr, rocprofiler_record_header_t* _hdr)
{
    if(_hdr->hash == typeid(Tp).hash_code())
    {
        auto* _v = reinterpret_cast<Tp*>(_hdr->payload);
        _arr.emplace_back(*_v);
    }
    else
    {
        GTEST_FAIL() << __PRETTY_FUNCTION__ << " failed";
    }
}
}  // namespace

TEST(buffering, serial)
{
    // this test verifies that the buffering system is ordered properly
    // and does not suffer from data loss or data corruption. We generate
    // 240 raw arrays of data where 120 of them are twice as large as the
    // the other 120 raw array and these two arrays contain data of different
    // types. For each iteration, we randomize whether the uint64_t array with
    // 32 elements or whether the double array with 64 elements gets inserted
    // first. We then pull all the data back out of the buffer and verify
    // that no arrays were lost and that none of the data was corrupted.

    uint64_t n = 120;

    // storage of the original data put into the buffer
    auto _ui_history = std::vector<uint_raw_array_t>{};
    auto _ui_result  = std::vector<uint_raw_array_t>{};

    // storage of the data extracted from the buffer
    auto _fp_history = std::vector<flt_raw_array_t>{};
    auto _fp_result  = std::vector<flt_raw_array_t>{};

    // a buffer to hold all the data
    auto _buffer = record_header_buffer_t{n * (sizeof(uint_raw_array_t) + sizeof(flt_raw_array_t))};

    // RNG use to make the ordering of the different sized records inconsistent
    auto _gen = std::mt19937_64{std::random_device{}()};
    auto _rng = std::uniform_int_distribution<short>{0, 1};
    for(uint64_t i = 0; i < n; ++i)
    {
        // generate a 32*8 byte array
        auto _u = generate_array<uint64_t, 32>();
        // generate a 64*8 byte array
        auto _f = generate_array<double, 64>();

        // store the original data
        _ui_history.emplace_back(_u);
        _fp_history.emplace_back(_f);

        EXPECT_EQ(_u, _ui_history.back()) << "uint not equal after emplace_back";
        EXPECT_EQ(_f, _fp_history.back()) << "float not equal after emplace_back";

        // randomize sequence of insertion into buffer
        if(_rng(_gen) % 2 == 0)
        {
            _buffer.emplace(_u);
            _buffer.emplace(_f);
        }
        else
        {
            _buffer.emplace(_f);
            _buffer.emplace(_u);
        }

        EXPECT_EQ(_u, _ui_history.back()) << "uint not equal after emplace_back";
        EXPECT_EQ(_f, _fp_history.back()) << "float not equal after emplace_back";
    }

    // designates that buffer should be cleared after invoking functor
    constexpr auto clear_buffer_v = std::true_type{};

    // get the records out of the buffer
    auto _num_headers = _buffer.process_record_headers(
        clear_buffer_v,
        [](auto&& _headers, auto& _ui_result_v, auto& _fp_result_v) {
            for(auto* itr : _headers)
            {
                ASSERT_TRUE(itr->payload) << "nullptr to payload not expected";

                if(itr->hash == typeid(uint_raw_array_t).hash_code())
                {
                    extract_header(_ui_result_v, itr);
                }
                else if(itr->hash == typeid(flt_raw_array_t).hash_code())
                {
                    extract_header(_fp_result_v, itr);
                }
                else
                {
                    GTEST_FAIL() << "unknown type id hash code: " << std::to_string(itr->hash);
                }
            }
        },
        _ui_result,
        _fp_result);

    ASSERT_EQ(_ui_history.size() + _fp_history.size(), _num_headers)
        << "UINT: " << _ui_history.size() << " + FLOAT: " << _fp_history.size()
        << " != HEADERS: " << _num_headers;
    // validate that we got the same number of records out that we put in
    ASSERT_EQ(_ui_history.size(), _ui_result.size())
        << "UINT: " << _ui_history.size() << " vs. " << _ui_result.size();
    ASSERT_EQ(_fp_history.size(), _fp_result.size())
        << "FLOAT: " << _fp_history.size() << " vs. " << _fp_result.size();

    // validate there was no data corruption or data loss from storage in the buffer
    for(size_t i = 0; i < n; ++i)
    {
        auto& _ui_lhs = _ui_history.at(i);
        auto& _ui_rhs = _ui_result.at(i);
        auto& _fp_lhs = _fp_history.at(i);
        auto& _fp_rhs = _fp_result.at(i);

        EXPECT_EQ(_ui_lhs, _ui_rhs) << "\n"
                                    << "UINT LHS:\n"
                                    << _ui_lhs.to_string() << "\n"
                                    << "UINT RHS:\n"
                                    << _ui_rhs.to_string() << "\n";

        EXPECT_EQ(_fp_lhs, _fp_rhs) << "\n"
                                    << "FLOAT LHS:\n"
                                    << _fp_lhs.to_string() << "\n"
                                    << "FLOAT RHS:\n"
                                    << _fp_rhs.to_string() << "\n";
    }
}
