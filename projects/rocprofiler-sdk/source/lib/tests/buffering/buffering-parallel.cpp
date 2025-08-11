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
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"

#include <gtest/gtest.h>
#include <pthread.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <typeinfo>
#include <utility>

namespace
{
namespace test  = ::rocprofiler::test;
namespace units = ::rocprofiler::common::units;
namespace mpl   = ::rocprofiler::common::mpl;

using record_header_buffer_t = rocprofiler::common::container::record_header_buffer;

// this function returns a random array of values specific to template instantiation
template <typename Tp, size_t N>
auto&
get_generated_array()
{
    static auto _value = []() {
        auto _v = test::raw_array<Tp, N>{};
        test::generate(_v, Tp{0}, std::numeric_limits<Tp>::max());
        return _v;
    }();
    return _value;
}

// these are the array size variants. we use the units to scale up
// but technically the data size of the raw_array will be multiplied
// by sizeof(Tp)
constexpr auto test_data_sizes = std::index_sequence<1 * units::byte,
                                                     2 * units::byte,
                                                     3 * units::byte,
                                                     4 * units::byte,
                                                     8 * units::byte,
                                                     16 * units::kilobyte,
                                                     20 * units::kilobyte,
                                                     24 * units::kilobyte,
                                                     32 * units::kilobyte,
                                                     56 * units::kilobyte,
                                                     64 * units::kilobyte,
                                                     91 * units::kilobyte,
                                                     128 * units::kilobyte,
                                                     387 * units::kilobyte,
                                                     693 * units::kilobyte>{};

// this is the list of array data types we will generate. Effectively, there
// will be one raw array for each combination of these types and the test data sizes
// (i.e. there will be unique 160 arrays of different types and sizes)
using test_data_types = mpl::type_list<int8_t,
                                       uint8_t,
                                       int16_t,
                                       uint16_t,
                                       int32_t,
                                       uint32_t,
                                       int64_t,
                                       uint64_t,
                                       float,
                                       double>;

// this function creates a thread for each data size for a given type.
// all threads are detached and will wait at the first barrier until all
// threads have reached it, race to emplace their data in the shared
// buffer and then wait at the second barrier until all the threads have
// emplacing the data and the main thread has also reached the second
// barrier.
template <typename Tp, size_t... Idx>
void
launch_threads(record_header_buffer_t& _buf,
               pthread_barrier_t&      _race_barrier,
               pthread_barrier_t&      _done_barrier,
               std::index_sequence<Idx...>)
{
    auto _launch =
        [](record_header_buffer_t* _buf_v, auto* _race_barrier_v, auto* _done_barrier_v, auto* _v) {
            pthread_barrier_wait(_race_barrier_v);
            EXPECT_TRUE(_buf_v->emplace(*_v));
            pthread_barrier_wait(_done_barrier_v);
        };
    (std::thread{_launch, &_buf, &_race_barrier, &_done_barrier, &get_generated_array<Tp, Idx>()}
         .detach(),
     ...);
}

// expansion for each type
template <typename... Tp, size_t... Idx>
void
launch_threads(record_header_buffer_t& _buf,
               pthread_barrier_t&      _race_barrier,
               pthread_barrier_t&      _done_barrier,
               mpl::type_list<Tp...>,
               std::index_sequence<Idx...> _seq)
{
    (launch_threads<Tp>(_buf, _race_barrier, _done_barrier, _seq), ...);
}

// computes the size of every raw_array size for a given type
template <typename Tp, size_t... Idx>
constexpr size_t get_data_size(std::index_sequence<Idx...>)
{
    size_t _v = 0;
    ((_v += sizeof(get_generated_array<Tp, Idx>())), ...);
    return _v;
}

// expansion for each type
template <typename... Tp, size_t... Idx>
constexpr size_t
get_data_size(mpl::type_list<Tp...>, std::index_sequence<Idx...> _seq)
{
    size_t _v = 0;
    ((_v += get_data_size<Tp>(_seq)), ...);
    return _v;
}

// validates that the raw array extracted out of the buffer is equal
// to the raw array that was placed in the buffer
template <typename Tp, size_t N>
void
validate(const std::vector<rocprofiler_record_header_t*>& _headers)
{
    using data_type = test::raw_array<Tp, N>;
    auto& _ref_data = get_generated_array<Tp, N>();
    for(auto* itr : _headers)
    {
        if(itr->hash == typeid(data_type).hash_code())
        {
            auto* _data = static_cast<data_type*>(itr->payload);
            EXPECT_EQ(_ref_data, *_data)
                << "\nREF: " << _ref_data.to_string() << "\nINP: " << _data->to_string();
        }
    }
}

// expansion for every raw array size for a given data type
template <typename Tp, size_t... Idx>
void
validate(const std::vector<rocprofiler_record_header_t*>& _headers, std::index_sequence<Idx...>)
{
    (validate<Tp, Idx>(_headers), ...);
}

// expansion for each raw array type
template <typename... Tp, size_t... Idx>
void
validate(const std::vector<rocprofiler_record_header_t*>& _headers,
         mpl::type_list<Tp...>,
         std::index_sequence<Idx...> _seq)
{
    (validate<Tp>(_headers, _seq), ...);
}
}  // namespace

TEST(buffering, parallel)
{
    // this test launches 160 threads, each with a randomly generated array of data
    // and has them contend for emplacing their data in the same buffer. The purpose
    // of this test is to validate that multiple threads can write to the same
    // (lock-free) buffer without any data corruption or loss.

    constexpr auto num_variants = test_data_types::size() * test_data_sizes.size();
    constexpr auto data_size    = get_data_size(test_data_types{}, test_data_sizes);

    EXPECT_EQ(num_variants, 150);

    // make a buffer large enough to hold all the data we generate
    auto _buffer = record_header_buffer_t{data_size};

    // create a barrier that all child threads will wait and then race to enqueue their data in the
    // buffer i.e., we want to maximize contention on inserting into buffer
    auto _data_race_barrier = pthread_barrier_t{};
    pthread_barrier_init(&_data_race_barrier, nullptr, num_variants);

    // a barrier to signal that all threads have completed placing their data in the buffer
    auto _emplaced_barrier = pthread_barrier_t{};
    pthread_barrier_init(&_emplaced_barrier, nullptr, num_variants + 1);

    // launch 160 threads
    launch_threads(
        _buffer, _data_race_barrier, _emplaced_barrier, test_data_types{}, test_data_sizes);

    // wait for all the threads to complete
    pthread_barrier_wait(&_emplaced_barrier);

    // designates that buffer should be cleared after invoking functor
    using clear_buffer_t = std::true_type;

    // verify the data pulled out the buffer matches the data put in by the threads
    _buffer.process_record_headers(clear_buffer_t{}, [](auto&& _records) {
        validate(_records, test_data_types{}, test_data_sizes);
    });
}
