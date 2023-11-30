// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <unordered_map>

#include <rocprofiler-sdk/fwd.h>

#include <glog/logging.h>

namespace rocprofiler
{
namespace counters
{
constexpr uint64_t COUNTER_BIT_LENGTH = 16;
constexpr uint64_t DIM_BIT_LENGTH     = 48;
constexpr uint64_t MAX_64             = std::numeric_limits<uint64_t>::max();
enum rocprofiler_profile_counter_instance_types
{
    ROCPROFILER_DIMENSION_NONE = 0,       ///< No dimension data, returns/sets 48 bit value as is
    ROCPROFILER_DIMENSION_XCC,            ///< XCC dimension of result
    ROCPROFILER_DIMENSION_SHADER_ENGINE,  ///< SE dimension of result
    ROCPROFILER_DIMENSION_AGENT,          ///< Agent dimension
    ROCPROFILER_DIMENSION_PMC_CHANNEL,    ///< PMC Channel dimension of result
    ROCPROFILER_DIMENSION_CU,             ///< Number of compute units
    ROCPROFILER_DIMENSION_LAST
};

const std::unordered_map<rocprofiler_profile_counter_instance_types, std::string>&
dimension_map();

inline rocprofiler_counter_id_t
rec_to_counter_id(rocprofiler_counter_instance_id_t id);
inline void
set_dim_in_rec(rocprofiler_counter_instance_id_t&         id,
               rocprofiler_profile_counter_instance_types dim,
               size_t                                     value);
inline void
set_counter_in_rec(rocprofiler_counter_instance_id_t& id, rocprofiler_counter_id_t value);

inline size_t
rec_to_dim_pos(rocprofiler_counter_instance_id_t          id,
               rocprofiler_profile_counter_instance_types dim);

}  // namespace counters
}  // namespace rocprofiler

inline rocprofiler_counter_id_t
rocprofiler::counters::rec_to_counter_id(rocprofiler_counter_instance_id_t id)
{
    return {.handle = id >> DIM_BIT_LENGTH};
}

inline void
rocprofiler::counters::set_dim_in_rec(rocprofiler_counter_instance_id_t&         id,
                                      rocprofiler_profile_counter_instance_types dim,
                                      size_t                                     value)
{
    size_t  bit_length = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;
    int64_t mask       = (MAX_64 >> (64 - bit_length)) << ((dim - 1) * bit_length);

    if(dim == ROCPROFILER_DIMENSION_NONE)
    {
        // Set all 48 bits of dimension
        id         = (id & ~(MAX_64 >> COUNTER_BIT_LENGTH)) | value;
        bit_length = DIM_BIT_LENGTH;
    }
    else
    {
        // Reset bits to 0 for dimension. Does so by getting the bit length as F's then
        // shifiting that into the position of dim. Not's that value and then and's it
        // with id.
        id = (id & ~(mask));
        // Set the value for the dimenstion
        id = id | (value << ((dim - 1) * bit_length));
    }

    CHECK(value <= (MAX_64 >> (64 - bit_length))) << "Dimension value exceeds max allowed";
}

inline void
rocprofiler::counters::set_counter_in_rec(rocprofiler_counter_instance_id_t& id,
                                          rocprofiler_counter_id_t           value)
{
    // Maximum counter value given the current setup.
    CHECK(value.handle <= 0xffff) << "Counter id exceeds max allowed";
    // Reset bits to 0 for counter id
    id = id & ~((MAX_64 >> (64 - DIM_BIT_LENGTH)) << (DIM_BIT_LENGTH));
    // Set the value for the dimenstion
    id = id | (value.handle << (DIM_BIT_LENGTH));
}

inline size_t
rocprofiler::counters::rec_to_dim_pos(rocprofiler_counter_instance_id_t          id,
                                      rocprofiler_profile_counter_instance_types dim)
{
    if(dim == ROCPROFILER_DIMENSION_NONE)
    {
        // read all 48 bits of dimension
        return id & (MAX_64 >> COUNTER_BIT_LENGTH);
    }

    size_t bit_length = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;
    id                = id & ((MAX_64 >> (64 - bit_length)) << ((dim - 1) * bit_length));
    return id >> ((dim - 1) * bit_length);
}
