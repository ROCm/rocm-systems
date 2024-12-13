// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "info_type.hpp"
#include "enumerated_list.hpp"
#include "get_availability.hpp"

#include "api.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/components/pthread_gotcha.hpp"

#include <timemory/components/definition.hpp>
#include <timemory/enum.h>
#include <timemory/utility/macros.hpp>

#include <utility>

template <size_t EndV>
std::vector<info_type>
get_component_info()
{
    using index_seq_t = std::make_index_sequence<EndV>;
    using enum_list_t = typename enumerated_list<tim::type_list<>, index_seq_t>::type;

    auto _info = std::vector<info_type>{};
    return get_availability<>{}(enum_list_t{}, _info);
}

template std::vector<info_type>
get_component_info<TIMEMORY_NATIVE_COMPONENTS_END>();

template std::vector<info_type>
get_component_info<TIMEMORY_COMPONENTS_END>();
