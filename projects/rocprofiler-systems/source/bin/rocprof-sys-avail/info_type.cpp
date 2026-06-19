// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
