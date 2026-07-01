// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/writer_types.hpp"
#include "writers/interfaces/api_writer_base.hpp"

namespace profiler_hub
{

template <typename Derived>
class pmc_event_writer_interface : public api_writer_base<Derived>
{
public:
    void insert(const writer_types::pmc_event_data_t&     data,
                const writer_types::pmc_info_unique_id_t& pmc_unique_id)
    {
        this->self().insert_impl(data, pmc_unique_id);
    }
};

}  // namespace profiler_hub
