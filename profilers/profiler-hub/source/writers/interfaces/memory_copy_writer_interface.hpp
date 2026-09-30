// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/writer_types.hpp"
#include "writers/interfaces/api_writer_base.hpp"

namespace profiler_hub
{

template <typename Derived>
class memory_copy_writer_interface : public api_writer_base<Derived>
{
public:
    void insert(const writer_types::memory_copy_data_t&  data,
                const writer_types::trace_environment_t& trace_env)
    {
        this->self().insert_impl(data, trace_env);
    }
};

}  // namespace profiler_hub
