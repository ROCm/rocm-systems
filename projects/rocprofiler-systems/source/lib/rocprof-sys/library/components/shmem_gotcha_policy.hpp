// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "library/components/category_region.hpp"
#include "library/components/comm_data.hpp"
#include "library/components/shmem_gotcha.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/components/gotcha/components.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>
#include <timemory/variadic/types.hpp>

namespace rocprofsys
{

struct DefaultSHMEMPolicy
{
    using comm_data       = component::comm_data;
    using gotcha_data     = tim::component::gotcha_data;
    using category_region = component::category_region<category::shmem>;

    using component_t = component::shmem_gotcha<DefaultSHMEMPolicy>;

    using shmem_bundle_t = tim::component_bundle<category::shmem, component_t, comm_data>;
    using shmem_gotcha_t = tim::component::gotcha<component_t::gotcha_capacity,
                                                  shmem_bundle_t, category::shmem>;

    using gotcha_bundle_t = tim::lightweight_tuple<shmem_gotcha_t>;
};
}  // namespace rocprofsys
