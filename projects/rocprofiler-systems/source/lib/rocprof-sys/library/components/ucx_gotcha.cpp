// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/ucx_gotcha.hpp"
#include "library/components/ucx_gotcha_policy.hpp"

#include <timemory/components/macros.hpp>

TIMEMORY_STORAGE_INITIALIZER(
    rocprofsys::component::ucx_gotcha<rocprofsys::DefaultUCXPolicy>)
