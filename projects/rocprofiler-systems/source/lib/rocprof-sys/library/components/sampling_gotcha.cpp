// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/sampling_gotcha.hpp"
#include "library/components/sampling_gotcha_policy.hpp"

#include <timemory/components/macros.hpp>

TIMEMORY_STORAGE_INITIALIZER(
    rocprofsys::component::sampling_gotcha<rocprofsys::DefaultSamplingPolicy>)
