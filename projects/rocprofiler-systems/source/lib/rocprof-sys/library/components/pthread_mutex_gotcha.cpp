// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/pthread_mutex_gotcha.hpp"
#include "library/components/pthread_mutex_gotcha_policy.hpp"

#include <timemory/components/macros.hpp>

TIMEMORY_STORAGE_INITIALIZER(
    rocprofsys::component::pthread_mutex_gotcha<rocprofsys::default_pthread_mutex_policy>)
