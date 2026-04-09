// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/shmem_gotcha.hpp"

#include <timemory/components/macros.hpp>

TIMEMORY_STORAGE_INITIALIZER(
    rocprofsys::component::shmem_gotcha<rocprofsys::DefaultSHMEMPolicy>)
