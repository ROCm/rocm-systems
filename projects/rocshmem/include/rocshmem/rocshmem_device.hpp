/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file rocshmem_device.hpp
 * @brief Additional includes needed for device-side Tile API usage
 *
 * This header should be included AFTER rocshmem.hpp when using the Tile API
 * in device code. It includes the necessary internal headers to make the
 * Tile API template implementations available.
 *
 * Usage:
 * #include <rocshmem/rocshmem.hpp>
 * #ifdef __HIP_DEVICE_COMPILE__
 * #include <rocshmem/rocshmem_device.hpp>
 * #endif
 */

#ifndef LIBRARY_INCLUDE_ROCSHMEM_DEVICE_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_DEVICE_HPP

#ifdef __HIP_DEVICE_COMPILE__

// Include internal context definitions needed for Tile API
#include "../src/context_incl.hpp"

// Include Tile API template implementations
#include "rocshmem_TILE_impl.hpp"

#endif  // __HIP_DEVICE_COMPILE__

#endif  // LIBRARY_INCLUDE_ROCSHMEM_DEVICE_HPP
