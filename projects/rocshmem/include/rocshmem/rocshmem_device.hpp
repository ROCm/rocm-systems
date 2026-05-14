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
 * in device code. It provides the template wrappers that forward to the
 * type-erased bitcode implementations.
 *
 * Usage:
 * #include <rocshmem/rocshmem.hpp>
 * #ifdef __HIP_DEVICE_COMPILE__
 * #include <rocshmem/rocshmem_device.hpp>
 * #endif
 *
 * NOTE: This header no longer includes internal context headers. The template
 * implementations in rocshmem_TILE_impl.hpp extract type information from
 * generic tensor types and forward to precompiled bitcode functions in
 * src/rocshmem_tile_gpu.cpp, which handle the backend dispatch internally.
 */

#ifndef LIBRARY_INCLUDE_ROCSHMEM_DEVICE_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_DEVICE_HPP

#ifdef __HIP_DEVICE_COMPILE__

// Include Tile API template implementations
// These forward to extern "C" bitcode functions, avoiding the need
// to expose internal context headers in the public API
#include "rocshmem_TILE_impl.hpp"

#endif  // __HIP_DEVICE_COMPILE__

#endif  // LIBRARY_INCLUDE_ROCSHMEM_DEVICE_HPP
