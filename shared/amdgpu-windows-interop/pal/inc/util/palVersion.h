/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palVersion.h
 * @brief Defines the Platform Abstraction Library (PAL) Interface Version
 ***********************************************************************************************************************
 */

#pragma once

/// Major interface version.  Note that the interface version is distinct from the PAL version itself, which is returned
/// in @ref Pal::PlatformProperties.
///
/// @attention Updates to the major version indicate an interface change that is not backward compatible and may require
///            action from each client during their next integration.  When determining if a change is backward
///            compatible, it is assumed that the client will default-initialize all structs.
///
/// @ingroup LibInit
#define PAL_INTERFACE_MAJOR_VERSION 986

/// Minimum major interface version. This is the minimum interface version PAL supports in order to support backward
/// compatibility. When it is equal to PAL_INTERFACE_MAJOR_VERSION, only the latest interface version is supported.
///
/// @ingroup LibInit
#define PAL_MINIMUM_INTERFACE_MAJOR_VERSION 916

/// Minimum supported major interface version for devdriver library. This is the minimum interface version of the
/// devdriver library that PAL is backwards compatible to.
///
/// @ingroup LibInit
#define PAL_MINIMUM_GPUOPEN_INTERFACE_MAJOR_VERSION 38

/**
 ***********************************************************************************************************************
 * @def     PAL_INTERFACE_VERSION
 * @ingroup LibInit
 * @brief   Current PAL interface version packed into a 32-bit unsigned integer. The low 16 bits are always zero.
 *          They used to contain the interface minor version and remain as a placeholder in case we add it back.
 *
 * @see PAL_INTERFACE_MAJOR_VERSION
 *
 * @hideinitializer
 ***********************************************************************************************************************
 */
#define PAL_INTERFACE_VERSION (PAL_INTERFACE_MAJOR_VERSION << 16)

#if PAL_COMPILE_TYPE != 0
// Static asserts to ensure clients define PAL_CLIENT_INTERFACE_MAJOR_VERSION and that it falls in the supported range.
#ifndef PAL_CLIENT_INTERFACE_MAJOR_VERSION
    static_assert(false, "The client must link against 'palUtil' or 'pal' in CMake!");
#else
    static_assert((PAL_CLIENT_INTERFACE_MAJOR_VERSION >= PAL_MINIMUM_INTERFACE_MAJOR_VERSION) &&
                  (PAL_CLIENT_INTERFACE_MAJOR_VERSION <= PAL_INTERFACE_MAJOR_VERSION),
                  "The specified PAL_CLIENT_INTERFACE_MAJOR_VERSION is not supported.");
#endif
#endif

