/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Common hipFile test functionality

#include "hipfile.h"
#include "magic-word.h"
#include "mhip.h"
#include "mmountinfo.h"
#include "msys.h"

#include <array>
#include <cassert>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <optional>

// ***********************************************************************
//  ERRORS AND ERROR HANDLING
// ***********************************************************************

// Set a particular hipFile error
constexpr hipFileError_t
HipFileHipError(hipError_t err)
{
    return {hipFileHipDriverError, err};
}

// Set a particular HIP error
constexpr hipFileError_t
HipFileOpError(hipFileOpError_t err)
{
    return {err, hipSuccess};
}

// == overload for hipFileError_t values
inline bool
operator==(const hipFileError_t &lhs, const hipFileError_t &rhs)
{
    return lhs.err == rhs.err && lhs.hip_drv_err == rhs.hip_drv_err;
}

// != overload for hipFileError_t values
inline bool
operator!=(const hipFileError_t &lhs, const hipFileError_t &rhs)
{
    return lhs.err != rhs.err || lhs.hip_drv_err != rhs.hip_drv_err;
}

// << overload for hipFileError_t values
//
// Unused in the test code, but kept here for iostream debugging
#include <ostream>
inline std::ostream &
operator<<(std::ostream &os, const hipFileError_t &rfe)
{
    return os << "hipFileError_t{ err: " << rfe.err << ", hip_drv_err: " << rfe.hip_drv_err << " }";
}

// Convenience "success" value
inline constexpr hipFileError_t HIPFILE_SUCCESS{hipFileSuccess, hipSuccess};

// Convenience "invalid argument" value
inline constexpr hipFileError_t HIPFILE_INVALID_VALUE{hipFileInvalidValue, hipSuccess};

// ***********************************************************************
//  BASE ERROR CLASSES
// ***********************************************************************

// Base class for tests that open the driver
struct HipFileOpened : public ::testing::Test {

    HipFileOpened()
    {
        assert(hipFileUseCount() == 0);
        assert(hipFileDriverOpen() == HIPFILE_SUCCESS);
    }

    virtual ~HipFileOpened() override
    {
        while (hipFileUseCount()) {
            assert(hipFileDriverClose() == HIPFILE_SUCCESS);
        }
        assert(hipFileUseCount() == 0);
    }
};

// Base class for tests that do NOT open the driver
struct HipFileUnopened : public ::testing::Test {

    HipFileUnopened()
    {
        assert(hipFileUseCount() == 0);
    }

    virtual ~HipFileUnopened() override
    {
        while (hipFileUseCount()) {
            assert(hipFileDriverClose() == HIPFILE_SUCCESS);
        }
        assert(hipFileUseCount() == 0);
    }
};

// ***********************************************************************
//  BUFFER FUNCTIONALITY
// ***********************************************************************

/// @brief Set up mocks for buffer registration
void expect_buffer_registration(hipFile::MHip &mhip, hipMemoryType memory_type);

// ***********************************************************************
//  FILE FUNCTIONALITY
// ***********************************************************************

/// @brief Setup mocks for file registration
///
/// Mock methods will return default values
void expect_file_registration(hipFile::MSys &msys, hipFile::MLibMountHelper &mlibmounthelper);

/// @brief Setup mocks for file registration
///
/// Mock methods will return the specified values
void expect_file_registration(hipFile::MSys &msys, hipFile::MLibMountHelper &mlibmounthelper,
                              struct statx stxbuf, int fcntl_flags, hipFile::MountInfo mountinfo);

// ***********************************************************************
//  ENUM VALUE HELPERS
// ***********************************************************************

constexpr std::array<hipMemoryType, 1> SupportedHipMemoryTypes{hipMemoryTypeDevice};
constexpr std::array<hipMemoryType, 5> UnsupportedHipMemoryTypes{
    hipMemoryTypeArray,   hipMemoryTypeHost,         hipMemoryTypeManaged,
    hipMemoryTypeUnified, hipMemoryTypeUnregistered,
};
