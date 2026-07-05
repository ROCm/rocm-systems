// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_status.h
/// @brief Status codes returned by the rocjitsu C API.

#ifndef ROCJITSU_BASE_RJ_STATUS_H_
#define ROCJITSU_BASE_RJ_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup base
/// @{

/// @brief Status codes for rocjitsu API functions.
typedef enum rj_status_e {
  /// @brief Operation completed successfully.
  ROCJITSU_STATUS_SUCCESS = 0,
  /// @brief Unspecified error.
  ROCJITSU_STATUS_ERROR = 1,
  /// @brief One or more arguments are invalid.
  ROCJITSU_STATUS_INVALID_ARGUMENT = 2,
  /// @brief Insufficient resources to complete the operation.
  ROCJITSU_STATUS_OUT_OF_RESOURCES = 3,
  /// @brief The supplied code object is malformed or unsupported.
  ROCJITSU_STATUS_INVALID_CODE_OBJECT = 4,
  /// @brief A required file could not be opened or read.
  ROCJITSU_STATUS_INVALID_FILE = 5
} rj_status_t;

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_BASE_RJ_STATUS_H_
