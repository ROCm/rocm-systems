/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_PROF_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_PROF_H

// HIP ROCclr Op IDs enumeration
enum HipVdiOpId {
  kHipVdiOpIdDispatch = 0,
  kHipVdiOpIdCopy = 1,
  kHipVdiOpIdBarrier = 2,
  kHipVdiOpIdNumber = 3
};

// Types of ROCclr commands
enum HipVdiCommandKind {
  kHipVdiCommandKernel = 0x11F0,
  kHipVdiCommandTask = 0x11F1,
  kHipVdiMemcpyDeviceToHost = 0x11F3,
  kHipHipVdiMemcpyHostToDevice = 0x11F4,
  kHipVdiMemcpyDeviceToDevice = 0x11F5,
  kHipVidMemcpyDeviceToHostRect = 0x1201,
  kHipVdiMemcpyHostToDeviceRect = 0x1202,
  kHipVdiMemcpyDeviceToDeviceRect = 0x1203,
  kHipVdiFillMemory = 0x1207,
};

/**
 * @brief Initializes activity callback
 *
 * @param [input] id_callback Event ID callback function
 * @param [input] op_callback Event operation callback function
 * @param [input] arg         Arguments passed into callback
 *
 * @returns None
 */
void hipInitActivityCallback(void* id_callback, void* op_callback, void* arg);

/**
 * @brief Enables activity callback
 *
 * @param [input] op      Operation, which will trigger a callback (@see HipVdiOpId)
 * @param [input] enable  Enable state for the callback
 *
 * @returns True if successful
 */
bool hipEnableActivityCallback(uint32_t op, bool enable);

/**
 * @brief Returns the description string for the operation kind
 *
 * @param [input] id      Command kind id (@see HipVdiCommandKind)
 *
 * @returns A pointer to a const string with the command description
 */
const char* hipGetCmdName(uint32_t id);

#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_PROF_H
