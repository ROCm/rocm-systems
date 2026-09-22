/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Conversions between hipFile and cuFile types

#pragma once

#include "hipfile.h"

#include <cufile.h>

// cuFile to hipFile ////////////////////////////////////////////////////////

hipFileOpError_t toHipFileOpError(CUfileOpError cu_status);

hipFileError_t toHipFileError(const CUfileError_t &cu_status);

hipFileOpError_t toHipFileDriverStatusFlags(CUfileDriverStatusFlags_t   cu_flags,
                                            hipFileDriverStatusFlags_t *hip_flags);

hipFileOpError_t toHipFileDriverControlFlags(CUfileDriverControlFlags_t   cu_flags,
                                             hipFileDriverControlFlags_t *hip_flags);

hipFileOpError_t toHipFileFeatureFlags(CUfileFeatureFlags_t cu_flags, hipFileFeatureFlags_t *hip_flags);

hipFileOpError_t toHipFileFileHandleType(enum CUfileFileHandleType cu_type,
                                         hipFileFileHandleType_t  *hip_type);

hipFileDriverProps_t toHipFileDriverProps(const CUfileDrvProps_t &cu_props);

hipFileOpError_t toHipFileRDMAInfo(const cufileRDMAInfo_t *cu_info, hipFileRDMAInfo_t *hip_info);

hipFileOpError_t toHipFileDescr(const CUfileDescr_t *cu_fd, hipFileDescr_t *hip_fd);

hipFileOpError_t toHipFileOpcode(CUfileOpcode_t cu_opcode, hipFileOpcode_t *hip_opcode);

hipFileStatus_t toHipFileStatus(CUfileStatus_t cu_status);

hipFileOpError_t toHipFileBatchMode(CUfileBatchMode_t cu_mode, hipFileBatchMode_t *hip_mode);

hipFileOpError_t toHipFileIOParams(const CUfileIOParams_t *cu_params, hipFileIOParams_t *hip_params);

hipFileIOEvents_t toHipFileIOEvents(const CUfileIOEvents_t &cu_events);

// hipFile to cuFile ////////////////////////////////////////////////////////

CUfileOpError toCUfileOpError(hipFileOpError_t hip_status);

CUfileError_t toCUfileError(hipFileError_t hip_status);

hipFileOpError_t toCUfileDriverStatusFlags(hipFileDriverStatusFlags_t hip_flags,
                                           CUfileDriverStatusFlags_t *cu_flags);

hipFileOpError_t toCUfileDriverControlFlags(hipFileDriverControlFlags_t hip_flags,
                                            CUfileDriverControlFlags_t *cu_flags);

hipFileOpError_t toCUfileFeatureFlags(hipFileFeatureFlags_t hip_flags, CUfileFeatureFlags_t *cu_flags);

CUfileFileHandleType toCUfileFileHandleType(hipFileFileHandleType_t hip_type);

hipFileOpError_t toCUfileDrvProps(const hipFileDriverProps_t *hip_props, CUfileDrvProps_t *cu_props);

hipFileOpError_t toCufileRDMAInfo(const hipFileRDMAInfo_t *hip_info, cufileRDMAInfo_t *cu_info);

CUfileDescr_t toCUfileDescr(const hipFileDescr_t &hip_fd);

CUfileOpcode_t toCUfileOpcode(hipFileOpcode_t hip_opcode);

hipFileOpError_t toCUfileStatus(hipFileStatus_t hip_status, CUfileStatus_t *cu_status);

CUfileBatchMode_t toCUfileBatchMode(hipFileBatchMode_t hip_mode);

CUfileIOParams_t toCUfileIOParams(const hipFileIOParams_t &hip_params);

hipFileOpError_t toCUfileIOEvents(const hipFileIOEvents_t *hip_events, CUfileIOEvents_t *cu_events);

CUFileSizeTConfigParameter_t toCUFileSizeTConfigParameter(hipFileSizeTConfigParameter_t param);

CUFileBoolConfigParameter_t toCUFileBoolConfigParameter(hipFileBoolConfigParameter_t param);

CUFileStringConfigParameter_t toCUFileStringConfigParameter(hipFileStringConfigParameter_t param);
