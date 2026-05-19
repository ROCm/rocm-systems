/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_TUNER_H_
#define NCCL_INT_TUNER_H_

#include "nccl_tuner.h"
#include "comm.h"

// Tuning plugin to override NCCL's default algorithm/protocol tuning.

// Built-in CSV tuner - compiled into librccl.so (follows rocmNetIb pattern)
extern ncclTuner_t rcclCsvTuner;

// Find CSV config file path. Returns path if found, nullptr if not.
// gpuArch: GPU architecture string (e.g., "gfx950") for arch-specific config lookup
const char* rcclCsvTunerFindConfig(const char* gpuArch);

// Reset CSV tuner config path discovery (for testing)
void rcclCsvTunerResetConfigPath();

// Attempts to load NCCL tuner from environmental variable.
// Returns ncclSuccess if the correct tuner symbol has been found and
// successully loaded.  Otherwise returns an error and also logs the error.
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm);

// Cleans up NCCL tuner plugin.
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm);
#endif
