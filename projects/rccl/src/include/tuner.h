/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_TUNER_H_
#define NCCL_INT_TUNER_H_

#include <map>
#include <string>

#include "nccl_tuner.h"
#include "comm.h"

// Tuning plugin to override NCCL's default algorithm/protocol tuning.

// Built-in CSV tuner - compiled into librccl.so (follows rocmNetIb pattern)
extern ncclTuner_t rcclCsvTuner;

// Find the CSV config source. Returns a file path, an "<embedded>/<name>" label
// for a config compiled into librccl.so, or nullptr if neither is available.
// gpuArch: GPU architecture string (e.g., "gfx950") for arch-specific config lookup
const char* rcclCsvTunerFindConfig(const char* gpuArch);

// CSV configs compiled in from the source tree's tuner/ directory, keyed by file name.
const std::map<std::string, std::string>& rcclCsvTunerEmbeddedConfigs();

// Reset CSV tuner config source discovery (for testing)
void rcclCsvTunerResetConfigPath();

// Attempts to load NCCL tuner from environmental variable.
// Returns ncclSuccess if the correct tuner symbol has been found and
// successully loaded.  Otherwise returns an error and also logs the error.
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm);

// Cleans up NCCL tuner plugin.
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm);
#endif
