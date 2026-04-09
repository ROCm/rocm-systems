/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include <unordered_set>

// Catch Test Features
typedef enum CTFeatures {
  CT_FEATURE_FINEGRAIN_HWSUPPORT    = 0x0, // FINEGRAIN Supported Hardware.
  CT_FEATURE_HMM                    = 0x1, // HMM Enabled
  CT_FEATURE_TEXTURES_NOT_SUPPORTED = 0x2, // Textures not supported
  CT_FEATURE_LAST                   = 0x3
} CTFeatures;

bool CheckIfFeatSupported(enum CTFeatures test_feat, std::string gcn_arch);
bool getGenericTarget(const std::string& agentTarget, std::string& genericTarget);
bool isGenericTargetSupported(char* gcnArchName = nullptr, int deviceId = 0);
