/*
Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
