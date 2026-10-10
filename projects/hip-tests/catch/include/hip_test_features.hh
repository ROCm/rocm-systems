/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>

bool getGenericTarget(const std::string& agentTarget, std::string& genericTarget);
bool isGenericTargetSupported(char* gcnArchName = nullptr, int deviceId = 0);
