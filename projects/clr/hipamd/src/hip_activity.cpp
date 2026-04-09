/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/activity.hpp"
#include <hip/hip_runtime_api.h>

extern "C" const char* hipGetCmdName(unsigned op) {
  return amd::activity_prof::getOclCommandKindString(static_cast<cl_command_type>(op));
}