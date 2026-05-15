/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "top.hpp"
#include "utils/debug.hpp"
#include "device/appprofile.hpp"
#include "device/pal/palappprofile.hpp"

namespace amd::pal {

AppProfile::AppProfile()
    : amd::AppProfile(), enableHighPerformanceState_(true), reportAsOCL12Device_(false) {
  propertyDataMap_.insert(
      {"HighPerfState", PropertyData(DataType_Boolean, &enableHighPerformanceState_)});

  propertyDataMap_.insert({"OCL12Device", PropertyData(DataType_Boolean, &reportAsOCL12Device_)});
}
}  // namespace amd::pal
