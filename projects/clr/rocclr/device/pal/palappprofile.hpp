/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>
#include <map>

namespace amd::pal {

class AppProfile : public amd::AppProfile {
 public:
  AppProfile();

  //! return the value of enableHighPerformanceState_
  bool enableHighPerformanceState() const { return enableHighPerformanceState_; }
  bool reportAsOCL12Device() const { return reportAsOCL12Device_; }

 private:
  bool enableHighPerformanceState_;
  bool reportAsOCL12Device_;
};
}  // namespace amd::pal
