/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace amd::roc {

class AppProfile : public amd::AppProfile {
 public:
  AppProfile() : amd::AppProfile() {}

 protected:
  //! parse application profile based on application file name
  virtual bool ParseApplicationProfile();
};
}  // namespace amd::roc
