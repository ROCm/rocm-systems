/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "top.hpp"
#include "device/device.hpp"
#include "device/appprofile.hpp"
#include "device/rocm/rocappprofile.hpp"

#include <algorithm>

amd::AppProfile* rocCreateAppProfile() {
  amd::AppProfile* appProfile = new amd::roc::AppProfile;

  if ((appProfile == nullptr) || !appProfile->init()) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_INIT,
             "App Profile init failed, appProfile: 0x%x \n", appProfile);
    return nullptr;
  }

  return appProfile;
}

namespace amd::roc {

bool AppProfile::ParseApplicationProfile() {
  std::string appName("Explorer");

  std::transform(appName.begin(), appName.end(), appName.begin(), ::tolower);
  std::transform(appFileName_.begin(), appFileName_.end(), appFileName_.begin(), ::tolower);

  if (appFileName_.compare(appName) == 0) {
    gpuvmHighAddr_ = false;
    profileOverridesAllSettings_ = true;
  }

  return true;
}
}  // namespace amd::roc
