/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "top.hpp"
#include "os/os.hpp"
#include "utils/flags.hpp"
#include "appprofile.hpp"
#include <cstdlib>
#include <cstring>


#define GETPROCADDRESS(_adltype_, _adlfunc_) (_adltype_) amd::Os::getSymbol(adlHandle_, #_adlfunc_);

namespace amd {

AppProfile::AppProfile() : gpuvmHighAddr_(false), profileOverridesAllSettings_(false) {
  amd::Os::getAppPathAndFileName(appFileName_, appPathAndFileName_);
  propertyDataMap_.insert(
      DataMap::value_type("BuildOptsAppend", PropertyData(DataType_String, &buildOptsAppend_)));
}

AppProfile::~AppProfile() {}

bool AppProfile::init() {
  if (appFileName_.empty()) {
    return false;
  }

  // Convert appName to wide char for X2_Search ADL interface
  size_t strLength = appFileName_.length() + 1;
  size_t strPathLength = appPathAndFileName_.length() + 1;
  wchar_t* appName = new wchar_t[strPathLength];

  size_t success = mbstowcs(appName, appFileName_.c_str(), strLength);
  if (success > 0) {
    // mbstowcs was able to convert to wide character successfully.
    appName[strLength - 1] = L'\0';
  }

  wsAppFileName_ = appName;

  success = mbstowcs(appName, appPathAndFileName_.c_str(), strPathLength);
  if (success > 0) {
    // mbstowcs was able to convert to wide character successfully.
    appName[strPathLength - 1] = L'\0';
  }

  wsAppPathAndFileName_ = appName;

  delete[] appName;

  ParseApplicationProfile();

  return true;
}

bool AppProfile::ParseApplicationProfile() { return true; }
}  // namespace amd
