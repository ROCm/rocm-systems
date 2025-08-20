/* Copyright (c) 2014 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
