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

#ifndef APPPROFILE_HPP_
#define APPPROFILE_HPP_

#include <unordered_map>
#include <string>

namespace amd {

class AppProfile {
 public:
  AppProfile();
  virtual ~AppProfile();

  bool init();

  const std::string& GetBuildOptsAppend() const { return buildOptsAppend_; }

  const std::string& appFileName() const { return appFileName_; }
  const std::wstring& wsAppPathAndFileName() const { return wsAppPathAndFileName_; }

 protected:
  enum DataTypes {
    DataType_Unknown = 0,
    DataType_Boolean,
    DataType_String,
  };

  struct PropertyData {
    PropertyData(DataTypes type, void* data) : type_(type), data_(data) {}
    DataTypes type_;  //!< Data type
    void* data_;      //!< Pointer to the data
  };

  typedef std::unordered_map<std::string, PropertyData> DataMap;

  DataMap propertyDataMap_;
  std::string appFileName_;  // without extension
  std::wstring wsAppFileName_;
  std::string appPathAndFileName_;  // with path and extension
  std::wstring wsAppPathAndFileName_;

  virtual bool ParseApplicationProfile();

  bool gpuvmHighAddr_;                // Currently not used.
  bool profileOverridesAllSettings_;  // Overrides hint flags and env.var.
  std::string buildOptsAppend_;
};
}  // namespace amd
#endif
