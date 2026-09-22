/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
