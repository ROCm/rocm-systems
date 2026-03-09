/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "os/os.hpp"
#include <string>
#include <utility>
namespace amd::device {
// Interface for HSA/PAL Uri Locators
class UriLocator {
 public:
  struct UriInfo {
    std::string uriPath;
    int64_t loadAddressDiff;
  };

  virtual ~UriLocator() {}
  virtual UriInfo lookUpUri(uint64_t device_pc) = 0;
  virtual std::pair<uint64_t, uint64_t> decodeUriAndGetFd(UriInfo& uri,
                                                          amd::Os::FileDesc* uri_fd) = 0;
};
}  // namespace amd::device
#endif
#endif
