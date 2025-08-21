/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

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
