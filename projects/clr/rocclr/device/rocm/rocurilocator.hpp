/* Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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
#include "device/devurilocator.hpp"
#include "rocrctx.hpp"
#include <vector>
namespace amd::roc {
class UriLocator : public device::UriLocator {
  bool init_ = false;
  struct UriRange {
    uint64_t startAddr_, endAddr_;
    int64_t elfDelta_;
    std::string Uri_;
  };
  std::vector<UriRange> rangeTab_;
  hsa_ven_amd_loader_1_03_pfn_t fn_table_;

  hsa_status_t createUriRangeTable();

 public:
  virtual ~UriLocator() {}
  virtual UriInfo lookUpUri(uint64_t device_pc) override;
  virtual std::pair<uint64_t, uint64_t> decodeUriAndGetFd(UriInfo& uri_path,
                                                          amd::Os::FileDesc* uri_fd) override;
};
}  // namespace amd::roc
#endif
#endif
