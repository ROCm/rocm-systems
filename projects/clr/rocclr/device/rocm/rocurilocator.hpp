/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
