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
#include <mutex>
#include <vector>
namespace amd::roc {
class UriLocator : public device::UriLocator {
  struct UriRange {
    uint64_t startAddr_, endAddr_;
    int64_t elfDelta_;
    std::string Uri_;
  };

  // Records code object ranges at load time so a leaked PC can still be resolved at
  // teardown after its code object is unloaded. Never destroyed (kept reachable) so it
  // survives the teardown scan, which runs from a static destructor.
  static std::vector<UriRange>& table();
  static std::mutex& tableMutex();

 public:
  virtual ~UriLocator() {}
  static void recordCodeObjects(hsa_executable_t exec);
  virtual UriInfo lookUpUri(uint64_t device_pc) override;
  virtual std::pair<uint64_t, uint64_t> decodeUriAndGetFd(UriInfo& uri_path,
                                                          amd::Os::FileDesc* uri_fd) override;
};
}  // namespace amd::roc
#endif
#endif
