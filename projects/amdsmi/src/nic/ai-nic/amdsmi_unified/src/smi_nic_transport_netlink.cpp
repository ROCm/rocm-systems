// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Netlink and auto-selecting transport backends.
 */

#include "smi_nic_transport.h"

#ifdef HAVE_LIBNL3

#include <cerrno>
#include <cstring>
#include <memory>
#include <string>

#include "netlink_ethtool.h"

namespace amd::smi::nic::transport {

/**
 * Ethtool netlink transport backend (kernel 5.6+). Reports unavailable if the
 * client fails to initialize; driver info, statistics, and permanent address
 * are not exposed over netlink.
 */
class NetlinkBackend : public NicTransport {
 public:
  NetlinkBackend() : client_(), initialized_(false) { initialized_ = (client_.init() == 0); }

  ~NetlinkBackend() override = default;

  NetlinkBackend(const NetlinkBackend&) = delete;
  NetlinkBackend& operator=(const NetlinkBackend&) = delete;
  NetlinkBackend(NetlinkBackend&&) = delete;
  NetlinkBackend& operator=(NetlinkBackend&&) = delete;

  Result<PauseParams> get_pause_params(const std::string& iface) override {
    if (!initialized_) {
      return {false, {}, ENOTSUP};
    }
    return client_.get_pause_params(iface);
  }

  Result<LinkSettings> get_link_settings(const std::string& iface) override {
    if (!initialized_) {
      return {false, {}, ENOTSUP};
    }
    return client_.get_link_settings(iface);
  }

  Result<DriverInfo> get_driver_info(const std::string& /* iface */) override {
    return {false, {}, ENOTSUP};
  }

  Result<VendorStatistics> get_statistics(const std::string& /* iface */) override {
    return {false, {}, ENOTSUP};
  }

  Result<PermanentAddress> get_permanent_address(const std::string& /* iface */) override {
    return {false, {}, ENOTSUP};
  }

  std::string backend_name() const override {
    return initialized_ ? "netlink" : "netlink (unavailable)";
  }

  bool is_available() const { return initialized_; }

 private:
  ::amd::nic::netlink::EthtoolNetlinkClient client_;
  bool initialized_;
};

namespace ioctl_internal {
extern std::shared_ptr<NicTransport> create_ioctl_backend();
}

/**
 * Prefers netlink for the operations it supports and falls back to ioctl for
 * everything else (and entirely when netlink is unavailable).
 */
class AutoBackend : public NicTransport {
 public:
  AutoBackend() : netlink_backend_(), ioctl_backend_(ioctl_internal::create_ioctl_backend()) {
    try {
      auto nl_backend = std::make_unique<NetlinkBackend>();
      if (nl_backend->is_available()) {
        netlink_backend_ = std::move(nl_backend);
      }
    } catch (...) {
      netlink_backend_ = nullptr;
    }
  }

  ~AutoBackend() override = default;

  AutoBackend(const AutoBackend&) = delete;
  AutoBackend& operator=(const AutoBackend&) = delete;
  AutoBackend(AutoBackend&&) = delete;
  AutoBackend& operator=(AutoBackend&&) = delete;

  Result<PauseParams> get_pause_params(const std::string& iface) override {
    if (netlink_backend_) {
      auto result = netlink_backend_->get_pause_params(iface);
      if (result.success) {
        return result;
      }
    }
    return ioctl_backend_->get_pause_params(iface);
  }

  Result<LinkSettings> get_link_settings(const std::string& iface) override {
    if (netlink_backend_) {
      auto result = netlink_backend_->get_link_settings(iface);
      if (result.success) {
        return result;
      }
    }
    return ioctl_backend_->get_link_settings(iface);
  }

  Result<DriverInfo> get_driver_info(const std::string& iface) override {
    return ioctl_backend_->get_driver_info(iface);
  }

  Result<VendorStatistics> get_statistics(const std::string& iface) override {
    return ioctl_backend_->get_statistics(iface);
  }

  Result<PermanentAddress> get_permanent_address(const std::string& iface) override {
    return ioctl_backend_->get_permanent_address(iface);
  }

  std::string backend_name() const override {
    return netlink_backend_ ? "auto (netlink preferred, ioctl fallback)" : "auto (ioctl only)";
  }

 private:
  std::unique_ptr<NetlinkBackend> netlink_backend_;
  std::shared_ptr<NicTransport> ioctl_backend_;
};

namespace internal {

std::shared_ptr<NicTransport> create_netlink_backend() {
  return std::make_shared<NetlinkBackend>();
}

std::shared_ptr<NicTransport> create_auto_backend() { return std::make_shared<AutoBackend>(); }

}  // namespace internal

}  // namespace amd::smi::nic::transport

#endif  // HAVE_LIBNL3
