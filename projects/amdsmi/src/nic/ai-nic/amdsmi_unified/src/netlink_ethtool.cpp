// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Ethtool netlink client implementation.
 */

#include "netlink_ethtool.h"

#include <linux/ethtool_netlink.h>
#include <netlink/errno.h>

#include <cerrno>
#include <cstring>

namespace amd::nic::netlink {

namespace transport = ::amd::smi::nic::transport;

template <typename T>
struct QueryContext {
  transport::Result<T>* result;
  std::string iface;
};

EthtoolNetlinkClient::EthtoolNetlinkClient() : client_(), family_id_(-1), initialized_(false) {}

int EthtoolNetlinkClient::init() {
  if (initialized_) {
    return 0;
  }

  int ret = client_.connect();
  if (ret < 0) {
    return ret;
  }

  auto family = client_.resolve_family_id("ethtool");
  if (!family.has_value()) {
    return -ENOENT;
  }

  family_id_ = family.value();
  initialized_ = true;
  return 0;
}

int EthtoolNetlinkClient::build_header(NLMessage& msg, const std::string& iface, int header_attr) {
  NLAttributes attrs(msg);

  /**
   * Every ethtool request carries its device name inside the per-message
   * header nest (ETHTOOL_A_*_HEADER); the kernel rejects a request whose
   * header attribute id does not match the message type.
   */
  auto* header = attrs.nest_start(header_attr);
  if (!header) {
    return -NLE_NOMEM;
  }

  int ret = attrs.put_string(ETHTOOL_A_HEADER_DEV_NAME, iface);
  if (ret < 0) {
    return ret;
  }

  attrs.nest_end(header);
  return 0;
}

/**
 * Link Settings
 */
static int link_settings_handler(struct nl_msg* msg, void* arg) {
  auto* ctx = static_cast<QueryContext<transport::LinkSettings>*>(arg);
  if (!ctx || !ctx->result) {
    return NL_SKIP;
  }

  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct nlattr* tb[ETHTOOL_A_LINKMODES_MAX + 1];

  if (NLAttributes::parse(nlh, GENL_HDRLEN, tb, ETHTOOL_A_LINKMODES_MAX, nullptr) < 0) {
    ctx->result->success = false;
    ctx->result->error_code = EINVAL;
    return NL_SKIP;
  }

  transport::LinkSettings settings{};

  if (NLAttributes::exists(tb[ETHTOOL_A_LINKMODES_SPEED])) {
    settings.speed = NLAttributes::get_u32(tb[ETHTOOL_A_LINKMODES_SPEED]);
  }

  if (NLAttributes::exists(tb[ETHTOOL_A_LINKMODES_DUPLEX])) {
    settings.duplex = NLAttributes::get_u8(tb[ETHTOOL_A_LINKMODES_DUPLEX]);
  }

  /**
   * Autoneg is the only field a consumer reads over this path; a reply that
   * omits it is useless, so report failure to let AutoBackend try ioctl.
   */
  if (!NLAttributes::exists(tb[ETHTOOL_A_LINKMODES_AUTONEG])) {
    ctx->result->success = false;
    ctx->result->error_code = ENODATA;
    return NL_SKIP;
  }
  settings.autoneg = NLAttributes::get_u8(tb[ETHTOOL_A_LINKMODES_AUTONEG]);

  ctx->result->success = true;
  ctx->result->value = settings;
  ctx->result->error_code = 0;

  return NL_OK;
}

transport::Result<transport::LinkSettings> EthtoolNetlinkClient::get_link_settings(
    const std::string& iface) {
  transport::Result<transport::LinkSettings> result{false, {}, ENOTSUP};

  if (!initialized_) {
    result.error_code = ENOTCONN;
    return result;
  }

  QueryContext<transport::LinkSettings> ctx{&result, iface};

  auto build_fn = [this, &iface](NLMessage& msg) -> int {
    return build_header(msg, iface, ETHTOOL_A_LINKMODES_HEADER);
  };

  int ret = client_.query(family_id_, ETHTOOL_MSG_LINKMODES_GET, 1, build_fn, link_settings_handler,
                          &ctx);

  if (ret < 0 && !result.success) {
    result.error_code = -ret;
  }

  return result;
}

/**
 * Pause Parameters
 */
static int pause_handler(struct nl_msg* msg, void* arg) {
  auto* ctx = static_cast<QueryContext<transport::PauseParams>*>(arg);
  if (!ctx || !ctx->result) {
    return NL_SKIP;
  }

  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct nlattr* tb[ETHTOOL_A_PAUSE_MAX + 1];

  if (NLAttributes::parse(nlh, GENL_HDRLEN, tb, ETHTOOL_A_PAUSE_MAX, nullptr) < 0) {
    ctx->result->success = false;
    ctx->result->error_code = EINVAL;
    return NL_SKIP;
  }

  transport::PauseParams pause{};

  /**
   * All three fields are surfaced together; a reply missing any of them would
   * silently report a default, so fall back to ioctl unless all are present.
   */
  if (!NLAttributes::exists(tb[ETHTOOL_A_PAUSE_AUTONEG]) ||
      !NLAttributes::exists(tb[ETHTOOL_A_PAUSE_RX]) ||
      !NLAttributes::exists(tb[ETHTOOL_A_PAUSE_TX])) {
    ctx->result->success = false;
    ctx->result->error_code = ENODATA;
    return NL_SKIP;
  }
  pause.autoneg = NLAttributes::get_u8(tb[ETHTOOL_A_PAUSE_AUTONEG]) != 0;
  pause.rx_pause = NLAttributes::get_u8(tb[ETHTOOL_A_PAUSE_RX]) != 0;
  pause.tx_pause = NLAttributes::get_u8(tb[ETHTOOL_A_PAUSE_TX]) != 0;

  ctx->result->success = true;
  ctx->result->value = pause;
  ctx->result->error_code = 0;

  return NL_OK;
}

transport::Result<transport::PauseParams> EthtoolNetlinkClient::get_pause_params(
    const std::string& iface) {
  transport::Result<transport::PauseParams> result{false, {}, ENOTSUP};

  if (!initialized_) {
    result.error_code = ENOTCONN;
    return result;
  }

  QueryContext<transport::PauseParams> ctx{&result, iface};

  auto build_fn = [this, &iface](NLMessage& msg) -> int {
    return build_header(msg, iface, ETHTOOL_A_PAUSE_HEADER);
  };

  int ret = client_.query(family_id_, ETHTOOL_MSG_PAUSE_GET, 1, build_fn, pause_handler, &ctx);

  if (ret < 0 && !result.success) {
    result.error_code = -ret;
  }

  return result;
}

/**
 * Driver info and statistics are not implemented over netlink; callers fall
 * back to the ioctl backend for these.
 */
transport::Result<transport::DriverInfo> EthtoolNetlinkClient::get_driver_info(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

transport::Result<transport::VendorStatistics> EthtoolNetlinkClient::get_statistics(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

}  // namespace amd::nic::netlink
