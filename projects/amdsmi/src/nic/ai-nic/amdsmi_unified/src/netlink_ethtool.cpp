// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
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

int EthtoolNetlinkClient::build_header(NLMessage& msg, const std::string& iface, int header_attr,
                                       uint32_t flags) {
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

  if (flags != 0) {
    ret = attrs.put_u32(ETHTOOL_A_HEADER_FLAGS, flags);
    if (ret < 0) {
      return ret;
    }
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

  if (NLAttributes::is_present(tb[ETHTOOL_A_LINKMODES_SPEED])) {
    settings.speed = NLAttributes::get_u32(tb[ETHTOOL_A_LINKMODES_SPEED]);
  }

  if (NLAttributes::is_present(tb[ETHTOOL_A_LINKMODES_DUPLEX])) {
    settings.duplex = NLAttributes::get_u8(tb[ETHTOOL_A_LINKMODES_DUPLEX]);
  }

  /**
   * Autoneg is the only field a consumer reads over this path; a reply that
   * omits it is useless, so report failure to let AutoBackend try ioctl.
   */
  if (!NLAttributes::is_present(tb[ETHTOOL_A_LINKMODES_AUTONEG])) {
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

  if ((ret < 0) && !result.success) {
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
  if (!NLAttributes::is_present(tb[ETHTOOL_A_PAUSE_AUTONEG]) ||
      !NLAttributes::is_present(tb[ETHTOOL_A_PAUSE_RX]) ||
      !NLAttributes::is_present(tb[ETHTOOL_A_PAUSE_TX])) {
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

  if ((ret < 0) && !result.success) {
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

/**
 * FEC Statistics
 *
 * ETHTOOL_A_FEC_STAT_CORRECTED/_UNCORR/_CORR_BITS are each a packed array of
 * u64 (index 0 is the port total; further elements are per-lane, only present
 * on some drivers/kernels). This slice reads index 0 only. The 16-entry
 * per-lane symbol-error histogram (fec_symbols_err_0..15, kernel 6.18+) has no
 * netlink attribute in headers available to build against here
 * (verified: absent from /usr/include/linux/ethtool_netlink.h on this host,
 * kernel 6.8); the ionic StatTable_t lists those 16 names so they will
 * surface automatically once a future change adds the histogram attribute and
 * this kernel floor moves; until then they are silently absent, same as any
 * other table entry the driver does not expose.
 */
static int fec_stats_handler(struct nl_msg* msg, void* arg) {
  auto* ctx = static_cast<QueryContext<transport::FecStatistics_t>*>(arg);
  if (!ctx || !ctx->result) {
    return NL_SKIP;
  }

  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct nlattr* tb[ETHTOOL_A_FEC_MAX + 1];

  if (NLAttributes::parse(nlh, GENL_HDRLEN, tb, ETHTOOL_A_FEC_MAX, nullptr) < 0) {
    ctx->result->success = false;
    ctx->result->error_code = EINVAL;
    return NL_SKIP;
  }

  if (!NLAttributes::is_present(tb[ETHTOOL_A_FEC_STATS])) {
    ctx->result->success = false;
    ctx->result->error_code = ENODATA;
    return NL_SKIP;
  }

  struct nlattr* stat_tb[ETHTOOL_A_FEC_STAT_MAX + 1];
  if (NLAttributes::parse_nested(tb[ETHTOOL_A_FEC_STATS], stat_tb, ETHTOOL_A_FEC_STAT_MAX,
                                 nullptr) < 0) {
    ctx->result->success = false;
    ctx->result->error_code = EINVAL;
    return NL_SKIP;
  }

  transport::FecStatistics_t stats;
  auto add_stat = [&](int attr_id, const char* name) {
    if (!NLAttributes::is_present(stat_tb[attr_id])) {
      return;
    }
    auto values = NLAttributes::get_u64_array(stat_tb[attr_id]);
    if (values.empty()) {
      return;
    }
    stats.names.emplace_back(name);
    stats.values.push_back(values[0]);
  };

  add_stat(ETHTOOL_A_FEC_STAT_CORRECTED, "corrected_blocks");
  add_stat(ETHTOOL_A_FEC_STAT_UNCORR, "uncorrectable_blocks");
  add_stat(ETHTOOL_A_FEC_STAT_CORR_BITS, "corrected_bits");

  ctx->result->success = true;
  ctx->result->value = stats;
  ctx->result->error_code = 0;

  return NL_OK;
}

transport::Result<transport::FecStatistics_t> EthtoolNetlinkClient::get_fec_statistics(
    const std::string& iface) {
  transport::Result<transport::FecStatistics_t> result{false, {}, ENOTSUP};

  if (!initialized_) {
    result.error_code = ENOTCONN;
    return result;
  }

  QueryContext<transport::FecStatistics_t> ctx{&result, iface};

  auto build_fn = [this, &iface](NLMessage& msg) -> int {
    return build_header(msg, iface, ETHTOOL_A_FEC_HEADER, ETHTOOL_FLAG_STATS);
  };

  int ret = client_.query(family_id_, ETHTOOL_MSG_FEC_GET, 1, build_fn, fec_stats_handler, &ctx);

  if ((ret < 0) && !result.success) {
    result.error_code = -ret;
  }

  return result;
}

}  // namespace amd::nic::netlink
