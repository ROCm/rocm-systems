/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * This translation unit is compiled only when libnl-3 is available; the build
 * excludes *netlink*.cpp otherwise (see the amdsmi_unified CMakeLists).
 */

#include "smi_devlink_netlink.h"

#include <cerrno>

#ifdef HAVE_LIBNL3

#include <linux/devlink.h>
#include <netlink/errno.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace amd::nic::netlink {

namespace {

// State threaded through the health-reporter dump callback.
struct HealthDumpContext {
  std::string dev;                    // target devlink device (PCI BDF)
  std::vector<DevlinkReporter>* out;  // reporters accumulated across messages
};

/**
 * devlink encodes the reporter state as a u8: 0 == healthy, non-zero == error.
 * The named enum (DEVLINK_HEALTH_REPORTER_STATE_*) is absent from some UAPI
 * headers, so compare against the literal instead.
 */
constexpr uint8_t kStateHealthy = 0;

/**
 * Invoked once per reporter across the dump. Filters to the target device and
 * appends each of its reporters to the context vector.
 */
int health_dump_handler(struct nl_msg* msg, void* arg) {
  auto* ctx = static_cast<HealthDumpContext*>(arg);
  if (!ctx || !ctx->out) {
    return NL_SKIP;
  }

  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct nlattr* tb[DEVLINK_ATTR_MAX + 1];
  if (NLAttributes::parse(nlh, GENL_HDRLEN, tb, DEVLINK_ATTR_MAX, nullptr) < 0) {
    return NL_SKIP;
  }

  // A dump can span multiple devices; keep only the requested one.
  if (!NLAttributes::exists(tb[DEVLINK_ATTR_DEV_NAME]) ||
      ctx->dev != NLAttributes::get_string(tb[DEVLINK_ATTR_DEV_NAME])) {
    return NL_SKIP;
  }

  struct nlattr* rep = tb[DEVLINK_ATTR_HEALTH_REPORTER];
  if (!NLAttributes::exists(rep)) {
    return NL_SKIP;
  }

  // The reporter's fields live in a nested attribute.
  struct nlattr* rtb[DEVLINK_ATTR_MAX + 1];
  if (nla_parse_nested(rtb, DEVLINK_ATTR_MAX, rep, nullptr) < 0 ||
      !NLAttributes::exists(rtb[DEVLINK_ATTR_HEALTH_REPORTER_NAME])) {
    return NL_SKIP;
  }

  DevlinkReporter out{};
  std::snprintf(out.name, sizeof(out.name), "%s",
                NLAttributes::get_string(rtb[DEVLINK_ATTR_HEALTH_REPORTER_NAME]));

  out.healthy = 1;
  if (NLAttributes::exists(rtb[DEVLINK_ATTR_HEALTH_REPORTER_STATE])) {
    out.healthy =
        NLAttributes::get_u8(rtb[DEVLINK_ATTR_HEALTH_REPORTER_STATE]) == kStateHealthy ? 1 : 0;
  }

  out.error_count = 0;
  if (NLAttributes::exists(rtb[DEVLINK_ATTR_HEALTH_REPORTER_ERR_COUNT])) {
    const uint64_t errs = nla_get_u64(rtb[DEVLINK_ATTR_HEALTH_REPORTER_ERR_COUNT]);
    out.error_count = errs > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(errs);
  }

  ctx->out->push_back(out);
  return NL_OK;
}

}  // namespace

int DevlinkNetlinkClient::init() {
  if (initialized_) {
    return 0;
  }
  int ret = client_.connect();
  if (ret < 0) {
    return ret;
  }
  auto family = client_.resolve_family_id(DEVLINK_GENL_NAME);
  if (!family.has_value()) {
    return -NLE_OBJ_NOTFOUND;
  }
  family_id_ = family.value();
  initialized_ = true;
  return 0;
}

transport::Result<std::vector<DevlinkReporter>> DevlinkNetlinkClient::get_health_reporters(
    const std::string& dev) {
  transport::Result<std::vector<DevlinkReporter>> result{false, {}, ENOTSUP};

  int ret = init();
  if (ret < 0) {
    result.error_code = -ret;
    return result;
  }

  std::vector<DevlinkReporter> reporters;
  HealthDumpContext ctx{dev, &reporters};

  auto build_fn = [&dev](NLMessage& msg) -> int {
    NLAttributes attrs(msg);
    int r = attrs.put_string(DEVLINK_ATTR_BUS_NAME, "pci");
    if (r < 0) {
      return r;
    }
    return attrs.put_string(DEVLINK_ATTR_DEV_NAME, dev);
  };

  ret = client_.query(family_id_, DEVLINK_CMD_HEALTH_REPORTER_GET, DEVLINK_GENL_VERSION, build_fn,
                      health_dump_handler, &ctx, NLM_F_REQUEST | NLM_F_DUMP);
  if (ret < 0) {
    result.error_code = -ret;
    return result;
  }

  result.success = true;
  result.value = std::move(reporters);
  result.error_code = 0;
  return result;
}

namespace {

/**
 * State threaded through the port dump callback. A device may list several ports
 * (physical/SF/VF); split state is a property of the physical port, so prefer
 * the first physical-flavour port and fall back to the first port otherwise.
 */
struct PortDumpContext {
  std::string dev;  // target devlink device (PCI BDF)
  bool found_physical = false;
  bool found_any = false;
  DevlinkPortSplit split{};
};

int port_dump_handler(struct nl_msg* msg, void* arg) {
  auto* ctx = static_cast<PortDumpContext*>(arg);
  if (!ctx) {
    return NL_SKIP;
  }
  if (ctx->found_physical) {
    return NL_OK;  // already have the port we want
  }

  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct nlattr* tb[DEVLINK_ATTR_MAX + 1];
  if (NLAttributes::parse(nlh, GENL_HDRLEN, tb, DEVLINK_ATTR_MAX, nullptr) < 0) {
    return NL_SKIP;
  }

  if (!NLAttributes::exists(tb[DEVLINK_ATTR_DEV_NAME]) ||
      ctx->dev != NLAttributes::get_string(tb[DEVLINK_ATTR_DEV_NAME])) {
    return NL_SKIP;
  }

  const bool physical =
      NLAttributes::exists(tb[DEVLINK_ATTR_PORT_FLAVOUR]) &&
      NLAttributes::get_u16(tb[DEVLINK_ATTR_PORT_FLAVOUR]) == DEVLINK_PORT_FLAVOUR_PHYSICAL;
  if (!physical && ctx->found_any) {
    return NL_OK;  // keep the first non-physical port only until a physical one appears
  }

  DevlinkPortSplit split{};
  if (NLAttributes::exists(tb[DEVLINK_ATTR_PORT_SPLITTABLE])) {
    split.splittable = NLAttributes::get_u8(tb[DEVLINK_ATTR_PORT_SPLITTABLE]);
  }
  if (NLAttributes::exists(tb[DEVLINK_ATTR_PORT_SPLIT_COUNT])) {
    split.split_count = NLAttributes::get_u32(tb[DEVLINK_ATTR_PORT_SPLIT_COUNT]);
  }

  ctx->split = split;
  ctx->found_any = true;
  if (physical) {
    ctx->found_physical = true;
  }
  return NL_OK;
}

}  // namespace

transport::Result<DevlinkPortSplit> DevlinkNetlinkClient::get_port_split(const std::string& dev) {
  transport::Result<DevlinkPortSplit> result{false, {}, ENOTSUP};

  int ret = init();
  if (ret < 0) {
    result.error_code = -ret;
    return result;
  }

  PortDumpContext ctx;
  ctx.dev = dev;

  auto build_fn = [&dev](NLMessage& msg) -> int {
    NLAttributes attrs(msg);
    int r = attrs.put_string(DEVLINK_ATTR_BUS_NAME, "pci");
    if (r < 0) {
      return r;
    }
    return attrs.put_string(DEVLINK_ATTR_DEV_NAME, dev);
  };

  ret = client_.query(family_id_, DEVLINK_CMD_PORT_GET, DEVLINK_GENL_VERSION, build_fn,
                      port_dump_handler, &ctx, NLM_F_REQUEST | NLM_F_DUMP);
  if (ret < 0) {
    result.error_code = -ret;
    return result;
  }
  if (!ctx.found_any) {
    return result;  // device exposes no port object (e.g. pds_core): ENOTSUP
  }

  result.success = true;
  result.value = ctx.split;
  result.error_code = 0;
  return result;
}

namespace {

// State threaded through the device-info dump callback.
struct DeviceInfoContext {
  std::string dev;         // target devlink device (PCI BDF)
  DevlinkDeviceInfo* out;  // populated in place
  bool found = false;      // set once the target device's info is seen
};

// Parses one nested version attr ({NAME, VALUE}) and appends it, tagged with its
// class. Silently drops malformed entries and anything past the fixed capacity.
void append_version(struct nlattr* version_attr, DevlinkVersionType type, DevlinkDeviceInfo* out) {
  if (out->version_count >= kMaxDevlinkVersions) {
    return;
  }
  struct nlattr* vtb[DEVLINK_ATTR_MAX + 1];
  if (nla_parse_nested(vtb, DEVLINK_ATTR_MAX, version_attr, nullptr) < 0 ||
      !NLAttributes::exists(vtb[DEVLINK_ATTR_INFO_VERSION_NAME]) ||
      !NLAttributes::exists(vtb[DEVLINK_ATTR_INFO_VERSION_VALUE])) {
    return;
  }
  DevlinkVersion& v = out->versions[out->version_count];
  v.type = static_cast<uint8_t>(type);
  std::snprintf(v.name, sizeof(v.name), "%s",
                NLAttributes::get_string(vtb[DEVLINK_ATTR_INFO_VERSION_NAME]));
  std::snprintf(v.value, sizeof(v.value), "%s",
                NLAttributes::get_string(vtb[DEVLINK_ATTR_INFO_VERSION_VALUE]));
  ++out->version_count;
}

int device_info_handler(struct nl_msg* msg, void* arg) {
  auto* ctx = static_cast<DeviceInfoContext*>(arg);
  if (!ctx || !ctx->out) {
    return NL_SKIP;
  }

  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct nlattr* tb[DEVLINK_ATTR_MAX + 1];
  if (NLAttributes::parse(nlh, GENL_HDRLEN, tb, DEVLINK_ATTR_MAX, nullptr) < 0) {
    return NL_SKIP;
  }

  // A dump can span multiple devices; keep only the requested one.
  if (!NLAttributes::exists(tb[DEVLINK_ATTR_DEV_NAME]) ||
      ctx->dev != NLAttributes::get_string(tb[DEVLINK_ATTR_DEV_NAME])) {
    return NL_SKIP;
  }

  DevlinkDeviceInfo& info = *ctx->out;
  if (NLAttributes::exists(tb[DEVLINK_ATTR_INFO_DRIVER_NAME])) {
    std::snprintf(info.driver_name, sizeof(info.driver_name), "%s",
                  NLAttributes::get_string(tb[DEVLINK_ATTR_INFO_DRIVER_NAME]));
  }
  if (NLAttributes::exists(tb[DEVLINK_ATTR_INFO_SERIAL_NUMBER])) {
    std::snprintf(info.serial_number, sizeof(info.serial_number), "%s",
                  NLAttributes::get_string(tb[DEVLINK_ATTR_INFO_SERIAL_NUMBER]));
  }
  if (NLAttributes::exists(tb[DEVLINK_ATTR_INFO_BOARD_SERIAL_NUMBER])) {
    std::snprintf(info.board_serial_number, sizeof(info.board_serial_number), "%s",
                  NLAttributes::get_string(tb[DEVLINK_ATTR_INFO_BOARD_SERIAL_NUMBER]));
  }

  // Version attributes repeat (one nested block per version); a flat parse()
  // index keeps only the last of each type, so walk the raw stream instead.
  struct nlattr* nla;
  int rem;
  nlmsg_for_each_attr(nla, nlh, GENL_HDRLEN, rem) {
    switch (nla_type(nla)) {
      case DEVLINK_ATTR_INFO_VERSION_FIXED:
        append_version(nla, DevlinkVersionType::Fixed, &info);
        break;
      case DEVLINK_ATTR_INFO_VERSION_RUNNING:
        append_version(nla, DevlinkVersionType::Running, &info);
        break;
      case DEVLINK_ATTR_INFO_VERSION_STORED:
        append_version(nla, DevlinkVersionType::Stored, &info);
        break;
      default:
        break;
    }
  }

  ctx->found = true;
  return NL_OK;
}

}  // namespace

transport::Result<DevlinkDeviceInfo> DevlinkNetlinkClient::get_device_info(const std::string& dev) {
  transport::Result<DevlinkDeviceInfo> result{false, {}, ENOTSUP};

  int ret = init();
  if (ret < 0) {
    result.error_code = -ret;
    return result;
  }

  DevlinkDeviceInfo info{};
  DeviceInfoContext ctx{dev, &info};

  auto build_fn = [&dev](NLMessage& msg) -> int {
    NLAttributes attrs(msg);
    int r = attrs.put_string(DEVLINK_ATTR_BUS_NAME, "pci");
    if (r < 0) {
      return r;
    }
    return attrs.put_string(DEVLINK_ATTR_DEV_NAME, dev);
  };

  ret = client_.query(family_id_, DEVLINK_CMD_INFO_GET, DEVLINK_GENL_VERSION, build_fn,
                      device_info_handler, &ctx, NLM_F_REQUEST | NLM_F_DUMP);
  if (ret < 0) {
    result.error_code = -ret;
    return result;
  }
  if (!ctx.found) {
    return result;  // device has no devlink instance: ENOTSUP
  }

  result.success = true;
  result.value = info;
  result.error_code = 0;
  return result;
}

}  // namespace amd::nic::netlink

#endif  // HAVE_LIBNL3
