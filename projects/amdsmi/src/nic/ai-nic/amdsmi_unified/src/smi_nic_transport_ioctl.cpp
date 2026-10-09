// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
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

#include <linux/ethtool.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "smi_ethtool_ioctl.h"
#include "smi_nic_driver_stats.h"
#include "smi_nic_transport.h"

namespace amd::smi::nic::transport {

namespace {

// ETHTOOL_GLINKSETTINGS trails the struct with supported, advertising and lp_advertising masks.
constexpr size_t kLinkModeMaskCount = 3;
constexpr size_t kMacLen = ETH_ALEN;
static_assert(sizeof(PermanentAddress::mac) == kMacLen, "PermanentAddress::mac must hold one MAC");

struct FreeDeleter {
  void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
using ZeroedPtr_t = std::unique_ptr<T, FreeDeleter>;

// Zero-filled heap block of `bytes` for the variable-length ethtool structs.
template <typename T>
ZeroedPtr_t<T> make_zeroed(size_t bytes) {
  return ZeroedPtr_t<T>(static_cast<T*>(std::calloc(1, bytes)));
}

}  // namespace

Result<PermanentAddress> parse_perm_addr(uint32_t reported_size, const uint8_t* mac_bytes) {
  PermanentAddress result{};
  if (reported_size != result.mac.size()) {
    return {false, {}, ENODATA};
  }
  std::copy(mac_bytes, mac_bytes + result.mac.size(), result.mac.begin());
  return {true, result, 0};
}

namespace ioctl_internal {

namespace {

// The driver's count can change between the three requests. Kernels that check the input count
// answer 0 on a mismatch; older ones write their current count whatever size we allocated, so the
// headroom is best effort there. Either way a mismatch is a reason to start over.
constexpr int kMaxStatsAttempts = 3;
constexpr size_t kStatsHeadroom = 64;

}  // namespace

Result<VendorStatistics> read_driver_statistics(const DriverStatsIoctls_t& ioctls) {
  for (int attempt = 0; attempt < kMaxStatsAttempts; ++attempt) {
    auto count = ioctls.get_count();
    if (!count.success) {
      return {false, {}, count.error_code};
    }

    const uint32_t stats_num = count.value;
    if (stats_num == 0) {
      return {true, {{}, {}}, 0};
    }

    /**
     * ethtool_gstrings/ethtool_stats are variable-length: a fixed header
     * followed by `capacity` trailing entries.
     */
    const size_t capacity = static_cast<size_t>(stats_num) + kStatsHeadroom;
    auto strings =
        make_zeroed<ethtool_gstrings>(sizeof(ethtool_gstrings) + capacity * ETH_GSTRING_LEN);
    auto stats = make_zeroed<ethtool_stats>(sizeof(ethtool_stats) + capacity * sizeof(uint64_t));
    if (!strings || !stats) {
      return {false, {}, ENOMEM};
    }

    strings->cmd = ETHTOOL_GSTRINGS;
    strings->string_set = ETH_SS_STATS;
    strings->len = static_cast<__u32>(stats_num);
    if (ioctls.get_strings(strings.get(), capacity) != 0) {
      return {false, {}, errno};
    }

    stats->cmd = ETHTOOL_GSTATS;
    stats->n_stats = static_cast<__u32>(stats_num);
    if (ioctls.get_stats(stats.get(), capacity) != 0) {
      return {false, {}, errno};
    }

    if ((strings->len != stats_num) || (stats->n_stats != stats_num)) {
      continue;
    }

    VendorStatistics result;
    result.names.reserve(stats_num);
    result.values.reserve(stats_num);

    for (uint32_t i = 0; i < stats_num; i++) {
      const char* name_ptr = reinterpret_cast<const char*>(&strings->data[i * ETH_GSTRING_LEN]);
      result.names.emplace_back(name_ptr, strnlen(name_ptr, ETH_GSTRING_LEN));
      result.values.push_back(stats->data[i]);
    }

    return {true, result, 0};
  }

  return {false, {}, EAGAIN};
}

}  // namespace ioctl_internal

namespace {

/**
 * Ioctl (SIOCETHTOOL) transport backend. Works on all ethtool-capable kernels.
 * Each call is self-contained, so instances are thread-safe.
 */
class IoctlBackend : public NicTransport {
 public:
  IoctlBackend() = default;
  ~IoctlBackend() override = default;

  IoctlBackend(const IoctlBackend&) = delete;
  IoctlBackend& operator=(const IoctlBackend&) = delete;
  IoctlBackend(IoctlBackend&&) = delete;
  IoctlBackend& operator=(IoctlBackend&&) = delete;

  Result<PauseParams> get_pause_params(const std::string& iface) override {
    struct ethtool_pauseparam pause;
    std::memset(&pause, 0, sizeof(pause));
    pause.cmd = ETHTOOL_GPAUSEPARAM;

    int ret = smi_ethtool_ioctl(iface, &pause);
    if (ret != 0) {
      return {false, {}, errno};
    }

    return {true,
            {static_cast<bool>(pause.autoneg), static_cast<bool>(pause.rx_pause),
             static_cast<bool>(pause.tx_pause)},
            0};
  }

  Result<LinkSettings> get_link_settings(const std::string& iface) override {
    /**
     * GLINKSETTINGS is a two-step handshake: the first call zeroes the base
     * fields and reports the required link-mode word count as a negative
     * value; only a second call with a matching, correctly sized buffer
     * returns valid speed/duplex/autoneg.
     */
    struct ethtool_link_settings probe;
    std::memset(&probe, 0, sizeof(probe));
    probe.cmd = ETHTOOL_GLINKSETTINGS;

    int ret = smi_ethtool_ioctl(iface, &probe);
    if (ret != 0) {
      return {false, {}, errno};
    }

    const int32_t nwords = -probe.link_mode_masks_nwords;
    if (nwords <= 0) {
      return {false, {}, EINVAL};
    }

    size_t len = sizeof(ethtool_link_settings) +
                 (kLinkModeMaskCount * static_cast<size_t>(nwords) * sizeof(uint32_t));
    auto link = make_zeroed<ethtool_link_settings>(len);

    if (!link) {
      return {false, {}, ENOMEM};
    }

    link->cmd = ETHTOOL_GLINKSETTINGS;
    link->link_mode_masks_nwords = static_cast<__s8>(nwords);

    ret = smi_ethtool_ioctl(iface, link.get());
    if (ret != 0) {
      return {false, {}, errno};
    }

    /**
     * supported/advertising are multi-word bitmaps that neither fit the
     * single-word transport fields nor have a consumer; leave them zero.
     */
    LinkSettings settings{};
    settings.speed = link->speed;
    settings.duplex = link->duplex;
    settings.autoneg = link->autoneg;
    return {true, settings, 0};
  }

  Result<DriverInfo> get_driver_info(const std::string& iface) override {
    struct ethtool_drvinfo drvinfo;
    std::memset(&drvinfo, 0, sizeof(drvinfo));
    drvinfo.cmd = ETHTOOL_GDRVINFO;

    int ret = smi_ethtool_ioctl(iface, &drvinfo);
    if (ret != 0) {
      return {false, {}, errno};
    }

    // The kernel null-terminates these fixed-size fields.
    return {true,
            {std::string(reinterpret_cast<const char*>(drvinfo.driver)),
             std::string(reinterpret_cast<const char*>(drvinfo.version)),
             std::string(reinterpret_cast<const char*>(drvinfo.fw_version)),
             std::string(reinterpret_cast<const char*>(drvinfo.bus_info)), drvinfo.n_stats},
            0};
  }

  Result<VendorStatistics> get_statistics(const std::string& iface) override {
    ioctl_internal::DriverStatsIoctls_t ioctls;
    ioctls.get_count = [this, &iface]() -> Result<uint32_t> {
      const auto drvinfo = get_driver_info(iface);
      return {drvinfo.success, drvinfo.value.n_stats, drvinfo.error_code};
    };
    ioctls.get_strings = [&iface](ethtool_gstrings* strings, size_t) {
      return smi_ethtool_ioctl(iface, strings);
    };
    ioctls.get_stats = [&iface](ethtool_stats* stats, size_t) {
      return smi_ethtool_ioctl(iface, stats);
    };
    return ioctl_internal::read_driver_statistics(ioctls);
  }

  Result<FecStatistics_t> get_fec_statistics(const std::string& /* iface */) override {
    return {false, {}, ENOTSUP};
  }

  Result<PermanentAddress> get_permanent_address(const std::string& iface) override {
    /**
     * ethtool_perm_addr is followed by the MAC bytes; size the buffer for
     * both. Heap-allocate so the storage is suitably aligned for the struct's
     * __u32 fields (a plain uint8_t[] would be alignment-1).
     */
    std::vector<uint8_t> buffer(sizeof(ethtool_perm_addr) + kMacLen, 0);

    auto* perm = reinterpret_cast<ethtool_perm_addr*>(buffer.data());
    perm->cmd = ETHTOOL_GPERMADDR;
    perm->size = kMacLen;

    int ret = smi_ethtool_ioctl(iface, perm);
    if (ret != 0) {
      return {false, {}, errno};
    }

    const uint8_t* mac_data = buffer.data() + sizeof(ethtool_perm_addr);
    return parse_perm_addr(perm->size, mac_data);
  }

  std::string backend_name() const override { return "ioctl"; }
};

}  // namespace

namespace ioctl_internal {
std::shared_ptr<NicTransport> create_ioctl_backend() { return std::make_shared<IoctlBackend>(); }
}  // namespace ioctl_internal

#ifdef HAVE_LIBNL3
namespace internal {
std::shared_ptr<NicTransport> create_netlink_backend();
std::shared_ptr<NicTransport> create_auto_backend();
}  // namespace internal
#endif

std::shared_ptr<NicTransport> create_transport(NicBackend_t backend) {
#ifdef HAVE_LIBNL3
  if (backend == NicBackend_t::Netlink) {
    return with_debug_logging(internal::create_netlink_backend());
  }
  if (backend == NicBackend_t::Auto) {
    return with_debug_logging(internal::create_auto_backend());
  }
#endif

  // Ioctl, or Netlink/Auto without libnl-3: the always-available fallback.
  return with_debug_logging(std::make_shared<IoctlBackend>());
}

}  // namespace amd::smi::nic::transport
