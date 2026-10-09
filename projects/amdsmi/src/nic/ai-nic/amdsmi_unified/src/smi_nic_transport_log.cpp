// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "smi_nic_log.h"
#include "smi_nic_transport.h"

namespace amd::smi::nic::transport {

namespace {

// "xx:xx:xx:xx:xx:xx" plus NUL
constexpr auto kMacTextLen = size_t{18};
// " supported=0x........ advertising=0x........" plus NUL
constexpr auto kLinkMaskTextLen = size_t{48};

auto mac_text(const std::array<uint8_t, 6>& mac) -> std::string {
  char buf[kMacTextLen];
  std::snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x", mac[0], mac[1], mac[2], mac[3],
                mac[4], mac[5]);
  return buf;
}

auto describe(const PauseParams& v) -> std::string {
  return "autoneg=" + std::to_string(v.autoneg) + " rx_pause=" + std::to_string(v.rx_pause) +
         " tx_pause=" + std::to_string(v.tx_pause);
}

auto describe(const LinkSettings& v) -> std::string {
  char masks[kLinkMaskTextLen];
  std::snprintf(masks, sizeof(masks), " supported=0x%x advertising=0x%x", v.supported,
                v.advertising);
  return "speed=" + std::to_string(v.speed) + " duplex=" + std::to_string(v.duplex) +
         " autoneg=" + std::to_string(v.autoneg) + masks;
}

auto describe(const DriverInfo& v) -> std::string {
  return "driver=" + v.driver_name + " version=" + v.version + " fw_version=" + v.fw_version +
         " bus_info=" + v.bus_info + " n_stats=" + std::to_string(v.n_stats);
}

auto describe(const PermanentAddress& v) -> std::string {
  return "addr=" + amd::smi::nic::log::mask_tail(mac_text(v.mac));
}

// Summary line first, then one line per counter so each value is greppable.
template <typename Stats_t>
auto log_counters(const char* func, const std::string& backend, const std::string& op,
                  const std::string& iface, const Stats_t& stats) -> void {
  const auto prefix = "transport " + backend + " " + op + " " + iface + " ";
  amd::smi::nic::log::debug(func,
                            prefix + "-> SUCCESS count=" + std::to_string(stats.names.size()));
  const auto num_counters = std::min(stats.names.size(), stats.values.size());
  for (auto i = size_t{0}; i < num_counters; ++i) {
    amd::smi::nic::log::debug(func,
                              prefix + stats.names[i] + "=" + std::to_string(stats.values[i]));
  }
}

template <typename T>
auto log_result(const char* func, const std::string& backend, const std::string& op,
                const std::string& iface, const Result<T>& result) -> void {
  if (!amd::smi::nic::log::is_enabled()) {
    return;
  }
  const auto head = "transport " + backend + " " + op + " " + iface + " -> ";
  if (!result.success) {
    amd::smi::nic::log::debug(func, head + "FAIL code=" + std::to_string(result.error_code) + " " +
                                        std::system_category().message(result.error_code));
    return;
  }
  amd::smi::nic::log::debug(func, head + "SUCCESS " + describe(result.value));
}

class LoggingTransport_t : public NicTransport {
 public:
  explicit LoggingTransport_t(std::shared_ptr<NicTransport> inner) : m_inner(std::move(inner)) {}

  auto get_pause_params(const std::string& iface) -> Result<PauseParams> override {
    const auto result = m_inner->get_pause_params(iface);
    log_result(__PRETTY_FUNCTION__, m_inner->backend_name(), "pause", iface, result);
    return result;
  }

  auto get_link_settings(const std::string& iface) -> Result<LinkSettings> override {
    const auto result = m_inner->get_link_settings(iface);
    log_result(__PRETTY_FUNCTION__, m_inner->backend_name(), "link", iface, result);
    return result;
  }

  auto get_driver_info(const std::string& iface) -> Result<DriverInfo> override {
    const auto result = m_inner->get_driver_info(iface);
    log_result(__PRETTY_FUNCTION__, m_inner->backend_name(), "driver", iface, result);
    return result;
  }

  auto get_statistics(const std::string& iface) -> Result<VendorStatistics> override {
    const auto result = m_inner->get_statistics(iface);
    log_counters_result(__PRETTY_FUNCTION__, "stats", iface, result);
    return result;
  }

  auto get_fec_statistics(const std::string& iface) -> Result<FecStatistics_t> override {
    const auto result = m_inner->get_fec_statistics(iface);
    log_counters_result(__PRETTY_FUNCTION__, "fec", iface, result);
    return result;
  }

  auto get_permanent_address(const std::string& iface) -> Result<PermanentAddress> override {
    const auto result = m_inner->get_permanent_address(iface);
    log_result(__PRETTY_FUNCTION__, m_inner->backend_name(), "perm_addr", iface, result);
    return result;
  }

  auto backend_name() const -> std::string override { return m_inner->backend_name(); }

 private:
  template <typename Stats_t>
  auto log_counters_result(const char* func, const std::string& op, const std::string& iface,
                           const Result<Stats_t>& result) const -> void {
    if (!amd::smi::nic::log::is_enabled()) {
      return;
    }
    const auto backend = m_inner->backend_name();
    if (!result.success) {
      amd::smi::nic::log::debug(func, "transport " + backend + " " + op + " " + iface +
                                          " -> FAIL code=" + std::to_string(result.error_code) +
                                          " " + std::system_category().message(result.error_code));
      return;
    }
    log_counters(func, backend, op, iface, result.value);
  }

  std::shared_ptr<NicTransport> m_inner;
};

}  // namespace

auto with_debug_logging(std::shared_ptr<NicTransport> inner) -> std::shared_ptr<NicTransport> {
  return std::make_shared<LoggingTransport_t>(std::move(inner));
}

}  // namespace amd::smi::nic::transport
