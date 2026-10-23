// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_interface.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "amd_smi/impl/nic/amd_smi_nic_interface.h"
#include "smi_devlink_netlink.h"
#include "smi_ethtool_ioctl.h"
#include "smi_nic_system.h"
#include "smi_nic_telemetry.h"
#include "smi_nic_testing.h"
#include "smi_sysfs.h"
#include "vendors/vendor_registry.h"

struct smi_nic_ctx {
  std::unique_ptr<SmiNicSystem> nic_system;
  std::mutex ctx_mutex;
  std::atomic<bool> init;

  smi_nic_ctx() : init(false) {}
};

static constexpr const char* kNotAvailable = "N/A";

// Empty values render as "N/A", matching the optional::value_or("N/A") fields.
template <size_t N>
static void copy_or_na(char (&dst)[N], const std::string& value) {
  std::snprintf(dst, N, "%s", value.empty() ? kNotAvailable : value.c_str());
}

// The list holds at most SMI_NIC_MAX_STATISTICS entries, so the count cannot promise more.
static uint32_t capped_stat_count(size_t num_stats) {
  return static_cast<uint32_t>(std::min(num_stats, static_cast<size_t>(SMI_NIC_MAX_STATISTICS)));
}

static SmiNicSystem* get_nic_system_from_context(smi_nic_ctx* ctx) {
  if (!ctx || !ctx->init || !ctx->nic_system) {
    return nullptr;
  }
  return ctx->nic_system.get();
}

// The null-ctx and null-out-parameter guards on the entry points below have no
// test covering them: they sit behind the C ABI with no injection seam, so
// deleting one leaves the suites green. Change them on inspection, not on CI.
extern "C" {
smi_nic_status_t smi_nic_create_context(smi_nic_ctx_t* ctx, bool is_ainic_only) {
  try {
    if (!ctx) {
      return SMI_NIC_STATUS_WRONG_PARAM;
    }

    auto context = std::make_unique<smi_nic_ctx>();
    context->nic_system = std::make_unique<SmiNicSystem>();
    context->init = true;

    // Publish only once discovery has succeeded: throwing after the release
    // would both leak the context and leave the caller holding a handle to it.
    context->nic_system->discover_nics(is_ainic_only);
    *ctx = context.release();

    return SMI_NIC_STATUS_SUCCESS;

  } catch (const std::bad_alloc&) {
    return SMI_NIC_STATUS_NO_RESOURCE;
  } catch (...) {
    return SMI_NIC_STATUS_ERROR;
  }
}

smi_nic_status_t smi_nic_destroy_context(smi_nic_ctx_t ctx) {
  try {
    if (!ctx) {
      return SMI_NIC_STATUS_WRONG_PARAM;
    }

    {
      std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
      ctx->init = false;
      ctx->nic_system.reset();
    }

    delete ctx;
    return SMI_NIC_STATUS_SUCCESS;
  } catch (...) {
    return SMI_NIC_STATUS_ERROR;
  }
}

namespace {

auto to_shim_vendor(NicVendor vendor) -> smi_nic_vendor_t {
  switch (vendor) {
    case NicVendor::AMD:
      return SMI_NIC_VENDOR_AMD;
    case NicVendor::Broadcom:
      return SMI_NIC_VENDOR_BROADCOM;
    case NicVendor::Unknown:
      break;
  }
  return SMI_NIC_VENDOR_UNKNOWN;
}

}  // namespace

namespace {

// Resolves the NIC under the context lock and hands it to read; the shim getters that only
// copy one value out of a NIC share this.
auto read_nic_value(smi_nic_ctx_t ctx, uint64_t device,
                    const std::function<void(const SmiNic&)>& read) -> smi_nic_status_t {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }
  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }
  const auto lock = std::lock_guard<std::mutex>{ctx->ctx_mutex};
  const auto* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }
  read(*nic);
  return SMI_NIC_STATUS_SUCCESS;
}

}  // namespace

smi_nic_status_t smi_get_nic_kind(smi_nic_ctx_t ctx, uint64_t device, uint8_t* kind) {
  if (!kind) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }
  return read_nic_value(ctx, device,
                        [&](const SmiNic& nic) { *kind = static_cast<uint8_t>(nic.kind()); });
}

smi_nic_status_t smi_get_nic_capability(smi_nic_ctx_t ctx, uint64_t device, uint32_t* capability) {
  if (!capability) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }
  return read_nic_value(ctx, device, [&](const SmiNic& nic) { *capability = nic.capabilities(); });
}

static_assert(static_cast<int>(amd::smi::nic::pci::LinkType_t::Unknown) ==
              SMI_NIC_LINK_TYPE_UNKNOWN);
static_assert(static_cast<int>(amd::smi::nic::pci::LinkType_t::Pcie) == SMI_NIC_LINK_TYPE_PCIE);
static_assert(static_cast<int>(amd::smi::nic::pci::LinkType_t::Numa) == SMI_NIC_LINK_TYPE_NUMA);
static_assert(static_cast<int>(amd::smi::nic::pci::LinkType_t::Xnuma) == SMI_NIC_LINK_TYPE_XNUMA);

smi_nic_status_t smi_topo_get_nic_link_type(smi_nic_ctx_t ctx, uint64_t device_src,
                                            uint64_t device_dst, smi_nic_link_type_t* type) {
  if (!type) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }
  return read_nic_value(ctx, device_src, [&](const SmiNic& nic) {
    // LinkType_t carries the shim enumerators' values.
    *type = static_cast<smi_nic_link_type_t>(nic.link_type(device_dst));
  });
}

smi_nic_status_t smi_nic_driver_loaded(smi_nic_vendor_t vendor) {
  if ((vendor != SMI_NIC_VENDOR_AMD) && (vendor != SMI_NIC_VENDOR_BROADCOM)) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  // A vendor can have several plugins (Pensando and UALoE are both AMD); any one loaded counts.
  try {
    for (const auto& plugin : make_default_vendor_plugins()) {
      if ((to_shim_vendor(plugin->vendor()) == vendor) &&
          plugin->is_driver_present(DriverType::Main)) {
        return SMI_NIC_STATUS_SUCCESS;
      }
    }
  } catch (const std::exception&) {
    return SMI_NIC_STATUS_ERROR;
  }
  return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
}

smi_nic_status_t smi_discover_nics(smi_nic_ctx_t ctx, smi_nic_discovery_t* discovery) {
  if (!ctx || !discovery) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  discovery->count = 0;
  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

  try {
    const auto& nics = nic_system->get_nics();
    if (nics.empty()) {
      return SMI_NIC_STATUS_NO_DATA;
    }

    if (nics.size() > SMI_NIC_MAX_DEVICES) {
      return SMI_NIC_STATUS_NO_RESOURCE;
    }

    uint32_t index = 0;
    for (const auto* nic : nics) {
      std::snprintf(discovery->devices[index].bdf, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    nic->bdf().c_str());
      discovery->devices[index].vendor = to_shim_vendor(nic->vendor());
      index++;
    }

    discovery->count = static_cast<uint32_t>(nics.size());
    return SMI_NIC_STATUS_SUCCESS;

  } catch (const std::exception&) {
    discovery->count = 0;
    return SMI_NIC_STATUS_ERROR;
  }
}

smi_nic_status_t smi_get_nic_driver_info(smi_nic_ctx_t ctx, uint64_t device,
                                         smi_nic_driver_info_t* info) {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  if (!info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  // A fwctl-only NIC has no netdev to query via ethtool for driver name/version;
  // report NO_DATA (non-fatal upstream) rather than dropping the device.
  if (ports.empty()) {
    return SMI_NIC_STATUS_NO_DATA;
  }

  if (!nic_system->is_driver_loaded(ports[0].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  *info = {};
  struct ethtool_drvinfo drvinfo{};
  drvinfo.cmd = ETHTOOL_GDRVINFO;

  int ret = smi_ethtool_ioctl(ports[0].interface(), &drvinfo);
  if (ret != 0) {
    return SMI_NIC_STATUS_ERROR;
  }

  std::snprintf(info->name, SMI_NIC_MAX_STRING_LENGTH, "%s", drvinfo.driver);
  std::snprintf(info->version, SMI_NIC_MAX_STRING_LENGTH, "%s", drvinfo.version);

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_asic_info(smi_nic_ctx_t ctx, uint64_t device,
                                       smi_nic_asic_info_t* info) {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  if (!info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  // A fwctl-only NIC (no host netdev) exposes no port; its PCI-sysfs identity is
  // still valid. Require a loaded netdev driver only when a port exists.
  if (!ports.empty() && !nic_system->is_driver_loaded(ports[0].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  *info = {};
  info->vendor_id = nic->vendor_id().value_or(std::numeric_limits<uint16_t>::max());
  info->subvendor_id = nic->subvendor_id().value_or(std::numeric_limits<uint16_t>::max());
  info->device_id = nic->device_id().value_or(std::numeric_limits<uint16_t>::max());
  info->subsystem_id = nic->subsystem_id().value_or(std::numeric_limits<uint16_t>::max());
  info->revision = nic->revision().value_or(std::numeric_limits<uint8_t>::max());

  std::snprintf(info->permanent_address, SMI_NIC_MAX_STRING_LENGTH, "%s",
                nic->perm_address().value_or("N/A").c_str());
  std::snprintf(info->product_name, SMI_NIC_MAX_STRING_LENGTH, "%s",
                nic->product_name().value_or("N/A").c_str());
  std::snprintf(info->vendor_name, SMI_NIC_MAX_STRING_LENGTH, "%s",
                nic->vendor_name().value_or("N/A").c_str());
  std::snprintf(info->part_number, SMI_NIC_MAX_STRING_LENGTH, "%s",
                nic->part_number().value_or("N/A").c_str());

  // VPD is the primary serial source; when it was readable but carried no
  // serial keyword, fall back to the board serial reported over devlink. Skip
  // the fallback when VPD could not be opened at all, since the devlink round
  // trip costs ~1s per NIC and has never yet produced a serial in that case.
  std::string serial = nic->serial_number().value_or("");
  if (serial.empty() && nic->is_vpd_readable()) {
    // The devlink client allocates netlink messages that throw; the serial is optional here.
    try {
      auto devlink = amd::nic::netlink::create_devlink_client();
      auto dev_info = devlink->get_device_info(nic->telemetry_bdf());
      if (dev_info.success && (dev_info.value.board_serial_number[0] != '\0')) {
        serial = dev_info.value.board_serial_number;
      }
    } catch (...) {
    }
  }
  std::snprintf(info->serial_number, SMI_NIC_MAX_STRING_LENGTH, "%s",
                serial.empty() ? kNotAvailable : serial.c_str());

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_bus_info(smi_nic_ctx_t ctx, uint64_t device,
                                      smi_nic_bus_info_t* info) {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  if (!info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  // A fwctl-only NIC (no host netdev) exposes no port; its PCI-sysfs identity is
  // still valid. Require a loaded netdev driver only when a port exists.
  if (!ports.empty() && !nic_system->is_driver_loaded(ports[0].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  *info = {};
  info->bdf = device;
  info->max_pcie_width = nic->max_pcie_width().value_or(std::numeric_limits<uint8_t>::max());
  info->max_pcie_speed = nic->max_pcie_speed().value_or(std::numeric_limits<uint32_t>::max());
  std::snprintf(info->pcie_interface_version, SMI_NIC_MAX_STRING_LENGTH, "%s", "N/A");
  std::snprintf(info->slot_type, SMI_NIC_MAX_STRING_LENGTH, "%s", "N/A");

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_numa_info(smi_nic_ctx_t ctx, uint64_t device,
                                       smi_nic_numa_info_t* info) {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  if (!info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  // A fwctl-only NIC (no host netdev) exposes no port; its PCI-sysfs identity is
  // still valid. Require a loaded netdev driver only when a port exists.
  if (!ports.empty() && !nic_system->is_driver_loaded(ports[0].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  *info = {};
  info->node = nic->numa_node().value_or(std::numeric_limits<uint8_t>::max());
  std::snprintf(info->affinity, SMI_NIC_MAX_STRING_LENGTH, "%s",
                nic->numa_affinity(info->node).value_or("N/A").c_str());

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_port_info(smi_nic_ctx_t ctx, uint64_t device,
                                       smi_nic_port_info_t* info) {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  if (!info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  // A fwctl-only NIC has no host netdev/port to report link state for; report
  // NO_DATA (non-fatal upstream) rather than dropping the device.
  if (ports.empty()) {
    return SMI_NIC_STATUS_NO_DATA;
  }

  *info = {};

  uint32_t bound_count = 0;
  const auto slot_count =
      static_cast<uint32_t>(std::min(ports.size(), static_cast<size_t>(SMI_NIC_MAX_PORTS)));
  /**
   * port.autoneg()/pause_params() route through the netlink transport, whose
   * RAII message/callback allocations throw; keep that off the C ABI boundary.
   */
  try {
    for (uint32_t i = 0; i < slot_count; i++) {
      const auto& port = ports[i];
      // Slot i is nic_ports()[i], the index the statistics entry points take.
      smi_nic_port_t* port_info = &info->ports[i];

      port_info->bdf = parse_bdf(port.bdf());
      port_info->port_num = port.port_num().value_or(std::numeric_limits<uint32_t>::max());
      copy_or_na(port_info->netdev, port.interface());

      if (!nic_system->is_driver_loaded(port.bdf(), DriverType::Main)) {
        // Keep the identity; everything the driver would answer reads N/A.
        copy_or_na(port_info->type, "");
        copy_or_na(port_info->flavour, "");
        copy_or_na(port_info->mac_address, "");
        copy_or_na(port_info->link_state, "");
        copy_or_na(port_info->autoneg, "");
        copy_or_na(port_info->pause_autoneg, "");
        copy_or_na(port_info->pause_rx, "");
        copy_or_na(port_info->pause_tx, "");
        port_info->carrier = std::numeric_limits<uint8_t>::max();
        port_info->mtu = std::numeric_limits<uint16_t>::max();
        port_info->link_speed = std::numeric_limits<uint32_t>::max();
        port_info->active_fec = std::numeric_limits<uint32_t>::max();
        continue;
      }

      copy_or_na(port_info->type, port.port_type());
      copy_or_na(port_info->flavour, port.flavour());

      // The public field is one byte; an index that does not fit reads 0, and
      // smi_get_nic_port_ifindex() returns the whole value.
      const auto ifindex = port.ifindex().value_or(0);
      port_info->ifindex = (ifindex <= std::numeric_limits<uint8_t>::max())
                               ? static_cast<uint8_t>(ifindex)
                               : uint8_t{0};

      copy_or_na(port_info->mac_address, port.mac_address().value_or(""));

      port_info->carrier = port.carrier().value_or(std::numeric_limits<uint8_t>::max());
      port_info->mtu = port.mtu().value_or(std::numeric_limits<uint16_t>::max());

      copy_or_na(port_info->link_state, port.link_state().value_or(""));

      port_info->link_speed = port.link_speed().value_or(std::numeric_limits<uint32_t>::max());

      /**
       * FEC stays on the direct ioctl (not the transport): the netlink
       * ETHTOOL_A_FEC_ACTIVE attribute is a link-mode bit index, a different
       * domain than ioctl active_fec's ETHTOOL_FEC_* bitmask that this field's
       * API contract promises. The dead netlink FEC path was removed rather
       * than left to report the wrong domain.
       */
      struct ethtool_fecparam fecparam_info{};
      fecparam_info.cmd = ETHTOOL_GFECPARAM;
      port_info->active_fec = (smi_ethtool_ioctl(port.interface(), &fecparam_info) == 0)
                                  ? fecparam_info.active_fec
                                  : std::numeric_limits<uint32_t>::max();

      /** Autoneg and pause route through the transport (netlink-first, ioctl fallback). */
      auto on_off = [](std::optional<bool> v) -> const char* {
        return v.has_value() ? (v.value() ? "ON" : "OFF") : kNotAvailable;
      };
      std::snprintf(port_info->autoneg, SMI_NIC_MAX_STRING_LENGTH, "%s", on_off(port.autoneg()));

      auto pause = port.pause_params();
      std::snprintf(port_info->pause_autoneg, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    on_off(pause ? std::optional<bool>(pause->is_autoneg_enabled) : std::nullopt));
      std::snprintf(port_info->pause_rx, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    on_off(pause ? std::optional<bool>(pause->is_rx_pause_enabled) : std::nullopt));
      std::snprintf(port_info->pause_tx, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    on_off(pause ? std::optional<bool>(pause->is_tx_pause_enabled) : std::nullopt));

      bound_count++;
    }
  } catch (const std::bad_alloc&) {
    return SMI_NIC_STATUS_NO_RESOURCE;
  } catch (...) {
    return SMI_NIC_STATUS_ERROR;
  }

  info->num_ports = slot_count;
  if (bound_count == 0) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_rdma_dev_info(smi_nic_ctx_t ctx, uint64_t device,
                                           smi_nic_rdma_devices_info_t* info) {
  if (!ctx) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  if (!info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  *info = {};

  uint8_t rdma_count = 0;
  bool driver_not_loaded = false;
  for (uint32_t i = 0; i < ports.size() && rdma_count < SMI_NIC_MAX_RDMA_DEV; i++) {
    const auto& port = ports[i];
    if (!nic_system->is_driver_loaded(port.bdf(), DriverType::Rdma)) {
      driver_not_loaded = true;
      continue;
    }

    const auto& ibs = port.infiniband();
    for (uint8_t j = 0; j < ibs.size() && rdma_count < SMI_NIC_MAX_RDMA_DEV; j++) {
      const auto& ib = ibs[j];
      smi_nic_rdma_dev_info_t* rdma_dev = &info->rdma_dev_info[rdma_count];

      std::snprintf(rdma_dev->rdma_dev, SMI_NIC_MAX_STRING_LENGTH, "%s", ib.rdma_dev().c_str());
      std::snprintf(rdma_dev->node_guid, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    ib.node_guid().value_or("N/A").c_str());
      std::snprintf(rdma_dev->node_type, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    ib.node_type().value_or("N/A").c_str());
      std::snprintf(rdma_dev->sys_image_guid, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    ib.sys_image_guid().value_or("N/A").c_str());
      std::snprintf(rdma_dev->fw_ver, SMI_NIC_MAX_STRING_LENGTH, "%s",
                    ib.fw_ver().value_or("N/A").c_str());

      const auto& ib_ports = ib.ports();
      /**
       * Report only the entries the loop below actually writes: the array holds
       * at most SMI_NIC_MAX_PORTS, and the loop is bounded by ib_ports.size().
       * Using ib.ports_num() (a sysfs-reported count that can exceed either)
       * would let a consumer over-read rdma_port_info[].
       */
      rdma_dev->num_rdma_ports =
          static_cast<uint8_t>(std::min(ib_ports.size(), static_cast<size_t>(SMI_NIC_MAX_PORTS)));
      for (uint8_t k = 0; k < ib_ports.size() && k < SMI_NIC_MAX_PORTS; k++) {
        const auto& ib_port = ib_ports[k];
        smi_nic_rdma_port_info_t* port_info = &rdma_dev->rdma_port_info[k];

        std::snprintf(port_info->netdev, SMI_NIC_MAX_STRING_LENGTH, "%s", port.interface().c_str());
        std::snprintf(port_info->state, SMI_NIC_MAX_STRING_LENGTH, "%s",
                      ib_port.state().value_or("N/A").c_str());
        port_info->rdma_port = ib_port.port_num().value_or(std::numeric_limits<uint8_t>::max());
        port_info->max_mtu = ib_port.max_mtu().value_or(std::numeric_limits<uint16_t>::max());
        port_info->active_mtu = ib_port.active_mtu().value_or(std::numeric_limits<uint16_t>::max());
      }
      rdma_count++;
    }
  }

  info->num_rdma_dev = rdma_count;
  if (driver_not_loaded && rdma_count == 0) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (rdma_count == 0) {
    return SMI_NIC_STATUS_NO_DATA;
  }

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_port_ifindex(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index,
                                          uint32_t* ifindex) {
  if (!ctx || !ifindex) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  *ifindex = ports[port_index].ifindex().value_or(0);
  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_port_statistics_count(smi_nic_ctx_t ctx, uint64_t device,
                                                   uint32_t port_index, uint32_t* count) {
  if (!ctx || !count) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  if (!nic_system->is_driver_loaded(ports[port_index].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& port = ports[port_index];
  port.collect_standard_statistics();
  const auto& stats_map = port.get_standard_stats_map();
  *count = capped_stat_count(stats_map.size());

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_port_statistics_list(smi_nic_ctx_t ctx, uint64_t device,
                                                  uint32_t port_index, smi_nic_stat_info_t* stats) {
  if (!ctx || !stats) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  if (!nic_system->is_driver_loaded(ports[port_index].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& port = ports[port_index];
  port.collect_standard_statistics();
  const auto& stats_map = port.get_standard_stats_map();

  stats->count = capped_stat_count(stats_map.size());
  uint32_t i = 0;
  for (const auto& stat_pair : stats_map) {
    if (i >= stats->count) {
      break;
    }
    std::snprintf(stats->stats[i].name, SMI_NIC_MAX_STRING_LENGTH, "%s", stat_pair.first.c_str());
    stats->stats[i].value = stat_pair.second;
    i++;
  }

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_vendor_statistics_count(smi_nic_ctx_t ctx, uint64_t device,
                                                     uint32_t port_index, uint32_t* count) {
  return smi_get_nic_vendor_statistics_count_with_scope(ctx, device, port_index,
                                                        SMI_NIC_STAT_SCOPE_DEFAULT, count);
}

smi_nic_status_t smi_get_nic_vendor_statistics_count_with_scope(smi_nic_ctx_t ctx, uint64_t device,
                                                                uint32_t port_index,
                                                                smi_nic_stat_scope_t scope,
                                                                uint32_t* count) {
  if (!ctx || !count) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  if (!nic_system->is_driver_loaded(ports[port_index].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& port = ports[port_index];
  if (port.collect_vendor_statistics() != 0) {
    return SMI_NIC_STATUS_ERROR;
  }
  const auto tier =
      (scope == SMI_NIC_STAT_SCOPE_EXTENDED) ? StatTier_t::Extended : StatTier_t::Default;
  *count = capped_stat_count(port.get_vendor_stats_map(tier).size());

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_vendor_statistics_list(smi_nic_ctx_t ctx, uint64_t device,
                                                    uint32_t port_index,
                                                    smi_nic_stat_info_t* stats) {
  return smi_get_nic_vendor_statistics_list_with_scope(ctx, device, port_index,
                                                       SMI_NIC_STAT_SCOPE_DEFAULT, stats);
}

smi_nic_status_t smi_get_nic_vendor_statistics_list_with_scope(smi_nic_ctx_t ctx, uint64_t device,
                                                               uint32_t port_index,
                                                               smi_nic_stat_scope_t scope,
                                                               smi_nic_stat_info_t* stats) {
  if (!ctx || !stats) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  if (!nic_system->is_driver_loaded(ports[port_index].bdf(), DriverType::Main)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& port = ports[port_index];
  if (port.collect_vendor_statistics() != 0) {
    return SMI_NIC_STATUS_ERROR;
  }
  const auto tier =
      (scope == SMI_NIC_STAT_SCOPE_EXTENDED) ? StatTier_t::Extended : StatTier_t::Default;
  const auto stats_map = port.get_vendor_stats_map(tier);

  stats->count = capped_stat_count(stats_map.size());
  uint32_t i = 0;
  for (const auto& stat_pair : stats_map) {
    if (i >= stats->count) {
      break;
    }
    std::snprintf(stats->stats[i].name, SMI_NIC_MAX_STRING_LENGTH, "%s", stat_pair.first.c_str());
    stats->stats[i].value = stat_pair.second;
    i++;
  }

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_rdma_port_statistics_count(smi_nic_ctx_t ctx, uint64_t device,
                                                        uint32_t port_index, uint32_t ib_index,
                                                        uint32_t rdma_port_index, uint32_t* count) {
  if (!ctx || !count) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  if (!nic_system->is_driver_loaded(ports[port_index].bdf(), DriverType::Rdma)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& ibs = ports[port_index].infiniband();
  if (ib_index >= ibs.size()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& ib_ports = ibs[ib_index].ports();
  if (rdma_port_index >= ib_ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ib_port = ib_ports[rdma_port_index];
  // Counters change while running; the map from discovery time would be stale.
  ib_port.collect_hw_counters();
  const auto& stats_map = ib_port.get_hw_counters_map();
  *count = capped_stat_count(stats_map.size());

  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_rdma_port_statistics_list(smi_nic_ctx_t ctx, uint64_t device,
                                                       uint32_t port_index, uint32_t ib_index,
                                                       uint32_t rdma_port_index,
                                                       smi_nic_stat_info_t* stats) {
  if (!ctx || !stats) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ports = nic->nic_ports();
  if (ports.empty()) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  if (port_index >= ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  if (!nic_system->is_driver_loaded(ports[port_index].bdf(), DriverType::Rdma)) {
    return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  }

  const auto& ibs = ports[port_index].infiniband();
  if (ib_index >= ibs.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ib_ports = ibs[ib_index].ports();
  if (rdma_port_index >= ib_ports.size()) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  const auto& ib_port = ib_ports[rdma_port_index];
  // Counters change while running; the map from discovery time would be stale.
  ib_port.collect_hw_counters();
  const auto& stats_map = ib_port.get_hw_counters_map();

  stats->count = capped_stat_count(stats_map.size());
  uint32_t i = 0;
  for (const auto& stat_pair : stats_map) {
    if (i >= stats->count) {
      break;
    }
    std::snprintf(stats->stats[i].name, SMI_NIC_MAX_STRING_LENGTH, "%s", stat_pair.first.c_str());
    stats->stats[i].value = stat_pair.second;
    i++;
  }

  return SMI_NIC_STATUS_SUCCESS;
}

}  // extern "C"

smi_nic_status_t smi_nic_create_context_for_testing(smi_nic_ctx_t* ctx,
                                                    std::unique_ptr<SmiNicSystem> system) {
  if ((ctx == nullptr) || (system == nullptr)) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }
  auto context = std::make_unique<smi_nic_ctx>();
  context->nic_system = std::move(system);
  context->init = true;
  *ctx = context.release();
  return SMI_NIC_STATUS_SUCCESS;
}

void smi_nic_telemetry_from_snapshot_for_testing(
    const amd::smi::nic::telemetry::NicTelemetrySnapshot& snapshot,
    smi_nic_telemetry_t* telemetry) {
  *telemetry = {};
  telemetry->temperature.asic_temp_c = snapshot.temperature.asic_temp_c;
  telemetry->temperature.transceiver_temp_c = snapshot.temperature.transceiver_temp_c;
  telemetry->temperature.board_temp_c = snapshot.temperature.board_temp_c;
  telemetry->health.state = snapshot.health.state;
  telemetry->health.error_count = snapshot.health.error_count;
  std::snprintf(telemetry->health.reporter, sizeof(telemetry->health.reporter), "%s",
                snapshot.health.reporter);
  telemetry->port_split.splittable = snapshot.port_split.splittable;
  telemetry->port_split.split_count = snapshot.port_split.split_count;
}

smi_nic_status_t smi_get_nic_telemetry(smi_nic_ctx_t ctx, uint64_t device,
                                       smi_nic_telemetry_t* telemetry) {
  if (!ctx || !telemetry) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  // get_snapshot is total: an unexposed metric comes back as its reserved value, never
  // as a failure, so there is no unsuccessful result to branch on here.
  // The devlink client allocates netlink messages that throw; keep that off the C ABI boundary.
  try {
    amd::smi::nic::telemetry::NicTelemetry telemetry_reader(
        amd::nic::netlink::create_devlink_client());
    const auto snapshot = telemetry_reader.get_snapshot(*nic).value;
    smi_nic_telemetry_from_snapshot_for_testing(snapshot, telemetry);
  } catch (const std::bad_alloc&) {
    return SMI_NIC_STATUS_NO_RESOURCE;
  } catch (...) {
    return SMI_NIC_STATUS_ERROR;
  }
  return SMI_NIC_STATUS_SUCCESS;
}

namespace {

auto to_shim_fw_version_type(uint8_t devlink_type) -> smi_nic_fw_version_type_t {
  switch (static_cast<amd::nic::netlink::DevlinkVersionType>(devlink_type)) {
    case amd::nic::netlink::DevlinkVersionType::Running:
      return SMI_NIC_FW_VERSION_TYPE_RUNNING;
    case amd::nic::netlink::DevlinkVersionType::Stored:
      return SMI_NIC_FW_VERSION_TYPE_STORED;
    case amd::nic::netlink::DevlinkVersionType::Fixed:
      break;
  }
  return SMI_NIC_FW_VERSION_TYPE_FIXED;
}

}  // namespace

void smi_nic_fw_info_from_devlink_for_testing(const amd::nic::netlink::DevlinkDeviceInfo& device,
                                              smi_nic_fw_info_t* info) {
  *info = {};
  const auto count = std::min<size_t>(device.version_count, SMI_NIC_MAX_FW_VERSIONS);
  for (size_t i = 0; i < count; ++i) {
    info->versions[i].type = to_shim_fw_version_type(device.versions[i].type);
    std::snprintf(info->versions[i].name, sizeof(info->versions[i].name), "%s",
                  device.versions[i].name);
    std::snprintf(info->versions[i].version, sizeof(info->versions[i].version), "%s",
                  device.versions[i].value);
  }
  info->count = static_cast<uint32_t>(count);
}

smi_nic_status_t smi_get_nic_fw_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_fw_info_t* info) {
  if (!ctx || !info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  // DEVLINK_CMD_INFO_GET is a live FW/NVM mailbox round trip (~1s); the lock is
  // released before it runs so concurrent per-NIC callers overlap that wait
  // instead of serializing on it. create_devlink_client() below is a fresh,
  // unshared socket per call, so nothing here needs ctx_mutex's protection.
  std::string telemetry_bdf;
  {
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
    const SmiNic* nic = nic_system->get_nic_by_bdf(device);
    if (!nic) {
      return SMI_NIC_STATUS_NOT_FOUND;
    }
    telemetry_bdf = nic->telemetry_bdf();
  }

  try {
    auto devlink = amd::nic::netlink::create_devlink_client();
    auto dev = devlink->get_device_info(telemetry_bdf);
    if (!dev.success) {
      return SMI_NIC_STATUS_NOT_SUPPORTED;
    }

    smi_nic_fw_info_from_devlink_for_testing(dev.value, info);
  } catch (const std::bad_alloc&) {
    return SMI_NIC_STATUS_NO_RESOURCE;
  } catch (...) {
    return SMI_NIC_STATUS_ERROR;
  }
  return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_nic_sensor_info_from_records_for_testing(
    const std::vector<amd::smi::nic::hwmon::SensorRecord_t>& records, smi_nic_sensor_info_t* info) {
  *info = {};
  const auto count = std::min<size_t>(records.size(), SMI_NIC_MAX_SENSORS);
  for (size_t i = 0; i < count; ++i) {
    const auto& record = records[i];
    auto& sensor = info->sensors[i];
    sensor.category = static_cast<uint32_t>(record.category);
    sensor.attribute = static_cast<uint32_t>(record.attribute);
    sensor.unit = static_cast<uint32_t>(record.unit);
    sensor.channel = record.channel;
    sensor.instance = record.instance;
    std::snprintf(sensor.label, sizeof(sensor.label), "%s", record.label.c_str());
    sensor.value = record.value;
  }
  info->count = static_cast<uint32_t>(count);
  return (records.size() > SMI_NIC_MAX_SENSORS) ? SMI_NIC_STATUS_NO_RESOURCE
                                                : SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_hwmon_sensors(smi_nic_ctx_t ctx, uint64_t device,
                                           smi_nic_sensor_info_t* info) {
  if (!ctx || !info) {
    return SMI_NIC_STATUS_WRONG_PARAM;
  }

  auto* nic_system = get_nic_system_from_context(ctx);
  if (!nic_system) {
    return SMI_NIC_STATUS_NOT_INIT;
  }

  std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
  const SmiNic* nic = nic_system->get_nic_by_bdf(device);
  if (!nic) {
    return SMI_NIC_STATUS_NOT_FOUND;
  }

  // The scan allocates strings and vectors; keep a throw off the C ABI boundary.
  try {
    return smi_nic_sensor_info_from_records_for_testing(nic->hwmon_sensors(), info);
  } catch (const std::bad_alloc&) {
    return SMI_NIC_STATUS_NO_RESOURCE;
  } catch (...) {
    return SMI_NIC_STATUS_ERROR;
  }
  return SMI_NIC_STATUS_SUCCESS;
}
