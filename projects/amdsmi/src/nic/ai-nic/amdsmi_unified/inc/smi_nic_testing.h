// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_TESTING_H_
#define AMDSMI_UNIFIED_NIC_TESTING_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "amd_smi/impl/nic/amd_smi_nic_interface.h"
#include "smi_devlink_netlink.h"
#include "smi_ibverbs.h"
#include "smi_nic_hwmon.h"
// Included bare: the header opens its own extern "C".
#include "smi_nic_interface.h"
#include "smi_nic_system.h"
#include "smi_nic_telemetry.h"

// Library-local test seam: wraps a prepared system in a context, so the smi_get_nic_* entry
// points can be driven on a host with no NIC. Not part of the C ABI.
smi_nic_status_t smi_nic_create_context_for_testing(smi_nic_ctx_t* ctx,
                                                    std::unique_ptr<SmiNicSystem> system);

// Library-local test seam: the devlink-to-shim conversion that smi_get_nic_fw_info() applies.
void smi_nic_fw_info_from_devlink_for_testing(const amd::nic::netlink::DevlinkDeviceInfo& device,
                                              smi_nic_fw_info_t* info);

// Library-local test seam: replaces the devlink client smi_get_nic_telemetry() creates per call;
// an empty function restores the real one.
using DevlinkFactory_t = std::function<std::shared_ptr<amd::nic::netlink::IDevlinkClient>()>;
void smi_nic_set_devlink_factory_for_testing(DevlinkFactory_t factory);

// Library-local test seam: replaces the libibverbs MTU query behind SmiInfiniBandPort; an empty
// function restores it.
using PortMtuQuery_t =
    std::function<std::optional<smi_ibverbs::PortMtu>(const std::string& rdma_dev, uint8_t port)>;
void smi_nic_set_port_mtu_query_for_testing(PortMtuQuery_t query);

// Library-local test seams: the snapshot and record conversions behind the telemetry and hwmon
// shim calls.
void smi_nic_telemetry_from_snapshot_for_testing(
    const amd::smi::nic::telemetry::NicTelemetrySnapshot& snapshot, smi_nic_telemetry_t* telemetry);
smi_nic_status_t smi_nic_sensor_info_from_records_for_testing(
    const std::vector<amd::smi::nic::hwmon::SensorRecord_t>& records, smi_nic_sensor_info_t* info);

#endif  // AMDSMI_UNIFIED_NIC_TESTING_H_
