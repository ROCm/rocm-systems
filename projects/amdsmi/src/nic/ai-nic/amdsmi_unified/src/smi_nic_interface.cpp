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

#include <cstring>
#include <cstdlib>
#include <vector>
#include <variant>
#include <string>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic>
#include <limits>

#include "smi_sysfs.h"
#include "smi_ethtool_ioctl.h"
#include "smi_nic_interface.h"
#include "smi_nic_system.h"

struct smi_nic_ctx {
	std::unique_ptr<SmiNicSystem> nic_system;
	std::mutex ctx_mutex;
	std::atomic<bool> init;

	smi_nic_ctx() : init(false) {}
};

static SmiNicSystem* get_nic_system_from_context(smi_nic_ctx *ctx)
{
	if (!ctx || !ctx->init || !ctx->nic_system) {
		return nullptr;
	}
	return ctx->nic_system.get();
}

extern "C" {
smi_nic_status_t smi_nic_create_context(smi_nic_ctx_t *ctx)
{
	try {
		if (!ctx) {
			return SMI_NIC_STATUS_WRONG_PARAM;
		}

		auto context = std::make_unique<smi_nic_ctx>();
		context->nic_system = std::make_unique<SmiNicSystem>();
		context->init = true;
		*ctx = context.release();

		(*ctx)->nic_system->discover_nics();

		return SMI_NIC_STATUS_SUCCESS;

	} catch (const std::bad_alloc&) {
		return SMI_NIC_STATUS_NO_RESOURCE;
	} catch (...) {
		return SMI_NIC_STATUS_ERROR;
	}
}

smi_nic_status_t smi_nic_destroy_context(smi_nic_ctx_t ctx)
{
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

smi_nic_status_t smi_discover_nics(smi_nic_ctx_t ctx, smi_nic_discovery_t *discovery)
{
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
			std::snprintf(discovery->devices[index].bdf, SMI_NIC_MAX_STRING_LENGTH,
						"%s", nic->bdf().c_str());
			index++;
		}

		discovery->count = static_cast<uint32_t>(nics.size());
		return SMI_NIC_STATUS_SUCCESS;

	} catch (const std::exception&) {
		discovery->count = 0;
		return SMI_NIC_STATUS_ERROR;
	}
}

smi_nic_status_t smi_get_nic_driver_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_driver_info_t *info)
{
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
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
	if (!nic) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ports = nic->nic_ports();
	if (ports.empty()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	if (!nic_system->driver_loaded(ports[0].bdf(), DriverType::IONIC)) {
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

smi_nic_status_t smi_get_nic_asic_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_asic_info_t *info)
{
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
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
	if (!nic) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ports = nic->nic_ports();
	if (ports.empty()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	if (!nic_system->driver_loaded(ports[0].bdf(), DriverType::IONIC)) {
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
	std::snprintf(info->serial_number, SMI_NIC_MAX_STRING_LENGTH, "%s",
		nic->serial_number().value_or("N/A").c_str());

	return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_bus_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_bus_info_t *info)
{
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
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
	if (!nic) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ports = nic->nic_ports();
	if (ports.empty()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	if (!nic_system->driver_loaded(ports[0].bdf(), DriverType::IONIC)) {
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

smi_nic_status_t smi_get_nic_numa_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_numa_info_t *info)
{
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
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
	if (!nic) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ports = nic->nic_ports();
	if (ports.empty()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	if (!nic_system->driver_loaded(ports[0].bdf(), DriverType::IONIC)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	*info = {};
	info->node = nic->numa_node().value_or(std::numeric_limits<uint8_t>::max());
	std::snprintf(info->affinity, SMI_NIC_MAX_STRING_LENGTH, "%s",
		nic->numa_affinity(info->node).value_or("N/A").c_str());

	return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_port_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_port_info_t *info)
{
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
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
	if (!nic) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ports = nic->nic_ports();
	if (ports.empty()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	*info = {};

	uint32_t port_count = 0;
	bool driver_not_loaded = false;
	for (uint32_t i = 0; i < ports.size() && port_count < SMI_NIC_MAX_PORTS; i++) {
		const auto& port = ports[i];

		if (!nic_system->driver_loaded(port.bdf(), DriverType::IONIC)) {
			driver_not_loaded = true;
			continue;
		}

		smi_nic_port_t *port_info = &info->ports[port_count];

		port_info->bdf = parse_bdf(port.bdf());
		port_info->port_num = port.port_num().value_or(std::numeric_limits<uint32_t>::max());

		auto port_type = port.port_type();
		std::snprintf(port_info->type, SMI_NIC_MAX_STRING_LENGTH, "%s",
			!port_type.empty() ? port_type.c_str() : "N/A");

		std::string flavour = port.flavour();
		std::snprintf(port_info->flavour, SMI_NIC_MAX_STRING_LENGTH, "%s",
			!flavour.empty() ? flavour.c_str() : "N/A");

		const std::string& netdev = port.interface();
		std::snprintf(port_info->netdev, SMI_NIC_MAX_STRING_LENGTH, "%s",
			!netdev.empty() ? netdev.c_str() : "N/A");

		port_info->ifindex = port.ifindex().value_or(std::numeric_limits<uint8_t>::max());

		auto mac = port.mac_address();
		std::snprintf(port_info->mac_address, SMI_NIC_MAX_STRING_LENGTH, "%s",
			(mac.has_value() && !mac.value().empty()) ? mac.value().c_str() : "N/A");

		port_info->carrier = port.carrier().value_or(std::numeric_limits<uint8_t>::max());
		port_info->mtu = port.mtu().value_or(std::numeric_limits<uint16_t>::max());

		auto link_state = port.link_state();
		std::snprintf(port_info->link_state, SMI_NIC_MAX_STRING_LENGTH, "%s",
			(link_state.has_value() && !link_state.value().empty()) ? link_state.value().c_str() : "N/A");

		port_info->link_speed = port.link_speed().value_or(std::numeric_limits<uint32_t>::max());

		struct ethtool_fecparam fecparam_info{};
		fecparam_info.cmd = ETHTOOL_GFECPARAM;
		port_info->active_fec = (smi_ethtool_ioctl(port.interface(), &fecparam_info) == 0)
			? fecparam_info.active_fec : std::numeric_limits<uint32_t>::max();

		struct ethtool_link_settings link_settings{};
		link_settings.cmd = ETHTOOL_GLINKSETTINGS;
		if (smi_ethtool_ioctl(port.interface(), &link_settings) == 0) {
			std::snprintf(port_info->autoneg, SMI_NIC_MAX_STRING_LENGTH, "%s",
				link_settings.autoneg ? "ON" : "OFF");
		} else {
			std::snprintf(port_info->autoneg, SMI_NIC_MAX_STRING_LENGTH, "%s", "N/A");
		}

		struct ethtool_pauseparam pause_info{};
		pause_info.cmd = ETHTOOL_GPAUSEPARAM;
		if (smi_ethtool_ioctl(port.interface(), &pause_info) == 0) {
			std::snprintf(port_info->pause_autoneg, SMI_NIC_MAX_STRING_LENGTH, "%s",
				pause_info.autoneg ? "ON" : "OFF");
			std::snprintf(port_info->pause_rx, SMI_NIC_MAX_STRING_LENGTH, "%s",
				pause_info.rx_pause ? "ON" : "OFF");
			std::snprintf(port_info->pause_tx, SMI_NIC_MAX_STRING_LENGTH, "%s",
				pause_info.tx_pause ? "ON" : "OFF");
		} else {
			std::snprintf(port_info->pause_autoneg, SMI_NIC_MAX_STRING_LENGTH, "%s", "N/A");
			std::snprintf(port_info->pause_rx, SMI_NIC_MAX_STRING_LENGTH, "%s", "N/A");
			std::snprintf(port_info->pause_tx, SMI_NIC_MAX_STRING_LENGTH, "%s", "N/A");
		}

		port_count++;
	}

	info->num_ports = port_count;
	if (driver_not_loaded && port_count == 0) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_rdma_dev_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_rdma_devices_info_t *info)
{
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
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
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
		if (!nic_system->driver_loaded(port.bdf(), DriverType::IONIC_RDMA)) {
			driver_not_loaded = true;
			continue;
		}

		const auto& ibs = port.infiniband();
		for (uint8_t j = 0; j < ibs.size() && rdma_count < SMI_NIC_MAX_RDMA_DEV; j++) {
			const auto& ib = ibs[j];
			smi_nic_rdma_dev_info_t *rdma_dev = &info->rdma_dev_info[rdma_count];

			std::snprintf(rdma_dev->rdma_dev, SMI_NIC_MAX_STRING_LENGTH, "%s",
				ib.rdma_dev().c_str());
			std::snprintf(rdma_dev->node_guid, SMI_NIC_MAX_STRING_LENGTH, "%s",
				ib.node_guid().value_or("N/A").c_str());
			std::snprintf(rdma_dev->node_type, SMI_NIC_MAX_STRING_LENGTH, "%s",
				ib.node_type().value_or("N/A").c_str());
			std::snprintf(rdma_dev->sys_image_guid, SMI_NIC_MAX_STRING_LENGTH, "%s",
				ib.sys_image_guid().value_or("N/A").c_str());
			std::snprintf(rdma_dev->fw_ver, SMI_NIC_MAX_STRING_LENGTH, "%s",
				ib.fw_ver().value_or("N/A").c_str());

			const auto& ib_ports = ib.ports();
			rdma_dev->num_rdma_ports = ib.ports_num();
			for (uint8_t k = 0; k < ib_ports.size() && k < SMI_NIC_MAX_PORTS; k++) {
				const auto& ib_port = ib_ports[k];
				smi_nic_rdma_port_info_t *port_info = &rdma_dev->rdma_port_info[k];

				std::snprintf(port_info->netdev, SMI_NIC_MAX_STRING_LENGTH, "%s",
					port.interface().c_str());
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

smi_nic_status_t smi_get_nic_port_statistics_count(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t *count)
{
	if (!ctx || !count) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	auto* nic_system = get_nic_system_from_context(ctx);
	if (!nic_system) {
		return SMI_NIC_STATUS_NOT_INIT;
	}

	std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
	if (!nic) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ports = nic->nic_ports();
	if (ports.empty()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	if (port_index >= ports.size()) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	if (!nic_system->driver_loaded(ports[port_index].bdf(), DriverType::IONIC)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	const auto& port = ports[port_index];
	const auto& stats_map = port.get_standard_stats_map();
	*count = static_cast<uint32_t>(stats_map.size());

	return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_port_statistics_list(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, smi_nic_stat_info_t *stats)
{
	if (!ctx || !stats) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	auto* nic_system = get_nic_system_from_context(ctx);
	if (!nic_system) {
		return SMI_NIC_STATUS_NOT_INIT;
	}

	std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
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

	if (!nic_system->driver_loaded(ports[port_index].bdf(), DriverType::IONIC)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	const auto& port = ports[port_index];
	const auto& stats_map = port.get_standard_stats_map();

	if (stats_map.empty()) {
		return SMI_NIC_STATUS_NO_DATA;
	}

	stats->count = static_cast<uint32_t>(std::min(stats_map.size(), (size_t)SMI_NIC_MAX_STATISTICS));
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

smi_nic_status_t smi_get_nic_vendor_statistics_count(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t *count)
{
	if (!ctx || !count) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	auto* nic_system = get_nic_system_from_context(ctx);
	if (!nic_system) {
		return SMI_NIC_STATUS_NOT_INIT;
	}

	std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
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

	if (!nic_system->driver_loaded(ports[port_index].bdf(), DriverType::IONIC)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}


	const auto& port = ports[port_index];
	const auto& stats_map = port.get_vendor_stats_map();
	*count = static_cast<uint32_t>(stats_map.size());

	return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_vendor_statistics_list(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, smi_nic_stat_info_t *stats)
{
	if (!ctx || !stats) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	auto* nic_system = get_nic_system_from_context(ctx);
	if (!nic_system) {
		return SMI_NIC_STATUS_NOT_INIT;
	}

	std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
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

	if (!nic_system->driver_loaded(ports[port_index].bdf(), DriverType::IONIC)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	const auto& port = ports[port_index];
	const auto& stats_map = port.get_vendor_stats_map();

	if (stats_map.empty()) {
		return SMI_NIC_STATUS_NO_DATA;
	}

	stats->count = static_cast<uint32_t>(std::min(stats_map.size(), (size_t)SMI_NIC_MAX_STATISTICS));
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

smi_nic_status_t smi_get_nic_rdma_port_statistics_count(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t ib_index, uint32_t rdma_port_index, uint32_t *count)
{
	if (!ctx || !count) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	auto* nic_system = get_nic_system_from_context(ctx);
	if (!nic_system) {
		return SMI_NIC_STATUS_NOT_INIT;
	}

	std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
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

	if (!nic_system->driver_loaded(ports[port_index].bdf(), DriverType::IONIC_RDMA)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	const auto& ibs = ports[port_index].infiniband();
	if (ib_index >= (uint32_t)ibs.size()) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	const auto& ib_ports = ibs[ib_index].ports();
	if (rdma_port_index >= (uint32_t)ib_ports.size()) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	const auto& ib_port = ib_ports[rdma_port_index];
	const auto& stats_map = ib_port.get_hw_counters_map();
	*count = static_cast<uint32_t>(stats_map.size());

	return SMI_NIC_STATUS_SUCCESS;
}

smi_nic_status_t smi_get_nic_rdma_port_statistics_list(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t ib_index, uint32_t rdma_port_index, smi_nic_stat_info_t *stats)
{
	if (!ctx || !stats) {
		return SMI_NIC_STATUS_WRONG_PARAM;
	}

	auto* nic_system = get_nic_system_from_context(ctx);
	if (!nic_system) {
		return SMI_NIC_STATUS_NOT_INIT;
	}

	std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
	const SmiNic *nic = nic_system->get_nic_by_bdf(device);
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

	if (!nic_system->driver_loaded(ports[port_index].bdf(), DriverType::IONIC_RDMA)) {
		return SMI_NIC_STATUS_DRIVER_NOT_LOADED;
	}

	const auto& ibs = ports[port_index].infiniband();
	if (ib_index >= (uint32_t)ibs.size()) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ib_ports = ibs[ib_index].ports();
	if (rdma_port_index >= (uint32_t)ib_ports.size()) {
		return SMI_NIC_STATUS_NOT_FOUND;
	}

	const auto& ib_port = ib_ports[rdma_port_index];
	const auto& stats_map = ib_port.get_hw_counters_map();

	if (stats_map.empty()) {
		return SMI_NIC_STATUS_NO_DATA;
	}

	stats->count = static_cast<uint32_t>(std::min(stats_map.size(), (size_t)SMI_NIC_MAX_STATISTICS));
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

} // extern "C"
