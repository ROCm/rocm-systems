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

#ifndef __SMI_NIC_INTERFACE_H__
#define __SMI_NIC_INTERFACE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <linux/ethtool.h>

#define SMI_NIC_MAX_STRING_LENGTH 256
#define SMI_NIC_MAX_DEVICES 64
#define SMI_NIC_MAX_STATISTICS 64
#define SMI_NIC_MAX_PORTS 32
#define SMI_NIC_MAX_RDMA_DEV 32

typedef enum {
	SMI_NIC_STATUS_SUCCESS = 0,		/**< API completed successfully */
	SMI_NIC_STATUS_ERROR = 1,		/**< Generic error */
	SMI_NIC_STATUS_WRONG_PARAM = 2,		/**< Wrong parameter provided */
	SMI_NIC_STATUS_NOT_FOUND = 3,		/**< NIC not found */
	SMI_NIC_STATUS_NO_RESOURCE = 4,		/**< Memory allocation failed */
	SMI_NIC_STATUS_NOT_SUPPORTED = 5,	/**< API not supported */
	SMI_NIC_STATUS_NOT_INIT = 6,		/**< Not initialized */
	SMI_NIC_STATUS_NO_DATA = 7,		/**< Requested data not found */
	SMI_NIC_STATUS_DRIVER_NOT_LOADED = 8	/**< Required driver not loaded */
} smi_nic_status_t;

/**
 * @struct smi_nic_discovery_t
 * @brief Structure about discovered NIC devices
 *
 * Contains information about detected network interface cards, including their count
 * and details for each device such as PCI BDF.
 *
 * @var smi_nic_discovery_t::count
 * Number of NIC devices discovered
 * @var smi_nic_discovery_t::devices
 * Array containing details for each discovered NIC device
 */
typedef struct {
	uint32_t count;
	struct {
		char bdf[SMI_NIC_MAX_STRING_LENGTH];		/**< PCI BDF */
	} devices[SMI_NIC_MAX_DEVICES];
} smi_nic_discovery_t;

/**
 * @brief Opaque handle for thread-safe NIC context
 *
 * This handle represents a thread-safe NIC context.
 * Multiple contexts can be created and used concurrently from different threads.
 */
typedef struct smi_nic_ctx *smi_nic_ctx_t;

/**
 * @struct smi_nic_stat_t
 * @brief Structure representing a single statistic name-value pair
 *
 * Contains a statistic name and its corresponding 64-bit value.
 */
typedef struct {
	char name[SMI_NIC_MAX_STRING_LENGTH];
	uint64_t value;
} smi_nic_stat_t;

/**
 * @struct smi_nic_stat_info_t
 * @brief Structure containing an array of statistics
 *
 * Contains the count and array of statistic name-value pairs.
 */
typedef struct {
	uint32_t count;
	smi_nic_stat_t stats[SMI_NIC_MAX_STATISTICS];
} smi_nic_stat_info_t;

/**
 * @struct smi_nic_driver_info_t
 * @brief Structure containing NIC driver information
 *
 * Contains driver name and version information.
 */
typedef struct {
	char name[SMI_NIC_MAX_STRING_LENGTH];
	char version[SMI_NIC_MAX_STRING_LENGTH];
} smi_nic_driver_info_t;

/**
 * @struct smi_nic_asic_info_t
 * @brief Structure containing NIC ASIC information
 *
 * Contains ASIC information including vendor IDs, device IDs,
 * MAC address, product details, and serial number, etc.
 */
typedef struct {
	uint16_t vendor_id;
	uint16_t subvendor_id;
	uint16_t device_id;
	uint16_t subsystem_id;
	uint8_t revision;
	char permanent_address[SMI_NIC_MAX_STRING_LENGTH];
	char product_name[SMI_NIC_MAX_STRING_LENGTH];
	char part_number[SMI_NIC_MAX_STRING_LENGTH];
	char serial_number[SMI_NIC_MAX_STRING_LENGTH];
	char vendor_name[SMI_NIC_MAX_STRING_LENGTH];
} smi_nic_asic_info_t;

/**
 * @struct smi_nic_bus_info_t
 * @brief Structure containing NIC bus/PCIe information
 *
 * Contains PCIe bus information including width, speed, interface version, and slot type.
 */
typedef struct {
	uint64_t bdf;
	uint8_t max_pcie_width;
	uint32_t max_pcie_speed;
	char pcie_interface_version[SMI_NIC_MAX_STRING_LENGTH];
	char slot_type[SMI_NIC_MAX_STRING_LENGTH];
} smi_nic_bus_info_t;

/**
 * @struct smi_nic_numa_info_t
 * @brief Structure containing NIC NUMA information
 *
 * Contains NUMA node and CPU affinity information.
 */
typedef struct {
	uint8_t node;
	char affinity[SMI_NIC_MAX_STRING_LENGTH];
} smi_nic_numa_info_t;

/**
 * @struct smi_nic_port_t
 * @brief Structure containing information for a single NIC port
 *
 * Contains information about a single network port.
 */
typedef struct {
	uint64_t bdf;
	uint32_t port_num;
	char type[SMI_NIC_MAX_STRING_LENGTH];
	char flavour[SMI_NIC_MAX_STRING_LENGTH];
	char netdev[SMI_NIC_MAX_STRING_LENGTH];
	uint8_t ifindex;
	char mac_address[SMI_NIC_MAX_STRING_LENGTH];
	uint8_t carrier;
	uint16_t mtu;
	char link_state[SMI_NIC_MAX_STRING_LENGTH];
	uint32_t link_speed;
	uint32_t active_fec;
	char autoneg[SMI_NIC_MAX_STRING_LENGTH];
	char pause_autoneg[SMI_NIC_MAX_STRING_LENGTH];
	char pause_rx[SMI_NIC_MAX_STRING_LENGTH];
	char pause_tx[SMI_NIC_MAX_STRING_LENGTH];
} smi_nic_port_t;

/**
 * @struct smi_nic_port_info_t
 * @brief Structure containing information for all NIC ports
 *
 * Contains the count and array of port information.
 */
typedef struct {
	uint32_t num_ports;
	smi_nic_port_t ports[SMI_NIC_MAX_PORTS];
} smi_nic_port_info_t;

/**
 * @struct smi_nic_rdma_port_info_t
 * @brief Structure containing information for a single RDMA port
 */
typedef struct {
	char netdev[SMI_NIC_MAX_STRING_LENGTH];
	char state[SMI_NIC_MAX_STRING_LENGTH];
	uint8_t rdma_port;
	uint16_t max_mtu;
	uint16_t active_mtu;
} smi_nic_rdma_port_info_t;

/**
 * @struct smi_nic_rdma_dev_info_t
 * @brief Structure containing information for a single RDMA device
 */
typedef struct {
	char rdma_dev[SMI_NIC_MAX_STRING_LENGTH];
	char node_guid[SMI_NIC_MAX_STRING_LENGTH];
	char node_type[SMI_NIC_MAX_STRING_LENGTH];
	char sys_image_guid[SMI_NIC_MAX_STRING_LENGTH];
	char fw_ver[SMI_NIC_MAX_STRING_LENGTH];
	uint8_t num_rdma_ports;
	smi_nic_rdma_port_info_t rdma_port_info[SMI_NIC_MAX_PORTS];
} smi_nic_rdma_dev_info_t;

/**
 * @struct smi_nic_rdma_devices_info_t
 * @brief Structure containing information for all RDMA devices
 */
typedef struct {
	uint8_t num_rdma_dev;
	smi_nic_rdma_dev_info_t rdma_dev_info[SMI_NIC_MAX_RDMA_DEV];
} smi_nic_rdma_devices_info_t;

/**
 * @brief Create a new thread-safe NIC context
 *
 * Creates a new context handle. Each context maintains its own state and
 * can be used concurrently from different threads.
 *
 * @param[out] ctx Pointer to store the created context handle
 *
 * @return ::SMI_NIC_STATUS_SUCCESS if context created successfully
 * @return ::SMI_NIC_STATUS_WRONG_PARAM if ctx is NULL
 * @return ::SMI_NIC_STATUS_NO_RESOURCE if memory allocation failed
 *
 * @note This function is thread-safe
 * @note The context must be destroyed with smi_nic_destroy_context()
 *
 */
smi_nic_status_t smi_nic_create_context(smi_nic_ctx_t *ctx);

/**
 * @brief Destroy a NIC context and free its resources
 *
 * Destroys the specified context and frees all associated resources.
 *
 * @param[in] ctx Context handle to destroy
 *
 * @return ::SMI_NIC_STATUS_SUCCESS if context destroyed successfully
 * @return ::SMI_NIC_STATUS_WRONG_PARAM if ctx is NULL
 *
 * @note This function is thread-safe
 * @note Do not use the context handle after calling this function
 */
smi_nic_status_t smi_nic_destroy_context(smi_nic_ctx_t ctx);

/**
 * @brief Discover available NICs and their BDFs.
 *
 * Discovers all available network interface cards.
 *
 * @param ctx Context handle
 * @param discovery Pointer to structure that will be filled with discovered NIC info.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 *
 * @note This function is thread-safe when using separate contexts
 * @note Maximum of SMI_NIC_MAX_DEVICES devices can be discovered
 */
smi_nic_status_t smi_discover_nics(smi_nic_ctx_t ctx, smi_nic_discovery_t *discovery);

/**
 * @brief Retrieve NIC driver information.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param info Pointer to smi_nic_driver_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 */
smi_nic_status_t smi_get_nic_driver_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_driver_info_t *info);

/**
 * @brief Retrieve NIC ASIC information.
 *
 * This function retrieves ASIC related information, including
 * vendor IDs, device IDs, revision, MAC address, and product information.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param info Pointer to smi_nic_asic_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 */
smi_nic_status_t smi_get_nic_asic_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_asic_info_t *info);

/**
 * @brief Retrieve NIC bus/PCIe information.
 *
 * This function retrieves bus related information, including
 * PCIe width, speed, interface version, and slot type.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param info Pointer to smi_nic_bus_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 */
smi_nic_status_t smi_get_nic_bus_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_bus_info_t *info);

/**
 * @brief Retrieve NIC NUMA information.
 *
 * This function retrieves NUMA node and CPU affinity information.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param info Pointer to smi_nic_numa_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 */
smi_nic_status_t smi_get_nic_numa_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_numa_info_t *info);

/**
 * @brief Retrieve NIC port information for all ports.
 *
 * This function retrieves information for all ports on the NIC, including
 * interface names, BDFs, link status, speeds, pause settings, etc.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param info Pointer to smi_nic_port_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 */
smi_nic_status_t smi_get_nic_port_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_port_info_t *info);

/**
 * @brief Retrieve RDMA device information for a NIC.
 *
 * @param ctx Context handle
 * @param device BDF of the NIC device
 * @param info Pointer to structure that will be filled with RDMA device info
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 *
 * @note This function aggregates all RDMA/InfiniBand devices across all ports
 * @note Returns SMI_NIC_STATUS_DRIVER_NOT_LOADED if no RDMA driver found on any port
 * @note Returns SMI_NIC_STATUS_NO_DATA if no RDMA devices found
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on fail
 */
smi_nic_status_t smi_get_nic_rdma_dev_info(smi_nic_ctx_t ctx, uint64_t device, smi_nic_rdma_devices_info_t *info);

/**
 * @brief Get the count of available standard port statistics for a specified NIC port.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based).
 * @param count Pointer to uint32_t to store the number of available statistics.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_port_statistics_count(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t *count);

/**
 * @brief Retrieve standard port statistics list for a specified NIC port.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based).
 * @param stats Pointer to smi_nic_stat_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_port_statistics_list(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, smi_nic_stat_info_t *stats);

/**
 * @brief Get the count of available vendor statistics for a specified NIC port.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based).
 * @param count Pointer to uint32_t to store the number of available statistics.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_vendor_statistics_count(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t *count);

/**
 * @brief Retrieve vendor statistics list for a specified NIC port.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based).
 * @param stats Pointer to smi_nic_stat_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_vendor_statistics_list(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, smi_nic_stat_info_t *stats);

/**
 * @brief Get the count of available RDMA hardware counters for a specified InfiniBand port.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based).
 * @param ib_index Index of the InfiniBand device (0-based).
 * @param rdma_port_index Index of the RDMA port (0-based).
 * @param count Pointer to uint32_t to store the number of available counters.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_rdma_port_statistics_count(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t ib_index, uint32_t rdma_port_index, uint32_t *count);

/**
 * @brief Retrieve RDMA hardware counters list for a specified InfiniBand port.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based).
 * @param ib_index Index of the InfiniBand device (0-based).
 * @param rdma_port_index Index of the RDMA port (0-based).
 * @param stats Pointer to smi_nic_stat_info_t structure to be filled.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_rdma_port_statistics_list(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index, uint32_t ib_index, uint32_t rdma_port_index, smi_nic_stat_info_t *stats);

#ifdef __cplusplus
}
#endif

#endif // __SMI_NIC_INTERFACE_H__
