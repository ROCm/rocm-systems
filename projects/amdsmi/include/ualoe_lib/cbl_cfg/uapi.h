/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef IFOE_UAPI_H
#define IFOE_UAPI_H

#include <linux/if_ether.h>
#include <linux/ioctl.h>
#include <linux/time_types.h>
#include <linux/types.h>

#define CFG_LABEL_SIZE 32
#define CFG_PCI_ADDR_MAX_LEN 32

struct cfg_label {
  char text[CFG_LABEL_SIZE];
};

struct cfg_version {
  __u32 major;
  __u32 minor;
  __u32 patch;
};

struct cfg_version_cmd {
  struct cfg_version fw_version;
  struct cfg_version telemetry_version;
};

#define CFG_GET_VERSION _IOR(CFG_IOC_MAGIC, 0x1, struct cfg_version_cmd)

#define CFG_IFOE_MAX_SUPPORTED_ACCELERATORS 1024
#define CFG_IFOE_MAX_PATHS_PER_STATION 4

struct cfg_capabilities {
  __u32 num_configured_stations;
  __u32 max_accelerators;
  __u32 num_netports_per_station;
  __u32 num_paths_per_station;
};

#define CFG_GET_CAPABILITIES _IOR(CFG_IOC_MAGIC, 0x2, struct cfg_capabilities)

struct cfg_station_desc {
  struct cfg_label label;
  __u32 logical_id;
  __u32 physical_id;
};

struct cfg_station_list {
  struct cfg_station_desc* list;
  int count;
};

#define CFG_GET_STATION_LIST _IOWR(CFG_IOC_MAGIC, 0x3, struct cfg_station_list)

#define CFG_SET_IDENTITY _IOW(CFG_IOC_MAGIC, 0x4, __u32)

struct cfg_accelerator_config {
  /**
   * The size of the following bitmasks in bytes
   *
   * Should be a multiple of the size of one bitmask element (u32).
   */
  __u32 bitmask_size;
  __u32* active_accelerator_bitmask;
  __u32* local_accelerator_bitmask;
};

#define CFG_SET_ACCELERATOR_CONFIG _IOW(CFG_IOC_MAGIC, 0x5, struct cfg_accelerator_config)

enum cfg_ifoe_virt_mode { CFG_IFOE_VIRT_MODE_VIRTUALIZED, CFG_IFOE_VIRT_MODE_BARE_METAL };

enum cfg_ifoe_encap_type {
  CFG_IFOE_ENCAP_TYPE_ETHERNET,
  CFG_IFOE_ENCAP_TYPE_TF1,
  CFG_IFOE_ENCAP_TYPE_IP
};

enum cfg_ifoe_failover_mode { CFG_IFOE_FAILOVER_DISABLED, CFG_IFOE_FAILOVER_ENABLED };

enum cfg_ifoe_loopback_mode {
  CFG_IFOE_LOOPBACK_OFF,
  CFG_IFOE_LOOPBACK_IFOE,
  CFG_IFOE_LOOPBACK_IFOE_PRE_MAC
};

struct cfg_ifoe_config {
  enum cfg_ifoe_virt_mode v_mode;
  enum cfg_ifoe_encap_type e_type;
  enum cfg_ifoe_failover_mode f_mode;
  enum cfg_ifoe_loopback_mode loopback_mode;
};

#define CFG_SET_IFOE_CONFIG _IOW(CFG_IOC_MAGIC, 0x6, struct cfg_ifoe_config)

enum cfg_config_phase {
  CFG_CONFIG_PHASE_SYSTEM,
  CFG_CONFIG_PHASE_PROVIDER,
  CFG_CONFIG_PHASE_TENANT,
  CFG_CONFIG_PHASE_SHOWTIME,
  CFG_CONFIG_PHASE_DIAGNOSTICS
};

#define CFG_NEXT_CONFIG_PHASE _IOR(CFG_IOC_MAGIC, 0x7, enum cfg_config_phase)

#define CFG_GET_CURRENT_CONFIG_PHASE _IOR(CFG_IOC_MAGIC, 0x8, enum cfg_config_phase)

enum cfg_network_addr_type {
  CFG_NETWORK_ADDR_TYPE_MAC = (1 << 0),
  CFG_NETWORK_ADDR_TYPE_IP = (1 << 1),
  CFG_NETWORK_ADDR_TYPE_MAC_IP = CFG_NETWORK_ADDR_TYPE_MAC | CFG_NETWORK_ADDR_TYPE_IP,
};

struct cfg_ifoe_accelerator_addr_map {
  __u32 accelerator_id;
  __u8 mac_addr[ETH_ALEN];
  __u32 ip_addr;
};

/* Enum defining the different categories of telemetry */
enum cfg_telemetry_category {
  CFG_TELEMETRY_CATEGORY_IFOE,
  CFG_TELEMETRY_CATEGORY_SWITCH,
  CFG_TELEMETRY_CATEGORY_CRYPTO,
  CFG_TELEMETRY_CATEGORY_PFC,
  CFG_TELEMETRY_CATEGORY_NETPORT,
  CFG_TELEMETRY_CATEGORY_DERIVED_IFOE,
  CFG_TELEMETRY_CATEGORY_DERIVED_NETPORT,
  CFG_TELEMETRY_CATEGORY_MAX
};

struct cfg_telemetry_item {
  __u64 id;
  __u64 value;
};

#define CFG_TELEMETRY_MAX_ITEMS 167
struct cfg_telemetry_instance {
  struct cfg_label name;
  __u32 logical_id;
  __u32 item_count;
  struct cfg_telemetry_item* items;
};

/* Maximum of 144 instances (36 stations × 4 netports each) */
#define CFG_TELEMETRY_MAX_INSTANCES 144
struct cfg_telemetry_dataset {
  enum cfg_telemetry_category category;
  __u64 generation_count;
#ifdef __KERNEL__
  struct __kernel_timespec timestamp;
#else
  struct timespec timestamp;
#endif
  __u32 instance_count;
  struct cfg_telemetry_instance* instances;
};

// TODO how to specify number of categories? rw ioctl?
struct cfg_telemetry {
  struct cfg_telemetry_dataset* datasets;
  int dev_id;
};

#define CFG_GET_TELEMETRY _IOR(CFG_IOC_MAGIC, 0x9, struct cfg_telemetry)

struct cfg_set_telemetry {
  unsigned int category_mask;
  int dev_id;
};

#define CFG_SET_TELEMETRY _IOW(CFG_IOC_MAGIC, 0xA, struct cfg_set_telemetry)

#define CFG_FREE_TELEMETRY _IOW(CFG_IOC_MAGIC, 0xB, int)

#define CFG_L2PING_MAX_PINGS 255

/**
 * struct cfg_l2ping_config - L2 ping test configuration and response
 * @dev_id: Device ID
 * @specify_accelerator: Target specific accelerator if true
 * @specify_netport: Target specific network port if true
 * @include_ifoe_req: Include IFoE Request traffic in test
 * @include_ifoe_resp: Include IFoE Response traffic in test
 * @include_non_ifoe: Include non-IFoE traffic in test
 * @accelerator_id: Target accelerator ID (if specify_accelerator is true)
 * @netport_idx: Target network port index (if specify_netport is true)
 * @num_pings: Number of pings to perform on each path (1-255)
 * @num_accelerators: [OUT] Number of accelerators in results
 * @num_netports: [OUT] Number of netports in results
 * @handle: [OUT] Handle for this ping request
 *
 * Input fields (dev_id through num_pings) are provided by
 * userspace to configure the ping test.
 * Output fields (num_accelerators, num_netports, handle) are populated
 * by the driver with information needed to allocate result buffers.
 */
struct cfg_l2ping_config {
  int dev_id;
  __u8 specify_accelerator;
  __u8 specify_netport;
  __u8 include_ifoe_req;
  __u8 include_ifoe_resp;
  __u8 include_non_ifoe;
  __u32 accelerator_id;
  __u32 netport_idx;
  __u32 num_pings;
  __u32 num_accelerators;
  __u32 num_netports;
  __u32 handle;
};

#define CFG_IFOE_L2PING_CONFIG _IOWR(CFG_IOC_MAGIC, 0xD, struct cfg_l2ping_config)

/**
 * struct cfg_l2ping_start - Start L2 ping test
 * @dev_id: Device ID
 * @handle: Handle from CFG_IFOE_L2PING_CONFIG identifying the ping request
 *
 * This ioctl starts the L2 ping test that was previously configured with
 * CFG_IFOE_L2PING_CONFIG.
 */
struct cfg_l2ping_start {
  int dev_id;
  __u32 handle;
};

#define CFG_IFOE_L2PING_START _IOW(CFG_IOC_MAGIC, 0xE, struct cfg_l2ping_start)

/**
 * struct cfg_l2ping_netport_result - L2 ping result for a netport
 * @req_failures: Count of ping failures on IFoE Request PFC channel
 * @resp_failures: Count of ping failures on IFoE Response PFC channel
 * @non_ifoe_failures: Count of ping failures on non-IFoE traffic PFC channel
 */
struct cfg_l2ping_netport_result {
  __u8 req_failures;
  __u8 resp_failures;
  __u8 non_ifoe_failures;
};

/**
 * struct cfg_l2ping_accel_result - L2 ping result for an accelerator
 * @id: Accelerator ID
 * @netports: Pointer to array of netport results
 */
struct cfg_l2ping_accel_result {
  __u32 id;
  struct cfg_l2ping_netport_result* netports;
};

/**
 * struct cfg_l2ping_update - Update L2 ping test status and results
 * @dev_id: Device ID
 * @handle: Handle from CFG_IFOE_L2PING_CONFIG identifying the ping request
 * @num_accelerators: Number of accelerators in results array
 * @num_netports: Number of netports per accelerator
 * @accels: Pointer to array of accelerator results
 * @complete: [OUT] Test completion flag
 * @progress: [OUT] Current progress counter
 * @total: [OUT] Total progress value
 *
 * This ioctl retrieves the current status and results of the L2 ping test.
 * The driver populates the results arrays and progress fields.
 *
 * When both num_accelerators and num_netports are set to 0 on entry,
 * the driver will return the actual number of accelerators and netports
 * from the device without copying result data. This allows userspace to
 * query the required buffer sizes before allocating memory for results.
 */
struct cfg_l2ping_update {
  int dev_id;
  __u32 handle;
  __u32 num_accelerators;
  __u32 num_netports;
  struct cfg_l2ping_accel_result* accels;
  __u8 complete;
  __u32 progress;
  __u32 total;
};

#define CFG_IFOE_L2PING_UPDATE _IOWR(CFG_IOC_MAGIC, 0xF, struct cfg_l2ping_update)

/**
 * struct cfg_l2ping_fini - Finalize L2 ping test
 * @dev_id: Device ID
 * @handle: Handle from CFG_IFOE_L2PING_CONFIG identifying the ping request
 *
 * This ioctl finalizes the L2 ping test, releasing any resources
 * allocated by the driver for this ping request.
 */
struct cfg_l2ping_fini {
  int dev_id;
  __u32 handle;
};

#define CFG_IFOE_L2PING_FINI _IOW(CFG_IOC_MAGIC, 0x10, struct cfg_l2ping_fini)

/**
 * struct cfg_connect_cmd - Command structure for CFG_CONNECT ioctl
 * @pci_addr: PCI address string input by userspace to identify the device
 * @dev_id: Device ID output by the driver after successful connection
 *
 * Userspace fills in the pci_addr field with the PCI address of the IFoE
 * device it wants to connect to. The driver processes this command, attempts
 * to connect to the specified device, and if successful, fills in the dev_id
 * field with a unique identifier for that device which will be used in
 * subsequent ioctl commands to refer to this device.
 */
struct cfg_connect_cmd {
  char pci_addr[CFG_PCI_ADDR_MAX_LEN]; /* input */
  int dev_id;                          /* output */
};

#define CFG_CONNECT _IOWR(CFG_IOC_MAGIC, 0x11, struct cfg_connect_cmd)

#define CFG_RESET _IO(CFG_IOC_MAGIC, 0xC)

#define CFG_IOC_MAGIC '#'

#define CFG_FAMILY_NAME "ifoe-cfg"
#define CFG_MC_GRP_NAME "ifoe-cfg-mcgrp"

/**
 * DOC: IFoE Config Driver-Userspace Event Interface
 *
 * The configuration driver can generate events to notify user space of
 * configuration phase changes. These events are sent via a netlink to a
 * userspace process that has joined the multicast group "ifoe-cfg-mcgrp".
 *
 * CFG_EVT_CMD_PHASE_EVENT:
 * - CFG_EVT_ATTR_TS: Timestamp of the event
 * - CFG_EVT_ATTR_PHASE_EVENT: New configuration phase
 */

enum {
  CFG_EVT_CMD_PHASE_EVENT = 1,
  CFG_EVT_CMD_IFOE_LINK_EVENT,
  CFG_EVT_CMD_NETPORT_LINK_EVENT,
  __CFG_EVT_CMD_MAX
};
#define CFG_EVT_CMD_MAX (__CFG_EVT_CMD_MAX - 1)

enum {
  CFG_EVT_ATTR_TS = 1,
  CFG_EVT_ATTR_PHASE_EVENT,
  CFG_EVT_ATTR_LABEL,
  CFG_EVT_ATTR_LOGICAL_IDX,
  CFG_EVT_ATTR_LINK_DOWN,
  CFG_EVT_ATTR_DX_ISOLATED,
  CFG_EVT_ATTR_NETPORT_COUNT,
  CFG_EVT_ATTR_NETPORT0,
  CFG_EVT_ATTR_NETPORT1,
  CFG_EVT_ATTR_NETPORT2,
  CFG_EVT_ATTR_NETPORT3,
  CFG_EVT_ATTR_DEV_ID,
  __CFG_EVT_ATTR_MAX
};
#define CFG_EVT_ATTR_MAX (__CFG_EVT_ATTR_MAX - 1)

enum { CFG_EVT_ATTR_TS_SECS = 1, CFG_EVT_ATTR_TS_NSECS, __CFG_EVT_ATTR_TS_MAX };
#define CFG_EVT_ATTR_TS_MAX (__CFG_EVT_ATTR_TS_MAX - 1)

enum {
  CFG_EVT_ATTR_NETPORT_LOGICAL_IDX = 1,
  CFG_EVT_ATTR_NETPORT_FAULT,
  CFG_EVT_ATTR_NETPORT_STREAMS_FAILOVER,
  CFG_EVT_ATTR_NETPORT_STREAMS_PAUSED,
  __CFG_EVT_ATTR_NETPORT_MAX
};
#define CFG_EVT_ATTR_NETPORT_MAX (__CFG_EVT_ATTR_NETPORT_MAX - 1)

/**
 * DOC: IFoE Config Userspace-Driver Command Interface
 *
 * The configuration driver exposes a set of commands via a netlink
 * interface for userspace processes to query and modify the IFoE
 * configuration.
 */
enum {
  CFG_CMD_CONNECT = 1,
  CFG_CMD_GET_VERSION,
  CFG_CMD_GET_IFOE_CONFIG,
  CFG_CMD_SET_ACCELERATOR_ID,
  CFG_CMD_SET_ACCELERATOR_CONFIG,
  CFG_CMD_SET_IFOE_CONFIG,
  CFG_CMD_GET_CAPABILITIES,
  CFG_CMD_GET_CURRENT_CONFIG_PHASE,
  CFG_CMD_SET_CONFIG_PHASE,
  CFG_CMD_SET_ENABLED_ACCELERATOR,
  CFG_CMD_RESET,
  CFG_CMD_GET_STATION_LIST,
  CFG_CMD_GET_NETPORT_LIST,
  CFG_CMD_STATION_CTRL,
  CFG_CMD_STATION_GET_STATE,
  CFG_CMD_NETPORT_CTRL,
  CFG_CMD_NETPORT_GET_STATE,
  CFG_CMD_NETPORT_SET_ADDR,
  CFG_CMD_NETPORT_SET_ACCELERATOR_ADDR_MAP,
  CFG_CMD_NETPORT_GET_PROPERTIES,
  CFG_CMD_SET_PATH_PORT_MAP,
  CFG_CMD_GET_PATH_PORT_MAP,
  CFG_CMD_IFOE_CONFIG_CRYPTO,
  CFG_CMD_IFOE_SET_TX_CRYPTO_KEY,
  CFG_CMD_IFOE_DISABLE_RX_CRYPTO_KEY,
  CFG_CMD_IFOE_SET_RX_CRYPTO_KEY,
  CFG_CMD_NETPORT_CONFIG_LINK_AUTO,
  CFG_CMD_NETPORT_CONFIG_LINK_MANUAL,
  __CFG_CMD_MAX
};
#define CFG_CMD_MAX (__CFG_CMD_MAX - 1)

enum {
  CFG_ATTR_IFOE_CONFIG = 1,
  CFG_ATTR_DEV_ID,
  CFG_ATTR_PCI_ADDR,
  CFG_ATTR_ACCEL_CONFIG,
  CFG_ATTR_ACCELERATOR_ID,
  CFG_ATTR_CAPABILITIES,
  CFG_ATTR_FW_VERSION,
  CFG_ATTR_TELEMETRY_VERSION,
  CFG_ATTR_CONFIG_PHASE,
  CFG_ATTR_DESC_COUNT,
  CFG_ATTR_BANDWIDTH,
  CFG_ATTR_STATION_LABEL,
  CFG_ATTR_STATION_LOGICAL_IDX,
  CFG_ATTR_STATION_PHYSICAL_IDX,
  CFG_ATTR_STATION_STATE,
  CFG_ATTR_STATION_LINK_DOWN,
  CFG_ATTR_STATION_DX_ISOLATED,
  CFG_ATTR_NETPORT_COUNT,
  CFG_ATTR_NETPORT_LABEL,
  CFG_ATTR_NETPORT_LOGICAL_IDX,
  CFG_ATTR_NETPORT_REL_IDX,
  CFG_ATTR_NETPORT_STATE,
  CFG_ATTR_NETPORT_AUTONEG_ENABLED,
  CFG_ATTR_NETPORT_PARALLEL_DETECT_ENABLED,
  CFG_ATTR_NETPORT_LINK_FLAGS,
  CFG_ATTR_NETPORT_LINK_TECHNOLOGY,
  CFG_ATTR_NETPORT_FEC_MODE,
  CFG_ATTR_NETPORT_LOOPBACK_MODE,
  CFG_ATTR_NETPORT_IFOE_ADDR_TYPE,
  CFG_ATTR_NETPORT_IFOE_MAC_ADDR,
  CFG_ATTR_NETPORT_IFOE_IP_ADDR,
  CFG_ATTR_NETPORT_PERM_ADDR,
  CFG_ATTR_NETPORT_ACCELERATOR_ADDR_MAP,
  CFG_ATTR_NETPORT_ETH_TECH_MASK,
  CFG_ATTR_NETPORT_NUM_LANES,
  CFG_ATTR_NETPORT_MAX_FRAME_LEN,
  CFG_ATTR_ENABLED_ACCEL,
  CFG_ATTR_ENABLED_ACCEL_SIZE,
  CFG_ATTR_REENABLE_STREAMS,
  CFG_ATTR_PATH_TO_PORT_MAP,
  CFG_ATTR_CRYPTO_MODE,
  CFG_ATTR_CRYPTO_KEY,
  CFG_ATTR_CRYPTO_KEY_LEN,
  CFG_ATTR_CRYPTO_KEY_ID,
  CFG_ATTR_NETPORT0,
  CFG_ATTR_NETPORT1,
  CFG_ATTR_NETPORT2,
  CFG_ATTR_NETPORT3,
  CFG_ATTR_CATEGORY_MASK,
  CFG_ATTR_TELEMETRY_ENABLE,
  __CFG_ATTR_MAX
};
#define CFG_ATTR_MAX (__CFG_ATTR_MAX - 1)

enum {
  CFG_ATTR_NETPORT_STATE_IDX = 1,
  CFG_ATTR_NETPORT_STATE_FAULT,
  CFG_ATTR_NETPORT_STATE_STREAMS_FAILOVER,
  CFG_ATTR_NETPORT_STATE_STREAMS_PAUSED,
  __CFG_ATTR_NETPORT_STATE_MAX
};
#define CFG_ATTR_NETPORT_STATE_MAX (__CFG_ATTR_NETPORT_STATE_MAX - 1)

enum {
  CFG_ATTR_IFOE_CONFIG_FLAGS = 1,
  CFG_ATTR_IFOE_CONFIG_ACCEL_ID,
  CFG_ATTR_IFOE_CONFIG_CURR_PHASE,
  CFG_ATTR_IFOE_CONFIG_VIRT_MODE,
  CFG_ATTR_IFOE_CONFIG_ENCAP_MODE,
  CFG_ATTR_IFOE_CONFIG_FAILOVER_MODE,
  CFG_ATTR_IFOE_CONFIG_CRYPTO_MODE,
  CFG_ATTR_IFOE_CONFIG_ENABLED_ACCEL,
  CFG_ATTR_IFOE_CONFIG_LOOPBACK_MODE,
  __CFG_ATTR_IFOE_CONFIG_MAX
};
#define CFG_ATTR_IFOE_CONFIG_MAX (__CFG_ATTR_IFOE_CONFIG_MAX - 1)

enum {
  CFG_ATTR_ACCEL_CONFIG_BITMASK_SIZE = 1,
  CFG_ATTR_ACCEL_CONFIG_ACTIVE_BITMASK,
  CFG_ATTR_ACCEL_CONFIG_LOCAL_BITMASK,
  __CFG_ATTR_ACCEL_CONFIG_MAX
};
#define CFG_ATTR_ACCEL_CONFIG_MAX (__CFG_ATTR_ACCEL_CONFIG_MAX - 1)

enum {
  CFG_ATTR_CAPS_IFOE_STATION_COUNT = 1,
  CFG_ATTR_CAPS_ACCELERATOR_COUNT,
  CFG_ATTR_CAPS_NETPORTS_PER_STATION,
  CFG_ATTR_CAPS_PATHS_PER_STATION,
  __CFG_ATTR_CAPABILITIES_MAX
};
#define CFG_ATTR_CAPABILITIES_MAX (__CFG_ATTR_CAPABILITIES_MAX - 1)

enum {
  CFG_ATTR_VERSION_MAJOR = 1,
  CFG_ATTR_VERSION_MINOR,
  CFG_ATTR_VERSION_PATCH,
  __CFG_ATTR_VERSION_MAX
};

#define CFG_ATTR_VERSION_MAX (__CFG_ATTR_VERSION_MAX - 1)

#endif /* IFOE_UAPI_H */
