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

#ifndef UALOE_LIB_H
#define UALOE_LIB_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "ifoe_telemetry_hostapp.h"
#include "ualoe_mcdi_defs.h"

/* UALoE library source compatibility version. This is incremented
 * whenever a definition is added, removed or changed such that a
 * client might need to guard its use with a compile-time check.
 */
#define UALOE_LIB_VERSION_MAJOR 0
#define UALOE_LIB_VERSION_MINOR 1
#define UALOE_LIB_VERSION_PATCH 0

/* ----------------- */
/* Types and Defines */
/* ----------------- */

/** Handle of an UALoE driver instance */
typedef int ualoe_handle_t;

/** Size of an IFoE textual label */
#define UALOE_LABEL_SIZE (32)

/** Maximum number of Network Ports per IFoE Station */
#define UALOE_MAX_NETPORTS_PER_IFOE_STATION (4)

/** Defines size of a crypto key */
#define UALOE_CRYPTO_KEY_SIZE (8)

/** Size of a MAC address */
#define UALOE_MAC_ADDRESS_SIZE (6)

/** Maximum number of pings supported per path for the L2Ping mechanism */
#define UALOE_L2PING_MAX_PINGS (255)

/** Enum defining different configuration phases for UALoE. Note that UALoE
 * enforces a strict order of phases
 *
 *        System -> Provider -> Tenant -> Showtime
 *                    \
 *                     \-> Diagnostics
 *
 * The only transitions permitted are to the next phase via an explicit
 * call to ualoe_next_config_phase() or by an entity reset which will
 * reset back to the Provider or Tenant configuration phase, depending
 * on whether a handle to the PF or VF is being used
 *
 * Additionally a diagnostics mode is supported to allow offline diagnostics
 * tests to be carried out e.g. Network Port PRBS. A PF FLR or entity reset
 * will reset back to the Provider configuration phase.
 *
 * System
 * ------
 * After the UALoE firmware boots, it will enter this phase waiting for the
 * UALoE station and network configuration. This configuration is provided by the
 * SoC before the host OS boots to allow IFoE to bring up and train the Network
 * Ports and therefore the host driver should not expect to ever see IFoE in
 * this state.
 *
 * Provider
 * --------
 * After the System configuration completes the firmware is in this state
 * waiting for the Provider to configure the IFoE datapath. This configuration
 * is provided from the AI Fabric Manager via the host kernel driver.
 * Configuration that the Provider should perform via the PF...
 *   ualoe_set_identity()
 *   ualoe_set_accelerator_config()
 *   ualoe_set_ifoe_config()
 *   ifoe_netport_set_accelerator_addr_map()
 *   ifoe_netport_set_addr()
 *
 * Tenant
 * ------
 * After the core Provider configuration is complete, the firmware enters this
 * state waiting to receive the Tenant configuration. Configuration that the
 * Tenant should perform via the VF in virtualized mode of via the PF in Bare
 * Metal mode...
 *   ualoe_config_crypto()
 *   ualoe_enable_accelerators()
 *
 * Showtime
 * --------
 * Configuration is complete and the UALoE datapath is active.
 *
 * Diagnostics
 * -----
 * Offline diagnostics mode
 */
typedef enum ualoe_config_phase {
  UALOE_CONFIG_PHASE_SYSTEM,
  UALOE_CONFIG_PHASE_PROVIDER,
  UALOE_CONFIG_PHASE_TENANT,
  UALOE_CONFIG_PHASE_SHOWTIME,
  UALOE_CONFIG_PHASE_DIAGNOSTICS
} ualoe_config_phase_e;

/** For the client there are 2 roles defined: Provider and Tenant.
 * The Provider is responsible for configuring the IFoE datapath, including
 * setting the set of the accelerators in the vpod, which of these is local
 * and configuring various global settings. The Tenant is responsible for
 * configuring encryption (if required), validating the Provider configuration,
 * enabling communications with remote accelerators and transitioning the
 * datapath to showtime.
 *
 * ifoe_virt_mode_e defines the virtualization mode in which IFoE is operating.
 * This controls whether IFoE is operating with a host hypervisor (PF) and
 * guest (VM) or in baremetal mode using the PF only.
 * In virtualized mode, the PF has the role of Provider and VF has the role of
 * Tenant. In bare-metal mode, the PF has both the role of Provider and Tenant
 * and the VF is not used.
 */
typedef enum ifoe_virt_mode {
  IFOE_VIRT_MODE_VIRTUALIZED,
  IFOE_VIRT_MODE_BARE_METAL
} ifoe_virt_mode_e;

/** Enum defining the encapsulation mode that will be used for IFoE traffic */
typedef enum ifoe_encap_type {
  IFOE_ENCAP_TYPE_ETHERNET,
  IFOE_ENCAP_TYPE_TF1,
  IFOE_ENCAP_TYPE_IP
} ifoe_encap_type_e;

/** Enum defining the failover mode that should be used
 *
 * When failover is disabled and a retransmission timeout occurs for an IFoE
 * stream, the stream will be paused and the caller notified. It is the
 * responsibility of the client to take action before the failed memory
 * transaction times out.
 *
 * When failover is enabled and a retransmission timeout occurs for an IFoE
 * stream, the firmware will remap the path for the failed stream to a
 * different network port (if possible) and notify the caller that a failover
 * has occurred. If a remap is not possible, for example because the device is
 * operating in 800G port mode, then the caller will simply be notified that
 * a stream has failed.
 */
typedef enum ifoe_failover_mode {
  IFOE_FAILOVER_DISABLED,
  IFOE_FAILOVER_ENABLED
} ifoe_failover_mode_e;

/** Enum defining the loopback modes for IFoE */
typedef enum ifoe_loopback_mode {
  IFOE_LOOPBACK_OFF,
  IFOE_LOOPBACK_IFOE,
  IFOE_LOOPBACK_IFOE_PRE_MAC
} ifoe_loopback_mode_e;

/** Enum defining the encryption mode to be used for UALoE traffic */
typedef enum ualoe_crypto_mode {
  UALOE_CRYPTO_MODE_OFF,
  UALOE_CRYPTO_MODE_AES_GCM_256
} ualoe_crypto_mode_e;

/** Enum used to specify the encryption key ID used for UALoE traffic */
typedef enum ualoe_crypto_key_id {
  UALOE_CRYPTO_KEY_ID_0,
  UALOE_CRYPTO_KEY_ID_1
} ualoe_crypto_key_id_e;

/** Enum defining the state of an IFoE Station */
typedef enum ifoe_station_state {
  /** In this state a station is disabled and Network Ports owned by the IFoE
   * station will be powered off */
  IFOE_STATION_DISABLED,
  /** In this state, the IFoE station is in a state where the datapath is not
   * not active and traffic originating from both SDP and the network will be
   * dropped. */
  IFOE_STATION_STOPPED,
  /** In this state, the IFoE datapath is enabled with traffic flow enabled
   * in both directions */
  IFOE_STATION_ACTIVE
} ifoe_station_state_e;

/** Enum defining the state of a Network Port */
typedef enum ifoe_netport_state {
  /** In this state the network port is disabled and powered off */
  IFOE_NETPORT_POWEROFF,
  /** In this state, the Network Port is in an idle state, powered on
   * but not being used for IFoE traffic flow */
  IFOE_NETPORT_IDLE,
  /** In this state, the network port is enabled for IFoE traffic */
  IFOE_NETPORT_ENABLED
} ifoe_netport_state_e;

/** Structure defining an UALoE version. Follows semantic versioning. */
typedef struct ualoe_version_s {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} ualoe_version_t;

/** Structure defining a textual label. Labels must be null terminated */
typedef struct ualoe_label_s {
  char text[UALOE_LABEL_SIZE];
} ualoe_label_t;

/** Structure defining UALoE device capabilities */
typedef struct ualoe_capabilities_s {
  /** Number of IFoE Stations configured for use */
  unsigned num_configured_stations;
  /** Number of accelerators supported by the device */
  unsigned max_accelerators;
  /** Number of Network Ports configured for each IFoE Station */
  unsigned num_netports_per_station;
  /** Number of IFoE Paths configured for each IFoE station */
  unsigned num_paths_per_station;
} ualoe_capabilities_t;

/** Structure to hold an encryption/decryption key */
typedef struct ualoe_crypto_key_s {
  uint32_t key[UALOE_CRYPTO_KEY_SIZE];
} ualoe_crypto_key_t;

/** Flags indicating items that have been configured by a client (as opposed
 * to being at the firmware default)
 */
#define IFOE_ACCELERATOR_ID_CONFIGURED (1 << 0)
#define IFOE_VIRT_MODE_CONFIGURED (1 << 2)
#define IFOE_ENCAP_MODE_CONFIGURED (1 << 3)
#define IFOE_FAILOVER_MODE_CONFIGURED (1 << 4)
#define IFOE_CRYPTO_MODE_CONFIGURED (1 << 5)
#define IFOE_LOOPBACK_MODE_CONFIGURED (1 << 6)
#define IFOE_ACTIVE_ACCELERATORS_CONFIGURED (1 << 7)
#define IFOE_LOCAL_ACCELERATORS_CONFIGURED (1 << 8)
#define IFOE_ENABLED_ACCELERATORS_CONFIGURED (1 << 9)

/** Structure defining the global IFoE configuration */
typedef struct ifoe_config_s {
  /** Status flags indicating what has been configured by a client. If clear,
   * this means that the configuration is currently the firmware default. Note
   * that if an item has been configured by a client, whether this was by the
   * Provider or by the Tenant, the flag will be set even if the configuration
   * item matches the firmware default. */
  unsigned configured_flags;
  /** The GPU accelerator ID of the GPU connected to this IFoE system. The
   * firmware default when unconfigured and/or following a PF FLR is 0. */
  unsigned accelerator_id;
  /** Current IFoE configuration phase. When unconfigured and/or following
   * a PF FLR, this will be UALOE_CONFIG_PHASE_PROVIDER. */
  ualoe_config_phase_e current_phase;
  /** Virtualization mode. When unconfigured and/or following a PF FLR,
   * this will be IFOE_VIRT_MODE_VIRTUALIZED. */
  ifoe_virt_mode_e virt_mode;
  /** Encapsulation mode. When unconfigured and/or following a PF FLR, this
   * will be IFOE_ENCAP_TYPE_ETHERNET. */
  ifoe_encap_type_e encap_mode;
  /** Failover mode. When unconfigured and/or following a PF FLR, this will
   * be IFOE_FAILOVER_ENABLED. */
  ifoe_failover_mode_e failover_mode;
  /** Encryption mode. When unconfigured and/or following a PF FLR, this
   * will be UALOE_CRYPTO_MODE_OFF. */
  ualoe_crypto_mode_e crypto_mode;
  /** Loopback mode */
  ifoe_loopback_mode_e loopback_mode;
} ifoe_config_t;

/** Structure defining an IFoE Station descriptor */
typedef struct ifoe_station_desc_s {
  /** Label of IFoE station */
  ualoe_label_t name;
  /** Logical index of the IFoE Station within the device. Logical indices are
   * numbered sequentially from zero. */
  unsigned logical_idx;
  /** Physical index of the IFoE Station within the device */
  unsigned physical_idx;
} ifoe_station_desc_t;

/** Structure defining the state of an IFoE Station
 *
 * The set of IFoE Stations is configured in the BIOS and will be set before
 * IFoE transitions to the Provider configuration phase. The initial state of
 * each IFoE Station at the start of the Provider phase is as follows:
 *   .state = IFOE_STATION_ACTIVE
 *   .dx_isolated = FALSE
 *   .netports[].fault = FALSE
 *   .netports[].streams_failover = 0
 *   .netports[].streams_paused = 0
 * The value of other fields depends on the station instance and the state of
 * the underlying network ports.
 */
typedef struct ifoe_station_state_s {
  /** IFoE Station state */
  ifoe_station_state_e state;
  /** An IFoE Station Logical Link represents the network link provided by the
   * set of underlying physical network ports. This boolean indicates that the
   * logical link is down. It will only be true if _all_ the underlying network
   * ports are down and traffic flow to the ethernet network is not possible. */
  bool link_down;
  /** Boolean indicating that that datapath interface between IFoE and data
   * fabric is isolated i.e. no data flow is disabled in both directions. */
  bool dx_isolated;
  /** Bandwidth of the IFoE logical link in Gbits/s. If the logical link
   * is degraded due to an issue with one of the underlying Network Ports,
   * this bandwidth will be reduced. */
  unsigned bandwidth;
  /** Logical index of the IFoE Station within the device. Logical indices are
   * numbered sequentially from zero. */
  unsigned logical_idx;
  /** Physical index of the IFoE Station within the device */
  unsigned physical_idx;
  /** Number of Network Port structures populated */
  unsigned netport_count;
  /** Array of Network Port information */
  struct {
    /** Logical index of the network port */
    unsigned logical_idx;
    /** When true, indicates that there is a fault on the network port. */
    bool fault;
    /** Count of number of streams currently in a failover state which have been
     * remapped to an alternative Network Port. */
    unsigned streams_failover;
    /** Count of number of streams currently in a paused state due to a
     * retransmission timeout where failover was either not possible or
     * disabled. */
    unsigned streams_paused;
  } netports[UALOE_MAX_NETPORTS_PER_IFOE_STATION];
} ifoe_station_state_t;

/** Structure defining Network Port properties */
typedef struct ualoe_netport_properties_s {
  /** Bitmask of ethernet technologies supported (see UALOE_ETH_TECH_MASK) */
  __uint128_t eth_tech_mask;
  /** Bitmask of FEC modes supported (see UALOE_FEC_MODE_MASK) */
  uint32_t fec_modes;
  /** Number of serdes lanes used in the configured port mode */
  unsigned num_lanes;
  /** Bitmask of loopback modes supported (see UALOE_NETPORT_LOOPBACK_MASK) */
  uint64_t loopback_modes;
  /** Maximum frame length supported */
  unsigned max_frame_len;
} ualoe_netport_properties_t;

/** Structure defining an Network Port descriptor */
typedef struct ifoe_netport_desc_s {
  /** Label of Network Port */
  ualoe_label_t name;
  /** Logical index of the Network Port */
  unsigned logical_idx;
  /** IFoE Station that Network Port is part of */
  ualoe_label_t station_name;
  /** Logical index of the IFoE Station */
  unsigned station_idx;
  /** Station-relative index for this Network Port. Within each IFoE station
   * Network Ports are numbered sequentially from zero. */
  unsigned station_rel_netport_idx;
} ifoe_netport_desc_t;

/** Structure containing the state of a Network Port
 *
 * The set of Network Ports is configured in the BIOS based on the port mode
 * and the number of configured IFoE stations and will be set before IFoE
 * transitions to the Provider configuration phase. The initial state of each
 * Network Port at the start of the Provider phase is as follows:
 *   .state = IFOE_NETPORT_ENABLED
 *   .autoneg_enabled = TRUE
 *   .parallel_detect_enabled = TRUE
 *   .loopback_mode = UALOE_NETPORT_LOOPBACK_NONE
 * The value of other fields depends on the network port link status and
 * the link technology and FEC mode negotiated with the link partner.
 */
typedef struct ifoe_netport_state_s {
  /** Network Port state */
  ifoe_netport_state_e state;
  /** Boolean indicating whether Auto-Negotiation is enabled */
  bool autoneg_enabled;
  /** Boolean indicating whether Parallel-Detect is enabled */
  bool parallel_detect_enabled;
  /** Network Port link status flags. For the definition of the flags, refer
   * to the defines UALOE_NETPORT_STATUS_FLAG_* in ualoe_mcdi_defs.h. */
  uint64_t link_flags;
  /** Configured link technology */
  ualoe_eth_tech_e link_technology;
  /** FEC mode in use */
  ualoe_fec_mode_e fec_mode;
  /** Loopback mode */
  ualoe_netport_loopback_mode_e loopback_mode;
  /** MAC address configured for IFoE traffic */
  uint8_t ifoe_mac_addr[UALOE_MAC_ADDRESS_SIZE];
  /** Permanent MAC address assigned to this Network Port. This can be used for
   * non-IFoE traffic. */
  uint8_t permanent_mac_addr[UALOE_MAC_ADDRESS_SIZE];
} ifoe_netport_state_t;

/** Enum defining the type of used network address */
typedef enum ifoe_network_addr_type {
  IFOE_NETWORK_ADDR_TYPE_MAC,
  IFOE_NETWORK_ADDR_TYPE_IP,
  IFOE_NETWORK_ADDR_TYPE_MAC_IP,
} ifoe_network_addr_type_e;

/** Structure defining a mapping from accelerator ID to destination address */
typedef struct ifoe_accelerator_addr_map_s {
  /** Accelerator ID */
  unsigned accelerator_id;
  /** MAC address */
  uint8_t mac_addr[UALOE_MAC_ADDRESS_SIZE];
  /** IP address (in Network byte order) */
  uint32_t ip_addr;
} ifoe_accelerator_addr_map_t;

/** Enum defining the different categories of telemetry */
typedef enum ualoe_telemetry_category {
  UALOE_TELEMETRY_CATEGORY_IFOE,
  UALOE_TELEMETRY_CATEGORY_SWITCH,
  UALOE_TELEMETRY_CATEGORY_CRYPTO,
  UALOE_TELEMETRY_CATEGORY_PFC,
  UALOE_TELEMETRY_CATEGORY_NETPORT,
  UALOE_TELEMETRY_CATEGORY_DERIVED_IFOE,
  UALOE_TELEMETRY_CATEGORY_DERIVED_NETPORT,
  UALOE_TELEMETRY_CATEGORY_MAX
} ualoe_telemetry_category_e;

/** Define used to construct a bitmask for selected Telemetry categories */
#define UALOE_TELEMETRY_CATEGORY_MASK(cat) (1U << UALOE_TELEMETRY_CATEGORY_##cat)

/** Definition of an item of telemetry */
typedef struct ualoe_telemetry_item_s {
  /** Identifier of the telemetry item */
  uint64_t id;
  /** Value of the telemetry item */
  uint64_t value;
} ualoe_telemetry_item_t;

/** Collection of telemetry data items for an instance of a cateogory of telemetry */
typedef struct ualoe_telemetry_instance_s {
  /** Name for this instance */
  ualoe_label_t name;
  /** Logical index for this instance */
  unsigned logical_idx;
  /** Number of telemetry items in the set */
  unsigned item_count;
  /** Set of telemetry items */
  ualoe_telemetry_item_t* items;
} ualoe_telemetry_instance_t;

/** Definition of a Telemetry dataset. This contains the all telemetry for one
 * category.
 */
typedef struct ualoe_telemetry_dataset_s {
  /** Telemetry category */
  ualoe_telemetry_category_e category;
  /** Sequence number incremented each time that the telemetry data is written */
  uint64_t generation_count;
  /** Timestamp at which the telemetry was captured. This is the UTC time */
  struct timespec timestamp;
  /** Number of instances of telemetry data for this category */
  unsigned instance_count;
  /** Array of instances of data */
  ualoe_telemetry_instance_t* instances;
} ualoe_telemetry_dataset_t;

/** Top level structure defining the telemetry data for IFoE. Telemetry data
 * is organised into datasets, one for each category of telemetry. This allows
 * a subset of the overall telemetry to be delivered to a client, according to
 * what the client requests and also what is permitted- the Tenant is only
 * allowed to receive the derived telemetry for IFoE Stations and Network
 * Ports. In the top level telemetry data structure, ualoe_telemetry_t, the
 * 'datasets' array is indexed by ualoe_telemetry_category_e. Where a category
 * is not available or has not been requested by the client, the array entry
 * will be Null.
 *
 * For each category, the data is organized into a dataset contains data for
 * each instance of the hardware. Instances are either per-IFoE station or
 * per-network ports, depending on the category of telemetry.
 * Each dataset includes:
 *   - A generation count which is incremented each time that the dataset is
 *     updated
 *   - A timestamp of when the dataset was last updated
 *   - The number of instances in the dataset
 *   - An array of pointers to the data for each instance
 * Each instance includes:
 *   - The textual label of the instance (this will match the equivalent sysfs name),
 *   - The logical index of the instance (either the IFoE Station or Network Port)
 *   - An array of telemetry items consisting of a unique ID and the current value
 *
 * The telemetry IDs are defined in YML which is then used to generate the header
 * file ifoe_telemetry.h provided in the same directory as this API.
 *
 * Almost all telemetry items are counters. The only exceptions are the items:
 *   IFOE_TELEM_ID_IFOE_LOGICAL_LINK_STATUS
 *   IFOE_TELEM_ID_NETPORT_LINK_STATUS
 * in the datasets for derived telemetry for IFoE Stations and Network Ports.
 * These two items are sets of flags defining the status. Refer to
 *   IFOE_TELEM_LINK_STATUS_*
 *   IFOE_TELEM_NETPORT_STATUS_*
 * for the definitions of the individual flags.
 */
typedef struct ualoe_telemetry_s {
  /** Dataset for each category of telemetry */
  ualoe_telemetry_dataset_t* datasets[UALOE_TELEMETRY_CATEGORY_MAX];
} ualoe_telemetry_t;

/** Type for an L2Ping handle. */
typedef unsigned ualoe_ping_handle_t;

/** Structure defining the specification for a L2 ping network connectivity
 * test. */
typedef struct ualoe_ping_spec_s {
  /** When indicates that the network ping check should be carried out on the
   * specified accelerator */
  bool specify_accelerator;
  /** When indicates that the network ping check should be carried out on the
   * specified Network Port */
  bool specify_netport;
  /** Include IFoE Request traffic in the ping test */
  bool include_ifoe_req;
  /** Include IFoE Response traffic in the ping test */
  bool include_ifoe_resp;
  /** Include non-IFoE traffic in the ping test */
  bool include_non_ifoe;
  /** Accelerator ID to target if specify_accelerator is true */
  unsigned accelerator_id;
  /** Network Port to target if specify_netport is true */
  unsigned netport_idx;
  /** Number of pings to perform on each path (max UALOE_L2PING_MAX_PINGS) */
  unsigned num_pings;
} ualoe_ping_spec_t;

/** Structure holding a single ping result for a specific accelerator and
 * netport. */
typedef struct ualoe_ping_netport_results_s {
  /** Count of ping failures on the IFoE Request PFC channel. A value of 0
   * indicates all pings completed successfully. */
  uint8_t req_failures;
  /** Count of ping failures on the IFoE Response PFC channel. A value of 0
   * indicates all pings completed successfully. */
  uint8_t resp_failures;
  /** Count of non-IFOE failures on the non-IFoE traffic PFC channel. A value
   * of 0 indicates all pings completed successfully. */
  uint8_t non_ifoe_failures;
} ualoe_ping_netport_result_t;

/** Structure holding a set of ping results for a specific accelerator. This
 * includes the ping results for all network paths to the accelerator. */
typedef struct ualoe_ping_accel_result_s {
  /** Accelerator ID */
  unsigned id;
  /** Array of network port results. This is either indexed by logical network
   * port index or, if a specific network port was specified for the ping test,
   * a single result for that network port. */
  ualoe_ping_netport_result_t* netports;
} ualoe_ping_accel_result_t;

/** Context for a L2Ping test. */
typedef struct ualoe_ping_s {
  /** Opaque handle for the ping test */
  ualoe_ping_handle_t handle;
  /** Specification of the test */
  ualoe_ping_spec_t spec;
  /** Indicates that the test is complete */
  bool test_complete;
  /** Indication of progress. This field counts up to 'total' */
  unsigned progress;
  /** Indication of progress. The field 'progress' will count up to this value. */
  unsigned total;
  /** Number of accelerators included in the ping test. This provides the size
   * of the accels array. */
  unsigned num_accelerators;
  /** Number of netports included in the ping test. This provides the size of
   * each ualoe_ping_netport_result_t array. */
  unsigned num_netports;
  /** Array of results for each accelerator included in the test. If a specific
   * accelerator was defined in the test spec, this will be an array of 1 entry. */
  ualoe_ping_accel_result_t* accels;
} ualoe_ping_t;

/** Enumeration defining the different UALoE events */
typedef enum {
  /** Notification that a Network Port link state has changed */
  UALOE_EVENT_NETPORT_LINK_CHANGE,
  /** Notification that a IFoE Logical Link state has changed */
  UALOE_EVENT_IFOE_LINK_CHANGE,
  /** Notification that the configuration phase of UALoE has changed */
  UALOE_EVENT_CONFIG_PHASE_CHANGE,
  /** Notification that the current L2 ping test is complete */
  UALOE_EVENT_L2PING_COMPLETE,
} ualoe_event_id_e;

/** Structure defining an UALoE event. This is passed to the caller
 * when an event occurs.
 */
typedef struct ualoe_event_s {
  /** ID of the event that occurred */
  ualoe_event_id_e id;
  /** UTC time for the event */
  struct timespec timestamp;

  /** Union containing event-specific data */
  union {
    /** Data associated with the event UALOE_EVENT_NETPORT_LINK_CHANGE */
    struct {
      /* Label of Network Port that changed link state */
      ualoe_label_t name;
      /** Network Port logical index */
      unsigned logical_idx;
      /* Boolean indicating if link is up or down */
      bool link_down;
    } netport_link_change;

    /** Data associated with the event UALOE_EVENT_IFOE_LINK_CHANGE */
    struct {
      /** IFoE Station name */
      ualoe_label_t name;
      /** IFoE Station logical index */
      unsigned logical_idx;
      /** Boolean indicating that the IFoE Logical Link is down */
      bool link_down;
      /** Boolean indicating that that datapath interface between IFoE and data
       * fabric is isolated i.e. no data flow is disabled in both directions. */
      bool dx_isolated;
      /** Number of Network Port structures populated */
      unsigned netport_count;
      /** Array of Network Port information */
      struct {
        /** Logical index of the network port */
        unsigned logical_idx;
        /** When true, indicates that there is a fault on the network port. */
        bool fault;
        /** Count of number of streams currently in a failover state which have
         * been remapped to an alternative Network Port. */
        unsigned streams_failover;
        /** Count of number of streams currently in a paused state due to a
         * retransmission timeout where failover was either not possible or
         * disabled. */
        unsigned streams_paused;
      } netports[UALOE_MAX_NETPORTS_PER_IFOE_STATION];
    } ifoe_link_change;

    /** Data associated with the event UALOE_EVENT_CONFIG_PHASE_CHANGE */
    struct {
      /** New configuration phase */
      ualoe_config_phase_e phase;
    } config_phase_change;

    /** Data associated with the event UALOE_EVENT_L2PING_COMPLETE */
    struct {
      /** Handle associated with the ping test */
      ualoe_ping_handle_t handle;
    } l2ping_complete;
  } u;
} ualoe_event_t;

/* ------------------- */
/* Function Prototypes */
/* ------------------- */

/**
 * @brief Open a connection to an UALoE driver. The UALoE device exposes
 * a driver instance for each PCI function (one PF and one VF),
 * identified by PCI address.
 *
 * @note This library is statically linked to the application. Multiple
 * clients can connect simultaneously as each has it's own copy of the
 * library code and data.
 *
 * @param pci_addr PCI Address of the UALoE device
 * @param handle Handle to be used for subsequent configuration calls
 * @return 0 on success. errno on failure.
 */
int ualoe_open(const char* pci_addr, ualoe_handle_t* handle);

/**
 * @brief Close a connection to an UALoE driver.
 *
 * @param handle Handle of the UAoE driver
 * @return 0 on success. errno on failure.
 */
int ualoe_close(ualoe_handle_t handle);

/**
 * @brief Get the UALoE firmware and telemetry versions
 * This command is permitted in all configuration phases.
 *
 * @param handle UALoE driver handle
 * @param lib_version Pointer to structure for the UALoE library version
 * @param fw_version Pointer to structure for FW version
 * @param telemetry_version Pointer to structure for Telemetry version
 * @return 0 on success. errno on failure.
 */
int ualoe_get_version(ualoe_handle_t handle, ualoe_version_t* lib_version,
                      ualoe_version_t* fw_version, ualoe_version_t* telemetry_version);

/**
 * @brief Reset the UALoE datapath and configuration. The behaviour of this
 * operation depends on whether the driver instance is associated with the PF
 * or the VF of IFoE.
 * For the PF, this operation will disable the IFoE datapath, reset all
 * stations and return the configuration phase back to the Provider phase. The
 * network ports will remain up and the VF will remain enabled but the crypto
 * keys will be disabled and erased. After a reset, the configuration will be
 * returned to the firmware defaults with the set of active and local
 * accelerators empty. For detail of the defaults for other configuration
 * items, refer to the definition of ifoe_config_t.
 * For the VF, this operation will disable communications with remote GPUs and
 * clear the set of enabled accelerators, clear the crypto keys, disable
 * encryption and return the configuration phase back to Tenant phase.
 * This is a blocking call and will only return when complete.
 *
 * @param handle UALoE driver handle
 * @return 0 on success. errno on failure.
 */
int ualoe_reset(ualoe_handle_t handle);

/**
 * @brief Get the capabilities of this UALoE device. This returns information
 * about the hardware configuration of the device including number of stations,
 * netports per station etc.
 * This command is permitted in all configuration phases.
 *
 * @param handle UALoE driver handle
 * @param capabilities Pointer to capabilities structure to be populated
 * @return 0 on success. errno on failure.
 */
int ualoe_get_capabilities(ualoe_handle_t handle, ualoe_capabilities_t* capabilities);

/**
 * @brief Set an UALoE instance's identity, specifically the ID for the local
 * accelerator.
 * This command is only permitted in the Provider configuration phase.
 *
 * @note This command is not required long-term as this information will be
 * retrieved through a different mechanism.
 *
 * @param handle IFoE driver handle
 * @param accelerator_id The GPU accelerator ID of the GPU connected to this
 * UALoE device
 * @return 0 on success. errno on failure.
 */
int ualoe_set_identity(ualoe_handle_t handle, unsigned accelerator_id);

/**
 * @brief Configure the Accelerator IDs for use in the VPod. The caller should
 * provide one bitmask of the active accelerators in the VPod and a second
 * bitmask that indicates which of these accelerators are local to the host
 * (this applies to any other GPUs that are part of the VPod and also attached
 * to this host processor). For each bit in the bitmask, the bit number
 * corresponds to the accelerator ID.
 * The set of local accelerators must be subset of the active accelerators,
 * noting that a configuration where all accelerators are local is valid.
 * This command is only permitted in the Provider configuration phase.
 *
 * @note This command is not required long-term as this information will be
 * retrieved through a different mechanism.
 *
 * @param handle UALoE driver handle
 * @param bitmask_size Size of the supplied bitmasks in bytes
 * @param active_accelerator_bitmask Bitmask of the active Accelerators in
 * the VPod
 * @param local_accelerator_bitmask Bitmask of the Accelerators local to the device
 * @return 0 on success. errno on failure.
 */
int ualoe_set_accelerator_config(ualoe_handle_t handle, unsigned bitmask_size,
                                 uint32_t active_accelerator_bitmask[],
                                 uint32_t local_accelerator_bitmask[]);

/**
 * @brief Set various IFoE configuration options. These are global settings
 * that apply to all IFoE Stations / Network Ports in the device.
 * This command is only permitted in the Provider configuration phase and
 * if attempted in any other phase, will fail with EBUSY.
 *
 * @param handle UALoE driver handle
 * @param virt_mode Configures IFoE is operating in virtualized or bare-metal mode
 * @param encap_type Specifies the network encapsulation that IFoE traffic will use
 * @param failover_mode Specifies whether failover will be be used
 * @param loopback_mode Network port loopback mode
 */
int ualoe_set_ifoe_config(ualoe_handle_t handle, ifoe_virt_mode_e virt_mode,
                          ifoe_encap_type_e encap_type, ifoe_failover_mode_e failover_mode,
                          ifoe_loopback_mode_e loopback_mode);

/**
 * @brief Get the IFoE configuration. The returned structure includes flags
 * indicating which items have been configured by the client, and consequently
 * which items have not been configured and are the firmware defaults.
 * This command is permitted in all phases.
 *
 * @param handle UALoE driver handle
 * @param config Pointer to structure where configuration should be written
 * @param bitmask_size Size of the supplied bitmasks in bytes
 * @param active_accelerator_bitmask Bitmask of the active Accelerators in
 * the VPod
 * @param local_accelerator_bitmask Bitmask of the Accelerators local to the device
 * @param enabled_accelerator_bitmask Bitmask of the Accelerators enabled for
 * communication
 */
int ualoe_get_ifoe_config(ualoe_handle_t handle, ifoe_config_t* config, unsigned bitmask_size,
                          uint32_t active_accelerator_bitmask[],
                          uint32_t local_accelerator_bitmask[],
                          uint32_t enabled_accelerator_bitmask[]);

/**
 * @brief Transition to the next configuration phase. The caller must specify
 * the next phase to avoid accidentally transitioning to the wrong phase. If
 * this does not match the strict ordering enforced by the firmware, the
 * request will fail with ERANGE. See the UALoE configuration phases definition
 * above for notes on the transition order.
 *
 * @param handle UALoE driver handle
 * @param next_phase Configuration phase to transition to
 * @return 0 on success. errno on failure
 */
int ualoe_next_config_phase(ualoe_handle_t handle, ualoe_config_phase_e next_phase);

/**
 * @brief Get the current configuration phase
 * This command is permitted in all configuration phases.
 *
 * @param handle UALoE driver handle
 * @param phase Pointer to location to return current configuration phase
 * @return 0 on success. errno on failure
 */
int ualoe_get_current_config_phase(ualoe_handle_t handle, ualoe_config_phase_e* phase);

/**
 * @brief Configure the set of accelerator IDs that should be enabled for
 * communication with this accelerator. Specified as a bitmask. This
 * operation is carried out by the tenant and confirms that the tenant has
 * validated that the remote accelerators are trusted, has programmed
 * encryption keys, if necessary, and is ready to enable the reception
 * of traffic. The set of enabled accelerators must be a subset of
 * the configured active accelerators. Note that the local accelerator ID
 * is implicitly active whether or not specified in the set.
 * This command is only permitted in the Tenant configuration phase and
 * if attempted in any other phase, will fail with EBUSY.
 *
 * @param handle UALoE driver handle
 * @param bitmask_size Size of the supplied bitmask in bytes
 * @param enabled_accelerator_bitmask Bitmask of the enabled Accelerators
 * in the VPod
 * @return 0 on success. errno on failure
 */
int ualoe_enable_accelerators(ualoe_handle_t handle, unsigned bitmask_size,
                              uint32_t enabled_accelerator_bitmask[]);

/**
 * @brief Configure the encryption mode for the UALoE datapath. This operation
 * is carried out by the tenant in virtualization mode. In bare-metal mode
 * mode the provider will carry out this operation but may do so only after
 * completing the Provider configuration and transitioning to the tenant
 * configuration phase. Encryption can be disabled or enabled with one of
 * the encryption modes supported by the datapath.
 * This command is only permitted in the Tenant configuration phase and if
 * attempted in any other phase, will fail with EBUSY.
 *
 * @param handle UALoE driver handle
 * @param mode Encryption mode to be used
 * @return 0 on success. errno on failure
 */
int ualoe_config_crypto(ualoe_handle_t handle, ualoe_crypto_mode_e mode);

/**
 * @brief Set the encryption key for UALoE TX / egress datapath. This command
 * is permitted when encryption is enabled and when UALoE is in the Tenant
 * configuration or Showtime phase. If attempted in any other phase, the
 * command will fail with EBUSY. If attempted when encryption is disabled
 * the command will fail with ENODEV.
 *
 * @param handle UALoE driver handle
 * @param key_id Key ID to be used for TX encryption
 * @param key Crypto key to be programmed
 * @return 0 on success. errno on failure
 */
int ualoe_set_tx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                            ualoe_crypto_key_t* key);

/**
 * @brief Disables the specified key for RX decryption. For the UALoE receive
 * datapath, two keys on the RX datapath to allow the key rotation. This
 * function disables one key ID such that the associated key can no longer
 * be used to decrypt packets.
 * This command is permitted when encryption is enabled and when UALoE is in
 * the Tenant configuration or Showtime phase. If attempted in any other
 * phase, the command will fail with EBUSY. If attempted when encryption is
 * disabled the command will fail with ENODEV.
 *
 * @param handle UALoE driver handle
 * @param key_id Key ID to be used for TX encryption
 * @return 0 on success. errno on failure
 */
int ualoe_disable_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id);

/**
 * @brief Set and enable an encryption key for UALoE RX / ingress datapath.
 * IFoE supports two keys on the RX datapath identified by a key ID. This
 * allows the keys to be rotated without interruption to traffic flow.
 * This command is permitted when encryption is enabled and when IFoE is
 * in the Tenant configuration or showtime phase. If attempted in any other
 * phase, the command will fail with EBUSY. If attempted when encryption is
 * disabled the command will fail with ENODEV.
 *
 * @param handle UALoE driver handle
 * @param key_id Key ID to be used for RX decryption
 * @param key Crypto key to be programmed
 * @return 0 on success. errno on failure
 */
int ualoe_set_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                            ualoe_crypto_key_t* key);

/**
 * @brief Get a set of descriptors for the IFoE Stations in this device. Each
 * descriptor contain information about the station including a textual label
 * and physical and logical indices.
 * This command is permitted in the Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param desc_count Number of entries in descriptor array provided by the
 * caller. This should match the number of configured IFoE stations- refer to
 * ualoe_get_capabilities().
 * @param descs Array of descriptors. This memory should be allocated by the
 * caller and will be populated by this function.
 * @return 0 on success. errno on failure
 */
int ifoe_get_station_list(ualoe_handle_t handle, unsigned desc_count, ifoe_station_desc_t descs[]);

/**
 * @brief Function to control an IFoE station. This can be used to
 * administratively enable and disable the station as well as placing it
 * in a stopped/drain state.
 * This command is only permitted in the Provider configuration phase.
 *
 * @param handle UALoE driver handle
 * @param station_idx Logical index of IFoE station
 * @param state The state to set the IFoE station into
 * @return 0 on success. errno on failure
 */
int ifoe_station_ctrl(ualoe_handle_t handle, unsigned station_idx, ifoe_station_state_e state);

/**
 * @brief Function to get information about the state of an IFoE Station.
 * This command is permitted in the Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param station_idx Logical index of IFoE station
 * @param state Pointer to structure to store the station state
 * @return 0 on success. errno on failure
 */
int ifoe_station_get_state(ualoe_handle_t handle, unsigned station_idx,
                           ifoe_station_state_t* state);

/**
 * @brief Set the Path to Port mapping for the IFoE datapath. This can be done
 * globally, per IFoE Station and per Accelerator ID, depending on the selected
 * flags.
 * This command is permitted in the Provider, Tenant and Showtime phases.
 *
 * @param handle UALoE driver handle
 * @param specify_station Boolean indicating if the mapping should be set for a
 * specific IFoE Station
 * @param specify_accelerator Boolean indicating if the mapping should be set
 * for a specific Accelerator.
 * @param reenable_streams Boolean indicating that any disabled streams
 * affected by this mapping request should be re-enabled.
 * @param station_idx Logical index of IFoE station
 * @param accelerator_id Accelerator ID
 * @param path_count Number of entries in paths array provided by the
 * caller. This must match the number IFoE Paths as returned by
 * ualoe_get_capabilities().
 * @param map Array of Network Port logical indices, indexed by the IFoE Path.
 * @return 0 on success. errno on failure
 */
int ifoe_set_path_to_port_map(ualoe_handle_t handle, bool specify_station, bool specify_accelerator,
                              bool reenable_streams, unsigned station_idx, unsigned accelerator_id,
                              unsigned path_count, unsigned map[]);

/**
 * @brief Get the Path to Port mapping for a specified IFoE Station and
 * Accelerator.
 * This command is permitted in the Provider, Tenant and Showtime phases.
 *
 * @param handle UALoE driver handle
 * @param station_idx Logical index of IFoE station
 * @param accelerator_id Accelerator ID
 * @param path_count Number of entries in paths array provided by the
 * caller. This must match the number IFoE Paths as returned by
 * ualoe_get_capabilities().
 * @param map Array of Network Port logical indices, indexed by IFoE Path.
 * @return 0 on success. errno on failure
 */
int ifoe_get_path_to_port_map(ualoe_handle_t handle, unsigned station_idx, unsigned accelerator_id,
                              unsigned path_count, unsigned map[]);

/**
 * @brief Get the Network Port properties. Note that the port properties
 * are common to all IFoE network ports and are fixed for a given port
 * mode.
 * This command is permitted in all configuration phases.
 *
 * @param handle UALoE driver handle
 * @param properties Pointer to location to write port properties
 * @return 0 on success. errno on failure
 */
int ifoe_get_netport_properties(ualoe_handle_t handle, ualoe_netport_properties_t* properties);

/**
 * @brief Get a set of descriptors for the Network Ports associated with the
 * IFoE device. Each descriptor contain information about the location network
 * port i.e. the IFoE station that it belongs to and it's station-relative
 * index.
 * This command is permitted in the Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param desc_count Number of entries in descriptor array provided by the
 * caller. This should match the number of configured network ports- refer to
 * ualoe_get_capabilities().
 * @param descs Array of descriptors. This memory should be allocated by the
 * caller and will be populated by this function.
 * @return 0 on success. errno on failure
 */
int ifoe_get_netport_list(ualoe_handle_t handle, unsigned desc_count, ifoe_netport_desc_t descs[]);

/**
 * @brief Function to control a Network Port. This can be used to disable a
 * Network Port and remove it as a part of the logical link of the associated
 * IFoE Station.
 * This command is only permitted in the Provider configuration phase.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param state The state to set the Network Port into
 * @return 0 on success. errno on failure
 */
int ifoe_netport_ctrl(ualoe_handle_t handle, unsigned netport_idx, ifoe_netport_state_e state);

/**
 * @brief Function to configure a Network Port link to use Auto-Negotiation.
 * The caller should supply the set of ethernet technologies supported in
 * as well as the requested FEC mode, noting that a weaker FEC mode may be
 * negotiated depending on the link parter's capabilities. With AN enabled,
 * parallel detect can used in which case the local link will attempt to
 * determine the link configuration being used by the partner.
 * This command is only permitted in the Provider and Diagnostics configuration
 * phases.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param parallel_detect_enable Enable Parallel Detect
 * @param advertised_eth_techs Bitmask of supported ethernet technologies
 * @param requested_fec_mode Requested FEC mode to use
 * @return 0 on success. errno on failure
 */
int ifoe_netport_config_link_auto(ualoe_handle_t handle, unsigned netport_idx,
                                  bool parallel_detect_enable, __uint128_t advertised_eth_techs,
                                  ualoe_fec_mode_e requested_fec_mode);

/**
 * @brief Function to configure a Network Port link with Auto-Negotiation
 * disabled and a fixed link configuration. The caller should specify the
 * ethernet technology, FEC mode and the loopback mode (can be none).
 * This command is only permitted in the Provider and Diagnostics configuration
 * phases.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param eth_tech Ethernet technology to configure
 * @param fec_mode FEC mode to use
 * @param loopback_mode Loopback mode
 * @return 0 on success. errno on failure
 */
int ifoe_netport_config_link_manual(ualoe_handle_t handle, unsigned netport_idx,
                                    ualoe_eth_tech_e eth_tech, ualoe_fec_mode_e fec_mode,
                                    ualoe_netport_loopback_mode_e loopback_mode);

/**
 * @brief Function to configure the MAC and/or IP addresses to use
 * for IFoE traffic.
 * This command is only permitted in the Provider configuration phase.
 *
 * @note The mac_addr and ip_addr parameters are optional. Their presence or
 * absence is determined by the addr_type parameter. A "missing" parameter can
 * take any value, and the parameter's value will be ignored during the call.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param addr_type The type of network address to use for IFoE traffic
 * @param mac_addr The MAC address to use for IFoE traffic
 * @param ip_addr The IP address to use for IFoE traffic in Network byte order
 * @return 0 on success. errno on failure
 *
 * @note addr_type must match the configured encapsulation mode:
 * - IFOE_NETWORK_ADDR_TYPE_MAC can be used only with IFOE_ENCAP_TYPE_ETHERNET;
 * - IFOE_NETWORK_ADDR_TYPE_IP with IFOE_ENCAP_TYPE_TF1;
 * - IFOE_NETWORK_ADDR_TYPE_MAC_IP with IFOE_ENCAP_TYPE_IP.
 *
 * The command will fail with error EINVAL when attempting to pass an address
 * type that is not compatible with the currently used encapsulation mode.
 */
int ifoe_netport_set_addr(ualoe_handle_t handle, unsigned netport_idx,
                          ifoe_network_addr_type_e addr_type, uint8_t mac_addr[], uint32_t ip_addr);

/**
 * @brief Function to set the Accelerator to destination MAC address mapping
 * for a Network Port.
 * This command is only permitted in the Provider configuration phase.
 *
 * @note To be discussed, should this be provided as a more global mapping i.e.
 * a struct containing netport, accelerator ID, dest mac address?
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param map_addr_type The type of network address in map array
 * @param map_count Number of mapping entries provided
 * @param map Array of accelerator to address mappings
 *
 * @note All details about the use of the address type and its values from
 * ifoe_netport_set_addr apply to this function and the address values used
 * in ifoe_accelerator_addr_map_t.
 */
int ifoe_netport_set_accelerator_addr_map(ualoe_handle_t handle, unsigned netport_idx,
                                          ifoe_network_addr_type_e map_addr_type,
                                          unsigned map_count, ifoe_accelerator_addr_map_t map[]);

/**
 * @brief Function to get IFoE-specific state of a Network Port.
 * This command is permitted in the Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param state Pointer to structure to store the station state
 * @return 0 on success. errno on failure
 */
int ifoe_netport_get_state(ualoe_handle_t handle, unsigned netport_idx,
                           ifoe_netport_state_t* state);

/**
 * @brief Allocate storage for Telemetry. This is called specify a set of
 * telemetry categories for which telemetry should be retrieved and the call
 * will allocate the required storage for the telemetry. The allocated
 * storage can then be reused each time that the caller wants to get an
 * updated snapshot of the data.
 * This functionality can be used by both the Provider and Tenant but the
 * tenant is restricted in the categories that it can receive to the
 * DERIVED_IFOE and DERIVED_NETPORT categories.
 * This command is permitted in Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param category_mask Bitmask of the Telemetry categories required,
 * constructed using UALOE_telemetry_CATEGORY_MASK(cat)
 * @param telemetry Returned telemetry structure
 * @return 0 on success. errno on failure
 */
int ualoe_telemetry_alloc(ualoe_handle_t handle, unsigned category_mask,
                          ualoe_telemetry_t** telemetry);

/**
 * @brief Get a copy of the latest telemetry data snapshot
 * This command is permitted in Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param telemetry Storage for telemetry pre-allocated using
 * ualoe_telemetry_alloc()
 * @return 0 on success. errno on failure
 */
int ualoe_telemetry_get(ualoe_handle_t handle, ualoe_telemetry_t* telemetry);

/**
 * @brief Free the storage allocated for the telemetry data
 * This command is permitted in Provider, Tenant, Showtime and Diagnostics
 * phases.
 *
 * @param handle UALoE driver handle
 * @param telemetry Telemetry data to be freed
 * @return 0 on success. errno on failure
 */
int ualoe_telemetry_free(ualoe_handle_t handle, ualoe_telemetry_t* telemetry);

/**
 * @brief Configure an L2 Ping test, allocate the context for the test and
 * memory required to hold the results before starting the test. The test will
 * take some time and continue in the background. When the test is complete,
 * the notification UALOE_EVENT_L2PING_COMPLETE will be sent.
 * ualoe_l2ping_update() can be called during the ping test to get an update
 * on progress and to retrieve the full results at the end of the test.
 * ualoe_l2ping_free() must be called at the end of the test to free the
 * allocated resources.
 * This command is only permitted in Showtime and Diagnostics phases.
 *
 * @param handle UALoE driver handle
 * @param spec Specification for the ping test
 * @param ping Pointer to the ping context which will be allocated for the test
 * @return 0 on success. errno on failure
 */
int ualoe_l2ping_start(ualoe_handle_t handle, ualoe_ping_spec_t* spec, ualoe_ping_t** ping);

/**
 * @brief Update the ping context with the status and progress. If the ping
 * test is complete the context will be marked as such and the full results
 * populated.
 * This command is only permitted in Showtime and Diagnostics phases.
 *
 * @param handle UALoE driver handle
 * @param ping Pointer to the context for the ping currently in progress
 * @return 0 on success. errno on failure
 */
int ualoe_l2ping_update(ualoe_handle_t handle, ualoe_ping_t* ping);

/**
 * @brief Stop the current ping test if not already finished and free the
 * resources allocated for the test.
 * This command is allowed in all phases.
 *
 * @param handle UALoE driver handle
 * @param ping Pointer to the ping context to be freed
 * @return 0 on success. errno on failure
 */
int ualoe_l2ping_fini(ualoe_handle_t handle, ualoe_ping_t* ping);

/**
 * @brief Prototype of callback function to notify the client that
 * an event has occurred
 *
 * @param user_context Context supplied by the user when the callback
 * was subscribed to
 * @param handle UALoE driver handle
 * @param event Structure containing the event
 */
typedef void (*ualoe_event_callback_t)(void* user_context, ualoe_handle_t handle,
                                       ualoe_event_t event);

/**
 * @brief Register a callback to be notified of UALoE events
 * This command is permitted in all configuration phases.
 *
 * @param handle UALoE driver handle
 * @param callback Callback function
 * @param user_context Context that will be passed back in callbacks
 * @return 0 on success. errno on failure
 */
int ualoe_register_event_callback(ualoe_handle_t handle, ualoe_event_callback_t callback,
                                  void* user_context);

/** Functions for diagnostics operations */

/** Structure used to hold PRBS test results */
typedef struct ualoe_prbs_results {
  /** Indicates whether the PRBS results are valid */
  bool valid;
  /** Indicates that the requested RX EQ operation was successful */
  bool rxeq_success;
  /** Indicates whether CDR is locked */
  bool cdr_lock;
  /** Indicates that the specified PRBS pattern is detected. If false,
   * indicates that there has been no detection of the pattern. */
  bool pattern_lock;
  /** Indicates that the error count has overflowed since the previous
   * time that the results were retrieved. */
  bool overflow;
  /** Interval since the last call to collect the results (or since PRBS
   * RX was enabled if this is the first call to retrieve results) */
  struct timespec interval;
  /** Number of bit errors detected */
  uint64_t error_count;
} ualoe_prbs_results_t;

/**
 * @brief Configure a network port PMA differential lane in preparation for
 * PRBS (pseudo-randon binary sequence) testing.
 * This operation can only be performed in Diagnostics mode.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param lane_idx Netport relative index of the serdes lane to configure
 * @param enable Controls if PMA lane should be enabled or powered down
 * @param pma_rate Specifies the PMA rate
 * @param loopback_mode Specifies loopback mode
 * @param tx_polarity Specifies polarity of TX lane
 * @param rx_polarity Specifies polarity of RX lane
 * @return 0 on success. errno on failure
 */
int ualoe_diag_config_pma_lane(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                               bool enable, ualoe_pma_rate_e pma_rate,
                               ualoe_netport_loopback_mode_e loopback_mode,
                               ualoe_pma_polarity_e tx_polarity, ualoe_pma_polarity_e rx_polarity);

/**
 * @brief Configure PRBS TX on a network port serdes lane. This operation will
 * configure the PMA to start transmitting the selected PRBS sequence. Before
 * enabling PRBS, the lane should be configured using the function
 * ualoe_diag_config_pma_lane(). To stop PRBS, the function can be called
 * again with enable false, whereby the PMA will be reconfigured to the
 * normal state where it processes transitions from the PCS.
 * This operation can only be performed in Diagnostics mode.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param lane_idx Netport relative index of the serdes lane to configure
 * @param enable Boolean indicating whether to enable/disable PRBS
 * @param pattern PRBS pattern to emit
 * @param user_pattern User supplied pattern if pattern is set to user defined
 * @return 0 on success. errno on failure
 */
int ualoe_diag_config_prbs_tx(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                              bool enable, ualoe_prbs_pattern_e pattern, __uint128_t user_pattern);

/**
 * @brief Start PRBS RX on a network port serdes lane. This operation will
 * configure the PMA to listen for the specified PRBS sequence. Before enabling
 * PRBS, the lane should be configured using the function
 * ualoe_diag_config_pma_lane(). When enabling PRBS RX, a preliminary RX
 * equalization using the resync flag. After enabling PRBS, IFoE will continue
 * to collect results until PRBS is disabled by a subsequent call with enable
 * set to false. While running, results can be fetched using the function
 * ualoe_diag_get_prbs_results().
 * This operation can only be performed in Diagnostics mode.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param lane_idx Netport relative index of the serdes lane to configure
 * @param enable Boolean indicating whether to enable/disable PRBS
 * @param resync If true, perform an RX equalization operation on the lane
 * before starting PRBS RX.
 * @param pattern PRBS pattern to emit
 * @param user_pattern User supplied pattern if pattern is set to user defined
 * @return 0 on success. errno on failure
 */
int ualoe_diag_config_prbs_rx(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                              bool enable, bool resync, ualoe_prbs_pattern_e pattern,
                              __uint128_t user_pattern);

/**
 * @brief When PRBS RX is enabled, this operation gets a snapshot of the
 * results. Note that taking a snapshot of the results will clear the
 * counters in the firmware to minimize the possibility of overflow.
 * Therefore, if cumulative results are required this is the responsibility
 * of the client.
 * This operation can only be performed in Diagnostics mode.
 *
 * @param handle UALoE driver handle
 * @param netport_idx Logical index of the Network Port
 * @param lane_idx Netport relative index of the serdes lane to configure
 * @param results Pointer to structure where results will be written.
 * @return 0 on success. errno on failure
 */
int ualoe_diag_get_prbs_results(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                                ualoe_prbs_results_t* results);

#endif  // UALOE_LIB_H
