/*
 * MIT License
 *
 * Copyright (c) 2020 Open Compute Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OAM_INCLUDE_OAM_OAM_MAPI_H_
#define OAM_INCLUDE_OAM_OAM_MAPI_H_

/**
 * \file    oam_mapi.h
 * \brief   OAM management and monitoring library API definitions
 */

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

/**
 * \struct  oam_mapi_version_t
 * \brief   OAM library API version
 * \details TBD
 *	All the libraries versions are expected to be backward compatible.
 *	The major version increment indicates a new API has been added.
 *	Minor version increment indicates an interface change.
 */
typedef struct oam_mapi_version {
	uint32_t major;
	uint32_t minor;
} oam_mapi_version_t;

/**
 * \struct oam_dev_properties_t
 * \brief Network identifier for the device
 * \details Immutable network identifier for the device.
 *	This is unique across the entire network.
 */
typedef struct oam_net_dev_id {
	/*!< unique network identifier for the device */
	int network_id;
} oam_net_dev_id_t;

/*
 * various lengths for device properties
 */
#define DEVICE_VENDOR_LEN	128
#define DEVICE_NAME_LEN	128
#define DEVICE_SKU_LEN	128
#define BOARD_NAME_LEN	128
#define BOARD_REVISION_LEN	128
#define BOARD_SERIAL_NUM_LEN	128

/**
 * \struct  oam_dev_properties_t
 * \brief   TBD
 * \details TBD
 */
typedef struct oam_dev_properties {
	/*!< Immutable local identifier for the device */
	uint32_t device_id;
	/*!< vendor name */
	char device_vendor[DEVICE_VENDOR_LEN];
	/*!< Device name */
	char device_name[DEVICE_NAME_LEN];
	/*!< SKU name */
	char sku_name[DEVICE_SKU_LEN];
	/*!< Board name */
	char board_name[BOARD_NAME_LEN];
	/*!< Board revision */
	char board_revision[BOARD_REVISION_LEN];
	/*!<
	 * Board Serial Number or UUID any other identifier, which can be used
	 * to identify devices uniquely and physically.
	 */
	char board_serial_number[BOARD_SERIAL_NUM_LEN];
} oam_dev_properties_t;

/**
 * \struct  oam_sensor_count_t
 * \brief   TBD
 * \details TBD
 *          Various sensor related information
 */
typedef struct oam_sensor_count {
	uint32_t num_temperature_sensors;
	uint32_t num_power_sensors;
	uint32_t num_voltage_sensors;
	uint32_t num_current_sensors;
	uint32_t num_fans;
} oam_sensor_count_t;

/**
 * \enum    oam_sensor_type_t
 * \brief   Sensor types
 * \details This enumerated type defines available sensors types.
 */
typedef enum oam_sensor_type {
	OAM_SENSOR_TYPE_POWER = 0,
	OAM_SENSOR_TYPE_VOLTAGE,
	OAM_SENSOR_TYPE_CURRENT,
	OAM_SENSOR_TYPE_TEMP,
	OAM_SENSOR_TYPE_FAN_SPEED,
	OAM_SENSOR_TYPE_UNKNOWN
} oam_sensor_type_t;

/**
 * \enum    oam_power_sensor_scale_t
 * \brief   scale for power measurements
 * \details This enumerated type defines available scales for power measurements
 */
typedef enum oam_power_sensor_scale {
	OAM_POWER_SCALE_uW = 0,
	OAM_POWER_SCALE_mW,
	OAM_POWER_SCALE_W,
} oam_power_sensor_scale_t;

/**
 * \enum    oam_voltage_sensor_scale_t
 * \brief   scale for voltage measurements
 * \details This enumerated type defines available scales for voltage measurements
 */
typedef enum oam_voltage_sensor_scale {
	OAM_VOLTAGE_SCALE_uV = 0,
	OAM_VOLTAGE_SCALE_mV,
	OAM_VOLTAGE_SCALE_V,
} oam_voltage_sensor_scale_t;

/**
 * \enum    oam_current_sensor_scale_t
 * \brief   scale for current measurements
 * \details This enumerated type defines available scales for current measurements
 */
typedef enum oam_current_sensor_scale {
	OAM_CURRENT_SCALE_uA = 0,
	OAM_CURRENT_SCALE_mA,
	OAM_CURRENT_SCALE_A,
} oam_current_sensor_scale_t;

/**
 * \enum    oam_temp_sensor_scale_t
 * \brief   scale for temp measurements
 * \details This enumerated type defines available scales for temp measurements
 */
typedef enum oam_temp_sensor_scale {
	OAM_TEMP_SCALE_C = 0,
	OAM_TEMP_SCALE_F
} oam_temp_sensor_scale_t;

/**
 * \enum    oam_fan_sensor_scale_t
 * \brief   scale for power measurements
 * \details This enumerated type defines available scales for power measurements
 */
typedef enum oam_fan_sensor_scale {
	OAM_FAN_SPEED_Hz = 0,
	OAM_FAN_SPEED_KHz,
	OAM_FAN_SPEED_MHz
} oam_fan_sensor_scale_t;

typedef union oam_sensor_scale {
	oam_power_sensor_scale_t power_scale;
	oam_voltage_sensor_scale_t volate_scale;
	oam_current_sensor_scale_t current_scale;
	oam_temp_sensor_scale_t temp_scale;
	oam_fan_sensor_scale_t fan_scale;
} oam_sensor_scale_t;

/**
 * \struct  oam_dev_handle_t
 * \brief   Device handle
 * \details Device handle obtained using open call
 *          The same handle is used by all the APIs which are used to perform
 *          specific operation on that device.
 */
typedef struct oam_dev_handle {
	void *handle;
} oam_dev_handle_t;

/**
 * \enum    oam_dev_mode_t
 * \brief   Device open modes
 * \details This enumerated type defines modes in which the device can be opened
 *          For some operations e.g. health check user should open the device
 *          in exclusive mode, so that if there are many applications using the same
 *          device there are no side effects.
 */
typedef enum oam_dev_mode {
	OAM_DEV_MODE_EXCLUSIVE = 0,
	OAM_DEV_MODE_NONEXLUSIVE = 1,
	OAM_DEV_MODE_UNKNOWN = 0xFF
} oam_dev_mode_t;

/**
 * \def     OAM_SENSOR_NAME_MAX
 * \brief   length of sensor name
 */
#define OAM_SENSOR_NAME_MAX  256

/**
 * \struct  oam_sensor_info_t
 * \brief   Sensor information
 * \details Structure to store various info of sensors.
 */
typedef struct oam_sensor_info {
	char sensor_name[OAM_SENSOR_NAME_MAX];
	oam_sensor_type_t sensor_type;
	oam_sensor_scale_t scale;
	int64_t value;
} oam_sensor_info_t;

/**
 * \struct  oam_dev_error_count_t
 * \brief   Device error information
 * \details Various types of errors reported by device.
 */
typedef struct oam_dev_error_count {
	uint32_t total_error_count;
	uint32_t fatal_error_count;
	uint32_t unknown_error_count;
	uint32_t ecc_error_count;
} oam_dev_error_count_t;

/**
 * \struct  oam_firmware_version_t
 * \brief   Device error information
 * \details Structure to store various firmware versions of OAM module
 */
typedef struct oam_firmware_version {
	oam_mapi_version_t device_boot_fw_version;
	oam_mapi_version_t device_fw_version;
	oam_mapi_version_t board_boot_fw_version;
	oam_mapi_version_t board_fw_version;
} oam_firmware_version_t;

/**
 * \struct  oam_pci_info_t
 * \brief   PCI information for the device
 * \details Structure to store PCI (Domain, BDF) information of the device
 */
typedef struct oam_pci_info {
	uint16_t domain;
	uint8_t bus;
	uint8_t device;
	uint8_t function;
} oam_pci_info_t;

/**
 * \enum    oam_net_port_state_t
 * \brief   Network port state
 * \details This enumerated type defines various states of the network port
 */
typedef enum oam_net_port_state {
	OAM_NET_PORT_DISABLED = 0,
	OAM_NET_PORT_ENABLED = 1
} oam_net_port_state_t;

/**
 * \enum    oam_net_port_status_t
 * \brief   Network port status
 * \details This enumerated type defines various status of the network port
 */
typedef enum oam_net_port_status {
	OAM_NET_PORT_UP = 0,
	OAM_NET_PORT_DOWN = 1,
} oam_net_port_status_t;

/**
 * \enum    oam_net_port_id_t
 * \brief   Network port identifiers
 * \details This enumerated type defines various identifiers for network ports
 */
typedef enum oam_net_port_id {
	OAM_NET_PORT0 = 0,
	OAM_NET_PORT1 = 1,
	OAM_NET_PORT2 = 2,
	OAM_NET_PORT_MAX = 0xFFFF
} oam_net_port_id_t;

/**
 * \enum    oam_firmware_modes_t
 * \brief   Supported mode to update firmware on device
 * \details This enumerated type defines various modes which are supported by
 *          the device to update firmware.
 */
typedef enum oam_firmware_modes {
	OAM_DOWNLOAD_ONLY = 0,
	OAM_DOWNLOAD_ACTIVATE = 1
} oam_firmware_modes_t;

/**
 * \def     OAM_NET_PORT_NAME
 * \brief   length of network port name
 */
#define OAM_NET_PORT_NAME  256

/**
 * \struct  oam_net_port_desc
 * \brief   Network port description
 * \details Structure to store additional details about the network port
 */
typedef struct oam_net_port_desc {
	char name[OAM_NET_PORT_NAME];
} oam_net_port_desc_t;

/**
 * \def     OAM_DEV_HOST_NAME
 * \brief   length of host name
 */
#define OAM_DEV_HOST_NAME  256

/**
 * \struct  oam_net_dev_info_t
 * \brief   Information about the device on a network
 * \details Structure to store additional details about the network device
 *          on a particular network.
 */
typedef struct oam_net_dev_info {
	oam_net_dev_id_t net_dev_id;
	char host_name[OAM_DEV_HOST_NAME];
	oam_pci_info_t pci_info;
} oam_net_dev_info_t;

/**
 * \struct  oam_neighbour_info_t
 * \brief   Information about device neighburs
 * \details Structure to store information about device neighbours on the
 *          network
 */
typedef struct oam_neighbour_info {
	oam_net_port_id_t device_port;
	oam_net_dev_info_t device_info;
} oam_neighbour_info_t;

/**
 * \enum    oam_dev_tpc_id_t
 * \brief   TPC identifiers
 * \details This enumerated type defines various identifiers for TPCs
 */
typedef enum oam_dev_tpc_id {
	OAM_DEV_TPC0,
	OAM_DEV_TPC1,
	OAM_DEV_TPC2,
	OAM_DEV_TPC_MAX
} oam_dev_tpc_id_t;

/**
 * \def     OAM_TPC_NAME
 * \brief   length of TPC name
 */
#define OAM_TPC_NAME  256

/**
 * \struct  oam_tpc_desc_t
 * \brief   TPC description
 * \details Structure to store information about TPC e.g. name corresponding
 *          to the id etc.
 */
typedef struct oam_tpc_desc {
	char name[256];
} oam_tpc_desc_t;

/**
 * \struct  oam_dev_tpc_stats_t
 * \brief   TPC statistical information
 * \details Structure to store information about TPC statistical information
 *          e.g. TPC utilization
 */
typedef struct oam_dev_tpc_stats {
	double util;
} oam_dev_tpc_stats_t;

/**
 * \enum    oam_dev_mem_id_t
 * \brief   Device memory identifiers
 * \details This enumerated type defines various identifiers for device memories
 */
typedef enum oam_dev_mem_id {
	OAM_DEV_MEM0,
	OAM_DEV_MEM1,
	OAM_DEV_MEM2,
	OAM_DEV_MEM_MAX
} oam_dev_mem_id_t;

/**
 * \struct  oam_mem_desc_t
 * \brief   Device memory description
 * \details Structure to store additional details about device memories port
 */
typedef struct oam_mem_desc {
	char name[256];
} oam_mem_desc_t;

/**
 * \struct  oam_dev_mem_stats_t
 * \brief   Device memory statistical information
 * \details Structure to store various statastical information about device
 *          memory.
 */
typedef struct oam_dev_mem_stats {
	uint32_t total_mem;
	uint32_t allocated_mem;
	uint32_t free_mem;
} oam_dev_mem_stats_t;

/**
 * \struct  oam_net_port_pkt_stats_t
 * \brief   Device network port statistical information
 * \details Structure to store various statastical information about the network
 *          packets on a given port.
 */
typedef struct oam_net_port_pkt_stats {
	uint64_t rx_count;
	uint64_t tx_count;
	uint64_t rx_errors;
	uint64_t tx_errors;
} oam_net_port_pkt_stats_t;

/**
 * \struct  oam_ops_t
 * \brief   OAM Device operations
 * \details Structure provides list of APIs which needs to be
 *          supported by the OAM library.
 */
typedef struct oam_ops {
	/*!<
	 * to initialise library instance and perform version compatibility
	 * check
	 */
	int (*init)(void);
	int (*free)(void);

	/*!<
	 * To get error description from the error code
	 */
	int (*get_error_description)(int error_code, const char **error_description);

	/*!<
	 * To retrieve the OAM Management interface version
	 */
	int (*get_mapi_version)(oam_mapi_version_t *version);

	/*!<
	 * To retrieve the number of devices present/discovered by the library
	 */
	int (*discover_devices)(uint32_t *device_count);

	/*!<
	 * To retrieve device properties for each discovered devices
	 */
	int (*get_dev_properties)(uint32_t device_count,
				oam_dev_properties_t *devices);

	/*!<
	 * To retrieve PCI properties of the device
	 */
	int (*get_pci_properties)(uint32_t device_id, oam_pci_info_t *pci_info);

	/*!<
	 * To query the number of various sensors present
	 */
	int (*get_sensors_count)(uint32_t device_id,
				 oam_sensor_count_t *sensor_count);

	/*!<
	 * Open the device and obtain handle
	 */
	int (*open_device)(uint32_t *dev_id, oam_dev_mode_t mode,
				 oam_dev_handle_t *handle);
	int (*close_device)(oam_dev_handle_t *handle);


	/*!<
	 * To read various sensor values for a given sensor type
	 */
	int (*get_sensors_info)(uint32_t device_id,
				oam_sensor_type_t type,
				uint32_t num_sensors,
				oam_sensor_info_t sensor_info[]);
	/*!<
	 * To read current error count of the device
	 */
	int (*get_device_error_count)(oam_dev_handle_t *handle,
							oam_dev_error_count_t *count);

	/*!<
	 * To update firmware on the device
	 * fw_image contains a null terminated string which specifies complete
	 * path where the firmware image is located
	 */
	int (*download_firmware)(uint32_t *device_id, char *fw_image,
				 oam_firmware_modes_t mode);

	/*!<
	 * To query firmware versions
	 */
	int (*get_firmware_version)(uint32_t *device_id,
						oam_firmware_version_t *version);


	/*!<
	 * to get network id from device id
	 */
	int (*get_net_dev_id)(uint32_t *device_id, oam_net_dev_id_t *net_device);

	/*!<
	 * Network management APIs.
	 */

	/*!<
	 * discover network.
	 */
	int (*discover_network)(int *net_dev_count);
	int (*get_dev_net_properties)(oam_net_dev_info_t *net_dev_info);

	int (*get_neighbour_count)(uint32_t *device,
					 oam_net_port_id_t local_port_id,
					 uint32_t *neighbor_count);

	int (*get_neighbours_info)(uint32_t *device,
					 oam_net_port_id_t local_port_id,
					 uint32_t *neighbors_count,
					 oam_neighbour_info_t *neighbours_info);

	int (*configure_network)(oam_net_dev_id_t *net_devices,
				 uint32_t *net_device_count,
				 char *network_name);

	int (*destroy_network)(char *network_name);

	int (*query_network)(char *network_name, oam_net_dev_info_t *devices,
					 uint32_t *device_count);

	int (*get_network_count)(uint32_t *network_count);
	int (*list_networks)(char *network_names[]);

	/*!<
	 * Various statistics related to blocks
	 */

	/*!<
	 * To query number of ports
	 */
	int (*get_net_port_count)(oam_dev_handle_t *handle, uint32_t *count,
					oam_net_port_id_t *port_ids);

	int (*get_net_port_desc)(oam_dev_handle_t *handle, oam_net_port_id_t *port,
				 oam_net_port_desc_t *desc);

	int (*get_net_port_state)(oam_dev_handle_t *handle, oam_net_port_id_t *port,
					oam_net_port_state_t *state);

	int (*check_net_port_status)(oam_dev_handle_t *handle,
						 oam_net_port_id_t *port,
						 oam_net_port_status_t *status);
	int (*get_net_port_pkt_stats)(oam_dev_handle_t *handle,
				oam_net_port_id_t *port,
				uint32_t duration_sec,
				oam_net_port_pkt_stats_t *stats);

	int (*query_net_port_bandwidth)(oam_dev_handle_t *handle,
					oam_net_port_id_t *port,
					uint32_t duration_sec,
					double *bandwidth);

	int (*get_tpc_count)(oam_dev_handle_t *handle, uint32_t *count,
					 oam_dev_tpc_id_t *tpc_ids);

	int (*get_tpc_desc)(oam_dev_handle_t *handle, oam_dev_tpc_id_t *tpc_id,
					oam_tpc_desc_t *desc);

	int (*get_tpc_stats)(oam_dev_handle_t *handle,
					 oam_dev_tpc_id_t *port,
					 oam_dev_tpc_stats_t *stats,
					 uint32_t duration_sec);

	int (*get_mem_count)(oam_dev_handle_t *handle, uint32_t *count,
					 oam_dev_mem_id_t *mem_ids);

	int (*get_mem_desc)(oam_dev_handle_t *handle, oam_dev_mem_id_t *tpc_id,
					oam_mem_desc_t *desc);

	int (*get_mem_stats)(oam_dev_handle_t *handle, oam_dev_mem_id_t *mem_id,
					 oam_dev_mem_stats_t *stats);

	/*!<
	 * To check the health of the individual components, libraries
	 * generates test workload to check if the block is functioning properly
	 * or not. So no other workload should be running while calling these
	 * APIs
	 */
	int (*check_tpc_health)(uint32_t *device_id, oam_dev_tpc_id_t *tpc_id);
	int (*check_net_port_health)(uint32_t *device_id, oam_net_port_id_t *port);
	int (*check_mem_health)(uint32_t *device_id, oam_dev_mem_id_t *port);

	/*
	 * Following needs more attention, will work on in next
		int (*get_fan_speed)(oam_dev_t *oam);
		int (*set_fan_speed)(oam_dev_t *oam, int speed);

		int (*get_power_cap)(oam_dev_t *oam);
		int (*set_power_cap)(oam_dev_t *oam, int power);

		int (*get_telemetry)(oam_dev_t *oam);
	 */

} oam_ops_t;


#ifdef  __cplusplus
}
#endif

#endif  // OAM_INCLUDE_OAM_OAM_MAPI_H_
