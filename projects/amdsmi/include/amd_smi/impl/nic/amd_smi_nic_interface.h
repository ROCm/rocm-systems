// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __AMD_SMI_NIC_INTERFACE_H__
#define __AMD_SMI_NIC_INTERFACE_H__

// Shim entry points that only the bare-metal library has. smi_nic_interface.h is shared with
// amdsmi_unified and stays as it is there; what exists only here is declared in this header.

#include "amd_smi/impl/nic/amdsmi_unified/interface/smi_nic_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the kind of a NIC (a SmiNicKind value).
 *
 * @param ctx Context returned by ::smi_nic_create_context.
 * @param device Packed BDF, in the ::amdsmi_bdf_t as_uint encoding.
 * @param kind Filled on success; unmodified otherwise.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_kind(smi_nic_ctx_t ctx, uint64_t device, uint8_t* kind);

/**
 * @brief Get the capability bitmask of a NIC (SmiNicCapability bits).
 *
 * @param ctx Context returned by ::smi_nic_create_context.
 * @param device Packed BDF, in the ::amdsmi_bdf_t as_uint encoding.
 * @param capability Filled on success; unmodified otherwise.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_capability(smi_nic_ctx_t ctx, uint64_t device, uint32_t* capability);

/**
 * @brief Retrieve the full interface index of one NIC port.
 *
 * smi_nic_port_t holds the index in one byte; this returns the whole value.
 *
 * @param ctx Context handle
 * @param device BDF of the network device.
 * @param port_index Index of the NIC port (0-based), as in the statistics calls.
 * @param ifindex Pointer to uint32_t to store the interface index; 0 when unavailable.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, ::SMI_NIC_STATUS_NOT_FOUND
 * when @p port_index is past the last port, non-zero on other failure.
 */
smi_nic_status_t smi_get_nic_port_ifindex(smi_nic_ctx_t ctx, uint64_t device, uint32_t port_index,
                                          uint32_t* ifindex);

/** Board temperatures in whole degrees Celsius; UINT16_MAX when the NIC does not expose one. */
typedef struct {
  uint16_t asic_temp_c;
  uint16_t transceiver_temp_c;
  uint16_t board_temp_c;
} smi_nic_temperature_t;

/** Health of the NIC's devlink reporter. */
typedef struct {
  uint8_t state;        /**< a health state value, 0 when unknown */
  uint32_t error_count; /**< UINT32_MAX when not exposed */
  char reporter[64];    /**< devlink reporter name, NUL-terminated; "" if none */
} smi_nic_health_t;

/** Port splitting; UINT8_MAX when unknown. */
typedef struct {
  uint8_t splittable;
  uint8_t split_count;
} smi_nic_port_split_t;

/** All live telemetry in one read. A metric the NIC does not expose holds its reserved value. */
typedef struct {
  smi_nic_temperature_t temperature;
  smi_nic_health_t health;
  smi_nic_port_split_t port_split;
} smi_nic_telemetry_t;

/**
 * @brief Read the live telemetry of the NIC registered under @p device.
 *
 * Reads through the discovered NIC, with its vendor subclass and its ports, so devlink and hwmon
 * are addressed at the port function rather than the bridge the handle names.
 *
 * @param ctx Context returned by ::smi_nic_create_context.
 * @param device Packed BDF, in the ::amdsmi_bdf_t as_uint encoding.
 * @param telemetry Filled on success; unmodified otherwise.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_telemetry(smi_nic_ctx_t ctx, uint64_t device,
                                       smi_nic_telemetry_t* telemetry);

#define SMI_NIC_MAX_SENSORS 128

/** One hwmon attribute; category, attribute and unit carry the amdsmi enumerator values. */
typedef struct {
  uint32_t category;
  uint32_t attribute;
  uint32_t unit;
  uint32_t channel;
  uint32_t instance;
  char label[SMI_NIC_MAX_STRING_LENGTH]; /**< driver label, "" when none */
  int64_t value;                         /**< raw hwmon integer in the unit */
} smi_nic_sensor_t;

typedef struct {
  uint32_t count;
  smi_nic_sensor_t sensors[SMI_NIC_MAX_SENSORS];
} smi_nic_sensor_info_t;

/**
 * @brief Read every hwmon sensor attribute of the NIC registered under @p device.
 *
 * A NIC with no hwmon node yields an empty list, not an error. More than SMI_NIC_MAX_SENSORS
 * attributes keep the first SMI_NIC_MAX_SENSORS and report ::SMI_NIC_STATUS_NO_RESOURCE.
 *
 * @param ctx Context returned by ::smi_nic_create_context.
 * @param device Packed BDF, in the ::amdsmi_bdf_t as_uint encoding.
 * @param info Filled on success.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_hwmon_sensors(smi_nic_ctx_t ctx, uint64_t device,
                                           smi_nic_sensor_info_t* info);

/**
 * @brief Which tier of vendor statistics to return
 *
 * SMI_NIC_STAT_SCOPE_EXTENDED is a superset of SMI_NIC_STAT_SCOPE_DEFAULT.
 */
typedef enum {
  SMI_NIC_STAT_SCOPE_DEFAULT = 0,
  SMI_NIC_STAT_SCOPE_EXTENDED = 1
} smi_nic_stat_scope_t;

/**
 * @brief Get the count of vendor statistics of a NIC port in the given tier.
 *
 * Same as ::smi_get_nic_vendor_statistics_count, which reads the default tier.
 *
 * @param ctx Context returned by ::smi_nic_create_context.
 * @param device Packed BDF, in the ::amdsmi_bdf_t as_uint encoding.
 * @param port_index Index of the NIC port (0-based).
 * @param scope The statistics tier.
 * @param count Number of statistics available.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_vendor_statistics_count_with_scope(smi_nic_ctx_t ctx, uint64_t device,
                                                                uint32_t port_index,
                                                                smi_nic_stat_scope_t scope,
                                                                uint32_t* count);

/**
 * @brief Retrieve the vendor statistics of a NIC port in the given tier.
 *
 * Same as ::smi_get_nic_vendor_statistics_list, which reads the default tier.
 *
 * @param ctx Context returned by ::smi_nic_create_context.
 * @param device Packed BDF, in the ::amdsmi_bdf_t as_uint encoding.
 * @param port_index Index of the NIC port (0-based).
 * @param scope The statistics tier.
 * @param stats Filled on success.
 * @return ::smi_nic_status_t | ::SMI_NIC_STATUS_SUCCESS on success, non-zero on failure.
 */
smi_nic_status_t smi_get_nic_vendor_statistics_list_with_scope(smi_nic_ctx_t ctx, uint64_t device,
                                                               uint32_t port_index,
                                                               smi_nic_stat_scope_t scope,
                                                               smi_nic_stat_info_t* stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __AMD_SMI_NIC_INTERFACE_H__
