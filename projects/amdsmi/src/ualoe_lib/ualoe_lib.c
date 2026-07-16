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

#include "ualoe_lib.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "ualoe_cdev.h"
#include "ualoe_log.h"
#include "ualoe_nl.h"

/* Default logging state can be overridden at build time */
#ifndef UALOE_LOG_ENABLED_DEFAULT
#define UALOE_LOG_ENABLED_DEFAULT 1
#endif

/* Global variable for runtime logging control */
int ualoe_log_enabled = UALOE_LOG_ENABLED_DEFAULT;

/* Those defines should be moved to ualoe_lib.h if it's allowed */

/** Minimum size of accelerator bitmask in bytes */
#define UALOE_MIN_BITMASK_SIZE_BYTES (4)

/** Maximum size of accelerator bitmask in bytes */
#define UALOE_MAX_BITMASK_SIZE_BYTES (128)

/** Maximum length for device name (PCI address) */
#define UALOE_MAX_DEVICE_NAME_LEN (256)

/**
 * Validate device name (PCI address) for safety and correctness
 *
 * @param pci_addr: Device name to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_device_name(const char* pci_addr) {
  size_t len;

  if (pci_addr == NULL) {
    ualoe_log_error("%s: device name cannot be NULL\n", __func__);
    return EINVAL;
  }

  len = strlen(pci_addr);

  /* Check for empty device name */
  if (len == 0) {
    ualoe_log_error("%s: device name cannot be empty\n", __func__);
    return EINVAL;
  }

  /* Check for unreasonably long device name (prevents buffer overflows) */
  if (len > UALOE_MAX_DEVICE_NAME_LEN) {
    ualoe_log_error("%s: device name too long (max %d characters)\n", __func__,
                    UALOE_MAX_DEVICE_NAME_LEN);
    return EINVAL;
  }

  /* Basic PCI address format validation: should contain only valid PCI chars
   * Valid characters: 0-9, a-f, A-F, ':', '.'
   * This catches Unicode, special characters, and other malformed input
   */
  for (size_t i = 0; i < len; i++) {
    char c = pci_addr[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == ':' ||
          c == '.')) {
      ualoe_log_error("%s: invalid character in device name (expected PCI address format)\n",
                      __func__);
      return EINVAL;
    }
  }

  return 0;
}

/**
 * Validate ifoe_virt_mode_e enum value
 *
 * @param mode: Virtualization mode to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_virt_mode(ifoe_virt_mode_e mode) {
  switch (mode) {
    case IFOE_VIRT_MODE_VIRTUALIZED:
    case IFOE_VIRT_MODE_BARE_METAL:
      return 0;
  }
  ualoe_log_error("%s: invalid virt_mode value %d\n", __func__, mode);
  return EINVAL;
}

/**
 * Validate ifoe_encap_type_e enum value
 *
 * @param encap: Encapsulation type to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_encap_type(ifoe_encap_type_e encap) {
  switch (encap) {
    case IFOE_ENCAP_TYPE_ETHERNET:
    case IFOE_ENCAP_TYPE_TF1:
    case IFOE_ENCAP_TYPE_IP:
      return 0;
  }
  ualoe_log_error("%s: invalid encap_type value %d\n", __func__, encap);
  return EINVAL;
}

/**
 * Validate ifoe_failover_mode_e enum value
 *
 * @param mode: Failover mode to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_failover_mode(ifoe_failover_mode_e mode) {
  switch (mode) {
    case IFOE_FAILOVER_DISABLED:
    case IFOE_FAILOVER_ENABLED:
      return 0;
  }
  ualoe_log_error("%s: invalid failover_mode value %d\n", __func__, mode);
  return EINVAL;
}

/**
 * Validate ifoe_loopback_mode_e enum value
 *
 * @param mode: Loopback mode to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_loopback_mode(ifoe_loopback_mode_e mode) {
  switch (mode) {
    case IFOE_LOOPBACK_OFF:
    case IFOE_LOOPBACK_IFOE:
    case IFOE_LOOPBACK_IFOE_PRE_MAC:
      return 0;
  }
  ualoe_log_error("%s: invalid loopback_mode value %d\n", __func__, mode);
  return EINVAL;
}

/**
 * Validate ualoe_crypto_mode_e enum value
 *
 * @param mode: Crypto mode to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_crypto_mode(ualoe_crypto_mode_e mode) {
  switch (mode) {
    case UALOE_CRYPTO_MODE_OFF:
    case UALOE_CRYPTO_MODE_AES_GCM_256:
      return 0;
  }
  ualoe_log_error("%s: invalid crypto_mode value %d\n", __func__, mode);
  return EINVAL;
}

/**
 * Validate ualoe_crypto_key_id_e enum value
 *
 * @param key_id: Crypto key ID to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_crypto_key_id(ualoe_crypto_key_id_e key_id) {
  switch (key_id) {
    case UALOE_CRYPTO_KEY_ID_0:
    case UALOE_CRYPTO_KEY_ID_1:
      return 0;
  }
  ualoe_log_error("%s: invalid crypto_key_id value %d\n", __func__, key_id);
  return EINVAL;
}

/**
 * Validate ifoe_station_state_e enum value
 *
 * @param state: Station state to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_station_state(ifoe_station_state_e state) {
  switch (state) {
    case IFOE_STATION_DISABLED:
    case IFOE_STATION_STOPPED:
    case IFOE_STATION_ACTIVE:
      return 0;
  }
  ualoe_log_error("%s: invalid station_state value %d\n", __func__, state);
  return EINVAL;
}

/**
 * Validate ifoe_netport_state_e enum value
 *
 * @param state: Netport state to validate
 * @return: 0 on success, EINVAL on validation failure
 */
static int ualoe_validate_netport_state(ifoe_netport_state_e state) {
  switch (state) {
    case IFOE_NETPORT_POWEROFF:
    case IFOE_NETPORT_IDLE:
    case IFOE_NETPORT_ENABLED:
      return 0;
  }
  ualoe_log_error("%s: invalid netport_state value %d\n", __func__, state);
  return EINVAL;
}

int ualoe_open(const char* pci_addr, ualoe_handle_t* handle) {
  int rc;

  if (handle == NULL) {
    ualoe_log_error("%s: handle cannot be equal to NULL\n", __func__);
    return EINVAL;
  }

  /* Validate device name for safety (prevents crashes and confusion) */
  rc = ualoe_validate_device_name(pci_addr);
  if (rc != 0) {
    return rc;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_open(pci_addr, handle);
#else
  return ualoe_cdev_open(pci_addr, handle);
#endif /* UALOE_NETLINK */
}

int ualoe_close(ualoe_handle_t handle) {
#ifdef UALOE_NETLINK
  return ualoe_nl_close(handle);
#else
  return ualoe_cdev_close(handle);
#endif /* UALOE_NETLINK */
}

int ualoe_get_version(ualoe_handle_t handle, ualoe_version_t* lib_version,
                      ualoe_version_t* fw_version, ualoe_version_t* telemetry_version) {
  if (lib_version == NULL && fw_version == NULL && telemetry_version == NULL) {
    ualoe_log_error("%s: at least one version pointer must be present\n", __func__);
    return EINVAL;
  }
#ifdef UALOE_NETLINK
  return ualoe_nl_get_version(handle, lib_version, fw_version, telemetry_version);
#else
  return ualoe_cdev_get_version(handle, lib_version, fw_version, telemetry_version);
#endif /* UALOE_NETLINK */
}

int ualoe_reset(ualoe_handle_t handle) {
#ifdef UALOE_NETLINK
  return ualoe_nl_reset(handle);
#else
  return ualoe_cdev_reset(handle);
#endif /* UALOE_NETLINK */
}

int ualoe_get_capabilities(ualoe_handle_t handle, ualoe_capabilities_t* capabilities) {
  if (capabilities == NULL) {
    ualoe_log_error("%s: capabilities cannot be equal to NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_get_capabilities(handle, capabilities);
#else
  return ualoe_cdev_get_capabilities(handle, capabilities);
#endif /* UALOE_NETLINK */
}

int ualoe_set_identity(ualoe_handle_t handle, unsigned accelerator_id) {
#ifdef UALOE_NETLINK
  return ualoe_nl_set_identity(handle, accelerator_id);
#else
  return ualoe_cdev_set_identity(handle, accelerator_id);
#endif /* UALOE_NETLINK */
}

int ualoe_set_accelerator_config(ualoe_handle_t handle, unsigned bitmask_size,
                                 uint32_t active_accelerator_bitmask[],
                                 uint32_t local_accelerator_bitmask[]) {
  if (bitmask_size == 0) {
    ualoe_log_error("%s: bitmask_size cannot be zero (got %u)\n", __func__, bitmask_size);
    return EINVAL;
  }
  if (bitmask_size < UALOE_MIN_BITMASK_SIZE_BYTES || bitmask_size > UALOE_MAX_BITMASK_SIZE_BYTES) {
    ualoe_log_error("%s: bitmask_size must be between %u and %u (got %u)\n", __func__,
                    UALOE_MIN_BITMASK_SIZE_BYTES, UALOE_MAX_BITMASK_SIZE_BYTES, bitmask_size);
    return EINVAL;
  }
  if ((bitmask_size % 4) != 0) {
    ualoe_log_error("%s: bitmask_size must be a multiple of 4 (got %u)\n", __func__, bitmask_size);
    return EINVAL;
  }

  if (bitmask_size > 0 && active_accelerator_bitmask == NULL) {
    ualoe_log_error("%s: active_accelerator_bitmask cannot be NULL when bitmask_size > 0\n",
                    __func__);
    return EINVAL;
  }

  if (bitmask_size > 0 && local_accelerator_bitmask == NULL) {
    ualoe_log_error("%s: local_accelerator_bitmask cannot be NULL when bitmask_size > 0\n",
                    __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_set_accelerator_config(handle, bitmask_size, active_accelerator_bitmask,
                                         local_accelerator_bitmask);
#else
  return ualoe_cdev_set_accelerator_config(handle, bitmask_size, active_accelerator_bitmask,
                                           local_accelerator_bitmask);
#endif /* UALOE_NETLINK */
}

int ualoe_set_ifoe_config(ualoe_handle_t handle, ifoe_virt_mode_e virt_mode,
                          ifoe_encap_type_e encap_type, ifoe_failover_mode_e failover_mode,
                          ifoe_loopback_mode_e loopback_mode) {
  int rc;

  rc = ualoe_validate_virt_mode(virt_mode);
  if (rc != 0) return rc;

  rc = ualoe_validate_encap_type(encap_type);
  if (rc != 0) return rc;

  rc = ualoe_validate_failover_mode(failover_mode);
  if (rc != 0) return rc;

  rc = ualoe_validate_loopback_mode(loopback_mode);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ualoe_nl_set_ifoe_config(handle, virt_mode, encap_type, failover_mode, loopback_mode);
#else
  return ualoe_cdev_set_ifoe_config(handle, virt_mode, encap_type, failover_mode, loopback_mode);
#endif /* UALOE_NETLINK */
}

int ualoe_get_ifoe_config(ualoe_handle_t handle, ifoe_config_t* config, unsigned bitmask_size,
                          uint32_t active_accelerator_bitmask[],
                          uint32_t local_accelerator_bitmask[],
                          uint32_t enabled_accelerator_bitmask[]) {
  if (config == NULL) {
    ualoe_log_error("%s: config parameter is NULL\n", __func__);
    return EINVAL;
  }
  if (bitmask_size != 0 && (bitmask_size < UALOE_MIN_BITMASK_SIZE_BYTES ||
                            bitmask_size > UALOE_MAX_BITMASK_SIZE_BYTES)) {
    ualoe_log_error("%s: bitmask_size must be 0 or between %u and %u (got %u)\n", __func__,
                    UALOE_MIN_BITMASK_SIZE_BYTES, UALOE_MAX_BITMASK_SIZE_BYTES, bitmask_size);
    return EINVAL;
  }
  if ((bitmask_size % 4) != 0) {
    ualoe_log_error("%s: bitmask_size must be a multiple of 4 (got %u)\n", __func__, bitmask_size);
    return EINVAL;
  }
  if (bitmask_size > 0) {
    const char* wrong_field = NULL;

    if (active_accelerator_bitmask == NULL)
      wrong_field = "active_accelerator_bitmask";
    else if (local_accelerator_bitmask == NULL)
      wrong_field = "local_accelerator_bitmask";
    else if (enabled_accelerator_bitmask == NULL)
      wrong_field = "enabled_accelerator_bitmask";

    if (wrong_field != NULL) {
      ualoe_log_error("%s: %s cannot be NULL when bitmask_size > 0\n", __func__, wrong_field);
      return EINVAL;
    }
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_get_ifoe_config(handle, config, bitmask_size, active_accelerator_bitmask,
                                  local_accelerator_bitmask, enabled_accelerator_bitmask);
#else
  return EOPNOTSUPP;
#endif /* UALOE_NETLINK */
}

int ualoe_next_config_phase(ualoe_handle_t handle, ualoe_config_phase_e next_phase) {
#ifdef UALOE_NETLINK
  return ualoe_nl_set_config_phase(handle, next_phase);
#else
  return ualoe_cdev_next_config_phase(handle, next_phase);
#endif /* UALOE_NETLINK */
}

int ualoe_get_current_config_phase(ualoe_handle_t handle, ualoe_config_phase_e* phase) {
  if (phase == NULL) {
    ualoe_log_error("%s: phase cannot be equal to NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_get_current_config_phase(handle, phase);
#else
  return ualoe_cdev_get_current_config_phase(handle, phase);
#endif /* UALOE_NETLINK */
}

int ualoe_enable_accelerators(ualoe_handle_t handle, unsigned bitmask_size,
                              uint32_t enabled_accelerator_bitmask[]) {
  if (bitmask_size == 0) {
    ualoe_log_error("%s: bitmask_size cannot be zero (got %u)\n", __func__, bitmask_size);
    return EINVAL;
  }
  if (bitmask_size < UALOE_MIN_BITMASK_SIZE_BYTES || bitmask_size > UALOE_MAX_BITMASK_SIZE_BYTES) {
    ualoe_log_error("%s: bitmask_size must be between %u and %u (got %u)\n", __func__,
                    UALOE_MIN_BITMASK_SIZE_BYTES, UALOE_MAX_BITMASK_SIZE_BYTES, bitmask_size);
    return EINVAL;
  }
  if ((bitmask_size % 4) != 0) {
    ualoe_log_error("%s: bitmask_size must be a multiple of 4 (got %u)\n", __func__, bitmask_size);
    return EINVAL;
  }
  if (enabled_accelerator_bitmask == NULL) {
    ualoe_log_error("%s: enabled_accelerator_bitmask cannot be NULL when bitmask_size > 0\n",
                    __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_enable_accelerators(handle, bitmask_size, enabled_accelerator_bitmask);
#else
  return ualoe_cdev_enable_accelerators(handle, bitmask_size, enabled_accelerator_bitmask);
#endif /* UALOE_NETLINK */
}

int ualoe_config_crypto(ualoe_handle_t handle, ualoe_crypto_mode_e mode) {
  int rc;

  rc = ualoe_validate_crypto_mode(mode);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ualoe_nl_config_crypto(handle, mode);
#else
  return ualoe_cdev_config_crypto(handle, mode);
#endif /* UALOE_NETLINK */
}

int ualoe_set_tx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                            ualoe_crypto_key_t* key) {
  int rc;

  if (key == NULL) {
    ualoe_log_error("%s: key cannot be NULL\n", __func__);
    return EINVAL;
  }

  rc = ualoe_validate_crypto_key_id(key_id);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ualoe_nl_set_tx_crypto_key(handle, key_id, key);
#else
  return ualoe_cdev_set_tx_crypto_key(handle, key_id, key);
#endif /* UALOE_NETLINK */
}

int ualoe_disable_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id) {
  int rc;

  rc = ualoe_validate_crypto_key_id(key_id);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ualoe_nl_disable_rx_crypto_key(handle, key_id);
#else
  return ualoe_cdev_disable_rx_crypto_key(handle, key_id);
#endif /* UALOE_NETLINK */
}

int ualoe_set_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                            ualoe_crypto_key_t* key) {
  int rc;

  if (key == NULL) {
    ualoe_log_error("%s: key cannot be NULL\n", __func__);
    return EINVAL;
  }

  rc = ualoe_validate_crypto_key_id(key_id);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ualoe_nl_set_rx_crypto_key(handle, key_id, key);
#else
  return ualoe_cdev_set_rx_crypto_key(handle, key_id, key);
#endif /* UALOE_NETLINK */
}

int ifoe_get_station_list(ualoe_handle_t handle, unsigned desc_count, ifoe_station_desc_t descs[]) {
  if (desc_count == 0) {
    ualoe_log_error("%s: desc_count must be greater than 0\n", __func__);
    return EINVAL;
  }

  if (descs == NULL) {
    ualoe_log_error("%s: descs cannot be NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_get_station_list(handle, desc_count, descs);
#else
  return ifoe_cdev_get_station_list(handle, desc_count, descs);
#endif /* UALOE_NETLINK */
}

int ifoe_station_ctrl(ualoe_handle_t handle, unsigned station_idx, ifoe_station_state_e state) {
  int rc;

  rc = ualoe_validate_station_state(state);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ifoe_nl_station_ctrl(handle, station_idx, state);
#else
  return ifoe_cdev_station_ctrl(handle, station_idx, state);
#endif /* UALOE_NETLINK */
}

int ifoe_station_get_state(ualoe_handle_t handle, unsigned station_idx,
                           ifoe_station_state_t* state) {
  if (state == NULL) {
    ualoe_log_error("%s: state cannot be equal to NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_station_get_state(handle, station_idx, state);
#else
  return ifoe_cdev_station_get_state(handle, station_idx, state);
#endif /* UALOE_NETLINK */
}

int ifoe_set_path_to_port_map(ualoe_handle_t handle, bool specify_station, bool specify_accelerator,
                              bool reenable_streams, unsigned station_idx, unsigned accelerator_id,
                              unsigned path_count, unsigned map[]) {
  if (path_count > 0 && map == NULL) {
    ualoe_log_error("%s: map cannot be NULL when path_count > 0\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_set_path_to_port_map(handle, specify_station, specify_accelerator,
                                      reenable_streams, station_idx, accelerator_id, path_count,
                                      map);
#else
  return ifoe_cdev_set_path_to_port_map(handle, specify_station, specify_accelerator,
                                        reenable_streams, station_idx, accelerator_id, path_count,
                                        map);
#endif /* UALOE_NETLINK */
}

int ifoe_get_path_to_port_map(ualoe_handle_t handle, unsigned station_idx, unsigned accelerator_id,
                              unsigned path_count, unsigned map[]) {
  if (path_count > 0 && map == NULL) {
    ualoe_log_error("%s: map cannot be NULL when path_count > 0\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_get_path_to_port_map(handle, station_idx, accelerator_id, path_count, map);
#else
  return ifoe_cdev_get_path_to_port_map(handle, station_idx, accelerator_id, path_count, map);
#endif /* UALOE_NETLINK */
}

int ifoe_get_netport_properties(ualoe_handle_t handle, ualoe_netport_properties_t* properties) {
  if (properties == NULL) {
    ualoe_log_error("%s: properties cannot be NULL\n", __func__);
    return EINVAL;
  }
#ifdef UALOE_NETLINK
  return ifoe_nl_get_netport_properties(handle, properties);
#else
  return EOPNOTSUPP;
#endif /* UALOE_NETLINK */
}

int ifoe_get_netport_list(ualoe_handle_t handle, unsigned desc_count, ifoe_netport_desc_t descs[]) {
  if (desc_count == 0) {
    ualoe_log_error("%s: desc_count must be greater than 0\n", __func__);
    return EINVAL;
  }

  if (descs == NULL) {
    ualoe_log_error("%s: descs cannot be NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_get_netport_list(handle, desc_count, descs);
#else
  return ifoe_cdev_get_netport_list(handle, desc_count, descs);
#endif /* UALOE_NETLINK */
}

int ifoe_netport_ctrl(ualoe_handle_t handle, unsigned netport_idx, ifoe_netport_state_e state) {
  int rc;

  rc = ualoe_validate_netport_state(state);
  if (rc != 0) return rc;

#ifdef UALOE_NETLINK
  return ifoe_nl_netport_ctrl(handle, netport_idx, state);
#else
  return ifoe_cdev_netport_ctrl(handle, netport_idx, state);
#endif /* UALOE_NETLINK */
}

int ifoe_netport_config_link_auto(ualoe_handle_t handle, unsigned netport_idx,
                                  bool parallel_detect_enable, __uint128_t advertised_eth_techs,
                                  ualoe_fec_mode_e requested_fec_mode) {
#ifdef UALOE_NETLINK
  return ifoe_nl_netport_config_link_auto(handle, netport_idx, parallel_detect_enable,
                                          advertised_eth_techs, requested_fec_mode);
#else
  return EOPNOTSUPP;
#endif
}

int ifoe_netport_config_link_manual(ualoe_handle_t handle, unsigned netport_idx,
                                    ualoe_eth_tech_e eth_tech, ualoe_fec_mode_e fec_mode,
                                    ualoe_netport_loopback_mode_e loopback_mode) {
#ifdef UALOE_NETLINK
  return ifoe_nl_netport_config_link_manual(handle, netport_idx, eth_tech, fec_mode, loopback_mode);
#else
  return EOPNOTSUPP;
#endif
}

int ifoe_netport_set_addr(ualoe_handle_t handle, unsigned netport_idx,
                          ifoe_network_addr_type_e addr_type, uint8_t mac_addr[],
                          uint32_t ip_addr) {
  if ((addr_type == IFOE_NETWORK_ADDR_TYPE_MAC || addr_type == IFOE_NETWORK_ADDR_TYPE_MAC_IP) &&
      mac_addr == NULL) {
    ualoe_log_error("%s: mac_addr cannot be NULL for the specified addr_type\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_netport_set_addr(handle, netport_idx, addr_type, mac_addr, ip_addr);
#else
  return ifoe_cdev_netport_set_addr(handle, netport_idx, addr_type, mac_addr, ip_addr);
#endif /* UALOE_NETLINK */
}

#ifndef UALOE_MAX_ACCELERATORS
/* FIXME: IFOESW-804: The value 1024 is a temporary placeholder for the maximum
 * number of accelerators. Update this value to match hardware or system
 * constraints when available.
 */
#define UALOE_MAX_ACCELERATORS 1024
#endif

int ifoe_netport_set_accelerator_addr_map(ualoe_handle_t handle, unsigned netport_idx,
                                          ifoe_network_addr_type_e map_addr_type,
                                          unsigned map_count, ifoe_accelerator_addr_map_t map[]) {
  if (map_count == 0 || map_count > UALOE_MAX_ACCELERATORS) {
    ualoe_log_error("%s: map_count (%d) must be in range [1, %u]\n", __func__, map_count,
                    UALOE_MAX_ACCELERATORS);
    return EINVAL;
  }

  if (map == NULL) {
    ualoe_log_error("%s: map cannot be NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_netport_set_accelerator_addr_map(handle, netport_idx, map_addr_type, map_count,
                                                  map);
#else
  return ifoe_cdev_netport_set_accelerator_addr_map(handle, netport_idx, map_addr_type, map_count,
                                                    map);
#endif /* UALOE_NETLINK */
}

int ifoe_netport_get_state(ualoe_handle_t handle, unsigned netport_idx,
                           ifoe_netport_state_t* state) {
  if (state == NULL) {
    ualoe_log_error("%s: state cannot be equal to NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ifoe_nl_netport_get_state(handle, netport_idx, state);
#else
  return ifoe_cdev_netport_get_state(handle, netport_idx, state);
#endif /* UALOE_NETLINK */
}

int ualoe_telemetry_alloc(ualoe_handle_t handle, unsigned category_mask,
                          ualoe_telemetry_t** telemetry) {
  if (category_mask >= (1U << UALOE_TELEMETRY_CATEGORY_MAX)) {
    ualoe_log_error("%s: category_mask (0x%x) exceeds valid range [0, 0x%x]\n", __func__,
                    category_mask, (1U << UALOE_TELEMETRY_CATEGORY_MAX) - 1);
    return EINVAL;
  }
  if (telemetry == NULL) {
    ualoe_log_error("%s: telemetry parameter is NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_telemetry_alloc(handle, category_mask, telemetry);
#else
  return ualoe_cdev_telemetry_alloc(handle, category_mask, telemetry);
#endif /* UALOE_NETLINK */
}

int ualoe_telemetry_get(ualoe_handle_t handle, ualoe_telemetry_t* telemetry) {
  if (telemetry == NULL) {
    ualoe_log_error("%s: telemetry cannot be NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_telemetry_get(handle, telemetry);
#else
  return ualoe_cdev_telemetry_get(handle, telemetry);
#endif /* UALOE_NETLINK */
}

int ualoe_telemetry_free(ualoe_handle_t handle, ualoe_telemetry_t* telemetry) {
  if (telemetry == NULL) {
    ualoe_log_error("%s: telemetry cannot be NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_telemetry_free(handle, telemetry);
#else
  return ualoe_cdev_telemetry_free(handle, telemetry);
#endif /* UALOE_NETLINK */
}

int ualoe_l2ping_start(ualoe_handle_t handle, ualoe_ping_spec_t* spec, ualoe_ping_t** ping) {
  if (spec == NULL) {
    ualoe_log_error("%s: spec parameter is NULL\n", __func__);
    return EINVAL;
  }
  if (ping == NULL) {
    ualoe_log_error("%s: ping parameter is NULL\n", __func__);
    return EINVAL;
  }
  if (spec->num_pings == 0 || spec->num_pings > UALOE_L2PING_MAX_PINGS) {
    ualoe_log_error("%s: num_pings (%u) must be between 1 and %u\n", __func__, spec->num_pings,
                    UALOE_L2PING_MAX_PINGS);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_l2ping_start(handle, spec, ping);
#else
  return ualoe_cdev_l2ping_start(handle, spec, ping);
#endif /* UALOE_NETLINK */
}

int ualoe_l2ping_update(ualoe_handle_t handle, ualoe_ping_t* ping) {
  if (ping == NULL) {
    ualoe_log_error("%s: ping parameter is NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_l2ping_update(handle, ping);
#else
  return ualoe_cdev_l2ping_update(handle, ping);
#endif /* UALOE_NETLINK */
}

int ualoe_l2ping_fini(ualoe_handle_t handle, ualoe_ping_t* ping) {
  if (ping == NULL) {
    ualoe_log_error("%s: ping parameter is NULL\n", __func__);
    return EINVAL;
  }

#ifdef UALOE_NETLINK
  return ualoe_nl_l2ping_fini(handle, ping);
#else
  return ualoe_cdev_l2ping_fini(handle, ping);
#endif /* UALOE_NETLINK */
}

int ualoe_register_event_callback(ualoe_handle_t handle, ualoe_event_callback_t callback,
                                  void* user_context) {
#ifdef UALOE_NETLINK
  return ualoe_nl_register_event_callback(handle, callback, user_context);
#else
  return ualoe_cdev_register_event_callback(handle, callback, user_context);
#endif /* UALOE_NETLINK */
}

int ualoe_diag_config_pma_lane(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                               bool enable, ualoe_pma_rate_e pma_rate,
                               ualoe_netport_loopback_mode_e loopback_mode,
                               ualoe_pma_polarity_e tx_polarity, ualoe_pma_polarity_e rx_polarity) {
  return EOPNOTSUPP;
}

int ualoe_diag_config_prbs_tx(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                              bool enable, ualoe_prbs_pattern_e pattern, __uint128_t user_pattern) {
  return EOPNOTSUPP;
}

int ualoe_diag_config_prbs_rx(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                              bool enable, bool resync, ualoe_prbs_pattern_e pattern,
                              __uint128_t user_pattern) {
  return EOPNOTSUPP;
}

int ualoe_diag_get_prbs_results(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                                ualoe_prbs_results_t* results) {
  return EOPNOTSUPP;
}
