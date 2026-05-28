/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)
 *
 * This file contains confidential and proprietary information
 * of Advanced Micro Devices, Inc. and is protected under U.S.
 * and international copyright and other intellectual property
 * laws.
 *
 * DISCLAIMER
 * This disclaimer is not a license and does not grant any
 * rights to the materials distributed herewith. Except as
 * otherwise provided in a valid license issued to you by
 * AMD, and to the maximum extent permitted by applicable
 * law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
 * WITH ALL FAULTS, AND AMD HEREBY DISCLAIMS ALL WARRANTIES
 * AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
 * BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
 * INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
 * (2) AMD shall not be liable (whether in contract or tort,
 * including negligence, or under any other theory of
 * liability) for any loss or damage of any kind or nature
 * related to, arising under or in connection with these
 * materials, including for any direct, or any indirect,
 * special, incidental, or consequential loss or damage
 * (including loss of data, profits, goodwill, or any type of
 * loss or damage suffered as a result of any action brought
 * by a third party) even if such damage or loss was
 * reasonably foreseeable or AMD had been advised of the
 * possibility of the same.
 *
 * CRITICAL APPLICATIONS
 * AMD products are not designed or intended to be fail-
 * safe, or for use in any application requiring fail-safe
 * performance, such as life-support or safety devices or
 * systems, Class III medical devices, nuclear facilities,
 * applications related to the deployment of airbags, or any
 * other applications that could lead to death, personal
 * injury, or severe property or environmental damage
 * (individually and collectively, "Critical
 * Applications"). Customer assumes the sole risk and
 * liability of any use of AMD products in Critical
 * Applications, subject only to applicable laws and
 * regulations governing limitations on product liability.
 *
 * THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
 * PART OF THIS FILE AT ALL TIMES
 */

#include "ualoe_cdev.h"

#include <cbl_cfg/uapi.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/queue.h>
#include <unistd.h>

#include "ualoe_cb.h"
#include "ualoe_lib.h"
#include "ualoe_log.h"

struct ualoe_cdev_handle {
  int fd;
  int dev_id;
  LIST_ENTRY(ualoe_cdev_handle) lentry;
};

LIST_HEAD(cdev_handles, ualoe_cdev_handle);
static struct cdev_handles open_cdev_handles = LIST_HEAD_INITIALIZER(cdev_handles);
static pthread_mutex_t cdev_handle_lock = PTHREAD_MUTEX_INITIALIZER;

static int ualoe_cdev_connect(int fd, const char* pci_addr, int* dev_id) {
  struct cfg_connect_cmd cmd = {0};
  int rc;

  strncpy(cmd.pci_addr, pci_addr, sizeof(cmd.pci_addr) - 1);
  rc = ioctl(fd, CFG_CONNECT, &cmd);
  if (rc == -1) return errno;
  *dev_id = cmd.dev_id;
  return 0;
}

static int ualoe_cdev_find_handle(ualoe_handle_t handle, struct ualoe_cdev_handle** cdev_handle) {
  struct ualoe_cdev_handle* iter;

  *cdev_handle = NULL;
  pthread_mutex_lock(&cdev_handle_lock);
  LIST_FOREACH(iter, &open_cdev_handles, lentry) {
    if (iter->fd == handle) {
      *cdev_handle = iter;
      break;
    }
  }
  pthread_mutex_unlock(&cdev_handle_lock);

  return *cdev_handle ? 0 : ENOENT;
}

/* Validate that cfg_l2ping_netport_result and ualoe_ping_netport_result_t
 * have the same memory layout so we can cast between them */
_Static_assert(sizeof(struct cfg_l2ping_netport_result) == sizeof(ualoe_ping_netport_result_t),
               "cfg_l2ping_netport_result and ualoe_ping_netport_result_t "
               "must have the same size");
_Static_assert(offsetof(struct cfg_l2ping_netport_result, req_failures) ==
                   offsetof(ualoe_ping_netport_result_t, req_failures),
               "req_failures field offset mismatch");
_Static_assert(offsetof(struct cfg_l2ping_netport_result, resp_failures) ==
                   offsetof(ualoe_ping_netport_result_t, resp_failures),
               "resp_failures field offset mismatch");
_Static_assert(offsetof(struct cfg_l2ping_netport_result, non_ifoe_failures) ==
                   offsetof(ualoe_ping_netport_result_t, non_ifoe_failures),
               "non_ifoe_failures field offset mismatch");

/* Validate that cfg_l2ping_accel_result and ualoe_ping_accel_result_t
 * have the same memory layout so we can cast between them */
_Static_assert(sizeof(struct cfg_l2ping_accel_result) == sizeof(ualoe_ping_accel_result_t),
               "cfg_l2ping_accel_result and ualoe_ping_accel_result_t "
               "must have the same size");
_Static_assert(offsetof(struct cfg_l2ping_accel_result, id) ==
                   offsetof(ualoe_ping_accel_result_t, id),
               "id field offset mismatch");
_Static_assert(offsetof(struct cfg_l2ping_accel_result, netports) ==
                   offsetof(ualoe_ping_accel_result_t, netports),
               "netports field offset mismatch");

static int ualoe_cdev_find(const char* pci_addr, char** cdev_name) {
  /* TODO: define proper method of mapping from GPU UUID to char device
   * name. For testing just guess name.
   */
  *cdev_name = "/dev/cbl-cfg-ifoe.cfg.0";
  return 0;
}

int ualoe_cdev_open(const char* pci_addr, ualoe_handle_t* handle) {
  struct ualoe_cdev_handle* cdev_handle;
  char* cdev_name;
  int rc;

  cdev_handle = malloc(sizeof(*cdev_handle));
  if (!cdev_handle) return ENOMEM;

  rc = ualoe_cdev_find(pci_addr, &cdev_name);
  if (rc) {
    free(cdev_handle);
    return ENOENT;
  }

  rc = open(cdev_name, O_RDWR);
  if (rc == -1) {
    free(cdev_handle);
    return errno;
  }
  cdev_handle->fd = rc;

  rc = ualoe_cdev_connect(cdev_handle->fd, pci_addr, &cdev_handle->dev_id);
  if (rc) {
    close(cdev_handle->fd);
    free(cdev_handle);
    return rc;
  }

  pthread_mutex_lock(&cdev_handle_lock);
  LIST_INSERT_HEAD(&open_cdev_handles, cdev_handle, lentry);
  pthread_mutex_unlock(&cdev_handle_lock);

  *handle = cdev_handle->fd;
  return 0;
}

int ualoe_cdev_close(ualoe_handle_t handle) {
  struct ualoe_cdev_handle* cdev_handle;
  int rc;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (!rc) {
    pthread_mutex_lock(&cdev_handle_lock);
    LIST_REMOVE(cdev_handle, lentry);
    pthread_mutex_unlock(&cdev_handle_lock);
    free(cdev_handle);
  }

  ualoe_cb_fini(handle);

  if (close(handle)) return errno;

  return 0;
}

int ualoe_cdev_reset(ualoe_handle_t handle) {
  int rc;

  rc = ioctl(handle, CFG_RESET);
  if (rc == -1) return errno;

  return 0;
}

int ualoe_cdev_get_version(ualoe_handle_t handle, ualoe_version_t* lib_version,
                           ualoe_version_t* fw_version, ualoe_version_t* telemetry_version) {
  struct cfg_version_cmd cmd;
  int rc;

  rc = ioctl(handle, CFG_GET_VERSION, &cmd);
  if (rc == -1) return errno;

  if (lib_version != NULL) {
    lib_version->major = UALOE_LIB_VERSION_MAJOR;
    lib_version->minor = UALOE_LIB_VERSION_MINOR;
    lib_version->patch = UALOE_LIB_VERSION_PATCH;
  }
  if (fw_version != NULL) {
    fw_version->major = cmd.fw_version.major;
    fw_version->minor = cmd.fw_version.minor;
    fw_version->patch = cmd.fw_version.patch;
  }
  if (telemetry_version != NULL) {
    telemetry_version->major = cmd.telemetry_version.major;
    telemetry_version->minor = cmd.telemetry_version.minor;
    telemetry_version->patch = cmd.telemetry_version.patch;
  }

  return 0;
}

int ualoe_cdev_get_capabilities(ualoe_handle_t handle, ualoe_capabilities_t* capabilities) {
  struct cfg_capabilities cfg_capabilities;
  int rc;

  rc = ioctl(handle, CFG_GET_CAPABILITIES, &cfg_capabilities);
  if (rc == -1) return errno;

  capabilities->num_configured_stations = cfg_capabilities.num_configured_stations;
  capabilities->max_accelerators = cfg_capabilities.max_accelerators;
  capabilities->num_netports_per_station = cfg_capabilities.num_netports_per_station;
  capabilities->num_paths_per_station = cfg_capabilities.num_paths_per_station;

  return 0;
}

int ualoe_cdev_set_identity(ualoe_handle_t handle, unsigned accelerator_id) {
  int rc;

  rc = ioctl(handle, CFG_SET_IDENTITY, &accelerator_id);
  if (rc == -1) return errno;

  return 0;
}

int ualoe_cdev_set_accelerator_config(ualoe_handle_t handle, unsigned bitmask_size,
                                      uint32_t active_accelerator_bitmask[],
                                      uint32_t local_accelerator_bitmask[]) {
  struct cfg_accelerator_config accel_config;
  int rc;

  accel_config.bitmask_size = bitmask_size;

  accel_config.active_accelerator_bitmask = malloc(bitmask_size);
  if (!accel_config.active_accelerator_bitmask) return errno;

  accel_config.local_accelerator_bitmask = malloc(bitmask_size);
  if (!accel_config.local_accelerator_bitmask) {
    rc = errno;
    goto free_active_accel;
  }

  memcpy(accel_config.active_accelerator_bitmask, active_accelerator_bitmask, bitmask_size);
  memcpy(accel_config.local_accelerator_bitmask, local_accelerator_bitmask, bitmask_size);

  rc = ioctl(handle, CFG_SET_ACCELERATOR_CONFIG, &accel_config);
  if (rc == -1)
    rc = errno;
  else
    rc = 0;

  free(accel_config.local_accelerator_bitmask);
free_active_accel:
  free(accel_config.active_accelerator_bitmask);
  return rc;
}

int ualoe_cdev_set_ifoe_config(ualoe_handle_t handle, ifoe_virt_mode_e virt_mode,
                               ifoe_encap_type_e encap_type, ifoe_failover_mode_e failover_mode,
                               ifoe_loopback_mode_e loopback_mode) {
  struct cfg_ifoe_config config;
  int rc;

  config.v_mode = virt_mode;
  config.e_type = encap_type;
  config.f_mode = failover_mode;
  config.loopback_mode = loopback_mode;

  rc = ioctl(handle, CFG_SET_IFOE_CONFIG, &config);
  if (rc == -1) return errno;

  return 0;
}

int ualoe_cdev_next_config_phase(ualoe_handle_t handle, ualoe_config_phase_e next_phase) {
  enum cfg_config_phase n_phase;
  int rc;

  n_phase = next_phase;
  rc = ioctl(handle, CFG_NEXT_CONFIG_PHASE, &n_phase);
  if (rc == -1) return errno;

  return 0;
}

int ualoe_cdev_get_current_config_phase(ualoe_handle_t handle, ualoe_config_phase_e* phase) {
  enum cfg_config_phase curr_phase;
  int rc;

  rc = ioctl(handle, CFG_GET_CURRENT_CONFIG_PHASE, &curr_phase);
  if (rc == -1) return errno;

  *phase = curr_phase;
  return 0;
}

int ualoe_cdev_enable_accelerators(ualoe_handle_t handle, unsigned bitmask_size,
                                   uint32_t enable_accelerator_bitmask[]) {
  return 0;
}

int ualoe_cdev_config_crypto(ualoe_handle_t handle, ualoe_crypto_mode_e mode) { return 0; }

int ualoe_cdev_set_tx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                                 ualoe_crypto_key_t* key) {
  return 0;
}

int ualoe_cdev_disable_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id) {
  return 0;
}

int ualoe_cdev_set_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                                 ualoe_crypto_key_t* key) {
  return 0;
}

int ifoe_cdev_get_station_list(ualoe_handle_t handle, unsigned desc_count,
                               ifoe_station_desc_t descs[]) {
  struct cfg_station_list station_list;
  int i, rc;

  station_list.count = desc_count;
  station_list.list = malloc(sizeof(struct cfg_station_desc) * desc_count);
  if (!station_list.list) return ENOMEM;

  rc = ioctl(handle, CFG_GET_STATION_LIST, &station_list);
  if (rc == -1) {
    rc = errno;
    goto out;
  }

  for (i = 0; i < desc_count; i++) {
    descs[i].logical_idx = station_list.list[i].logical_id;
    descs[i].physical_idx = station_list.list[i].physical_id;
    strncpy(descs[i].name.text, station_list.list[i].label.text, UALOE_LABEL_SIZE - 1);
    descs[i].name.text[UALOE_LABEL_SIZE - 1] = '\0';
  }

  rc = 0;
out:
  free(station_list.list);
  return rc;
}

int ifoe_cdev_station_ctrl(ualoe_handle_t handle, unsigned station_idx,
                           ifoe_station_state_e state) {
  return 0;
}

int ifoe_cdev_station_get_state(ualoe_handle_t handle, unsigned station_idx,
                                ifoe_station_state_t* state) {
  /* TODO: Remove hardcoded mock */
  state->state = IFOE_STATION_DISABLED;
  state->link_down = false;
  state->dx_isolated = false;
  state->bandwidth = 0xFFFFFFFF;
  state->logical_idx = 0xFFFFFFFF;
  state->physical_idx = 0xFFFFFFFF;
  state->netport_count = UALOE_MAX_NETPORTS_PER_IFOE_STATION;

  for (int i = 0; i < UALOE_MAX_NETPORTS_PER_IFOE_STATION; i++) {
    state->netports[i].logical_idx = 0xFFFFFFFF;
    state->netports[i].fault = false;
    state->netports[i].streams_failover = 0xFFFFFFFF;
    state->netports[i].streams_paused = 0xFFFFFFFF;
  }

  return 0;
}

int ifoe_cdev_set_path_to_port_map(ualoe_handle_t handle, bool specify_station,
                                   bool specify_accelerator, bool reenable_streams,
                                   unsigned station_idx, unsigned accelerator_id,
                                   unsigned path_count, unsigned map[]) {
  return 0;
}

int ifoe_cdev_get_path_to_port_map(ualoe_handle_t handle, unsigned station_idx,
                                   unsigned accelerator_id, unsigned path_count, unsigned map[]) {
  /* TODO: Remove hardcoded mock */
  for (int i = 0; i < path_count; i++) map[i] = 0xFFFFFFFF;

  return 0;
}

int ifoe_cdev_get_netport_list(ualoe_handle_t handle, unsigned desc_count,
                               ifoe_netport_desc_t descs[]) {
  /* TODO: Remove hardcoded mock */
  for (int i = 0; i < desc_count; i++) {
    strncpy(descs[i].name.text, "DESC", UALOE_LABEL_SIZE - 1);
    descs[i].name.text[UALOE_LABEL_SIZE - 1] = '\0';
    descs[i].logical_idx = 0xFFFFFFFF;
    strncpy(descs[i].station_name.text, "STATION", UALOE_LABEL_SIZE - 1);
    descs[i].station_name.text[UALOE_LABEL_SIZE - 1] = '\0';
    descs[i].station_idx = 0xFFFFFFFF;
    descs[i].station_rel_netport_idx = 0xFFFFFFFF;
  }

  return 0;
}

int ifoe_cdev_netport_ctrl(ualoe_handle_t handle, unsigned netport_idx,
                           ifoe_netport_state_e state) {
  return 0;
}

int ifoe_cdev_netport_set_addr(ualoe_handle_t handle, unsigned netport_idx,
                               ifoe_network_addr_type_e addr_type, uint8_t mac_addr[],
                               uint32_t ip_addr) {
  return 0;
}

int ifoe_cdev_netport_set_accelerator_addr_map(ualoe_handle_t handle, unsigned netport_idx,
                                               ifoe_network_addr_type_e map_addr_type,
                                               unsigned map_count,
                                               ifoe_accelerator_addr_map_t map[]) {
  return 0;
}

int ifoe_cdev_netport_get_state(ualoe_handle_t handle, unsigned netport_idx,
                                ifoe_netport_state_t* state) {
  /* TODO: Remove hardcoded mock */
  uint8_t arr[UALOE_MAC_ADDRESS_SIZE] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

  state->state = IFOE_NETPORT_POWEROFF;
  state->autoneg_enabled = false;
  state->parallel_detect_enabled = false;
  state->link_flags = 0xFFFFFFFFFFFFFFFF;
  state->link_technology = UALOE_ETH_TECH_400GBASE_KR4;
  state->fec_mode = UALOE_FEC_MODE_RS;
  state->loopback_mode = UALOE_NETPORT_LOOPBACK_NONE;

  memcpy(state->ifoe_mac_addr, arr, sizeof(uint8_t) * UALOE_MAC_ADDRESS_SIZE);
  memcpy(state->permanent_mac_addr, arr, sizeof(uint8_t) * UALOE_MAC_ADDRESS_SIZE);

  return 0;
}

int ualoe_cdev_telemetry_alloc(ualoe_handle_t handle, unsigned category_mask,
                               ualoe_telemetry_t** telemetry) {
  struct ualoe_cdev_handle* cdev_handle;
  ualoe_telemetry_dataset_t* dset;
  ualoe_telemetry_instance_t* instance;
  struct cfg_set_telemetry cfg_set_tele;
  int i, j, rc;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;

  /* 1) allocate ualoe_telemetry_*t */
  *telemetry = calloc(1, sizeof(ualoe_telemetry_t));
  if (!*telemetry) return ENOMEM;

  for (i = 0; i < UALOE_TELEMETRY_CATEGORY_MAX; i++) {
    /* Skip categories not requested in category_mask */
    if (!((1U << i) & category_mask)) {
      /* Leave dataset pointer NULL for filtered categories */
      (*telemetry)->datasets[i] = NULL;
      continue;
    }

    dset = calloc(1, sizeof(*dset) + sizeof(dset->instances[0]) * CFG_TELEMETRY_MAX_INSTANCES);
    if (!dset) {
      rc = ENOMEM;
      goto free_dsets;
    }
    dset->instances = (ualoe_telemetry_instance_t*)&dset[1];

    (*telemetry)->datasets[i] = dset;

    for (j = 0; j < CFG_TELEMETRY_MAX_INSTANCES; j++) {
      instance = &dset->instances[j];

      /* allocate memory for array of items */
      instance->items = calloc(1, sizeof(instance->items[0]) * CFG_TELEMETRY_MAX_ITEMS);
      if (!instance->items) {
        rc = ENOMEM;
        goto free_dsets;
      }
    }
  }

  /* 2) call driver to 'select' and 'dma_cfg' */
  cfg_set_tele.dev_id = cdev_handle->dev_id;
  cfg_set_tele.category_mask = category_mask;
  rc = ioctl(handle, CFG_SET_TELEMETRY, &cfg_set_tele);
  if (rc == -1) {
    rc = errno;
    goto free_dsets;
  }

  return 0;

free_dsets:
  for (i = 0; i < UALOE_TELEMETRY_CATEGORY_MAX; i++) {
    if ((*telemetry)->datasets[i] != NULL) {
      for (j = 0; j < CFG_TELEMETRY_MAX_INSTANCES; j++) {
        if ((*telemetry)->datasets[i]->instances[j].items != NULL)
          free((*telemetry)->datasets[i]->instances[j].items);
      }
      free((*telemetry)->datasets[i]);
    }
  }
  free(*telemetry);
  return rc;
}

int ualoe_cdev_telemetry_get(ualoe_handle_t handle, ualoe_telemetry_t* telemetry) {
  struct ualoe_cdev_handle* cdev_handle;
  struct cfg_telemetry cfg_tele;  // TODO how to specify number of categories?
  int i, j, rc;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;

  cfg_tele.datasets = calloc(UALOE_TELEMETRY_CATEGORY_MAX, sizeof(struct cfg_telemetry_dataset));
  if (!cfg_tele.datasets) return ENOMEM;

  for (i = 0; i < CFG_TELEMETRY_CATEGORY_MAX; i++) {
    cfg_tele.datasets[i].instances =
        calloc(CFG_TELEMETRY_MAX_INSTANCES, sizeof(struct cfg_telemetry_instance));
    if (!cfg_tele.datasets[i].instances) {
      rc = ENOMEM;
      goto out_free_instances;
    }

    for (j = 0; j < CFG_TELEMETRY_MAX_INSTANCES; j++) {
      cfg_tele.datasets[i].instances[j].items =
          calloc(CFG_TELEMETRY_MAX_ITEMS, sizeof(struct cfg_telemetry_item));
      if (!cfg_tele.datasets[i].instances[j].items) {
        rc = ENOMEM;
        goto out_free_instances;
      }
    }
  }
  cfg_tele.dev_id = cdev_handle->dev_id;

  rc = ioctl(handle, CFG_GET_TELEMETRY, &cfg_tele);
  if (rc == -1) {
    rc = errno;
    goto out_free_instances;
  }

  /* Validate telemetry structure matches driver response */
  for (i = 0; i < UALOE_TELEMETRY_CATEGORY_MAX; i++) {
    /* Check if firmware returned unrequested category */
    if (cfg_tele.datasets[i].instance_count > 0 && telemetry->datasets[i] == NULL) {
      ualoe_log_error(
          "Failed to get telemetry, firmware returned category %d "
          "which was not requested in the original category_mask.\n",
          i);
      rc = EINVAL;
      goto out_free_instances;
    }

    /* Check if requested category has no data from firmware */
    if (telemetry->datasets[i] != NULL && cfg_tele.datasets[i].instance_count == 0) {
      ualoe_log_error(
          "Failed to get telemetry, category %d was requested "
          "but firmware returned no data for this category.\n",
          i);
      rc = EINVAL;
      goto out_free_instances;
    }
  }

  /* copy driver cfg_tele into userspace telemetry */
  for (i = 0; i < UALOE_TELEMETRY_CATEGORY_MAX; i++) {
    /* Skip NULL datasets (filtered categories) */
    if (telemetry->datasets[i] == NULL) continue;

    telemetry->datasets[i]->category = cfg_tele.datasets[i].category;
    telemetry->datasets[i]->generation_count = cfg_tele.datasets[i].generation_count;
    telemetry->datasets[i]->timestamp = cfg_tele.datasets[i].timestamp;
    telemetry->datasets[i]->instance_count = cfg_tele.datasets[i].instance_count;

    for (j = 0; j < telemetry->datasets[i]->instance_count; j++) {
      /* TODO test with populated 'text' strings */
      if (cfg_tele.datasets[i].instances[j].name.text[0] != '\0') {
        strncpy(telemetry->datasets[i]->instances[j].name.text,
                cfg_tele.datasets[i].instances[j].name.text, UALOE_LABEL_SIZE - 1);
        telemetry->datasets[i]->instances[j].name.text[UALOE_LABEL_SIZE - 1] = '\0';
      }

      telemetry->datasets[i]->instances[j].logical_idx =
          cfg_tele.datasets[i].instances[j].logical_id;
      telemetry->datasets[i]->instances[j].item_count =
          cfg_tele.datasets[i].instances[j].item_count;
      for (int k = 0; k < telemetry->datasets[i]->instances[j].item_count; k++) {
        telemetry->datasets[i]->instances[j].items[k].id =
            cfg_tele.datasets[i].instances[j].items[k].id;
        telemetry->datasets[i]->instances[j].items[k].value =
            cfg_tele.datasets[i].instances[j].items[k].value;
      }
    }
  }

out_free_instances:
  for (i = 0; i < CFG_TELEMETRY_CATEGORY_MAX; i++) {
    for (j = 0; j < CFG_TELEMETRY_MAX_INSTANCES; j++) free(cfg_tele.datasets[i].instances[j].items);
    free(cfg_tele.datasets[i].instances);
  }
  free(cfg_tele.datasets);

  return rc;
}

int ualoe_cdev_telemetry_free(ualoe_handle_t handle, ualoe_telemetry_t* telemetry) {
  struct ualoe_cdev_handle* cdev_handle;
  int rc;
  int i, j;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;

  rc = ioctl(handle, CFG_FREE_TELEMETRY, &cdev_handle->dev_id);
  if (rc == -1) return errno;

  for (i = UALOE_TELEMETRY_CATEGORY_MAX - 1; i >= 0; i--) {
    /* Skip NULL datasets (filtered categories) */
    if (telemetry->datasets[i] == NULL) continue;

    for (j = CFG_TELEMETRY_MAX_INSTANCES - 1; j >= 0; j--)
      free(telemetry->datasets[i]->instances[j].items);
    free(telemetry->datasets[i]);
  }
  free(telemetry);

  return 0;
}

int ualoe_cdev_l2ping_start(ualoe_handle_t handle, ualoe_ping_spec_t* spec, ualoe_ping_t** ping) {
  struct ualoe_cdev_handle* cdev_handle;
  struct cfg_l2ping_config cfg_config;
  struct cfg_l2ping_start cfg_start;
  int i, rc;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;

  *ping = NULL;

  /* Populate ioctl structure with spec fields */
  cfg_config.dev_id = cdev_handle->dev_id;
  cfg_config.specify_accelerator = spec->specify_accelerator ? 1 : 0;
  cfg_config.specify_netport = spec->specify_netport ? 1 : 0;
  cfg_config.include_ifoe_req = spec->include_ifoe_req ? 1 : 0;
  cfg_config.include_ifoe_resp = spec->include_ifoe_resp ? 1 : 0;
  cfg_config.include_non_ifoe = spec->include_non_ifoe ? 1 : 0;
  cfg_config.accelerator_id = spec->accelerator_id;
  cfg_config.netport_idx = spec->netport_idx;
  cfg_config.num_pings = spec->num_pings;

  /* Call driver to configure ping and get allocation info */
  rc = ioctl(handle, CFG_IFOE_L2PING_CONFIG, &cfg_config);
  if (rc == -1) return errno;

  /* Allocate ping context structure */
  *ping = calloc(1, sizeof(ualoe_ping_t));
  if (!*ping) return ENOMEM;

  /* Copy spec into context */
  memcpy(&(*ping)->spec, spec, sizeof(ualoe_ping_spec_t));

  /* Initialize progress fields */
  (*ping)->test_complete = false;
  (*ping)->progress = 0;
  (*ping)->total = 0;

  /* Store output fields from driver */
  (*ping)->handle = cfg_config.handle;
  (*ping)->num_accelerators = cfg_config.num_accelerators;
  (*ping)->num_netports = cfg_config.num_netports;

  /* Allocate array of accelerator results */
  (*ping)->accels = calloc(cfg_config.num_accelerators, sizeof(ualoe_ping_accel_result_t));
  if (!(*ping)->accels) {
    rc = ENOMEM;
    goto err_free_ping;
  }

  /* Allocate netport results array for each accelerator */
  for (i = 0; i < cfg_config.num_accelerators; i++) {
    (*ping)->accels[i].netports =
        calloc(cfg_config.num_netports, sizeof(ualoe_ping_netport_result_t));
    if (!(*ping)->accels[i].netports) {
      rc = ENOMEM;
      goto err_free_netports;
    }
  }

  /* Start the ping test */
  cfg_start.dev_id = cdev_handle->dev_id;
  cfg_start.handle = (*ping)->handle;

  rc = ioctl(handle, CFG_IFOE_L2PING_START, &cfg_start);
  if (rc == -1) {
    rc = errno;
    goto err_free_netports;
  }

  return 0;

err_free_netports:
  for (i = i - 1; i >= 0; i--) free((*ping)->accels[i].netports);
  free((*ping)->accels);
err_free_ping:
  free(*ping);
  *ping = NULL;
  return rc;
}

int ualoe_cdev_l2ping_update(ualoe_handle_t handle, ualoe_ping_t* ping) {
  struct ualoe_cdev_handle* cdev_handle;
  struct cfg_l2ping_update cfg_update;
  int i, rc;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;

  /* Populate ioctl structure */
  cfg_update.dev_id = cdev_handle->dev_id;
  cfg_update.handle = ping->handle;
  cfg_update.num_accelerators = 0;
  cfg_update.num_netports = 0;

  /* Call driver to get updated results */
  rc = ioctl(handle, CFG_IFOE_L2PING_UPDATE, &cfg_update);
  if (rc == -1) return errno;

  /* Update progress fields */
  ping->test_complete = cfg_update.complete ? true : false;
  ping->progress = cfg_update.progress;
  ping->total = cfg_update.total;

  ping->num_accelerators = cfg_update.num_accelerators;
  ping->num_netports = cfg_update.num_netports;

  /* Only allocate inner headers once complete */
  if (cfg_update.complete) {
    /* Allocate array of accelerator results if not already allocated */
    if (!ping->accels) {
      ping->accels = calloc(cfg_update.num_accelerators, sizeof(ualoe_ping_accel_result_t));
      if (!ping->accels) {
        rc = ENOMEM;
        goto ping_complete;
      }
    }

    /* Allocate netport results array for each accelerator if not already allocated */
    for (i = 0; i < cfg_update.num_accelerators; i++) {
      if (!ping->accels[i].netports) {
        ping->accels[i].netports =
            calloc(cfg_update.num_netports, sizeof(ualoe_ping_netport_result_t));
        if (!ping->accels[i].netports) {
          rc = ENOMEM;
          goto free_netports;
        }
      }
    }
    cfg_update.num_accelerators = ping->num_accelerators;
    cfg_update.num_netports = ping->num_netports;
    /* Safe cast - validated by _Static_assert at compile time */
    cfg_update.accels = (struct cfg_l2ping_accel_result*)ping->accels;

    /* Call driver again to get results, but now with allocated structs */
    rc = ioctl(handle, CFG_IFOE_L2PING_UPDATE, &cfg_update);
    if (rc == -1) goto free_netports;
  }

  return 0;

free_netports:
  for (i = i - 1; i >= 0; i--) free(ping->accels[i].netports);
  free(ping->accels);

ping_complete:
  return rc;
}

int ualoe_cdev_l2ping_fini(ualoe_handle_t handle, ualoe_ping_t* ping) {
  struct ualoe_cdev_handle* cdev_handle;
  struct cfg_l2ping_fini cfg_fini;
  int rc;
  int i;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;

  /* Call driver to finalize ping test */
  cfg_fini.dev_id = cdev_handle->dev_id;
  cfg_fini.handle = ping->handle;

  rc = ioctl(handle, CFG_IFOE_L2PING_FINI, &cfg_fini);
  if (rc == -1) return errno;

  /* Free allocated memory */
  for (i = ping->num_accelerators - 1; i >= 0; i--) {
    if (ping->accels[i].netports) free(ping->accels[i].netports);
  }

  if (ping->accels) free(ping->accels);

  if (ping) free(ping);

  return 0;
}

int ualoe_cdev_register_event_callback(ualoe_handle_t handle, ualoe_event_callback_t callback,
                                       void* user_context) {
  struct ualoe_cdev_handle* cdev_handle;
  int rc;

  rc = ualoe_cdev_find_handle(handle, &cdev_handle);
  if (rc) return rc;
  return ualoe_cb_init(handle, cdev_handle->dev_id, callback, user_context);
}
