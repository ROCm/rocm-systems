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

#ifndef UALOE_NL_H
#define UALOE_NL_H

#include "ualoe_lib.h"

int ualoe_nl_open(const char* gpu_uuid, ualoe_handle_t* handle);
int ualoe_nl_close(ualoe_handle_t handle);
int ualoe_nl_get_version(ualoe_handle_t handle, ualoe_version_t* lib_version,
                         ualoe_version_t* fw_version, ualoe_version_t* telemetry_version);
int ualoe_nl_reset(ualoe_handle_t handle);
int ualoe_nl_get_capabilities(ualoe_handle_t handle, ualoe_capabilities_t* capabilities);
int ualoe_nl_set_identity(ualoe_handle_t handle, unsigned accelerator_id);
int ualoe_nl_set_accelerator_config(ualoe_handle_t handle, unsigned bitmask_size,
                                    uint32_t active_accelerator_bitmask[],
                                    uint32_t local_accelerator_bitmask[]);
int ualoe_nl_set_ifoe_config(ualoe_handle_t handle, ifoe_virt_mode_e virt_mode,
                             ifoe_encap_type_e encap_type, ifoe_failover_mode_e failover_mode,
                             ifoe_loopback_mode_e loopback_mode);
int ualoe_nl_get_ifoe_config(ualoe_handle_t handle, ifoe_config_t* config, unsigned bitmask_size,
                             uint32_t active_accelerator_bitmask[],
                             uint32_t local_accelerator_bitmask[],
                             uint32_t enabled_accelerator_bitmask[]);
int ualoe_nl_set_config_phase(ualoe_handle_t handle, ualoe_config_phase_e next_phase);
int ualoe_nl_get_current_config_phase(ualoe_handle_t handle, ualoe_config_phase_e* phase);
int ualoe_nl_enable_accelerators(ualoe_handle_t handle, unsigned bitmask_size,
                                 uint32_t enabled_accelerator_bitmask[]);
int ualoe_nl_config_crypto(ualoe_handle_t handle, ualoe_crypto_mode_e mode);
int ualoe_nl_set_tx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                               ualoe_crypto_key_t* key);
int ualoe_nl_disable_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id);
int ualoe_nl_set_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                               ualoe_crypto_key_t* key);
int ifoe_nl_get_station_list(ualoe_handle_t handle, unsigned desc_count,
                             ifoe_station_desc_t descs[]);
int ifoe_nl_station_ctrl(ualoe_handle_t handle, unsigned station_idx, ifoe_station_state_e state);
int ifoe_nl_station_get_state(ualoe_handle_t handle, unsigned station_idx,
                              ifoe_station_state_t* state);
int ifoe_nl_set_path_to_port_map(ualoe_handle_t handle, bool specify_station,
                                 bool specify_accelerator, bool reenable_streams,
                                 unsigned station_idx, unsigned accelerator_id, unsigned path_count,
                                 unsigned map[]);
int ifoe_nl_get_path_to_port_map(ualoe_handle_t handle, unsigned station_idx,
                                 unsigned accelerator_id, unsigned path_count, unsigned map[]);
int ifoe_nl_get_netport_list(ualoe_handle_t handle, unsigned desc_count,
                             ifoe_netport_desc_t descs[]);
int ifoe_nl_netport_ctrl(ualoe_handle_t handle, unsigned netport_idx, ifoe_netport_state_e state);
int ifoe_nl_netport_set_accelerator_addr_map(ualoe_handle_t handle, unsigned netport_idx,
                                             ifoe_network_addr_type_e map_addr_type,
                                             unsigned map_count, ifoe_accelerator_addr_map_t map[]);
int ifoe_nl_netport_set_addr(ualoe_handle_t handle, unsigned netport_idx,
                             ifoe_network_addr_type_e addr_type, uint8_t mac_addr[],
                             uint32_t ip_addr);
int ifoe_nl_netport_config_link_auto(ualoe_handle_t handle, unsigned netport_idx,
                                     bool parallel_detect_enable, __uint128_t advertised_eth_techs,
                                     ualoe_fec_mode_e requested_fec_mode);
int ifoe_nl_netport_config_link_manual(ualoe_handle_t handle, unsigned netport_idx,
                                       ualoe_eth_tech_e eth_tech, ualoe_fec_mode_e fec_mode,
                                       ualoe_netport_loopback_mode_e loopback_mode);
int ifoe_nl_netport_get_state(ualoe_handle_t handle, unsigned netport_idx,
                              ifoe_netport_state_t* state);
int ifoe_nl_get_netport_properties(ualoe_handle_t handle, ualoe_netport_properties_t* properties);
int ualoe_nl_telemetry_alloc(ualoe_handle_t handle, unsigned category_mask,
                             ualoe_telemetry_t** telemetry);
int ualoe_nl_telemetry_get(ualoe_handle_t handle, ualoe_telemetry_t* telemetry);
int ualoe_nl_telemetry_free(ualoe_handle_t handle, ualoe_telemetry_t* telemetry);
int ualoe_nl_l2ping_start(ualoe_handle_t handle, ualoe_ping_spec_t* spec, ualoe_ping_t** ping);
int ualoe_nl_l2ping_update(ualoe_handle_t handle, ualoe_ping_t* ping);
int ualoe_nl_l2ping_fini(ualoe_handle_t handle, ualoe_ping_t* ping);
int ualoe_nl_register_event_callback(ualoe_handle_t handle, ualoe_event_callback_t callback,
                                     void* user_data);
#endif /* UALOE_NL_H */
