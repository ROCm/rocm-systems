/* SPDX-License-Identifier: MIT */
/*
 * Copyright (c) OdinLink-Five contributors
 * https://github.com/Geramy/OdinLink-Five
 *
 * OdinLink Thunderbolt 5 - Shared Type Definitions
 */
#ifndef ODL_TB5_TYPES_H
#define ODL_TB5_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* Connection states (mirrors kernel enum odl_tb5_conn_state) */
enum odl_tb5_conn_state {
	ODL_TB5_STATE_DISCONNECTED = 0,
	ODL_TB5_STATE_HANDSHAKE    = 1,
	ODL_TB5_STATE_CONNECTED    = 2,
	ODL_TB5_STATE_ERROR        = 3,
	ODL_TB5_STATE_READY        = 4,
};

/* Peer information */
struct odl_tb5_peer_info {
	uint8_t  uuid[16];
	uint32_t link_speed;
	uint32_t link_width;
	uint32_t state;
	uint32_t reserved;
	char     vendor_name[64];
	char     device_name[64];
};

struct odl_tb5_link_info {
	uint8_t  local_uuid[16];
	uint8_t  peer_uuid[16];
	uint32_t link_speed;
	uint32_t link_width;
	uint32_t state;
	uint32_t reserved;
	char     vendor_name[64];
	char     device_name[64];
};

/* Completion status */
struct odl_tb5_completion {
	uint32_t tx_completed;
	uint32_t rx_completed;
	uint32_t tx_submitted;
	uint32_t rx_submitted;
};

/* Buffer info */
struct odl_tb5_buf_info {
	uint64_t tx_buf_size;
	uint64_t rx_buf_size;
	uint32_t tx_buf_count;
	uint32_t rx_buf_count;
};

#endif /* ODL_TB5_TYPES_H */
