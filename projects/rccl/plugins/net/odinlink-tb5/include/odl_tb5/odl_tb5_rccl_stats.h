/*
 * OdinLink Thunderbolt 5 - RCCL Shared-Memory Stats Definition
 */
#ifndef ODL_TB5_RCCL_STATS_H
#define ODL_TB5_RCCL_STATS_H

#include <stdint.h>

#define ODL_RCCL_STATS_MAGIC   0x4F444C5253544154ULL
#define ODL_RCCL_STATS_VERSION 1
#define ODL_RCCL_STATS_DIR     "/run/odl_tb5"
#define ODL_RCCL_STATS_PATH    "/run/odl_tb5/rccl_stats"

struct odl_rccl_stats {
	uint64_t magic;
	uint64_t version;
	uint64_t tx_bytes;
	uint64_t rx_bytes;
	uint64_t tx_ops;
	uint64_t rx_ops;
	uint64_t start_time_ns;
	uint64_t last_update_ns;
	uint32_t active;
	uint32_t reserved[15];
};

#endif /* ODL_TB5_RCCL_STATS_H */
