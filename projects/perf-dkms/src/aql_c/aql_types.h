/**
 * @file aql_types.h
 * @brief Basic AQL types - bare bones version
 */

#ifndef AQL_TYPES_H
#define AQL_TYPES_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif

/* Basic result codes */
typedef enum {
	AQL_SUCCESS = 0,
	AQL_ERROR_INVALID_ARGUMENT = -1,
	AQL_ERROR_NO_MEMORY = -2,
	AQL_ERROR_UNSUPPORTED = -3,
} aql_result_t;

/* Basic packet structure */
typedef struct {
	uint32_t header;
	uint32_t *ib_base;
	uint32_t ib_size_dw;
} aql_pm4_ib_packet_t;

#endif /* AQL_TYPES_H */