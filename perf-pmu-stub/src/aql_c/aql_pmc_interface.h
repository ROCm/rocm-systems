/**
 * @file aql_pmc_interface.h
 * @brief Bare bones AQL PMC interface
 */

#ifndef AQL_PMC_INTERFACE_H
#define AQL_PMC_INTERFACE_H

#include "aql_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal event structure */
typedef struct {
    uint32_t block_id;
    uint32_t event_id;
} aql_pmc_event_t;

/* Minimal profile structure */
typedef struct {
    const aql_pmc_event_t* events;
    uint32_t event_count;
} aql_pmc_profile_t;

/* Minimal packet structure */
typedef struct {
    aql_pm4_ib_packet_t start_packet;
    aql_pm4_ib_packet_t stop_packet;
    aql_pm4_ib_packet_t read_packet;
} aql_pmc_packets_t;

/* Stub functions */
aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets,
                                   const aql_pmc_profile_t* profile);

aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets);

#ifdef __cplusplus
}
#endif

#endif /* AQL_PMC_INTERFACE_H */