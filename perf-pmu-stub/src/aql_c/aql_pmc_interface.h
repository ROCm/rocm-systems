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

/**
 * @brief Minimal event structure for performance counter specification
 *
 * Specifies a single performance counter event by hardware block and event ID.
 * This is the minimal interface for requesting counter monitoring.
 */
typedef struct {
    uint32_t block_id;               /* Hardware block identifier */
    uint32_t event_id;               /* Event ID to monitor within block */
} aql_pmc_event_t;

/**
 * @brief Minimal profile structure defining a set of counters to monitor
 *
 * Aggregates multiple counter events into a profile that can be started,
 * stopped, and read as a unit.
 */
typedef struct {
    const aql_pmc_event_t* events;   /* Array of events to monitor */
    uint32_t event_count;            /* Number of events in array */
} aql_pmc_profile_t;

/**
 * @brief Minimal packet structure containing PM4 command buffers
 *
 * Contains the three PM4 command packets needed for performance monitoring:
 * start (configure and enable counters), stop (disable counters), and
 * read (copy counter values to memory).
 */
typedef struct {
    aql_pm4_ib_packet_t start_packet; /* PM4 commands to start counters */
    aql_pm4_ib_packet_t stop_packet;  /* PM4 commands to stop counters */
    aql_pm4_ib_packet_t read_packet;  /* PM4 commands to read counter values */
} aql_pmc_packets_t;

/**
 * @brief Create PM4 packet sets for performance counter profiling (stub)
 *
 * Generates start/stop/read PM4 packet sequences for the specified profile.
 * Currently a stub implementation.
 *
 * @param packets Output structure to receive generated packet buffers
 * @param profile Input profile defining which counters to monitor
 * @return AQL_SUCCESS on success, AQL_ERROR_INVALID_ARGUMENT if parameters are NULL
 *
 * @see aql_pmc_delete_packets()
 */
aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets,
                                   const aql_pmc_profile_t* profile);

/**
 * @brief Delete PM4 packet sets and free resources (stub)
 *
 * Frees memory allocated for PM4 command buffers. Currently a stub implementation.
 *
 * @param packets Pointer to packet structure to clean up
 * @return AQL_SUCCESS on success, AQL_ERROR_INVALID_ARGUMENT if packets is NULL
 *
 * @see aql_pmc_create_packets()
 */
aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets);

#ifdef __cplusplus
}
#endif

#endif /* AQL_PMC_INTERFACE_H */