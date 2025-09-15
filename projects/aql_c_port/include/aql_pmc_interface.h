/**
 * @file aql_pmc_interface.h
 * @brief AQLProfile v2-compatible PMC interface for aql_c library
 *
 * This header provides an interface compatible with AQLProfile v2's
 * aqlprofile_pmc_create_packets functionality.
 */

#ifndef AQL_PMC_INTERFACE_H
#define AQL_PMC_INTERFACE_H

#include "aql_types.h"
#include "aql_arch_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief PMC event specification (compatible with aqlprofile v2)
 */
typedef struct {
    aql_block_id_t block_id;        /**< Block ID */
    uint32_t block_instance;        /**< Block instance/channel */
    uint32_t event_id;              /**< Event ID */
    uint32_t flags;                 /**< Event flags */

    // Debug information
    const char* block_name;         /**< Block name string */
    const char* event_name;         /**< Event name string */
} aql_pmc_event_t;

/**
 * @brief PMC profile configuration
 */
typedef struct {
    const char* arch_name;          /**< Architecture name (e.g., "gfx942") */
    const aql_pmc_event_t* events;  /**< Array of events to monitor */
    uint32_t event_count;           /**< Number of events */

    // Output buffer information
    void* output_buffer;            /**< Buffer for counter results */
    size_t output_buffer_size;      /**< Size of output buffer */
} aql_pmc_profile_t;

/**
 * @brief AQL packet set (compatible with aqlprofile v2 structure)
 */
typedef struct {
    aql_pm4_ib_packet_t start_packet;   /**< Reset counters and start incrementing */
    aql_pm4_ib_packet_t stop_packet;    /**< Pause counters from incrementing */
    aql_pm4_ib_packet_t read_packet;    /**< Retrieve results from device */

    // Internal state
    void* command_buffer;               /**< Command buffer memory */
    size_t command_buffer_size;         /**< Command buffer size */
    uint64_t handle;                    /**< Handle for cleanup */
} aql_pmc_packets_t;

/**
 * @brief Create PMC AQL packets (v2-compatible interface)
 *
 * This function creates start, stop, and read AQL packets for performance
 * counter monitoring, similar to aqlprofile_pmc_create_packets().
 *
 * @param[out] packets Pointer to packet structure to populate
 * @param[in] profile PMC profile configuration
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets,
                                   const aql_pmc_profile_t* profile);

/**
 * @brief Delete PMC AQL packets and free resources
 *
 * @param[in] packets Packet structure to clean up
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets);

/**
 * @brief Helper function to create a single counter event
 *
 * @param[out] event Event structure to populate
 * @param[in] block_name Block name string (e.g., "CPC", "GRBM")
 * @param[in] instance Block instance number
 * @param[in] event_id Event ID number
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_create_counter_event(aql_pmc_event_t* event,
                                     const char* block_name,
                                     uint32_t instance,
                                     uint32_t event_id);

#ifdef __cplusplus
}
#endif

#endif /* AQL_PMC_INTERFACE_H */