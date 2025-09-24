/**
 * @file aql_pmc_interface.c
 * @brief Bare bones AQL PMC interface implementation
 */

#include "aql_pmc_interface.h"

#ifdef __KERNEL__
#include <linux/string.h>
#else
#include <string.h>
#endif

aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets,
                                   const aql_pmc_profile_t* profile) {
    if (!packets || !profile) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Just zero out the structure */
    memset(packets, 0, sizeof(*packets));
    return AQL_SUCCESS;
}

aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets) {
    if (!packets) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Just zero out the structure */
    memset(packets, 0, sizeof(*packets));
    return AQL_SUCCESS;
}