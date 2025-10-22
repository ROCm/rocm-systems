/*
 * pmu_events.c - Event handling implementation for PMU Stub
 *
 * This file contains event-specific functionality and helpers
 * for the PMU stub driver. Events are dynamically generated from
 * the counter_registry to match available hardware counters.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/perf_event.h>
#include <linux/slab.h>

#include "amdgpu_pmu.h"
#include "aql_c/counter_registry.h"

/* Get event name from configuration */
const char *amdgpu_pmu_get_event_name(u64 config)
{
    const counter_def_t *counter;

    /* Config value is the counter_id */
    counter = lookup_counter_by_id((counter_id_t)config);
    if (counter) {
        return counter->name;
    }

    return "unknown";
}

/* Get event description from configuration */
const char *amdgpu_pmu_get_event_description(u64 config)
{
    const counter_def_t *counter;

    /* Config value is the counter_id */
    counter = lookup_counter_by_id((counter_id_t)config);
    if (counter) {
        /* Generate description from counter name and hardware block */
        static char description[128];
        const char *block_name;

        switch (counter->hw_block) {
        case HW_IP_BLOCK_GL2C:
            block_name = "GL2C (L2 Cache)";
            break;
        case HW_IP_BLOCK_SQ:
            block_name = "SQ (Shader Queue)";
            break;
        case HW_IP_BLOCK_TA:
            block_name = "TA (Texture Addressing)";
            break;
        case HW_IP_BLOCK_GRBM:
            block_name = "GRBM (Graphics Register Bus Manager)";
            break;
        default:
            block_name = "Unknown";
            break;
        }

        snprintf(description, sizeof(description), "%s - %s counter",
                 counter->name, block_name);
        return description;
    }

    return "Unknown event";
}

/* Validate event configuration */
bool amdgpu_pmu_is_valid_event(u64 config)
{
    /* Check if the counter ID exists in the registry */
    return lookup_counter_by_id((counter_id_t)config) != NULL;
}

/* Get total number of available events */
size_t amdgpu_pmu_get_event_count(void)
{
    return get_counter_count();
}

// TODO: Remove legacy stubs. 
/*
 * NOTE: Counter values are now managed by the AQL hardware layer.
 * The following functions are legacy stubs kept for compatibility.
 * Actual counter values are read from GPU hardware via AQL packets.
 */

/* Get counter value based on event type */
u64 amdgpu_pmu_get_counter_value(struct amdgpu_pmu *pmu, u64 config)
{
    /* Counter values are now read from hardware via AQL */
    pmu_debug("Legacy counter read for config 0x%llx - values managed by AQL\n", config);
    return 0;
}

/* Update counter value based on event type */
void amdgpu_pmu_update_counter(struct amdgpu_pmu *pmu, u64 config, s64 delta)
{
    /* Counter updates are now handled by hardware via AQL */
    pmu_debug("Legacy counter update for config 0x%llx - managed by AQL\n", config);
}

/* Reset all counters */
void amdgpu_pmu_reset_counters(struct amdgpu_pmu *pmu)
{
    /* Counter resets are now handled by hardware via AQL */
    pmu_debug("Legacy counter reset - managed by AQL\n");
}

/* Print event statistics (for debugging) */
void amdgpu_pmu_print_event_stats(struct amdgpu_pmu *pmu)
{
    pmu_info("Event Statistics:\n");
    pmu_info("  Total Events: %lld\n", atomic64_read(&pmu->total_events));
    pmu_info("  Hardware Events: %lld\n", atomic64_read(&pmu->hardware_events));
    pmu_info("  Simulation Events: %lld\n", atomic64_read(&pmu->simulation_events));
}

/* Export symbols if needed by other modules */
EXPORT_SYMBOL_GPL(amdgpu_pmu_get_event_name);
EXPORT_SYMBOL_GPL(amdgpu_pmu_get_event_description);
EXPORT_SYMBOL_GPL(amdgpu_pmu_is_valid_event);