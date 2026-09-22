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
		/* Return the counter name directly.
		 * The previous implementation used a static buffer which is
		 * unsafe for concurrent access from the perf subsystem. */
		return counter->name;
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


/* Export symbols if needed by other modules */
EXPORT_SYMBOL_GPL(amdgpu_pmu_get_event_name);
EXPORT_SYMBOL_GPL(amdgpu_pmu_get_event_description);
EXPORT_SYMBOL_GPL(amdgpu_pmu_is_valid_event);