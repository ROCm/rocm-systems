/*
 * pmu_events.c - Event handling implementation for PMU Stub
 *
 * This file contains event-specific functionality and helpers
 * for the PMU stub driver. Events correspond to GFX12 hardware counters.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/perf_event.h>
#include <linux/slab.h>

#include "pmu_stub.h"

/* Event configuration table - matches GFX12 hardware counters */
static const struct pmu_stub_event_config pmu_stub_events[] = {
    {
        .name = "sq_waves",
        .config = PMU_STUB_EVENT_SQ_WAVES,
        .config_mask = 0xFF,
        .description = "Number of waves in shader queues (GFX12_PERF_SEL_SQ_WAVES)"
    },
    {
        .name = "sq_instructions",
        .config = PMU_STUB_EVENT_SQ_INSTRUCTIONS,
        .config_mask = 0xFF,
        .description = "Number of shader instructions executed (GFX12_PERF_SEL_SQ_INSTS)"
    },
    {
        .name = "ta_busy",
        .config = PMU_STUB_EVENT_TA_BUSY,
        .config_mask = 0xFF,
        .description = "Texture unit busy cycles (GFX12_PERF_SEL_TA_BUSY)"
    },
};

/* Get event name from configuration */
const char *pmu_stub_get_event_name(u64 config)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(pmu_stub_events); i++) {
        if (pmu_stub_events[i].config == config) {
            return pmu_stub_events[i].name;
        }
    }

    return "unknown";
}

/* Get event description from configuration */
const char *pmu_stub_get_event_description(u64 config)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(pmu_stub_events); i++) {
        if (pmu_stub_events[i].config == config) {
            return pmu_stub_events[i].description;
        }
    }

    return "Unknown event";
}

/* Validate event configuration */
bool pmu_stub_is_valid_event(u64 config)
{
    return config < PMU_STUB_EVENT_MAX;
}

/* Get counter value based on event type */
u64 pmu_stub_get_counter_value(struct pmu_stub *pmu, u64 config)
{
    switch (config) {
    case PMU_STUB_EVENT_SQ_WAVES:
        return atomic64_read(&pmu->counter_sq_waves);
    case PMU_STUB_EVENT_SQ_INSTRUCTIONS:
        return atomic64_read(&pmu->counter_sq_instructions);
    case PMU_STUB_EVENT_TA_BUSY:
        return atomic64_read(&pmu->counter_ta_busy);
    default:
        return 0;
    }
}

/* Update counter value based on event type */
void pmu_stub_update_counter(struct pmu_stub *pmu, u64 config, s64 delta)
{
    switch (config) {
    case PMU_STUB_EVENT_SQ_WAVES:
        atomic64_add(delta, &pmu->counter_sq_waves);
        break;
    case PMU_STUB_EVENT_SQ_INSTRUCTIONS:
        atomic64_add(delta, &pmu->counter_sq_instructions);
        break;
    case PMU_STUB_EVENT_TA_BUSY:
        atomic64_add(delta, &pmu->counter_ta_busy);
        break;
    }
}

/* Reset all counters */
void pmu_stub_reset_counters(struct pmu_stub *pmu)
{
    atomic64_set(&pmu->counter_sq_waves, 0);
    atomic64_set(&pmu->counter_sq_instructions, 0);
    atomic64_set(&pmu->counter_ta_busy, 0);
}

/* Print event statistics (for debugging) */
void pmu_stub_print_event_stats(struct pmu_stub *pmu)
{
    pmu_info("Event Statistics:\n");
    pmu_info("  SQ Waves: %lld\n", atomic64_read(&pmu->counter_sq_waves));
    pmu_info("  SQ Instructions: %lld\n", atomic64_read(&pmu->counter_sq_instructions));
    pmu_info("  TA Busy: %lld\n", atomic64_read(&pmu->counter_ta_busy));
    pmu_info("  Total Events: %lld\n", atomic64_read(&pmu->total_events));
    pmu_info("  Total Samples: %lld\n", atomic64_read(&pmu->total_samples));
}

/* Export symbols if needed by other modules */
EXPORT_SYMBOL_GPL(pmu_stub_get_event_name);
EXPORT_SYMBOL_GPL(pmu_stub_get_event_description);
EXPORT_SYMBOL_GPL(pmu_stub_is_valid_event);