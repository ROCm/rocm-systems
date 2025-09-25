/**
 * @file gfx12_events.h
 * @brief GFX12 architecture-specific event mappings
 */

#ifndef GFX12_EVENTS_H
#define GFX12_EVENTS_H

#include "counter_registry.h"

/* GFX12 event mapping functions */
const arch_event_map_t* get_gfx12_events(void);
size_t get_gfx12_event_count(void);

#endif /* GFX12_EVENTS_H */