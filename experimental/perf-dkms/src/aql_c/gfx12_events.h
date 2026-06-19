/**
 * @file gfx12_events.h
 * @brief GFX12 architecture-specific event mappings
 */

#ifndef GFX12_EVENTS_H
#define GFX12_EVENTS_H

#include "counter_registry.h"

/**
 * @brief Get the GFX12 event mapping array
 *
 * Returns the array mapping counter IDs to GFX12 hardware event IDs.
 *
 * @return Pointer to const array of arch_event_map structures
 * @see get_gfx12_event_count()
 */
const struct arch_event_map *get_gfx12_events(void);

/**
 * @brief Get the number of GFX12 event mappings
 *
 * @return Number of event mappings in the GFX12 array
 * @see get_gfx12_events()
 */
size_t get_gfx12_event_count(void);

#endif /* GFX12_EVENTS_H */