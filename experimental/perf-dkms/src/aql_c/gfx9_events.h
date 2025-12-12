/**
 * @file gfx9_events.h
 * @brief GFX9 architecture-specific event mappings
 */

#ifndef GFX9_EVENTS_H
#define GFX9_EVENTS_H

#include "counter_registry.h"

/**
 * @brief Get the GFX9 event mapping array
 *
 * Returns the array mapping counter IDs to GFX9 hardware event IDs.
 *
 * @return Pointer to const array of arch_event_map structures
 * @see get_gfx9_event_count()
 */
const struct arch_event_map *get_gfx9_events(void);

/**
 * @brief Get the number of GFX9 event mappings
 *
 * @return Number of event mappings in the GFX9 array
 * @see get_gfx9_events()
 */
size_t get_gfx9_event_count(void);

#endif /* GFX9_EVENTS_H */
