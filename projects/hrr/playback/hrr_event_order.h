/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/* hrr_event_order.h — which replayed events run in global capture order. */

#include <cstdint>

#include "hrr/hrr_api_args.h"

// Events that create or destroy handles written into PlaybackContext maps.
// These must be submitted in global capture order so that handle translations
// are available before any thread that depends on them runs.
// Kernel launches and syncs are excluded — GPU stream ordering handles them.
inline bool hrr_needs_ordering(uint16_t etype) {
  (void)etype;
  return false;  // negative control
}
