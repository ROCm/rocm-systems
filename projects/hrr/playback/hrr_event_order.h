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
  switch (etype) {
    // Memory alloc / free
    case HRR_API_HIPMALLOC:
    case HRR_API_HIPEXTMALLOCWITHFLAGS:
    case HRR_API_HIPMALLOCASYNC:
    case HRR_API_HIPMALLOCFROMPOOLASYNC:
    case HRR_API_HIPMALLOCMANAGED:
    case HRR_API_HIPMALLOCHOST:
    case HRR_API_HIPMALLOC3D:
    case HRR_API_HIPMALLOC3DARRAY:
    case HRR_API_HIPMALLOCARRAY:
    case HRR_API_HIPMALLOCMIPMAPPEDARRAY:
    case HRR_API_HIPMALLOCPITCH:
    case HRR_API_HIPFREE:
    case HRR_API_HIPFREEASYNC:
    case HRR_API_HIPFREEARRAY:
    case HRR_API_HIPFREEHOST:
    case HRR_API_HIPFREEMIPMAPPEDARRAY:
    case HRR_API_HIPHOSTFREE:
    case HRR_API_HIPMEMADDRESSFREE:
    case HRR_API_HIPMEMRELEASE:
    // Host/pinned + VMM handle creation. These populate shared maps
    // (alloc_map / host_reg_bufs / vmm_va_map / vmm_handle_map) that later
    // consumers (e.g. hipMemMap, hipHostGetDevicePointer) translate against.
    // They must be ordered so a cross-thread consumer can never run before the
    // create populates the map (their destroy/free counterparts above are
    // already ordered — this restores the symmetry).
    case HRR_API_HIPHOSTREGISTER:
    case HRR_API_HIPHOSTUNREGISTER:
    case HRR_API_HIPHOSTGETDEVICEPOINTER:
    case HRR_API_HIPHOSTMALLOC:
    case HRR_API_HIPMEMADDRESSRESERVE:
    case HRR_API_HIPMEMCREATE:
    // Stream create / destroy
    case HRR_API_HIPSTREAMCREATE:
    case HRR_API_HIPSTREAMCREATEWITHFLAGS:
    case HRR_API_HIPSTREAMCREATEWITHPRIORITY:
    case HRR_API_HIPSTREAMDESTROY:
    // Event create / destroy
    case HRR_API_HIPEVENTCREATE:
    case HRR_API_HIPEVENTCREATEWITHFLAGS:
    case HRR_API_HIPEVENTDESTROY:
    // Module load / unload
    case HRR_API_HIPMODULELOAD:
    case HRR_API_HIPMODULELOADDATA:
    case HRR_API_HIPMODULELOADDATAEX:
    case HRR_API_HIPMODULELOADFATBINARY:
    case HRR_API_HIPMODULEUNLOAD:
    case HRR_API_HIPREGISTERFATBINARY:
    // Graph / graph-exec create
    case HRR_API_HIPSTREAMBEGINCAPTURE:
    case HRR_API_HIPSTREAMENDCAPTURE:
    // The other ways to open and close a capture: replay tracks which captures
    // are open, and a free on another thread has to see that in order.
    case HRR_API_HIPSTREAMBEGINCAPTURETOGRAPH:
    case HRR_API_HIPSTREAMBEGINCAPTURE_SPT:
    case HRR_API_HIPSTREAMENDCAPTURE_SPT:
    case HRR_API_HIPGRAPHINSTANTIATE:
    case HRR_API_HIPGRAPHINSTANTIATEWITHFLAGS:
    case HRR_API_HIPGRAPHINSTANTIATEWITHPARAMS:
    case HRR_API_HIPGRAPHEXECDESTROY:
    case HRR_API_HIPGRAPHDESTROY:
    case HRR_API_HIPLINKDESTROY:
    // MemPool create / destroy
    case HRR_API_HIPMEMPOOLCREATE:
    case HRR_API_HIPMEMPOOLDESTROY:
    // Array / mipmapped array create / destroy
    case HRR_API_HIPARRAY3DCREATE:
    case HRR_API_HIPARRAYCREATE:
    case HRR_API_HIPARRAYDESTROY:
    case HRR_API_HIPMIPMAPPEDARRAYCREATE:
    case HRR_API_HIPMIPMAPPEDARRAYDESTROY:
    // Texture / surface object create / destroy
    case HRR_API_HIPCREATETEXTUREOBJECT:
    case HRR_API_HIPCREATESURFACEOBJECT:
    case HRR_API_HIPDESTROYSURFACEOBJECT:
    case HRR_API_HIPDESTROYTEXTUREOBJECT:
    case HRR_API_HIPTEXOBJECTDESTROY:
      return true;
    default:
      return false;
  }
}
