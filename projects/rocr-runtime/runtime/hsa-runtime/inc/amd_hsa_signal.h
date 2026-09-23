////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef AMD_HSA_SIGNAL_H
#define AMD_HSA_SIGNAL_H

#include "amd_hsa_common.h"
#include "amd_hsa_queue.h"

// AMD Signal Kind Enumeration Values.
typedef int64_t amd_signal_kind64_t;
enum amd_signal_kind_t {
  AMD_SIGNAL_KIND_INVALID = 0,
  AMD_SIGNAL_KIND_USER = 1,
  AMD_SIGNAL_KIND_DOORBELL = -1,
  AMD_SIGNAL_KIND_LEGACY_DOORBELL = -2
};

// AMD Signal (v1). Size and offsets are frozen; the reserved spellings are
// retained in unions so existing references keep compiling.
#define AMD_SIGNAL_ALIGN_BYTES 64
#define AMD_SIGNAL_ALIGN __ALIGNED__(AMD_SIGNAL_ALIGN_BYTES)
typedef struct AMD_SIGNAL_ALIGN amd_signal_s {
  amd_signal_kind64_t kind;                /* 0x000 */
  union {
    volatile int64_t value;                /* 0x008 */
    volatile uint64_t* hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;              /* 0x010 */
  uint32_t event_id;                       /* 0x018 */
  uint32_t reserved1;                      /* 0x01c  zero; v2 properties */
  uint64_t start_ts;                       /* 0x020 */
  uint64_t end_ts;                         /* 0x028 */
  union {
    amd_queue_v2_t* queue_ptr;             /* 0x030  doorbell signals only */
    uint64_t reserved2;
  };
  uint32_t reserved3[2];                   /* 0x038 */
} amd_signal_t;

// AMD Signal v2. Line 0 is v1 field for field. The 1024-byte alignment is
// load-bearing: it leaves the low 10 bits of every v2 handle zero.
#define AMD_SIGNAL_V2_ALIGN_BYTES 1024
#define AMD_SIGNAL_V2_ALIGN __ALIGNED__(AMD_SIGNAL_V2_ALIGN_BYTES)
typedef struct AMD_SIGNAL_V2_ALIGN amd_signal_v2_s {
  /* Line 0: identical to v1. */
  amd_signal_kind64_t kind;                /* 0x000 */
  union {
    volatile int64_t value;                /* 0x008 */
    volatile uint64_t* hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;              /* 0x010 */
  uint32_t event_id;                       /* 0x018 */
  uint32_t properties;                     /* 0x01c  has AMD_SIGNAL_PROPERTY_FORMAT_V2 */
  uint64_t start_ts;                       /* 0x020  also the SDMA copy start */
  uint64_t end_ts;                         /* 0x028  also the SDMA copy end */
  union {
    amd_queue_v2_t* queue_ptr;             /* 0x030  doorbell signals only */
    uint64_t reserved2;
  };
  uint32_t reserved3[2];                   /* 0x038 */

  /* Line 1. */
  uint64_t reserved4;                      /* 0x040 */
  uint64_t runtime_private[2];             /* 0x048  runtime use; CP never reads */
  uint64_t reserved5[5];                   /* 0x058 - 0x07F */

  /* Lines 2-3. May be filled, in addition to start_ts/end_ts, for a
     completion signal hinted TS_SLOTS, one slot per hardware unit that ran
     the work. Unindexed: a zero slot is an unused one. Eight covers the most
     XCCs on any part today; the reserved lines below leave room to grow, for
     example to one slot per SDMA front end. */
  struct {
    uint64_t start_ts;
    uint64_t end_ts;
  } timestamp_slot[8];                     /* 0x080 - 0x0FF */

  /* Lines 4-15. */
  uint64_t reserved_v2_tail[96];           /* 0x100 - 0x3FF */
} amd_signal_v2_t;

// Signal handle hints. A handle points to a signal whose alignment leaves its
// low bits zero; a producer may set hints there so the CP can act on a
// completion signal without loading it. Hints live only in the handle, never
// in the signal, so they may differ per use. Zero is today's behaviour.
// Hints are purely optional: a consumer may ignore any of them, so a producer
// must be correct whether or not a hint is honoured.
typedef enum {
  /* Bit 0: FORMAT. Selects the hint mask; set only on a v2 handle that
     carries bits 6-9. */
  AMD_SIGNAL_HINT_FORMAT_V1     = 0u,
  AMD_SIGNAL_HINT_FORMAT_V2     = 1u,
  AMD_SIGNAL_HINT_FORMAT_MASK   = 0x1u,

  /* Bits 1-5: free under 64-byte alignment, so both formats carry them. */

  /* Bits 1-2, WHEN to interrupt. INTR_NONE on a signal with an event leaves
     a waiter sleeping on that event asleep. */
  AMD_SIGNAL_HINT_INTR_DEFAULT  = (0u << 1),  /* as today: if mailbox set */
  AMD_SIGNAL_HINT_INTR_NONE     = (1u << 1),  /* never */
  AMD_SIGNAL_HINT_INTR_ON_ZERO  = (2u << 1),  /* only non-zero -> zero */
  AMD_SIGNAL_HINT_INTR_MASK     = (3u << 1),

  /* Bit 3, HOW to update the value. Clear is the atomic add. */
  AMD_SIGNAL_HINT_VALUE_BINARY  = (1u << 3),  /* plain 64-bit store of 0 */

  /* Bits 4-5, WHETHER to capture timestamps. */
  AMD_SIGNAL_HINT_TS_INHERIT    = (0u << 4),  /* defer to the queue property */
  AMD_SIGNAL_HINT_TS_ON         = (1u << 4),  /* capture, whatever the queue */
  AMD_SIGNAL_HINT_TS_OFF        = (2u << 4),  /* suppress, whatever the queue */
  /* TS_ON, plus timestamps in timestamp_slot, one per hardware unit that ran
     the work, if available: only on a signal with PROPERTY_FORMAT_V2.
     Otherwise TS_ON. */
  AMD_SIGNAL_HINT_TS_SLOTS      = (3u << 4),
  AMD_SIGNAL_HINT_TS_MASK       = (3u << 4),

  /* Bits 6-9: v2 only; address bits on a v1 handle. */

  /* Bit 6, WHETHER to update the value at all; dominates VALUE_BINARY. */
  AMD_SIGNAL_HINT_VALUE_NONE    = (1u << 6),
  /* Bits 7-9 reserved. */
} amd_signal_hint_t;

#define AMD_SIGNAL_HINT_MASK_V1   ((uint64_t)0x03F)
#define AMD_SIGNAL_HINT_MASK_V2   ((uint64_t)0x3FF)
#define AMD_SIGNAL_HINT_WIDE_MASK ((uint64_t)0x3C0)

// Hint bits of a handle. A handle without FORMAT_V2 reads as v1, which is
// safe for an untagged v2 handle: its bits 6-9 are zero by alignment.
static inline uint64_t amd_signal_hint_mask(uint64_t handle) {
  return AMD_SIGNAL_HINT_MASK_V1 |
         (AMD_SIGNAL_HINT_WIDE_MASK & -(handle & AMD_SIGNAL_HINT_FORMAT_MASK));
}

static inline amd_signal_t* amd_signal_from_handle(uint64_t handle) {
  return (amd_signal_t*)(uintptr_t)(handle & ~amd_signal_hint_mask(handle));
}

// Signal properties, stored in amd_signal_v2_t::properties. Set at creation and
// fixed for the life of the signal. v1 keeps the word as reserved1, which is
// zero, so reading it from any signal is safe.
typedef enum {
  /* Set on every v2 signal, unlike the handle's format bit: the CP reads it
     with line 0 and honours TS_SLOTS only when it is set. */
  AMD_SIGNAL_PROPERTY_FORMAT_V2     = (1u << 0),

  /* The signal lives in device memory. Whether host stores there need an HDP
     flush and whether host atomics work there are properties of the owning
     agent: HSA_AMD_AGENT_INFO_HOST_STORES_NEED_HDP_FLUSH and
     HSA_AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED. */
  AMD_SIGNAL_PROPERTY_DEVICE_MEMORY = (1u << 1),
} amd_signal_property_t;

#if defined(__cplusplus)
static_assert(sizeof(amd_signal_t) == 64, "v1 ABI is frozen at 64 bytes.");
static_assert(sizeof(amd_signal_v2_t) == 1024, "v2 ABI is 1024 bytes.");
static_assert(alignof(amd_signal_v2_t) == AMD_SIGNAL_V2_ALIGN_BYTES,
              "v2 alignment is the hint mask width.");
static_assert(AMD_SIGNAL_HINT_MASK_V1 == AMD_SIGNAL_ALIGN_BYTES - 1 &&
                  AMD_SIGNAL_HINT_MASK_V2 == AMD_SIGNAL_V2_ALIGN_BYTES - 1,
              "Each hint mask is exactly the bits its alignment leaves zero.");

#define AMD_SIGNAL_SAME_OFFSET(field)                                          \
  static_assert(offsetof(amd_signal_v2_t, field) == offsetof(amd_signal_t, field), \
                "v2 line 0 must match v1 field for field.")
AMD_SIGNAL_SAME_OFFSET(kind);
AMD_SIGNAL_SAME_OFFSET(value);
AMD_SIGNAL_SAME_OFFSET(event_mailbox_ptr);
AMD_SIGNAL_SAME_OFFSET(event_id);
AMD_SIGNAL_SAME_OFFSET(start_ts);
AMD_SIGNAL_SAME_OFFSET(end_ts);
AMD_SIGNAL_SAME_OFFSET(queue_ptr);
AMD_SIGNAL_SAME_OFFSET(reserved3);
#undef AMD_SIGNAL_SAME_OFFSET
static_assert(offsetof(amd_signal_v2_t, properties) == offsetof(amd_signal_t, reserved1),
              "v2 properties overlays v1 reserved1.");
static_assert(offsetof(amd_signal_v2_t, timestamp_slot) == 0x80,
              "timestamp slots start line 2.");
#endif

#endif // AMD_HSA_SIGNAL_H
