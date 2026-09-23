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

// AMD Signal.
#define AMD_SIGNAL_ALIGN_BYTES 64
#define AMD_SIGNAL_ALIGN __ALIGNED__(AMD_SIGNAL_ALIGN_BYTES)
typedef struct AMD_SIGNAL_ALIGN amd_signal_s {
  amd_signal_kind64_t kind;
  union {
    volatile int64_t value;
    volatile uint64_t* hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;
  uint32_t event_id;
  uint32_t reserved1;
  uint64_t start_ts;
  uint64_t end_ts;
  union {
    amd_queue_v2_t* queue_ptr;
    uint64_t reserved2;
  };
  uint32_t reserved3[2];
} amd_signal_t;

// Signal handle hints. A signal is 64-byte aligned, so the low 6 bits of its
// handle are zero; a producer may set hints there so the CP can act on a
// completion signal without loading it. Hints live only in the handle, never
// in the signal, so they may differ per use. Zero is today's behaviour.
// Hints are purely optional: a consumer may ignore any of them, so a producer
// must be correct whether or not a hint is honoured.
typedef enum {
  /* Bit 0 reserved. */

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
  AMD_SIGNAL_HINT_TS_MASK       = (3u << 4),
} amd_signal_hint_t;

#define AMD_SIGNAL_HINT_MASK ((uint64_t)0x3F)

static inline amd_signal_t* amd_signal_from_handle(uint64_t handle) {
  return (amd_signal_t*)(uintptr_t)(handle & ~AMD_SIGNAL_HINT_MASK);
}

#if defined(__cplusplus)
static_assert(AMD_SIGNAL_HINT_MASK == AMD_SIGNAL_ALIGN_BYTES - 1,
              "The hint mask is exactly the bits the signal alignment leaves zero.");
#endif

#endif // AMD_HSA_SIGNAL_H
