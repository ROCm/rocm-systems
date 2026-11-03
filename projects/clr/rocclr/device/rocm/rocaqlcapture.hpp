/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "top.hpp"
#include "rocrctx.hpp"

#include <cstdint>

namespace amd::roc {

//! Everything the runtime hands to HSA for a single kernel dispatch, gathered at
//! the point the packet is published to the AQL ring. A record is written as one
//! JSON object per line (JSON Lines) so a truncated file - the normal outcome when
//! the capture target hangs - still parses up to the last complete dispatch.
//!
//! Enabled by ROC_AQL_CAPTURE_OUTPUT=<path>; the PID is appended to the path so
//! multi-process runs do not interleave.
struct AqlDispatchCaptureInput {
  const hsa_queue_t* queue;           //!< HW queue the packet was written to
  uint32_t device_index;              //!< Device ordinal
  int priority;                       //!< amd::CommandQueue::Priority
  uint64_t slot_index;                //!< Write index (wptr) of this packet
  uint16_t header;                    //!< Armed packet header
  uint16_t rest;                      //!< Dwords 16..31 of the header dword (setup/amd_format)
  bool ext_dispatch;                  //!< |packet| is hsa_amd_ext_kernel_dispatch_packet_t
  const void* packet;                 //!< Host copy of the AQL packet
  //! Metadata-prefetch packet for this slot, or nullptr when the queue has no
  //! metadata ring (DEBUG_CLR_ENABLE_KDQ=0 or pre-gfx12.5).
  const hsa_amd_metadata_kernel_dispatch_packet_t* metadata;
  const char* kernel_name;            //!< Demangled name, or nullptr
  uint32_t kernarg_segment_size;      //!< Kernel's kernarg segment size in bytes (0 if unknown)
};

class AqlCapture : public amd::AllStatic {
 public:
  //! True when ROC_AQL_CAPTURE_OUTPUT names a file and the record budget is not spent.
  //! Checked on the dispatch hot path, so the flag parse happens once.
  static bool Enabled();

  //! Append one JSON Lines record describing |in|. No-op when !Enabled().
  static void Record(const AqlDispatchCaptureInput& in);
};

}  // namespace amd::roc
