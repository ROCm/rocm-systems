/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include "hsa.h"

namespace amd::pal {

//! AmdAqlWrap slot state
enum AqlWrapState {
  AQL_WRAP_FREE = 0,
  AQL_WRAP_RESERVED,
  AQL_WRAP_READY,
  AQL_WRAP_MARKER,
  AQL_WRAP_BUSY,
  AQL_WRAP_DONE
};

struct AmdVQueueHeader {
  uint32_t aql_slot_num;     //!< [LRO/SRO] The total number of the AQL slots (multiple of 64).
  uint32_t event_slot_num;   //!< [LRO] The number of kernel events in the events buffer
  uint64_t event_slot_mask;  //!< [LRO] A pointer to the allocation bitmask array for the events
  uint64_t event_slots;      //!< [LRO] Pointer to a buffer for the events.
                             // Array of event_slot_num entries of AmdEvent
  uint64_t aql_slot_mask;    //!< [LRO/SRO]A pointer to the allocation bitmask for aql_warp slots
  uint32_t command_counter;  //!< [LRW] The global counter for the submitted commands into the queue
  uint32_t wait_size;        //!< [LRO] The wait list size (in clk_event_t)
  uint32_t arg_size;         //!< [LRO] The size of argument buffer (in bytes)
  uint32_t mask_groups;      //!< Processed mask groups by one thread
  uint64_t
      kernel_table;  //!< [LRO] Pointer to an array with all kernel objects (ulong for each entry)
  uint32_t reserved[2];  //!< For the future usage
};

struct AmdAqlWrap {
  uint32_t state;  //!< [LRW/SRW] The current state of the AQL wrapper:  FREE, RESERVED, READY,
                   // MARKER, BUSY and DONE. The block could be returned back to a free state.
  uint32_t enqueue_flags;  //!< [LWO/SRO] Contains the flags for the kernel execution start
  uint32_t command_id;     //!< [LWO/SRO] The unique command ID
  uint32_t child_counter;  //!< [LRW/SRW] Counter that determine the launches of child kernels.
                           // It's incremented on the
  // start and decremented on the finish. The parent kernel can be considered as
  // done when the value is 0 and the state is DONE
  uint64_t completion;   //!< [LWO/SRO] CL event for the current execution (clk_event_t)
  uint64_t parent_wrap;  //!< [LWO/SRO] Pointer to the parent AQL wrapper (AmdAqlWrap*)
  uint64_t wait_list;  //!< [LRO/SRO] Pointer to an array of clk_event_t objects (64 bytes default)
  uint32_t wait_num;   //!< [LWO/SRO] The number of cl_event_wait objects
  uint32_t reserved[5];              //!< For the future usage
  hsa_kernel_dispatch_packet_t aql;  //!< [LWO/SRO] AQL packet - 64 bytes AQL packet
};

struct AmdEvent {
  uint32_t state;        //!< [LRO/SRW] Event state: START, END, COMPLETE
  uint32_t counter;      //!< [LRW] Event retain/release counter. 0 means the event is free
  uint64_t timer[3];     //!< [LRO/SWO] Timer values for profiling for each state
  uint64_t captureInfo;  //!< [LRW/SRO] Profiling capture info for CLK_PROFILING_COMMAND_EXEC_TIME
};

struct SchedulerParam {
  uint32_t signal;          //!< Signal to stop the child queue(address must be 16 bytes aligned)
  uint32_t eng_clk;         //!< Engine clock in Mhz
  uint64_t hw_queue;        //!< Address to HW queue
  uint64_t hsa_queue;       //!< Address to HSA dummy queue
  uint32_t useATC;          //!< GPU access to shader program by ATC.
  uint32_t scratchSize;     //!< Scratch buffer size
  uint64_t scratch;         //!< GPU address to the scratch buffer
  uint32_t numMaxWaves;     //!< The max number of possible waves
  uint32_t releaseHostCP;   //!< Releases CP on the host queue
  uint64_t parentAQL;       //!< Host parent AmdAqlWrap packet
  uint32_t dedicatedQueue;  //!< Scheduler uses a dedicated queue
  uint32_t scratchOffset;   //!< Scratch buffer offset
  uint32_t reserved[2];     //!< Reserved
};

}  // namespace amd::pal
