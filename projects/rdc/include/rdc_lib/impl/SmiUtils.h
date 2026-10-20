/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INCLUDE_RDC_LIB_IMPL_RSMIUTILS_H_
#define INCLUDE_RDC_LIB_IMPL_RSMIUTILS_H_

#include <chrono>  // NOLINT(build/c++11)
#include <memory>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

rdc_status_t Smi2RdcError(amdsmi_status_t rsmi);

// Count how many of the retired/bad-page records are pending retirement.
// Returns 0 when records is null or count is 0.
uint64_t count_pending_bad_pages(const amdsmi_retired_page_record_t* records, uint32_t count);

// Derives the memory-activity percentage (0..100) used to estimate current
// memory bandwidth. Prefers the mem_activity_acc accumulator delta over the
// firmware-timestamp window (which also reflects DMA/copy traffic) and never
// drops below the instantaneous umc_activity_pct. Falls back to umc_activity_pct
// when there is no previous sample or the firmware clock and accumulator did not
// advance monotonically. Firmware timestamps are in 10ns units (1ms == 100000).
double derive_mem_activity_percent(double umc_activity_pct, bool have_prev,
                                   uint64_t prev_mem_activity_acc, uint64_t prev_firmware_ts,
                                   uint64_t cur_mem_activity_acc, uint64_t cur_firmware_ts);

// Physical/instance-0: gpu_id is a flat GPU index. Partition-instance: device_index is
// a socket index, instance_index the per-socket proc. These diverge in CPX. See SmiUtils.cc.
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle);
// The flat index of an available GPU's handle: its slot in the flat GPU table.
// AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS for any other handle.
amdsmi_status_t get_gpu_id_from_processor_handle(amdsmi_processor_handle processor_handle,
                                                 uint32_t* gpu_index);
// The slot an entity id holds in the flat GPU table, which is also its rocm_smi device index and
// so the number AMD SMI reports a process's GPUs by. A physical GPU (device type GPU, role
// physical, instance 0) is its own slot; any other GPU entity, such as a partition, is the slot
// of the entry with its socket index and processor index (the dual index above). A placeholder
// slot counts, so a GPU that AMD SMI cannot identify right now keeps its slot.
// AMDSMI_STATUS_NOT_FOUND for every other entity: another device type, a physical index beyond
// the table, or a socket and processor with no entry. AMDSMI_STATUS_INVAL for a null slot. It
// takes one snapshot of the table, which lists AMD SMI's processors only when the table is
// built or retried, and makes no other AMD SMI call.
amdsmi_status_t get_gpu_slot_from_entity(uint32_t entity_id, uint32_t* slot);
amdsmi_status_t get_processor_count(uint32_t& all_processor_count);
amdsmi_status_t get_socket_handles(std::vector<amdsmi_socket_handle>& sockets);
amdsmi_status_t get_processor_handles(amdsmi_socket_handle socket,
                                      std::vector<amdsmi_processor_handle>& processors);
// The KFD node of a GPU, which is the key ROCr lists its agents by, for an entity id
// (flat or partition). AMDSMI_STATUS_NOT_FOUND when AMD SMI does not know the entity,
// AMDSMI_STATUS_NOT_SUPPORTED when it reports no KFD node for it, AMDSMI_STATUS_INVAL
// for a null node_id, and amd-smi's own status for any other failure of the call.
// Nothing is cached: a node id goes stale across a partition change.
amdsmi_status_t get_kfd_node_id(uint32_t entity_id, uint32_t* node_id);
amdsmi_status_t get_kfd_partition_id(amdsmi_processor_handle proc, uint32_t* partition_id);
amdsmi_status_t get_num_partition(uint32_t index, uint16_t* num_partition);

struct GpuHandleEntry {
  amdsmi_processor_handle handle;
  uint32_t socket_index;
  uint32_t proc_index;
  // false: a placeholder for a processor whose type AMD SMI could not report (and that can only be
  // a GPU: it is on a GPU's socket, or AMD SMI lists no other kind of processor), or a GPU that
  // AMD SMI no longer lists. It keeps the indexes of the GPUs after it; looking it up returns
  // AMDSMI_STATUS_NOT_FOUND.
  bool available;
};

using GpuHandleTable = std::vector<GpuHandleEntry>;

// The flat GPU table (index = RDC GPU index, at most RDC_MAX_NUM_DEVICES slots), built on first
// use. Thread-safe; a snapshot can be read while it is held, and its handles are valid while an
// SmiSession lives. Slots never move and the table never shrinks until it is reset. While a
// placeholder may still be filled, or the first listing failed, the first use at least 5 s after
// the last listing lists the processors again: a placeholder whose processor is now a GPU takes
// its handle, one whose processor is of another kind is no longer retried, a GPU that is no
// longer listed becomes unavailable, and other slots stay as they are.
std::shared_ptr<const GpuHandleTable> get_flat_gpu_table();

// Drops the table; the next use builds a new one. SmiSession resets it whenever AMD SMI is
// initialized or shut down, since handles change then. Prefer a daemon restart after a
// partition-mode change.
void reset_flat_gpu_table();

// For tests: the least time between two listings of a table that is retried (default 5 s).
void set_flat_gpu_table_retry_period(std::chrono::milliseconds period);

// RDC's own reference on AMD SMI. The first live session calls amdsmi_init() and lists the GPUs;
// the last one drops the GPU table and calls amdsmi_shut_down(). AMD SMI counts its
// initializations, so a process that uses AMD SMI itself keeps its handles while RDC runs.
// The constructor throws RdcException(RDC_ST_FAIL_LOAD_MODULE) when amdsmi_init() fails, and
// passes on an exception from listing the GPUs after releasing AMD SMI.
class SmiSession {
 public:
  SmiSession();
  ~SmiSession();
  SmiSession(const SmiSession&) = delete;
  SmiSession& operator=(const SmiSession&) = delete;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RSMIUTILS_H_
