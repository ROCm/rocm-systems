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

#include "rdc_lib/impl/SmiUtils.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <set>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {

rdc_status_t Smi2RdcError(amdsmi_status_t rsmi) {
  switch (rsmi) {
    case AMDSMI_STATUS_SUCCESS:
      return RDC_ST_OK;

    case AMDSMI_STATUS_INVAL:
      return RDC_ST_BAD_PARAMETER;

    case AMDSMI_STATUS_NOT_SUPPORTED:
      return RDC_ST_NOT_SUPPORTED;

    case AMDSMI_STATUS_NOT_FOUND:
      return RDC_ST_NOT_FOUND;

    case AMDSMI_STATUS_OUT_OF_RESOURCES:
      return RDC_ST_INSUFF_RESOURCES;

    case AMDSMI_STATUS_FILE_ERROR:
      return RDC_ST_FILE_ERROR;

    case AMDSMI_STATUS_NO_DATA:
      return RDC_ST_NO_DATA;

    case AMDSMI_STATUS_NO_PERM:
      return RDC_ST_PERM_ERROR;

    case AMDSMI_STATUS_CORRUPTED_EEPROM:
      return RDC_ST_CORRUPTED_EEPROM;

    case AMDSMI_STATUS_BUSY:
    case AMDSMI_STATUS_UNKNOWN_ERROR:
    case AMDSMI_STATUS_INTERNAL_EXCEPTION:
    case AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS:
    case AMDSMI_STATUS_INIT_ERROR:
    case AMDSMI_STATUS_NOT_YET_IMPLEMENTED:
    case AMDSMI_STATUS_INSUFFICIENT_SIZE:
    case AMDSMI_STATUS_INTERRUPT:
    case AMDSMI_STATUS_UNEXPECTED_SIZE:
    case AMDSMI_STATUS_UNEXPECTED_DATA:
    case AMDSMI_STATUS_REFCOUNT_OVERFLOW:
    default:
      return RDC_ST_UNKNOWN_ERROR;
  }
}

namespace {

// The flat GPU table and its state, guarded by s_table_mutex.
std::mutex s_table_mutex;
std::shared_ptr<const GpuHandleTable> s_table;  // null until first use
std::vector<bool> s_retry_slot;  // per slot: a placeholder the next listing may still fill
bool s_table_listed = false;     // the table came from a listing that read every list
bool s_table_complete = false;   // listed, and no slot left to retry
std::chrono::steady_clock::time_point s_table_listed_at;
std::chrono::milliseconds s_retry_period(5000);

// RDC's AMD SMI references (SmiSession objects), guarded by s_session_mutex.
std::mutex s_session_mutex;
uint32_t s_session_count = 0;

enum class ProcessorKind { kGpu, kOther, kUnknown };

struct ListedProcessor {
  amdsmi_processor_handle handle;
  uint32_t socket_index;
  uint32_t proc_index;
  ProcessorKind kind;  // kUnknown: AMD SMI could not report the type
};

// Lists every processor of every socket, in (socket, processor) order, with its kind. Returns
// false when a socket or processor list cannot be read, since the slot count is then unknown.
bool list_processors(std::vector<ListedProcessor>* out) {
  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  std::vector<ListedProcessor> listed;
  for (uint32_t s = 0; s < socket_count && s < sockets.size(); s++) {
    uint32_t proc_count = 0;
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
      return false;
    }
    std::vector<amdsmi_processor_handle> procs(proc_count);
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, procs.data()) !=
        AMDSMI_STATUS_SUCCESS) {
      return false;
    }
    for (uint32_t p = 0; p < proc_count && p < procs.size(); p++) {
      processor_type_t proc_type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      ProcessorKind kind = ProcessorKind::kUnknown;
      if (amdsmi_get_processor_type(procs[p], &proc_type) == AMDSMI_STATUS_SUCCESS) {
        kind = proc_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU ? ProcessorKind::kGpu
                                                          : ProcessorKind::kOther;
      }
      listed.push_back({procs[p], s, p, kind});
    }
  }
  *out = std::move(listed);
  return true;
}

// The GPU slots of a listing: a slot for each GPU, and a placeholder for each processor of unknown
// type that can only be a GPU: one on a socket with a GPU, or on any socket when nothing listed is
// of another kind (AMD SMI was initialized for GPUs only). An unknown processor on a socket with a
// CPU, or on an unidentified socket next to CPUs, gets no slot. At most RDC_MAX_NUM_DEVICES slots.
GpuHandleTable gpu_slots(const std::vector<ListedProcessor>& listed) {
  std::set<uint32_t> gpu_sockets;
  std::set<uint32_t> other_sockets;
  for (const auto& proc : listed) {
    if (proc.kind == ProcessorKind::kGpu) {
      gpu_sockets.insert(proc.socket_index);
    } else if (proc.kind == ProcessorKind::kOther) {
      other_sockets.insert(proc.socket_index);
    }
  }
  GpuHandleTable table;
  for (const auto& proc : listed) {
    if (proc.kind == ProcessorKind::kOther) {
      continue;
    }
    if (proc.kind == ProcessorKind::kUnknown && gpu_sockets.count(proc.socket_index) == 0 &&
        !other_sockets.empty()) {
      RDC_LOG(RDC_ERROR, "AMD SMI cannot identify processor " << proc.proc_index << " of socket "
                                                              << proc.socket_index
                                                              << "; RDC leaves it out");
      continue;
    }
    if (table.size() == RDC_MAX_NUM_DEVICES) {
      RDC_LOG(RDC_ERROR, "AMD SMI lists more than " << RDC_MAX_NUM_DEVICES
                                                    << " GPUs; RDC uses the first of them");
      break;
    }
    table.push_back(
        {proc.handle, proc.socket_index, proc.proc_index, proc.kind == ProcessorKind::kGpu});
  }
  return table;
}

// Caller holds s_table_mutex. Publishes the GPU slots of a complete listing as a new table.
void publish_listing_locked(const std::vector<ListedProcessor>& listed) {
  GpuHandleTable table = gpu_slots(listed);
  if (table.empty()) {
    RDC_LOG(RDC_ERROR, "build_flat_gpu_table: no AMD GPUs found; GPU table is empty");
  }
  s_retry_slot.assign(table.size(), false);
  for (uint32_t i = 0; i < table.size(); i++) {
    if (!table[i].available) {
      RDC_LOG(RDC_ERROR, "AMD SMI cannot identify GPU " << i << "; it is unavailable until it can");
      s_retry_slot[i] = true;
    }
  }
  s_table_listed = true;
  s_table_complete =
      std::none_of(s_retry_slot.begin(), s_retry_slot.end(), [](bool r) { return r; });
  s_table = std::make_shared<const GpuHandleTable>(std::move(table));
}

// Caller holds s_table_mutex. Builds the table on first use.
void build_table_locked() {
  std::vector<ListedProcessor> listed;
  s_table_listed_at = std::chrono::steady_clock::now();
  if (list_processors(&listed)) {
    publish_listing_locked(listed);
    return;
  }
  RDC_LOG(RDC_ERROR, "AMD SMI could not list its processors; RDC sees no GPU until it can");
  s_retry_slot.clear();
  s_table_listed = false;
  s_table_complete = false;
  s_table = std::make_shared<const GpuHandleTable>();
}

// Caller holds s_table_mutex. Lists the processors again. A placeholder whose processor is now a
// GPU takes its handle; one whose processor is of another kind stays unavailable and is no longer
// retried; a GPU that is no longer listed becomes unavailable. Slots keep their places and the
// table keeps its size.
void retry_table_locked() {
  std::vector<ListedProcessor> listed;
  s_table_listed_at = std::chrono::steady_clock::now();
  if (!list_processors(&listed)) {
    return;  // keep the table; retry after the next period
  }
  if (!s_table_listed) {
    publish_listing_locked(listed);  // the first complete listing
    return;
  }
  auto table = std::make_shared<GpuHandleTable>(*s_table);
  for (uint32_t i = 0; i < table->size(); i++) {
    GpuHandleEntry& slot = (*table)[i];
    auto found = std::find_if(listed.begin(), listed.end(), [&slot](const ListedProcessor& proc) {
      return proc.socket_index == slot.socket_index && proc.proc_index == slot.proc_index;
    });
    if (found == listed.end()) {
      if (slot.available) {
        RDC_LOG(RDC_ERROR, "AMD SMI no longer lists GPU " << i << "; it is unavailable");
        slot.available = false;
      }
      s_retry_slot[i] = true;
    } else if (!slot.available && found->kind == ProcessorKind::kGpu) {
      bool taken = std::any_of(table->begin(), table->end(), [&found](const GpuHandleEntry& e) {
        return e.available && e.handle == found->handle;
      });
      if (taken) {
        continue;  // the listing moved: never give one GPU's handle to two indexes
      }
      slot.handle = found->handle;
      slot.available = true;
      s_retry_slot[i] = false;
      RDC_LOG(RDC_INFO, "AMD SMI now identifies GPU " << i);
    } else if (!slot.available && found->kind == ProcessorKind::kOther) {
      if (s_retry_slot[i]) {
        RDC_LOG(RDC_ERROR, "GPU " << i << " is a processor of another kind; it stays unavailable");
      }
      s_retry_slot[i] = false;
    }
  }
  s_table_complete =
      std::none_of(s_retry_slot.begin(), s_retry_slot.end(), [](bool r) { return r; });
  s_table = std::move(table);
}

// Caller holds s_table_mutex. The table, built on first use and retried when it is due. A retry
// lists AMD SMI's processors under the mutex; those calls only read AMD SMI's own lists.
const GpuHandleTable& current_table_locked() {
  if (!s_table) {
    build_table_locked();
  } else if (!s_table_complete &&
             std::chrono::steady_clock::now() - s_table_listed_at >= s_retry_period) {
    retry_table_locked();
  }
  return *s_table;
}

}  // namespace

std::shared_ptr<const GpuHandleTable> get_flat_gpu_table() {
  std::lock_guard<std::mutex> lock(s_table_mutex);
  current_table_locked();
  return s_table;
}

void reset_flat_gpu_table() {
  std::lock_guard<std::mutex> lock(s_table_mutex);
  s_table.reset();
  s_retry_slot.clear();
  s_table_listed = false;
  s_table_complete = false;
}

void set_flat_gpu_table_retry_period(std::chrono::milliseconds period) {
  std::lock_guard<std::mutex> lock(s_table_mutex);
  s_retry_period = period;
}

SmiSession::SmiSession() {
  std::lock_guard<std::mutex> lock(s_session_mutex);
  if (s_session_count == 0) {
    uint64_t init_flag = AMDSMI_INIT_AMD_GPUS;
#ifdef ENABLE_ESMI_LIB
    init_flag |= AMDSMI_INIT_AMD_CPUS;
#endif
    amdsmi_status_t ret = amdsmi_init(init_flag);
#ifdef ENABLE_ESMI_LIB
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_INFO, "Failed to initialize amdsmi with CPUs enabled.. Disabling CPUs.");
      init_flag &= ~AMDSMI_INIT_AMD_CPUS;
      ret = amdsmi_init(init_flag);
    }
#endif
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "amdsmi_init failed: " << ret);
      throw RdcException(RDC_ST_FAIL_LOAD_MODULE, "AMD SMI failed to initialize");
    }
    // Handles from an earlier session are no longer valid: list the GPUs again.
    try {
      reset_flat_gpu_table();
      get_flat_gpu_table();
    } catch (...) {
      amdsmi_shut_down();  // no session holds the reference amdsmi_init() took
      throw;
    }
  }
  s_session_count++;
}

SmiSession::~SmiSession() {
  std::lock_guard<std::mutex> lock(s_session_mutex);
  if (--s_session_count == 0) {
    reset_flat_gpu_table();
    amdsmi_shut_down();
  }
}

// Dual index semantics for gpu_id:
//   Physical/instance-0: device_index is a flat index into the flat GPU table.
//   Partition-instance:  device_index is a socket index, instance_index the per-socket proc.
// These coincide in SPX but diverge in CPX (multiple procs per socket), so don't pass a
// flat index where a socket index is expected.
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  if (processor_handle == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  rdc_entity_info_t info = rdc_get_info_from_entity_index(gpu_id);

  if (info.instance_index == 0 && info.entity_role == RDC_DEVICE_ROLE_PHYSICAL) {
    std::lock_guard<std::mutex> lock(s_table_mutex);
    const GpuHandleTable& table = current_table_locked();
    uint32_t flat_idx = info.device_index;
    if (flat_idx >= table.size()) {
      return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
    }
    if (!table[flat_idx].available) {
      return AMDSMI_STATUS_NOT_FOUND;  // logged when the slot became a placeholder
    }
    *processor_handle = table[flat_idx].handle;
    return AMDSMI_STATUS_SUCCESS;
  }

  // Entity-encoded index (partition instances): use socket/proc mapping
  uint32_t socket_count = 0;
  amdsmi_status_t ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t socket_index = info.device_index;
  uint32_t instance_index = info.instance_index;

  if (socket_index >= socket_count) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  uint32_t proc_count = 0;
  ret = amdsmi_get_processor_handles(sockets[socket_index], &proc_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (instance_index >= proc_count) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  std::vector<amdsmi_processor_handle> procs(proc_count);
  ret = amdsmi_get_processor_handles(sockets[socket_index], &proc_count, procs.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  *processor_handle = procs[instance_index];
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t get_gpu_id_from_processor_handle(amdsmi_processor_handle processor_handle,
                                                 uint32_t* gpu_index) {
  if (!gpu_index) {
    return AMDSMI_STATUS_INVAL;
  }

  // The GPU's slot in the table, so that the index matches the one lookups use.
  std::shared_ptr<const GpuHandleTable> table = get_flat_gpu_table();
  for (uint32_t i = 0; i < table->size(); i++) {
    if ((*table)[i].available && (*table)[i].handle == processor_handle) {
      *gpu_index = i;
      return AMDSMI_STATUS_SUCCESS;
    }
  }

  return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
}

amdsmi_status_t get_processor_count(uint32_t& all_processor_count) {
  uint32_t total_processor_count = 0;
  uint32_t socket_count = 0;
  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  for (auto& socket : sockets) {
    uint32_t processor_count = 0;
    ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }
    total_processor_count += processor_count;
  }
  all_processor_count = total_processor_count;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t get_socket_handles(std::vector<amdsmi_socket_handle>& sockets) {
  uint32_t socket_count = 0;
  amdsmi_status_t ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  sockets.resize(socket_count);

  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());

  return ret;
}

amdsmi_status_t get_processor_handles(amdsmi_socket_handle socket,
                                      std::vector<amdsmi_processor_handle>& processors) {
  uint32_t processor_count = 0;
  amdsmi_status_t ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  processors.resize(processor_count);

  ret = amdsmi_get_processor_handles(socket, &processor_count, processors.data());

  return ret;
}

namespace {

// What amd-smi reports for a KFD field it cannot read.
constexpr uint32_t kNoKfdValue = std::numeric_limits<uint32_t>::max();

// One amdsmi_get_gpu_kfd_info call, made afresh at every use: a node id that was kept
// would go stale across a partition change.
amdsmi_status_t get_kfd_info(amdsmi_processor_handle proc, amdsmi_kfd_info_t* info) {
  return amdsmi_get_gpu_kfd_info(proc, info);
}

}  // namespace

amdsmi_status_t get_kfd_node_id(uint32_t entity_id, uint32_t* node_id) {
  if (node_id == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  amdsmi_processor_handle proc = nullptr;
  amdsmi_status_t ret = get_processor_handle_from_id(entity_id, &proc);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    // Every cause reads the same to the caller: RDC cannot identify that GPU.
    RDC_LOG(RDC_DEBUG, "GPU " << entity_id << ": get_processor_handle_from_id fails with " << ret);
    return AMDSMI_STATUS_NOT_FOUND;
  }

  amdsmi_kfd_info_t kfd_info = {};
  ret = get_kfd_info(proc, &kfd_info);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_DEBUG, "GPU " << entity_id << ": amdsmi_get_gpu_kfd_info fails with " << ret);
    // A GPU AMD SMI lists but has no KFD node for is not one it does not know.
    return ret == AMDSMI_STATUS_NOT_FOUND ? AMDSMI_STATUS_NOT_SUPPORTED : ret;
  }
  if (kfd_info.node_id == kNoKfdValue) {
    RDC_LOG(RDC_DEBUG, "GPU " << entity_id << ": amdsmi_get_gpu_kfd_info reports no KFD node");
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  *node_id = kfd_info.node_id;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t get_kfd_partition_id(amdsmi_processor_handle proc, uint32_t* partition_id) {
  if (partition_id == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  amdsmi_kfd_info_t kfd_info = {};
  amdsmi_status_t ret = get_kfd_info(proc, &kfd_info);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  *partition_id = kfd_info.current_partition_id;
  return ret;
}

amdsmi_status_t get_num_partition(uint32_t index, uint16_t* num_partition) {
  if (num_partition == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  // Decode the entity index so both bare flat indices and partition-instance entity
  // indices resolve to the right socket (they diverge in CPX). See the dual-index note
  // on get_processor_handle_from_id().
  std::shared_ptr<const GpuHandleTable> table = get_flat_gpu_table();
  rdc_entity_info_t info = rdc_get_info_from_entity_index(index);

  uint32_t target_socket = 0;
  if (info.entity_role == RDC_DEVICE_ROLE_PHYSICAL && info.instance_index == 0) {
    // Physical/instance-0: device_index is a flat index into the GPU table.
    uint32_t flat_idx = info.device_index;
    if (flat_idx >= table->size()) {
      return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
    }
    target_socket = (*table)[flat_idx].socket_index;
  } else {
    // Partition-instance: device_index is the socket index.
    target_socket = info.device_index;
  }

  // Find the first processor in the same socket to query the physical GPU profile.
  amdsmi_processor_handle phys_handle = nullptr;
  for (const auto& entry : *table) {
    if (entry.available && entry.socket_index == target_socket) {
      phys_handle = entry.handle;
      break;
    }
  }

  if (phys_handle == nullptr) {
    return AMDSMI_STATUS_NOT_FOUND;
  }

  amdsmi_accelerator_partition_profile_t profile;
  memset(&profile, 0, sizeof(profile));
  uint32_t partition_id{};

  amdsmi_status_t ret =
      amdsmi_get_gpu_accelerator_partition_profile(phys_handle, &profile, &partition_id);
  // Non-partitionable GPUs report this as unsupported; treat them as a single
  // partition. All consumers handle a count of 1 correctly.
  if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
    *num_partition = 1;
    return AMDSMI_STATUS_SUCCESS;
  }
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (partition_id != 0) {
    return AMDSMI_STATUS_UNEXPECTED_DATA;
  }

  // Some driver/amdsmi versions return the current profile with num_partitions set to an
  // invalid sentinel (e.g. UINT32_MAX) even though the call succeeds. Fall back to the flat
  // table, which enumerates exactly one entry per partition in a socket, so the count stays
  // consistent with the rest of RDC's GPU view.
  if (profile.num_partitions == 0 || profile.num_partitions > RDC_MAX_NUM_PARTITIONS) {
    uint16_t socket_entries = 0;
    for (const auto& entry : *table) {
      if (entry.socket_index == target_socket) {  // placeholders keep partition indexes too
        socket_entries++;
      }
    }
    *num_partition = socket_entries;
    return AMDSMI_STATUS_SUCCESS;
  }

  *num_partition = profile.num_partitions;

  return ret;
}

uint64_t count_pending_bad_pages(const amdsmi_retired_page_record_t* records, uint32_t count) {
  if (records == nullptr) {
    return 0;
  }
  uint64_t pending = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (AMDSMI_MEM_PAGE_STATUS_PENDING == records[i].status) {
      ++pending;
    }
  }
  return pending;
}

double derive_mem_activity_percent(double umc_activity_pct, bool have_prev,
                                   uint64_t prev_mem_activity_acc, uint64_t prev_firmware_ts,
                                   uint64_t cur_mem_activity_acc, uint64_t cur_firmware_ts) {
  // In gpu_metrics the max value of the type means "not supported"; never treat the
  // sentinel as real accumulator/timestamp data (a huge unsigned delta would otherwise
  // clamp to 100%).
  constexpr uint64_t kNotSupported = std::numeric_limits<uint64_t>::max();
  // Bound the averaging window: a stale cached sample (for example a watcher that
  // paused for minutes or hours) would otherwise spread a recent burst across the
  // whole gap and dilute it to nearly zero. Beyond this the accumulator is ignored
  // and the caller re-primes the cache so the next sample uses a fresh, bounded
  // window. 60s comfortably exceeds common monitoring intervals (1-10s) while
  // rejecting multi-minute gaps.
  constexpr double kMaxSampleWindowMs = 60000.0;
  double activity_pct = umc_activity_pct;
  if (have_prev && cur_mem_activity_acc != kNotSupported && cur_firmware_ts != kNotSupported &&
      prev_mem_activity_acc != kNotSupported && prev_firmware_ts != kNotSupported &&
      cur_firmware_ts > prev_firmware_ts && cur_mem_activity_acc >= prev_mem_activity_acc) {
    // Firmware timestamps are 10ns units; the accumulator advances by the
    // activity percentage every firmware millisecond (1ms == 100000 units).
    const double elapsed_ms = static_cast<double>(cur_firmware_ts - prev_firmware_ts) / 100000.0;
    if (elapsed_ms > 0.0 && elapsed_ms <= kMaxSampleWindowMs) {
      double acc_pct =
          static_cast<double>(cur_mem_activity_acc - prev_mem_activity_acc) / elapsed_ms;
      if (acc_pct < 0.0) acc_pct = 0.0;
      if (acc_pct > 100.0) acc_pct = 100.0;
      // Never report below the instantaneous UMC-activity reading.
      if (acc_pct > activity_pct) activity_pct = acc_pct;
    }
  }
  return activity_pct;
}

}  // namespace rdc
}  // namespace amd
