/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_aie_umq_queue.h"

#include <atomic>
#include <cassert>
#include <cstring>

#include "inc/hsa_ext_amd_aie.h"
#include "core/inc/amd_xdna_driver.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {

static_assert(sizeof(hsa_amd_aie_kernel_dispatch_packet_t) == sizeof(core::AqlPacket),
              "hsa_amd_aie_kernel_dispatch_packet_t must be the same size as core::AqlPacket");

namespace {

/// @brief Mirrors the AIE4 kernel driver's `struct host_queue_header`
/// (drivers/accel/amdxdna/aie4_host_queue.h, confirmed against the installed xrt-amdxdna-2.26.0
/// source), which must open the BO passed to CREATE_HWCTX as `umq_bo`; the packet ring follows
/// immediately after, at @ref data_address. This layout is not a stable, versioned UAPI struct
/// (unlike amdxdna_drm_create_hwctx) -- only observed in one driver build, so a future version
/// could change it silently.
struct HostQueueHeader {
  uint64_t read_index;
  uint16_t version_major;
  uint16_t version_minor;
  uint32_t capacity;
  uint64_t padding0[6];
  uint64_t write_index;
  uint64_t padding1[6];
  uint64_t data_address;
};
static_assert(sizeof(HostQueueHeader) == 128,
              "HostQueueHeader must match the kernel driver's struct host_queue_header exactly");

/// @brief Matches aie4_host_queue.h's CTX_MAX_CMDS: the fixed slot count CERT itself validates
/// write_index/read_index against (the kernel driver's get_read_index()/valid_queue_index()),
/// regardless of whatever @ref HostQueueHeader::capacity we declare. The ring must be sized for
/// exactly this many packets.
constexpr uint32_t kCtxMaxCmds = 32;

constexpr size_t kHostQueueHeaderSize = sizeof(HostQueueHeader);

}  // namespace

AieUmqQueue::AieUmqQueue(core::SharedQueue* shared_queue, AieAgent* agent, size_t req_size_pkts,
                         uint32_t node_id, core::HsaEventCallback callback, void* err_data,
                         uint64_t flags)
    : Queue(shared_queue, flags, agent),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      // The ring the device reads is always kCtxMaxCmds packets, whatever req_size_pkts asks for
      // (CERT's own read/write-index validity check hardcodes that count; see HostQueueHeader),
      // plus the header the ring follows.
      queue_size_bytes_(static_cast<uint32_t>(
          kHostQueueHeaderSize + kCtxMaxCmds * sizeof(hsa_amd_aie_kernel_dispatch_packet_t))),
      suspended_(false),
      errors_callback_(callback),
      errors_data_(err_data) {
  if (agent->device_type() != core::Agent::DeviceType::kAmdAieDevice) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_AGENT,
                        "Attempting to create an AIE queue on a non-AIE agent.");
  }
  if (req_size_pkts != kCtxMaxCmds) {
    log_warning_n(10,
                  "AIE: UMQ queue size is fixed at %u packets (CERT's own read/write-index check "
                  "hardcodes it); the requested %zu is ignored.\n",
                  kCtxMaxCmds, req_size_pkts);
  }

  ring_buf_ =
      agent->system_allocator()(queue_size_bytes_, 4096, core::MemoryRegion::AllocateNoFlags);
  if (!ring_buf_) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
                        "Could not allocate a ring buffer for an AIE queue.");
  }
  MAKE_NAMED_SCOPE_GUARD(ring_buf_guard, [&] { agent->system_deallocator()(ring_buf_); });
  std::memset(ring_buf_, 0, queue_size_bytes_);

  // CERT reads its read/write indices and the packet ring straight out of this BO (passed below
  // as the hardware context's umq_bo); the header must open it, per HostQueueHeader. The packet
  // ring -- what the application actually writes into -- starts right after.
  auto* header = static_cast<HostQueueHeader*>(ring_buf_);
  header->version_major = 1;
  header->version_minor = 0;
  header->capacity = kCtxMaxCmds;
  // Under PASID/SVA the device walks the submitting process's own page tables (see
  // XdnaDriver::CreateUserModeQueue's comment), so the address CERT dereferences is this
  // process's own -- no separate device-address translation, unlike a blob placed into the
  // CODE_AGENT segment (e.g. AieKernelDescriptor::static_ctrl_code_dev_addr).
  void* packet_ring = static_cast<uint8_t*>(ring_buf_) + kHostQueueHeaderSize;
  header->data_address = reinterpret_cast<uint64_t>(packet_ring);

  // Populate hsa_queue_t fields.
  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_SINGLE;
  amd_queue_.hsa_queue.features = 0;  // UMQ queues do not support any features.
  amd_queue_.hsa_queue.size = kCtxMaxCmds;
  amd_queue_.hsa_queue.base_address = packet_ring;
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);
  amd_queue_.hsa_queue.id = GetQueueId();

  // Populate AMD queue fields.
  amd_queue_.write_dispatch_id = 0;
  amd_queue_.read_dispatch_id = 0;

  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.queue_ptr = &amd_queue_;

  auto& driver = static_cast<XdnaDriver&>(agent->driver());
  // The hardware context spans every column.
  hsa_status_t err =
      driver.CreateUserModeQueue(ring_buf_, *agent, agent->num_cols(), agent->num_core_rows(),
                                 &umq_metadata_, &doorbell_);
  if (err != HSA_STATUS_SUCCESS) {
    throw hsa_exception(err, "Failed to create UMQ metadata for the AIE queue.");
  }
  // Hardware doorbell supports AQL semantics, the same as a GPU queue. The UMQ doorbell register
  // is 32 bits (unlike signal_.hardware_doorbell_ptr's 64-bit pointee); this field is only used
  // for address reporting (HSA_AMD_QUEUE_INFO_DOORBELL_ID), never dereferenced through its own
  // type, so the width mismatch does not matter.
  signal_.hardware_doorbell_ptr = reinterpret_cast<volatile uint64_t*>(doorbell_);

  active_ = true;

  ring_buf_guard.Dismiss();
}

AieUmqQueue::~AieUmqQueue() {
  auto err = AieUmqQueue::Inactivate();
  assert(err == HSA_STATUS_SUCCESS && "Destroy queue failed.");
  (void)err;
  if (ring_buf_) {
    auto& agent = static_cast<AieAgent&>(*GetAgent());
    agent.system_deallocator()(ring_buf_);
  }
  if (shared_queue_) {
    core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);
  }
}

hsa_status_t AieUmqQueue::Inactivate() {
  bool active = active_.exchange(false, std::memory_order_relaxed);
  if (!active) {
    return HSA_STATUS_SUCCESS;
  }

  auto& driver = static_cast<XdnaDriver&>(GetAgent()->driver());
  hsa_status_t err = driver.DestroyUserModeQueue(umq_metadata_);
  umq_metadata_ = nullptr;
  atomic::Fence(std::memory_order_acquire);
  return err;
}

hsa_status_t AieUmqQueue::SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) {
  return HSA_STATUS_SUCCESS;
}

void AieUmqQueue::Destroy() { delete this; }

// Atomic Reads/Writes
uint64_t AieUmqQueue::LoadReadIndexRelaxed() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieUmqQueue::LoadReadIndexAcquire() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
}

uint64_t AieUmqQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieUmqQueue::LoadWriteIndexAcquire() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
}

void AieUmqQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_relaxed);
}

void AieUmqQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_release);
}

uint64_t AieUmqQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_relaxed);
}

uint64_t AieUmqQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acquire);
}

uint64_t AieUmqQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_release);
}

uint64_t AieUmqQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acq_rel);
}

uint64_t AieUmqQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_relaxed);
}

uint64_t AieUmqQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acquire);
}

uint64_t AieUmqQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_release);
}

uint64_t AieUmqQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acq_rel);
}

void AieUmqQueue::SubmitPackets() {
  if (!active_.load(std::memory_order_relaxed) || suspended_) {
    return;
  }

  auto& agent = static_cast<AieAgent&>(*GetAgent());
  auto& driver = static_cast<XdnaDriver&>(agent.driver());

  const uint64_t first_pkt_idx = LoadReadIndexRelaxed();
  const uint64_t last_pkt_idx = LoadWriteIndexAcquire();
  if (first_pkt_idx >= last_pkt_idx) {
    // No packets to submit.
    return;
  }

  auto* ring =
      static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(amd_queue_.hsa_queue.base_address);
  const hsa_status_t err = driver.TranslateUmqPackets(ring, kCtxMaxCmds, first_pkt_idx,
                                                      last_pkt_idx - first_pkt_idx, agent);
  if (err != HSA_STATUS_SUCCESS) {
    // Rejected before reaching the device: no packet in this batch was readied, so none counts
    // towards read_dispatch_id, and the doorbell is never rung.
    debug_print("AIE UMQ queue: packet translation failed (0x%x)\n", err);
    suspended_ = true;
    InvokeErrorCallback(err);
    return;
  }
  atomic::Store(&amd_queue_.read_dispatch_id, last_pkt_idx, std::memory_order_release);

  // CERT's own write_index is a monotonic count of readied packets, i.e. last_pkt_idx itself (an
  // index one past the last readied packet), the same convention amd_queue_.write_dispatch_id
  // already uses.
  auto* header = static_cast<HostQueueHeader*>(ring_buf_);
  atomic::Store(&header->write_index, last_pkt_idx, std::memory_order_release);
  store_fence();
  // Hardware doorbell supports AQL semantics, the same as a GPU queue: the value rung is the
  // index of the last readied packet, not a count.
  *doorbell_ = static_cast<uint32_t>(last_pkt_idx - 1);
}

void AieUmqQueue::StoreRelaxed(hsa_signal_value_t value) { SubmitPackets(); }

void AieUmqQueue::StoreRelease(hsa_signal_value_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}

hsa_status_t AieUmqQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
      *static_cast<hsa_agent_t*>(value) = GetAgent()->public_handle();
      break;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      // Hardware doorbell supports AQL semantics.
      *static_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(signal_.hardware_doorbell_ptr);
      break;
    case HSA_QUEUE_INFO_USE_COUNT:
      // AIE queues do not support counted queue features
      *static_cast<uint32_t*>(value) = static_cast<uint32_t>(-1);
      break;
    case HSA_QUEUE_INFO_HW_ID:
      *static_cast<uint32_t*>(value) = public_handle()->id;
      break;
    case HSA_AMD_QUEUE_INFO_ENGINE_TYPE:
      *static_cast<hsa_amd_queue_engine_t*>(value) = HSA_AMD_QUEUE_ENGINE_AIE;
      break;
    case HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieUmqQueue::GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t AieUmqQueue::SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

void AieUmqQueue::ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b, hsa_fence_scope_t acquireFence,
                             hsa_fence_scope_t releaseFence, hsa_signal_t* signal) {
  assert(false && "AIE UMQ queue does not support PM4 packets.");
}

} // namespace AMD
} // namespace rocr
