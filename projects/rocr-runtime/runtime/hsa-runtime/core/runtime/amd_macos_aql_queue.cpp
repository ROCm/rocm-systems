////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__

#include "core/inc/amd_macos_aql_queue.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <utility>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#include "core/inc/exceptions.h"
#include "core/util/utils.h"
#include "loader/AMDHSAKernelDescriptor.h"

namespace rocr {
namespace AMD {
namespace {

constexpr uint32_t PACKET3_DISPATCH_DIRECT = 0x15;
constexpr uint32_t PACKET3_EVENT_WRITE = 0x46;
constexpr uint32_t PACKET3_WRITE_DATA = 0x37;
constexpr uint32_t PACKET3_ACQUIRE_MEM = 0x58;
constexpr uint32_t PACKET3_SET_SH_REG = 0x76;

constexpr uint32_t WRITE_DATA_DST_SEL_MEM_ASYNC = 5;
constexpr uint32_t WRITE_DATA_WR_CONFIRM = 1u << 20;
constexpr uint32_t WRITE_DATA_ENGINE_SEL_ME = 0;
constexpr uint32_t WRITE_DATA_CACHE_POLICY_BYPASS = 3u << 25;

constexpr uint32_t CS_PARTIAL_FLUSH = 7;
constexpr uint32_t EVENT_INDEX_CS_PARTIAL_FLUSH = 4;

constexpr uint32_t SH_REG_BASE = 0x2C00;
constexpr uint32_t COMPUTE_PGM_LO = 0x2E0C;
constexpr uint32_t COMPUTE_PGM_RSRC1 = 0x2E12;
constexpr uint32_t COMPUTE_RESOURCE_LIMITS = 0x2E15;
constexpr uint32_t COMPUTE_TMPRING_SIZE = 0x2E18;
// gfx12 architected-flat-scratch dispatch base (gc_12_0_0 regCOMPUTE_DISPATCH_
// SCRATCH_BASE_LO/HI 0x1bb0/0x1bb1 + the 0x1260 MMIO->SH offset used by every
// other reg constant in this file). Value = backing VA >> 8 (256-byte units).
// Corroborated by tinygrad's working gfx11/12 direct-PM4 path which writes this
// per-dispatch for target >= gfx11.
constexpr uint32_t COMPUTE_DISPATCH_SCRATCH_BASE_LO = 0x2E10;
constexpr uint32_t COMPUTE_DISPATCH_SCRATCH_BASE_HI = 0x2E11;
// The base is written as a single 2-dword SET_SH_REG starting at _LO, which
// must land on _LO then _HI; assert they are consecutive.
static_assert(COMPUTE_DISPATCH_SCRATCH_BASE_HI == COMPUTE_DISPATCH_SCRATCH_BASE_LO + 1,
              "scratch base LO/HI must be consecutive SH registers");
constexpr uint32_t COMPUTE_RESTART_X = 0x2E1B;

// gfx11/gfx12 COMPUTE_TMPRING_SIZE.WAVESIZE counts 256-byte blocks per wave.
constexpr uint32_t kScratchGranularity = 256;
// Worst-case concurrent wave-slot bound for gfx1201 (Navi48 / RX 9070 XT):
// num_cu * MaxSlotsScratchCU. 64 * 32 = 2048, fits the 12-bit WAVES field.
constexpr uint32_t kGfx1201NumCu = 64;
constexpr uint32_t kMaxSlotsScratchCU = 32;
constexpr uint32_t kScratchMaxWaves = kGfx1201NumCu * kMaxSlotsScratchCU;
// Over-allocate the backing buffer to cover per-shader-engine scratch striping
// (the exact wave->offset layout is the one item to confirm on hardware); cheap
// insurance against an out-of-bounds scratch access faulting the CP.
constexpr uint32_t kScratchSeFactor = 4;
constexpr uint32_t COMPUTE_PGM_RSRC3_GFX12 = 0x2E28;
constexpr uint32_t COMPUTE_USER_DATA_0 = 0x2E40;
constexpr uint32_t COMPUTE_START_X = 0x2E04;

constexpr uint32_t COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT = 1;
constexpr uint32_t COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_MASK = 0x1Fu;
constexpr uint32_t COMPUTE_PGM_RSRC3_GFX10_PLUS_IMAGE_OP = 1u << 31;

constexpr uint16_t kPropPrivateSegmentBuffer = 1u << 0;
constexpr uint16_t kPropDispatchPtr = 1u << 1;
constexpr uint16_t kPropQueuePtr = 1u << 2;
constexpr uint16_t kPropKernargPtr = 1u << 3;
constexpr uint16_t kPropDispatchId = 1u << 4;
constexpr uint16_t kPropFlatScratchInit = 1u << 5;
constexpr uint16_t kPropPrivateSegmentSize = 1u << 6;
constexpr uint16_t kPropWave32 = 1u << 10;

uint64_t AlignUp(uint64_t value, uint64_t align) {
  return (value + align - 1) & ~(align - 1);
}

uint32_t Low32(uint64_t value) { return static_cast<uint32_t>(value); }
uint32_t High32(uint64_t value) { return static_cast<uint32_t>(value >> 32); }

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
  if (divisor == 0) return value;
  return (value + divisor - 1) / divisor;
}

void Pkt3(std::vector<uint32_t>& pm4, uint32_t opcode, const std::vector<uint32_t>& payload) {
  const uint32_t n = static_cast<uint32_t>(payload.size());
  pm4.push_back((3u << 30) | (((n - 1u) & 0x3FFFu) << 16) | (opcode << 8));
  pm4.insert(pm4.end(), payload.begin(), payload.end());
}

void SetShReg(std::vector<uint32_t>& pm4, uint32_t reg, const std::vector<uint32_t>& values) {
  std::vector<uint32_t> payload;
  payload.reserve(values.size() + 1);
  payload.push_back(reg >= SH_REG_BASE ? reg - SH_REG_BASE : reg);
  payload.insert(payload.end(), values.begin(), values.end());
  Pkt3(pm4, PACKET3_SET_SH_REG, payload);
}

void EventWrite(std::vector<uint32_t>& pm4, uint32_t event_type, uint32_t event_index) {
  Pkt3(pm4, PACKET3_EVENT_WRITE, { (event_type & 0x3Fu) | ((event_index & 0xFu) << 8) });
}

void WriteData(std::vector<uint32_t>& pm4, uint64_t addr, uint32_t value) {
  const uint32_t control = ((WRITE_DATA_DST_SEL_MEM_ASYNC & 0xFu) << 8) |
                           WRITE_DATA_CACHE_POLICY_BYPASS |
                           ((WRITE_DATA_ENGINE_SEL_ME & 0x3u) << 30) |
                           WRITE_DATA_WR_CONFIRM;
  Pkt3(pm4, PACKET3_WRITE_DATA, {control, Low32(addr), High32(addr), value});
}

uint32_t FullRangeGcrCntl() {
  return (1u << 16) |  // SEQ = FORWARD
         (1u << 15) |  // GL2_WB
         (1u << 14) |  // GL2_INV
         (1u << 9) |   // GL1_INV
         (1u << 8) |   // GLV_INV
         (1u << 7) |   // GLK_INV
         (1u << 6) |   // GLK_WB
         (1u << 5) |   // GLM_INV
         (1u << 4) |   // GLM_WB
         (1u << 0);    // GLI_INV = ALL
}

void AcquireMemGfx10(std::vector<uint32_t>& pm4, uint32_t gcr_cntl = FullRangeGcrCntl()) {
  constexpr uint32_t kFullRangeSize = 0xFFFFFFFFu;
  constexpr uint32_t kFullRangeSizeHi = 0xFFu;
  constexpr uint32_t kPollInterval = 4u;
  Pkt3(pm4, PACKET3_ACQUIRE_MEM,
       {0, kFullRangeSize, kFullRangeSizeHi, 0, 0, kPollInterval, gcr_cntl});
}

void DispatchDirect(std::vector<uint32_t>& pm4, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z,
                    uint32_t initiator) {
  Pkt3(pm4, PACKET3_DISPATCH_DIRECT, {dim_x, dim_y, dim_z, initiator});
}

void CopyToBar(void* dst, const void* src, size_t size) {
  auto* d8 = static_cast<volatile uint8_t*>(dst);
  const auto* s8 = static_cast<const uint8_t*>(src);
  size_t i = 0;
  for (; i + sizeof(uint32_t) <= size; i += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, s8 + i, sizeof(word));
    *reinterpret_cast<volatile uint32_t*>(d8 + i) = word;
  }
  for (; i < size; ++i) d8[i] = s8[i];
}

void CopyFromBar(void* dst, const void* src, size_t size) {
  auto* d8 = static_cast<uint8_t*>(dst);
  const auto* s8 = static_cast<const volatile uint8_t*>(src);
  size_t i = 0;
  for (; i + sizeof(uint32_t) <= size; i += sizeof(uint32_t)) {
    uint32_t word = *reinterpret_cast<const volatile uint32_t*>(s8 + i);
    std::memcpy(d8 + i, &word, sizeof(word));
  }
  for (; i < size; ++i) d8[i] = s8[i];
}

// Host-pointer kernarg staging: a kernarg value that is a raw host pointer (not
// a registered VRAM address HostToGpuAddress can translate) is copied into VRAM
// scratch so the GPU can read it; writable ranges are recorded for an optional
// post-dispatch copy-back (ROCR_MACOS_AQL_ENABLE_HOST_COPYBACK).
constexpr size_t kHostPointerStageBytes = 4096;

struct HostPointerStage {
  uint64_t host = 0;
  void* staged_cpu = nullptr;
  uint64_t staged_gpu = 0;
  size_t size = 0;
  size_t copy_back_size = 0;
};

bool LooksLikeDarwinUserPointer(uint64_t value) {
  // Filter GPU virtual addresses and small/aligned scalar values before asking
  // Mach about the process VM map.
  if ((value & 0xffffffffull) == 0) return false;
  return value >= 0x100000000ull && value < 0x0000800000000000ull &&
         (value & (alignof(uint64_t) - 1)) == 0;
}

bool HostPointerRangeWithProtection(const void* ptr, vm_prot_t required_prot,
                                    size_t* available_size) {
  if (ptr == nullptr || available_size == nullptr) return false;
  *available_size = 0;
  mach_vm_address_t query = reinterpret_cast<mach_vm_address_t>(ptr);
  mach_vm_address_t region = query;
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  kern_return_t kr = mach_vm_region(mach_task_self(), &region, &region_size,
                                    VM_REGION_BASIC_INFO_64,
                                    reinterpret_cast<vm_region_info_t>(&info), &count,
                                    &object_name);
  if (object_name != MACH_PORT_NULL) {
    mach_port_deallocate(mach_task_self(), object_name);
  }
  if (kr != KERN_SUCCESS) return false;
  if (query < region || query >= region + region_size) return false;
  if ((info.protection & required_prot) != required_prot) return false;
  *available_size = static_cast<size_t>(std::min<mach_vm_size_t>(
      region + region_size - query, kHostPointerStageBytes));
  return *available_size != 0;
}

bool ReadableHostPointerRange(const void* ptr, size_t* readable_size) {
  return HostPointerRangeWithProtection(ptr, VM_PROT_READ, readable_size);
}

bool WritableHostPointerRange(const void* ptr, size_t* writable_size) {
  return HostPointerRangeWithProtection(ptr, VM_PROT_READ | VM_PROT_WRITE,
                                        writable_size);
}

bool SignalSatisfied(hsa_signal_t signal) {
  if (signal.handle == 0) return true;
  core::Signal* s = core::Signal::Convert(signal);
  return s->LoadAcquire() <= 0;
}

bool TraceAql() { return std::getenv("ROCR_MACOS_TRACE_AQL") != nullptr; }

}  // namespace

MacAqlQueue::MacAqlQueue(core::SharedQueue* shared_queue, MacGpuAgent* agent,
                         size_t req_size_pkts, uint64_t flags,
                         core::HsaEventCallback callback, void* err_data)
    : Queue(shared_queue, flags, agent),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      agent_(*agent),
      driver_(static_cast<MacOsDriver&>(agent->driver())),
      errors_callback_(callback),
      errors_data_(err_data),
      queue_size_pkts_(static_cast<uint32_t>(req_size_pkts)),
      active_(false) {
  if (!agent) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_AGENT, "Darwin AQL queue needs a MacGpuAgent");
  }

  std::memset(&amd_queue_, 0, sizeof(amd_queue_));
  const size_t queue_size_bytes = req_size_pkts * sizeof(core::AqlPacket);
  ring_buf_ = core::Runtime::runtime_singleton_->system_allocator()(
      queue_size_bytes, 4096, core::MemoryRegion::AllocateNoFlags, agent_.node_id());
  if (ring_buf_ == nullptr) {
    throw hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                        "Could not allocate Darwin AQL ring buffer");
  }
  std::memset(ring_buf_, 0, queue_size_bytes);

  hsa_status_t status = driver_.CreateDirectComputeQueue(&direct_queue_);
  if (status != HSA_STATUS_SUCCESS) {
    core::Runtime::runtime_singleton_->system_deallocator()(ring_buf_);
    ring_buf_ = nullptr;
    throw hsa_exception(status, "Could not create Darwin direct compute queue");
  }

  status = driver_.AllocateVram(4096, 4096, &marker_cpu_base_, &marker_gpu_);
  if (status != HSA_STATUS_SUCCESS) {
    driver_.DestroyDirectComputeQueue(direct_queue_);
    core::Runtime::runtime_singleton_->system_deallocator()(ring_buf_);
    ring_buf_ = nullptr;
    throw hsa_exception(status, "Could not allocate Darwin queue marker");
  }
  marker_cpu_ = static_cast<volatile uint32_t*>(marker_cpu_base_);
  *marker_cpu_ = 0;

  status = driver_.AllocateVram(1024 * 1024, 4096, &scratch_cpu_, &scratch_gpu_);
  if (status != HSA_STATUS_SUCCESS) {
    driver_.FreeMemory(marker_cpu_base_, 4096);
    driver_.DestroyDirectComputeQueue(direct_queue_);
    core::Runtime::runtime_singleton_->system_deallocator()(ring_buf_);
    ring_buf_ = nullptr;
    throw hsa_exception(status, "Could not allocate Darwin dispatch scratch");
  }
  scratch_size_ = 1024 * 1024;
  scratch_offset_ = 0;

  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_MULTI;
  amd_queue_.hsa_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
  amd_queue_.hsa_queue.base_address = ring_buf_;
  amd_queue_.hsa_queue.doorbell_signal =
      core::Signal::Convert(static_cast<core::Signal*>(this));
  amd_queue_.hsa_queue.size = queue_size_pkts_;
  amd_queue_.hsa_queue.id = GetQueueId();
  amd_queue_.write_dispatch_id = 0;
  amd_queue_.read_dispatch_id = 0;

  signal_.hardware_doorbell_ptr = nullptr;
  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.queue_ptr = &amd_queue_;
  active_.store(true, std::memory_order_release);
}

MacAqlQueue::~MacAqlQueue() {
  MacAqlQueue::Inactivate();
  if (gpu_scratch_cpu_) driver_.FreeMemory(gpu_scratch_cpu_, gpu_scratch_size_);
  if (scratch_cpu_) driver_.FreeMemory(scratch_cpu_, scratch_size_);
  if (marker_cpu_base_) driver_.FreeMemory(marker_cpu_base_, 4096);
  if (ring_buf_) core::Runtime::runtime_singleton_->system_deallocator()(ring_buf_);
  if (shared_queue_) core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);
}

hsa_status_t MacAqlQueue::Inactivate() {
  const bool was_active = active_.exchange(false, std::memory_order_acq_rel);
  if (was_active) driver_.DestroyDirectComputeQueue(direct_queue_);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacAqlQueue::SetPriority(HSA::hsa_amd_queue_priority_internal_t) {
  return HSA_STATUS_SUCCESS;
}

void MacAqlQueue::Destroy() { delete this; }

uint64_t MacAqlQueue::LoadReadIndexRelaxed() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
}
uint64_t MacAqlQueue::LoadReadIndexAcquire() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
}
uint64_t MacAqlQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
}
uint64_t MacAqlQueue::LoadWriteIndexAcquire() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
}
void MacAqlQueue::StoreReadIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.read_dispatch_id, value, std::memory_order_relaxed);
}
void MacAqlQueue::StoreReadIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.read_dispatch_id, value, std::memory_order_release);
}
void MacAqlQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value, std::memory_order_relaxed);
}
void MacAqlQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value, std::memory_order_release);
}
uint64_t MacAqlQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_relaxed);
}
uint64_t MacAqlQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_acquire);
}
uint64_t MacAqlQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_release);
}
uint64_t MacAqlQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_acq_rel);
}
uint64_t MacAqlQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_relaxed);
}
uint64_t MacAqlQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_acquire);
}
uint64_t MacAqlQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_release);
}
uint64_t MacAqlQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_acq_rel);
}

void MacAqlQueue::StoreRelaxed(hsa_signal_value_t value) {
  hsa_status_t status = SubmitPackets(static_cast<uint64_t>(value));
  if (status != HSA_STATUS_SUCCESS) ReportAsyncError(status);
}

void MacAqlQueue::StoreRelease(hsa_signal_value_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}

hsa_status_t MacAqlQueue::SubmitPackets(uint64_t doorbell_value) {
  if (!active_.load(std::memory_order_acquire)) return HSA_STATUS_ERROR_INVALID_QUEUE;
  const uint64_t end = std::min<uint64_t>(LoadWriteIndexAcquire(), doorbell_value + 1);
  uint64_t cur = LoadReadIndexRelaxed();
  while (cur < end) {
    auto* packet = static_cast<core::AqlPacket*>(ring_buf_) + (cur & (queue_size_pkts_ - 1));
    const uint16_t header = packet->packet.header;
    const uint8_t type = core::AqlPacket::type(header);
    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      status = SubmitKernel(packet->dispatch);
    } else if (type == HSA_PACKET_TYPE_BARRIER_AND) {
      status = SubmitBarrier(packet->barrier_and, false);
    } else if (type == HSA_PACKET_TYPE_BARRIER_OR) {
      status = SubmitBarrier(packet->barrier_and, true);
    } else if (type == HSA_PACKET_TYPE_INVALID) {
      status = HSA_STATUS_SUCCESS;
    } else {
      status = HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }

    packet->packet.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
    cur++;
    StoreReadIndexRelease(cur);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacAqlQueue::AllocateDispatchScratch(size_t size, size_t align, void** cpu,
                                                  uint64_t* gpu) {
  if (cpu == nullptr || gpu == nullptr || size == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const uint64_t aligned = AlignUp(scratch_offset_, align);
  const uint64_t rounded = AlignUp(size, align);
  if (aligned + rounded > scratch_size_) {
    scratch_offset_ = 0;
  } else {
    scratch_offset_ = static_cast<size_t>(aligned);
  }
  if (scratch_offset_ + rounded > scratch_size_) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  *cpu = static_cast<char*>(scratch_cpu_) + scratch_offset_;
  *gpu = scratch_gpu_ + scratch_offset_;
  scratch_offset_ += static_cast<size_t>(rounded);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacAqlQueue::EnsureGpuScratch(size_t size) {
  if (size <= gpu_scratch_size_) return HSA_STATUS_SUCCESS;
  const size_t rounded = static_cast<size_t>(AlignUp(size, 64 * 1024));
  void* cpu = nullptr;
  uint64_t gpu = 0;
  hsa_status_t status = driver_.AllocateVram(rounded, 4096, &cpu, &gpu);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (gpu_scratch_cpu_) driver_.FreeMemory(gpu_scratch_cpu_, gpu_scratch_size_);
  gpu_scratch_cpu_ = cpu;
  gpu_scratch_gpu_ = gpu;
  gpu_scratch_size_ = rounded;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacAqlQueue::SubmitKernel(const hsa_kernel_dispatch_packet_t& packet) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  if (packet.kernel_object == 0) return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  const auto* kd_cpu = reinterpret_cast<const kernel_descriptor_t*>(
      static_cast<uintptr_t>(packet.kernel_object));
  const kernel_descriptor_t* kd = kd_cpu;
  const void* kd_shadow = nullptr;
  if (driver_.VramShadowAddress(kd_cpu, sizeof(*kd), &kd_shadow) == HSA_STATUS_SUCCESS) {
    kd = static_cast<const kernel_descriptor_t*>(kd_shadow);
  }

  const auto code_cpu = reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(kd_cpu) + kd->kernel_code_entry_byte_offset);
  uint64_t code_gpu = 0;
  hsa_status_t status = driver_.HostToGpuAddress(code_cpu, &code_gpu);
  if (status != HSA_STATUS_SUCCESS) return status;

  uint64_t kernarg_gpu = 0;
  std::vector<std::pair<uint64_t, uint64_t>> kernarg_translations;
  std::vector<HostPointerStage> host_pointer_stages;
  size_t kernarg_stage_size = kd->kernarg_size;
  // rocBLAS/Tensile UserArgs kernels request the kernarg-pointer SGPR
  // (kPropKernargPtr) but report kernarg_size==0; without a staged buffer they
  // dispatch with kernarg_gpu=0 and the kernel faults reading kernargs from
  // address 0 (queue 0x1000 abort, e.g. hipBLAS SGEMM). Stage a fallback-size
  // blob from the host kernargs so the kernarg-pointer SGPR is valid.
  if (packet.kernarg_address != nullptr && kd->kernarg_size == 0 &&
      (kd->kernel_code_properties & kPropKernargPtr) != 0) {
    const char* env = std::getenv("ROCR_MACOS_AQL_ZERO_KERNARG_SIZE");
    kernarg_stage_size =
        env != nullptr ? static_cast<size_t>(std::strtoul(env, nullptr, 0)) : 512;
  }
  if (packet.kernarg_address != nullptr && kernarg_stage_size != 0) {
    void* kernarg_cpu = nullptr;
    status = AllocateDispatchScratch(kernarg_stage_size, 16, &kernarg_cpu, &kernarg_gpu);
    if (status != HSA_STATUS_SUCCESS) return status;

    std::vector<uint8_t> kernargs(kernarg_stage_size);
    std::memcpy(kernargs.data(), packet.kernarg_address, kernargs.size());
    // macOS-egpu: translate only kernarg qwords that are interiors of a REGISTERED
    // VRAM allocation, and (by default) never stage host-looking scalars. The old
    // value-heuristic translated/staged ANY qword that fell in the BAR window or
    // "looked like" a host pointer, which corrupts large by-value struct kernargs
    // (e.g. ATen reduce_kernel's ~976B ReduceOp of IntDivider magic/stride scalars)
    // and hangs the GPU (HSA 0x1000). ROCR_MACOS_AQL_KERNARG_LEGACY=1 restores the
    // old behavior for A/B comparison.
    const bool kernarg_legacy = EnvEnabled("ROCR_MACOS_AQL_KERNARG_LEGACY");
    for (size_t off = 0; off + sizeof(uint64_t) <= kernargs.size(); off += sizeof(uint64_t)) {
      uint64_t value = 0;
      std::memcpy(&value, kernargs.data() + off, sizeof(value));
      const void* value_ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(value));
      uint64_t translated = 0;
      const bool translate_eligible =
          kernarg_legacy || driver_.IsRegisteredVramPointer(value_ptr);
      if (translate_eligible &&
          driver_.HostToGpuAddress(value_ptr, &translated) == HSA_STATUS_SUCCESS) {
        if (TraceAql() && translated != value) {
          kernarg_translations.emplace_back(value, translated);
        }
        std::memcpy(kernargs.data() + off, &translated, sizeof(translated));
        continue;
      }
      if (!kernarg_legacy) continue;
      // Legacy fallback: a raw host pointer not registered as VRAM. Stage it to VRAM
      // scratch so the kernel can read it, and record a writable range for the
      // optional post-dispatch copy-back.
      size_t readable_size = 0;
      if (LooksLikeDarwinUserPointer(value) &&
          ReadableHostPointerRange(reinterpret_cast<const void*>(static_cast<uintptr_t>(value)),
                                   &readable_size)) {
        void* staged_cpu = nullptr;
        uint64_t staged_gpu = 0;
        status = AllocateDispatchScratch(readable_size, 16, &staged_cpu, &staged_gpu);
        if (status != HSA_STATUS_SUCCESS) return status;
        CopyToBar(staged_cpu, reinterpret_cast<const void*>(static_cast<uintptr_t>(value)),
                  readable_size);
        size_t writable_size = 0;
        const bool writable = WritableHostPointerRange(
            reinterpret_cast<void*>(static_cast<uintptr_t>(value)), &writable_size);
        host_pointer_stages.push_back(
            {value, staged_cpu, staged_gpu, readable_size,
             writable ? std::min(readable_size, writable_size) : 0});
        if (TraceAql()) kernarg_translations.emplace_back(value, staged_gpu);
        std::memcpy(kernargs.data() + off, &staged_gpu, sizeof(staged_gpu));
      }
    }
    CopyToBar(kernarg_cpu, kernargs.data(), kernargs.size());

    if (TraceAql()) {
      const size_t qword_count = std::min<size_t>(kernargs.size() / sizeof(uint64_t), 8);
      std::fprintf(stderr,
                   "ROCR macOS AQL kernargs host=%p gpu=0x%llx size=%u qwords=",
                   packet.kernarg_address, static_cast<unsigned long long>(kernarg_gpu),
                   kd->kernarg_size);
      for (size_t i = 0; i < qword_count; ++i) {
        uint64_t value = 0;
        std::memcpy(&value, kernargs.data() + i * sizeof(value), sizeof(value));
        std::fprintf(stderr, "%s0x%016llx", i == 0 ? "" : ",",
                     static_cast<unsigned long long>(value));
      }
      if (kernarg_translations.empty()) {
        std::fprintf(stderr, " translations=none\n");
      } else {
        std::fprintf(stderr, " translations=");
        for (size_t i = 0; i < kernarg_translations.size(); ++i) {
          std::fprintf(stderr, "%s0x%016llx->0x%016llx", i == 0 ? "" : ",",
                       static_cast<unsigned long long>(kernarg_translations[i].first),
                       static_cast<unsigned long long>(kernarg_translations[i].second));
        }
        std::fprintf(stderr, "\n");
      }
    }
  }

  uint64_t dispatch_packet_gpu = 0;
  if ((kd->kernel_code_properties & kPropDispatchPtr) != 0) {
    void* dispatch_packet_cpu = nullptr;
    status = AllocateDispatchScratch(sizeof(packet), 64, &dispatch_packet_cpu, &dispatch_packet_gpu);
    if (status != HSA_STATUS_SUCCESS) return status;
    hsa_kernel_dispatch_packet_t gpu_packet = packet;
    gpu_packet.kernarg_address = reinterpret_cast<void*>(static_cast<uintptr_t>(kernarg_gpu));
    CopyToBar(dispatch_packet_cpu, &gpu_packet, sizeof(gpu_packet));
  }

  // The enable_sgpr_private_segment_buffer (V# in user SGPRs) path is genuinely
  // unimplemented and is not used by the gfx12 architected-flat-scratch dispatch
  // path; keep rejecting it.
  if ((kd->kernel_code_properties & kPropPrivateSegmentBuffer) != 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  // KNOWN-INCOMPLETE on this no-MES path. gfx12 uses architected flat scratch:
  // FLAT_SCRATCH is a readonly, SPI-initialized register, and the SPI sources
  // the per-wave scratch base from per-QUEUE state (MQD / amd_queue_t) that MES
  // normally programs. The register writes below (DISPATCH_SCRATCH_BASE +
  // TMPRING) plus the SH_MEM_BASES/CONFIG aperture set up in the bring-up are
  // necessary but NOT sufficient without that per-queue scratch state: a
  // scratch-using kernel still faults the CP with FLAT_SCRATCH=0 (GCVM L2
  // permission fault at VA 0). Until per-queue scratch is wired (via MES
  // submission or MQD scratch fields), the knob below defaults OFF and we reject
  // scratch-using kernels cleanly (OUT_OF_RESOURCES) instead of wedging the GPU.
  const bool enable_scratch = EnvEnabled("ROCR_MACOS_AQL_ENABLE_SCRATCH");
  // A kernel uses scratch iff it has a nonzero per-work-item private segment.
  // gfx12 architected scratch does NOT set kPropFlatScratchInit, so don't key on
  // that. Take the larger of packet and descriptor sizes. (RSRC2.ENABLE_PRIVATE_
  // SEGMENT can be set with a zero fixed size and no real spilling, so it is not
  // used as the trigger.)
  const uint32_t scratch_bytes_per_thread = std::max<uint32_t>(
      packet.private_segment_size, kd->private_segment_fixed_size);
  const bool wants_scratch = scratch_bytes_per_thread != 0;
  if (wants_scratch && !enable_scratch) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  uint64_t gpu_scratch_base_256 = 0;
  uint32_t compute_tmpring_size = 0;
  if (wants_scratch) {
    const uint32_t lanes =
        (kd->kernel_code_properties & kPropWave32) != 0 ? 32u : 64u;
    const uint32_t bytes_per_thread = static_cast<uint32_t>(
        AlignUp(scratch_bytes_per_thread, kScratchGranularity / lanes));
    const uint32_t wave_bytes = bytes_per_thread * lanes;
    const uint32_t wavesize_units = CeilDiv(wave_bytes, kScratchGranularity);
    const uint64_t backing =
        static_cast<uint64_t>(wave_bytes) * kScratchMaxWaves * kScratchSeFactor;
    status = EnsureGpuScratch(static_cast<size_t>(backing));
    if (status != HSA_STATUS_SUCCESS) return status;
    gpu_scratch_base_256 = gpu_scratch_gpu_ >> 8;
    compute_tmpring_size =
        (kScratchMaxWaves & 0xFFFu) | ((wavesize_units & 0x3FFFFu) << 12);
    if (TraceAql()) {
      std::fprintf(stderr,
                   "ROCR macOS AQL scratch base=0x%llx pss=%u lanes=%u bpt=%u "
                   "wave_bytes=%u wavesize=%u waves=%u tmpring=0x%x backing=%zu\n",
                   static_cast<unsigned long long>(gpu_scratch_gpu_),
                   scratch_bytes_per_thread, lanes, bytes_per_thread, wave_bytes,
                   wavesize_units, kScratchMaxWaves, compute_tmpring_size,
                   gpu_scratch_size_);
    }
  }

  std::vector<uint32_t> user_data;
  user_data.reserve(16);
  if ((kd->kernel_code_properties & kPropPrivateSegmentBuffer) != 0) {
    user_data.insert(user_data.end(), {0, 0, 0, 0});
  }
  if ((kd->kernel_code_properties & kPropDispatchPtr) != 0) {
    user_data.insert(user_data.end(), {Low32(dispatch_packet_gpu), High32(dispatch_packet_gpu)});
  }
  if ((kd->kernel_code_properties & kPropQueuePtr) != 0) {
    user_data.insert(user_data.end(), {0, 0});
  }
  if ((kd->kernel_code_properties & kPropKernargPtr) != 0) {
    user_data.insert(user_data.end(), {Low32(kernarg_gpu), High32(kernarg_gpu)});
  }
  if ((kd->kernel_code_properties & kPropDispatchId) != 0) {
    user_data.insert(user_data.end(), {0, 0});
  }
  if ((kd->kernel_code_properties & kPropFlatScratchInit) != 0) {
    user_data.insert(user_data.end(), {0, 0});
  }
  if ((kd->kernel_code_properties & kPropPrivateSegmentSize) != 0) {
    user_data.push_back(packet.private_segment_size);
  }

  const uint32_t expected_user_sgprs =
      (kd->compute_pgm_rsrc2 >> COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) &
      COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_MASK;
  if (user_data.size() < expected_user_sgprs) {
    user_data.resize(expected_user_sgprs, 0);
  }

  const uint32_t lds_blocks =
      static_cast<uint32_t>((static_cast<uint64_t>(packet.group_segment_size) + 511) / 512);
  const bool wave32 = (kd->kernel_code_properties & kPropWave32) != 0;
  const uint64_t pgm = code_gpu >> 8;
  uint32_t compute_pgm_rsrc3 = kd->compute_pgm_rsrc3;
  if (EnvEnabled("ROCR_MACOS_AQL_FORCE_RSRC3_IMAGE_OP")) {
    compute_pgm_rsrc3 |= COMPUTE_PGM_RSRC3_GFX10_PLUS_IMAGE_OP;
  }
  const bool phase13_style = EnvEnabled("ROCR_MACOS_AQL_PHASE13_STYLE");
  const bool use_thread_dims =
      EnvEnabled("ROCR_MACOS_AQL_USE_THREAD_DIMS") ||
      (!phase13_style && !EnvEnabled("ROCR_MACOS_AQL_WORKGROUP_DIMS"));
  const bool order_mode = phase13_style || EnvEnabled("ROCR_MACOS_AQL_ORDER_MODE");
  const bool resource_zero =
      phase13_style || EnvEnabled("ROCR_MACOS_AQL_RESOURCE_LIMITS_ZERO");
  const bool resource_skip = EnvEnabled("ROCR_MACOS_AQL_RESOURCE_LIMITS_SKIP");
  const uint32_t dispatch_dim_x =
      use_thread_dims ? packet.grid_size_x
                      : std::max<uint32_t>(1, CeilDiv(packet.grid_size_x,
                                                      packet.workgroup_size_x));
  const uint32_t dispatch_dim_y =
      use_thread_dims ? packet.grid_size_y
                      : std::max<uint32_t>(1, CeilDiv(packet.grid_size_y,
                                                      packet.workgroup_size_y));
  const uint32_t dispatch_dim_z =
      use_thread_dims ? packet.grid_size_z
                      : std::max<uint32_t>(1, CeilDiv(packet.grid_size_z,
                                                      packet.workgroup_size_z));
  uint32_t dispatch_initiator = (1u << 0) | (1u << 2);
  if (use_thread_dims) dispatch_initiator |= 1u << 5;
  if (order_mode) dispatch_initiator |= 1u << 6;
  if (wave32) dispatch_initiator |= 1u << 15;

  if (TraceAql()) {
    std::fprintf(stderr,
                 "ROCR macOS AQL kernel obj=0x%llx code_cpu=%p code_gpu=0x%llx "
                 "kernarg_gpu=0x%llx grid=%ux%ux%u block=%ux%ux%u rsrc=(0x%x,0x%x,0x%x) "
                 "props=0x%x wave32=%u user_sgprs=%zu/%u lds_blocks=%u dispatch=%ux%ux%u "
                 "initiator=0x%x resource=%s%s\n",
                 static_cast<unsigned long long>(packet.kernel_object), code_cpu,
                 static_cast<unsigned long long>(code_gpu),
                 static_cast<unsigned long long>(kernarg_gpu), packet.grid_size_x,
                 packet.grid_size_y, packet.grid_size_z, packet.workgroup_size_x,
                 packet.workgroup_size_y, packet.workgroup_size_z, kd->compute_pgm_rsrc1,
                 kd->compute_pgm_rsrc2, compute_pgm_rsrc3, kd->kernel_code_properties,
                 wave32 ? 1u : 0u, user_data.size(), expected_user_sgprs, lds_blocks, dispatch_dim_x,
                 dispatch_dim_y, dispatch_dim_z, dispatch_initiator,
                 resource_skip ? "skip" : (resource_zero ? "zero" : "wddm"),
                 use_thread_dims ? ":thread-dims" : ":workgroup-dims");
  }

  if (std::getenv("ROCR_MACOS_AQL_PREFLIGHT_MARKER") != nullptr) {
    status = SubmitPm4AndWait({});
    if (status != HSA_STATUS_SUCCESS) return status;
  }

  if (std::getenv("ROCR_MACOS_AQL_MARKER_ONLY") != nullptr) {
    status = SubmitPm4AndWait({});
    if (status == HSA_STATUS_SUCCESS) CompleteSignal(packet.completion_signal, 0);
    return status;
  }

  std::vector<uint32_t> pm4;
  pm4.reserve(96);
  if (!EnvEnabled("ROCR_MACOS_AQL_SKIP_PRE_ACQUIRE")) {
    AcquireMemGfx10(pm4);
  }
  SetShReg(pm4, COMPUTE_PGM_LO, {Low32(pgm), High32(pgm)});
  SetShReg(pm4, COMPUTE_PGM_RSRC1,
           {kd->compute_pgm_rsrc1, kd->compute_pgm_rsrc2 | (lds_blocks << 15)});
  SetShReg(pm4, COMPUTE_PGM_RSRC3_GFX12, {compute_pgm_rsrc3});
  if (gpu_scratch_base_256 != 0) {
    SetShReg(pm4, COMPUTE_DISPATCH_SCRATCH_BASE_LO,
             {Low32(gpu_scratch_base_256), High32(gpu_scratch_base_256)});
  }
  SetShReg(pm4, COMPUTE_TMPRING_SIZE, {compute_tmpring_size});
  SetShReg(pm4, COMPUTE_RESTART_X, {0, 0, 0});
  if (!user_data.empty()) SetShReg(pm4, COMPUTE_USER_DATA_0, user_data);
  if (!resource_skip) {
    if (resource_zero) {
      SetShReg(pm4, COMPUTE_RESOURCE_LIMITS, {0});
    } else {
      SetShReg(pm4, COMPUTE_RESOURCE_LIMITS,
               {0x3ff, 0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff});
    }
  }
  SetShReg(pm4, COMPUTE_START_X,
           {0, 0, 0, packet.workgroup_size_x, packet.workgroup_size_y,
            packet.workgroup_size_z, 0, 0});
  DispatchDirect(pm4, dispatch_dim_x, dispatch_dim_y, dispatch_dim_z, dispatch_initiator);
  EventWrite(pm4, CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH);
  // Post-dispatch GL2 writeback+invalidate so the kernel's results reach VRAM and
  // are visible to the host blit. MUST default ON: without it the kernel's output
  // stays in L2 and the host reads stale VRAM (e.g. hipBLAS SAXPY returns the
  // unmodified y). A tree update inverted this from the original SKIP-gated
  // default-on to an enable-gated default-off; restore default-on.
  if (!EnvEnabled("ROCR_MACOS_AQL_SKIP_POST_ACQUIRE")) {
    AcquireMemGfx10(pm4);
  }
  status = SubmitPm4AndWait(pm4);
  // Copy back writable raw-host-pointer kernargs from their VRAM shadows. Opt-in
  // (ROCR_MACOS_AQL_ENABLE_HOST_COPYBACK) because a too-broad writable range can
  // clobber host memory adjacent to a small kernarg object; copy_back_size is
  // clamped to the Mach-reported writable region.
  if (status == HSA_STATUS_SUCCESS && EnvEnabled("ROCR_MACOS_AQL_ENABLE_HOST_COPYBACK") &&
      !EnvEnabled("ROCR_MACOS_AQL_SKIP_HOST_COPYBACK")) {
    for (const HostPointerStage& stage : host_pointer_stages) {
      if (stage.copy_back_size == 0) continue;
      CopyFromBar(reinterpret_cast<void*>(static_cast<uintptr_t>(stage.host)),
                  stage.staged_cpu, stage.copy_back_size);
    }
  }
  if (status == HSA_STATUS_SUCCESS) CompleteSignal(packet.completion_signal, 0);
  return status;
}

hsa_status_t MacAqlQueue::SubmitBarrier(const hsa_barrier_and_packet_t& packet, bool is_or) {
  bool satisfied_any = false;
  for (;;) {
    bool all = true;
    satisfied_any = false;
    for (const auto& dep : packet.dep_signal) {
      if (dep.handle == 0) continue;
      const bool satisfied = SignalSatisfied(dep);
      satisfied_any = satisfied_any || satisfied;
      all = all && satisfied;
    }
    if ((!is_or && all) || (is_or && satisfied_any)) break;
    std::this_thread::yield();
  }
  CompleteSignal(packet.completion_signal, 0);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacAqlQueue::SubmitPm4AndWait(const std::vector<uint32_t>& input_pm4) {
  std::vector<uint32_t> pm4 = input_pm4;
  const bool rptr_only = EnvEnabled("ROCR_MACOS_AQL_RPTR_ONLY");
  if (!rptr_only) {
    marker_value_++;
    if (marker_value_ == 0) marker_value_ = 1;
    *marker_cpu_ = 0;
    WriteData(pm4, marker_gpu_, marker_value_);
  } else if (pm4.empty()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (TraceAql()) {
    std::fprintf(stderr,
                 "ROCR macOS PM4 submit queue=%u dwords=%zu marker_gpu=0x%llx "
                 "value=%u completion=%s\n",
                 direct_queue_.queue_id, pm4.size(),
                 static_cast<unsigned long long>(marker_gpu_), marker_value_,
                 rptr_only ? "rptr-only" : "write-data-marker");
  }

  hsa_status_t status = driver_.SubmitDirectCompute(direct_queue_, pm4.data(), pm4.size());
  if (status != HSA_STATUS_SUCCESS) return status;

  // ReadDirectComputeRptr returns the WRAPPED ring offset (CP_HQD_PQ_RPTR),
  // while direct_queue_.wptr is the MONOTONIC dword index. Comparing them
  // directly breaks at the first ring wrap: the wrapped rptr jumps backward
  // (e.g. 966 -> 69) when wptr crosses ring_dw (1024) while wptr keeps
  // counting, so the dispatch is never seen complete -> HSA_STATUS_ERROR. This
  // capped sustained dispatch at ~14 on the 4 KiB ring (the #21 macOS
  // multi-dispatch limit; the GPU/CP actually drains the ring fine). Fix:
  // reconstruct the monotonic rptr congruent to the wrapped value mod ring_dw
  // and <= mono_wptr before comparing.
  uint64_t ring_dw = direct_queue_.ring_size_bytes / sizeof(uint32_t);
  if (ring_dw == 0) ring_dw = 1;
  const uint64_t mono_wptr = direct_queue_.wptr;
  const uint32_t expected_rptr = static_cast<uint32_t>(mono_wptr);
  bool rptr_done = false;
  for (uint32_t i = 0; i < 50000; ++i) {
    uint32_t hw_rptr = 0;
    if (driver_.ReadDirectComputeRptr(direct_queue_, &hw_rptr) == HSA_STATUS_SUCCESS) {
      uint64_t mono_rptr = (mono_wptr - (mono_wptr % ring_dw)) + hw_rptr;
      if (mono_rptr > mono_wptr) mono_rptr -= ring_dw;
      if (mono_rptr >= mono_wptr) {
        rptr_done = true;
        if (rptr_only || *marker_cpu_ == marker_value_) {
          if (TraceAql()) {
            std::fprintf(stderr,
                         "ROCR macOS PM4 complete rptr=%u mono=%llu marker=%u "
                         "completion=%s\n",
                         hw_rptr, static_cast<unsigned long long>(mono_rptr),
                         marker_value_, rptr_only ? "rptr-only" : "write-data-marker");
          }
          return HSA_STATUS_SUCCESS;
        }
      }
    }
    ::usleep(100);
  }
  if (TraceAql()) {
    uint32_t rptr = 0;
    (void)driver_.ReadDirectComputeRptr(direct_queue_, &rptr);
    std::fprintf(stderr,
                 "ROCR macOS PM4 timeout current=%u expected=%u rptr_done=%u "
                 "marker=%u observed=%u\n",
                 rptr, expected_rptr, rptr_done ? 1u : 0u, marker_value_,
                 static_cast<uint32_t>(*marker_cpu_));
  }
  return HSA_STATUS_ERROR;
}

void MacAqlQueue::CompleteSignal(hsa_signal_t signal, hsa_signal_value_t value) {
  if (signal.handle == 0) return;
  core::Signal::Convert(signal)->StoreRelease(value);
}

void MacAqlQueue::ReportAsyncError(hsa_status_t status) {
  if (errors_callback_) errors_callback_(status, public_handle(), errors_data_);
}

hsa_status_t MacAqlQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  if (value == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
      *static_cast<hsa_agent_t*>(value) = agent_.public_handle();
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      *static_cast<uint64_t*>(value) = 0;
      return HSA_STATUS_SUCCESS;
    case HSA_QUEUE_INFO_USE_COUNT:
      *static_cast<uint32_t*>(value) = static_cast<uint32_t>(-1);
      return HSA_STATUS_SUCCESS;
    case HSA_QUEUE_INFO_HW_ID:
      *static_cast<uint32_t*>(value) = static_cast<uint32_t>(public_handle()->id);
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t MacAqlQueue::GetCUMasking(uint32_t, uint32_t*) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t MacAqlQueue::SetCUMasking(uint32_t, const uint32_t*) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

void MacAqlQueue::ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b, hsa_fence_scope_t,
                             hsa_fence_scope_t, hsa_signal_t* signal) {
  if (cmd_data == nullptr || cmd_size_b == 0 || (cmd_size_b % sizeof(uint32_t)) != 0) {
    if (signal) CompleteSignal(*signal, -1);
    return;
  }
  std::vector<uint32_t> pm4(cmd_data, cmd_data + cmd_size_b / sizeof(uint32_t));
  hsa_status_t status = SubmitPm4AndWait(pm4);
  if (signal) CompleteSignal(*signal, status == HSA_STATUS_SUCCESS ? 0 : -1);
}

}  // namespace AMD
}  // namespace rocr

#endif  // __APPLE__
