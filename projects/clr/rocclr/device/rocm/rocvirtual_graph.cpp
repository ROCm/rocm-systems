/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef HIP_GRAPH_DISPATCH_OPTIMIZED

#include "device/devhostcall.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/roccounters.hpp"
#include "platform/activity.hpp"
#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/command_utils.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <cinttypes>

#if defined(__AVX__)
#if defined(__MINGW64__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif

/**
 * HSA image object size in bytes (see HSA spec)
 */
#define HSA_IMAGE_OBJECT_SIZE 48

/**
 * HSA image object alignment in bytes (see HSA spec)
 */
#define HSA_IMAGE_OBJECT_ALIGNMENT 16

/**
 * HSA sampler object size in bytes (see HSA spec)
 */
#define HSA_SAMPLER_OBJECT_SIZE 32

/**
 * HSA sampler object alignment in bytes (see HSA spec)
 */
#define HSA_SAMPLER_OBJECT_ALIGNMENT 16

namespace amd::roc {
// (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) invalidates I, K and L1
// (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) invalidates L1, L2 and flushes
// L2

static constexpr uint16_t kInvalidAql = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

static constexpr uint16_t kBarrierPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kNopPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketAcquireHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketReleaseHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierVendorPacketHeader =
    (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierVendorPacketNopScopeHeader =
    (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
  return (v >> pos) & ((1 << width) - 1);
};

static inline void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
#if IS_WINDOWS
  std::atomic_ref<uint32_t> atomic_header(*packet);
  atomic_header.store(header | (rest << 16), std::memory_order_release);
#else
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
#endif
}

// ================================================================================================
#if IS_LINUX
__attribute__((optimize("unroll-all-loops"), always_inline)) static inline void nontemporalMemcpy(
    void* __restrict dst, const void* __restrict src, size_t size) {
#if defined(ATI_ARCH_X86)
#if defined(__AVX512F__)
  for (auto i = 0u; i != size / sizeof(__m512i); ++i) {
    _mm512_stream_si512(reinterpret_cast<__m512i* __restrict&>(dst)++,
                        *reinterpret_cast<const __m512i* __restrict&>(src)++);
  }
  size = size % sizeof(__m512i);
#endif

#if defined(__AVX__)
  for (auto i = 0u; i != size / sizeof(__m256i); ++i) {
    _mm256_stream_si256(reinterpret_cast<__m256i* __restrict&>(dst)++,
                        *reinterpret_cast<const __m256i* __restrict&>(src)++);
  }
  size = size % sizeof(__m256i);
#endif

  for (auto i = 0u; i != size / sizeof(__m128i); ++i) {
    _mm_stream_si128(reinterpret_cast<__m128i* __restrict&>(dst)++,
                     *(reinterpret_cast<const __m128i* __restrict&>(src)++));
  }
  size = size % sizeof(__m128i);

  for (auto i = 0u; i != size / sizeof(long long); ++i) {
    _mm_stream_si64(reinterpret_cast<long long* __restrict&>(dst)++,
                    *reinterpret_cast<const long long* __restrict&>(src)++);
  }
  size = size % sizeof(long long);

  for (auto i = 0u; i != size / sizeof(int); ++i) {
    _mm_stream_si32(reinterpret_cast<int* __restrict&>(dst)++,
                    *reinterpret_cast<const int* __restrict&>(src)++);
  }

  size = size % sizeof(int);
  // Copy remaining bytes for unaligned size
  std::memcpy(dst, src, size);

  // Add memory fence
  _mm_sfence();
#else
  std::memcpy(dst, src, size);
#endif
}
#else
static inline void nontemporalMemcpy(void* __restrict dst, const void* __restrict src,
                                     size_t size) {
  std::memcpy(dst, src, size);
}
#endif

// ================================================================================================
namespace {
struct CompletionCallbackCtx {
  void (*callback)(void*);
  void* user_data;
  VirtualGPU* gpu;
};

bool GraphCompletionHandler(hsa_signal_value_t, void* arg) {
  auto* ctx = reinterpret_cast<CompletionCallbackCtx*>(arg);
  ctx->callback(ctx->user_data);
  ctx->gpu->QueuedAsyncHandlers()--;
  ctx->gpu->release();
  delete ctx;
  return false;
}
}  // namespace

// ================================================================================================
bool VirtualGPU::EmitCompletionCallback(void* hw_event,
                                        void (*callback)(void*),
                                        void* user_data) {
  auto* ps = reinterpret_cast<ProfilingSignal*>(hw_event);
  if (ps == nullptr || gpu_queue_ == nullptr) {
    return false;
  }

  // If the previous async handler hasn't fired yet (signal still > 0),
  // the signal is in use — caller should fall back to the Marker path.
  if (Hsa::signal_load_relaxed(ps->signal_) > 0) {
    return false;
  }

  Hsa::signal_store_relaxed(ps->signal_, kInitSignalValueOne);
  {
    std::scoped_lock lock(execution());
    dispatchBarrierPacket(kBarrierPacketHeader, true, ps->signal_);
  }

  auto* ctx = new CompletionCallbackCtx{callback, user_data, this};
  QueuedAsyncHandlers()++;
  retain();
  hsa_status_t result = Hsa::signal_async_handler(
      ps->signal_, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne, &GraphCompletionHandler, ctx);
  if (HSA_STATUS_SUCCESS != result) {
    QueuedAsyncHandlers()--;
    release();
    delete ctx;
    return false;
  }
  return true;
}

// ================================================================================================
bool VirtualGPU::dispatchAqlPacketBatchFlat(const std::vector<uint8_t>& flatPacketData,
                                            const std::vector<uint32_t>& validFullHeaders,
                                            amd::AccumulateCommand* vcmd, bool attach_signal,
                                            const std::vector<const std::string*>* kernelNames,
                                            bool pre_patched, bool blocking,
                                            bool execution_locked, bool skip_profiling) {
  if (vcmd == nullptr || flatPacketData.empty() || validFullHeaders.empty()) {
    return false;
  }

  const size_t numPackets = validFullHeaders.size();
  if (flatPacketData.size() != numPackets * 64) {
    return false;
  }

  std::unique_lock<std::recursive_mutex> lock(execution(), std::defer_lock);
  if (!execution_locked) {
    lock.lock();
  }

  if (!skip_profiling) {
    profilingBegin(*vcmd);
    dispatchBlockingWait(nullptr);
  } else {
    command_ = vcmd;
    Barriers().ClearExternalSignals();
  }

  if (kernelNames != nullptr) {
    vcmd->setKernelNamesRef(kernelNames);
  }

  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;
  const uint32_t sw_queue_size = queueMask;
  static constexpr size_t kPacketSize = sizeof(hsa_kernel_dispatch_packet_t);

  // Unpack first/last headers; apply system-scope once for the whole batch.
  // validFullHeaders stores the AQL full_header dword: low 16 = header, high 16 = setup.
  uint16_t firstHeader = static_cast<uint16_t>(validFullHeaders[0]);
  uint16_t firstSetup  = static_cast<uint16_t>(validFullHeaders[0] >> 16);
  uint16_t lastHeader  = static_cast<uint16_t>(validFullHeaders[numPackets - 1]);
  if (addSystemScope_) {
    firstHeader &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    firstHeader |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    lastHeader &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    lastHeader |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    addSystemScope_ = false;
  }

  uint8_t* queueBase = static_cast<uint8_t*>(gpu_queue_->base_address);

  // Reserve ALL slots with a single wptr bump, then submit in kPeriod-sized chunks.
  // Per-chunk: yield if the queue is full (handles graphs larger than the queue), then
  // memcpy + per-packet fixups + headers + doorbell.  For graphs that fit in the queue
  // the yield never fires.
  uint64_t startIndex = Hsa::queue_add_write_index_screlease(gpu_queue_, numPackets);
  setFenceDirty(true);

  // Update cached fence state from the last packet's release scope.
  // Clear fence dirty if the last packet has system-scope release, matching
  // the single-dispatch path in dispatchGenericAqlPacket (set dirty on reserve,
  // then conditionally clear if system scope).
  auto expected_fence_state =
      extractAqlBits(lastHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                     HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
  if (expected_fence_state == amd::Device::kCacheStateSystem) {
    setFenceDirty(false);
  }
  fence_state_ = static_cast<Device::CacheState>(expected_fence_state);

  const size_t kPeriod = DEBUG_HIP_GRAPH_BATCH_SIZE;
  auto* first_loc = reinterpret_cast<uint32_t*>(
      queueBase + (startIndex & queueMask) * kPacketSize);

  for (size_t chunkStart = 0; chunkStart < numPackets; ) {
    const size_t chunkEnd  = std::min(chunkStart + kPeriod, numPackets);
    const size_t thisChunk = chunkEnd - chunkStart;
    const bool isFirstChunk = (chunkStart == 0);
    const bool isLastChunk  = (chunkEnd == numPackets);

    // Yield until this chunk's physical slots are free.
    while (((startIndex + chunkEnd - 1) - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >=
           sw_queue_size) {
      amd::Os::yield();
    }

    // Copy this chunk's packet bodies to the queue. Handles RB wrap-around.
    const size_t chunkSlot = (startIndex + chunkStart) & queueMask;
    const uint8_t* srcData = flatPacketData.data() + chunkStart * kPacketSize;
    if (chunkSlot + thisChunk <= queueSize) {
      memcpy(queueBase + chunkSlot * kPacketSize, srcData, thisChunk * kPacketSize);
    } else {
      const size_t firstCount = queueSize - chunkSlot;
      memcpy(queueBase + chunkSlot * kPacketSize, srcData, firstCount * kPacketSize);
      memcpy(queueBase, srcData + firstCount * kPacketSize,
             (thisChunk - firstCount) * kPacketSize);
    }

    // Per-packet fixups: profiling signals and kernel-name printing.
    if (timestamp_ != nullptr || IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2)) {
      for (size_t i = chunkStart; i < chunkEnd; ++i) {
        const uint64_t slotIdx = (startIndex + i) & queueMask;
        auto* slot = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
            queueBase + slotIdx * kPacketSize);
        const uint16_t hdr = static_cast<uint16_t>(validFullHeaders[i]);
        const uint8_t pktType =
            extractAqlBits(hdr, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
        if (timestamp_ != nullptr) {
          // When pre_patched, skip any slot whose completion_signal was already
          // written by ApplyHwEventPatches (non-zero means pre-patched).
          bool has_prepatched_signal = pre_patched && (slot->completion_signal.handle != 0);
          if (!has_prepatched_signal) {
            slot->completion_signal =
                Barriers().ActiveSignal(kInitSignalValueOne, timestamp_, true);
            if (pktType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
              if (amd::activity_prof::IsEnabled(OP_ID_DISPATCH)) {
                slot->reserved2 = timestamp_->command().profilingInfo().correlation_id_;
              }
              Barriers().GetLastSignal()->flags_.isPacketDispatch_ = true;
            }
          }
        }
        if (kernelNames != nullptr && i < kernelNames->size() &&
            pktType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2, "Graph ShaderName : %s, device id : %u",
                  (*kernelNames)[i]->c_str(), dev().index());
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL,
                  "SWq=0x%zx, HWq=0x%zx, id=%d, Dispatch Header = "
                  "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
                  "setup=%d, grid=[%u, %u, %u], workgroup=[%u, %u, %u], "
                  "private_seg_size=%u, group_seg_size=%u, kernel_obj=0x%zx, "
                  "kernarg_address=0x%zx, completion_signal=0x%zx, correlation_id=%zu, "
                  "rptr=%u, wptr=%u",
                  gpu_queue_, gpu_queue_->base_address, gpu_queue_->id, hdr, pktType,
                  extractAqlBits(hdr, HSA_PACKET_HEADER_BARRIER,
                                 HSA_PACKET_HEADER_WIDTH_BARRIER),
                  extractAqlBits(hdr, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                                 HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
                  extractAqlBits(hdr, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                 HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
                  slot->setup,
                  slot->grid_size_x, slot->grid_size_y, slot->grid_size_z,
                  slot->workgroup_size_x, slot->workgroup_size_y, slot->workgroup_size_z,
                  slot->private_segment_size, slot->group_segment_size,
                  slot->kernel_object, slot->kernarg_address,
                  slot->completion_signal, slot->reserved2,
                  Hsa::queue_load_read_index_scacquire(gpu_queue_), slotIdx);
        }
      }
    }

    auto* lastSlotPtr = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
        queueBase + ((startIndex + chunkEnd - 1) & queueMask) * kPacketSize);
    if (isLastChunk && (attach_signal || blocking) && timestamp_ == nullptr) {
      lastSlotPtr->completion_signal = Barriers().ActiveSignal();
    }

    // Write valid headers and ring the doorbell for this chunk.
    // Hold global packet-0's header back in the first chunk so the GPU sees a
    // fully-committed batch before starting (AQL protocol).  Subsequent chunks
    // write in forward order — packet 0 is already committed.
    for (size_t i = (isFirstChunk ? 1 : chunkStart); i < chunkEnd; ++i) {
      const uint64_t idx = startIndex + i;
      auto* aql_loc =
          reinterpret_cast<uint32_t*>(queueBase + (idx & queueMask) * kPacketSize);
      const uint32_t dword = validFullHeaders[i];
      const uint16_t hdr = (i == numPackets - 1) ? lastHeader : static_cast<uint16_t>(dword);
      packet_store_release(aql_loc, hdr, static_cast<uint16_t>(dword >> 16));
    }
    if (isFirstChunk) {
      packet_store_release(first_loc, firstHeader, firstSetup);
    }
    Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, startIndex + chunkEnd - 1);

    chunkStart = chunkEnd;
  }

  hasPendingDispatch_ = true;

  auto* finalLastSlot = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
      queueBase + ((startIndex + numPackets - 1) & queueMask) * kPacketSize);

  // Skip the pending dispatch only when both conditions are met: a completion
  // signal tracks the last packet and the fence is already clean (system scope).
  if (finalLastSlot->completion_signal.handle != 0 && !isFenceDirty()) {
    hasPendingDispatch_ = false;
  }

  TrackQueueProgress(*finalLastSlot, startIndex + numPackets - 1, pre_patched);

  if (blocking) {
    LogInfo("Running serialized as blocking is requested");
    if (!Barriers().WaitCurrent()) {
      LogPrintfError("Failed blocking queue wait with signal [0x%lx]",
                     finalLastSlot->completion_signal.handle);
      if (!skip_profiling) { profilingEnd(); } else { command_ = nullptr; }
      return false;
    }
  }

  if (!skip_profiling) { profilingEnd(); } else { command_ = nullptr; }
  return true;
}

// ================================================================================================
bool VirtualGPU::submitKernelInternal(const amd::NDRangeContainer& sizes, const amd::Kernel& kernel,
                                      const_address parameters, void* event_handle,
                                      uint32_t sharedMemBytes, amd::NDRangeKernelCommand* vcmd,
                                      hsa_kernel_dispatch_packet_t* aql_packet,
                                      bool attach_signal) {
  device::Kernel* devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(dev()));
  Kernel& gpuKernel = static_cast<Kernel&>(*devKernel);
  size_t ldsUsage = gpuKernel.WorkgroupGroupSegmentByteSize();
  bool imageBufferWrtBack = false;                  // Image buffer write back is required
  std::vector<device::Memory*> wrtBackImageBuffer;  // Array of images for write back

  bool isGraphCapture = command_ != nullptr && command_->getPktCapturingState();

  bool printfEnabled = false;
  if (!isGraphCapture) {
    // Check memory dependency and SVM objects
    bool coopGroups = (vcmd != nullptr) ? vcmd->cooperativeGroups() : false;
    if (!processMemObjects(kernel, parameters, ldsUsage, coopGroups, imageBufferWrtBack,
                           wrtBackImageBuffer)) {
      LogError("Wrong memory objects!");
      return false;
    }

    // Init PrintfDbg object if printf is enabled.
    printfEnabled = (gpuKernel.printfInfo().size() > 0) ? true : false;
    if (!printfDbg()->init(printfEnabled)) {
      LogError("\nPrintfDbg object initialization failed!");
      return false;
    }
  }

  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  ClPrint(amd::LOG_INFO, amd::LOG_KERN2, "ShaderName : %s", gpuKernel.getDemangledName().c_str());

  amd::NDRange local_size(sizes.local());
  address hidden_arguments = const_cast<address>(parameters);
  // Calculate local size if it wasn't provided
  devKernel->FindLocalWorkSize(sizes.dimensions(), sizes.global(), local_size);

  uint16_t local[3] = {1, 1, 1};
  uint32_t global[3] = {1, 1, 1};
  for (uint i = 0; i < sizes.dimensions(); i++) {
    global[i] = static_cast<uint32_t>(sizes.global()[i]);
    local[i] = static_cast<uint16_t>(local_size[i]);
  }
  uint64_t spVA = 0;
  // Check if runtime has to setup hidden arguments
  for (uint32_t i = signature.numParameters(); i < signature.numParametersAll(); ++i) {
    const auto& it = signature.at(i);
    switch (it.info_.oclObject_) {
      case amd::KernelParameterDescriptor::HiddenNone:
        break;
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetX: {
        WriteAqlArgAt(hidden_arguments, sizes.offset()[0], it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetY: {
        if (sizes.dimensions() >= 2) {
          WriteAqlArgAt(hidden_arguments, sizes.offset()[1], it.size_, it.offset_);
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ: {
        if (sizes.dimensions() >= 3) {
          WriteAqlArgAt(hidden_arguments, sizes.offset()[2], it.size_, it.offset_);
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenPrintfBuffer: {
        uintptr_t bufferPtr = reinterpret_cast<uintptr_t>(printfDbg()->dbgBuffer());
        if (printfEnabled && bufferPtr) {
          WriteAqlArgAt(hidden_arguments, bufferPtr, it.size_, it.offset_);
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenHostcallBuffer: {
        if (amd::IS_HIP) {
          if (dev().info().pcie_atomics_) {
            uintptr_t buffer = reinterpret_cast<uintptr_t>(getOrCreateHostcallBuffer());
            if (!buffer) {
              LogError("Kernel expects a hostcall buffer, but none found");
              return false;
            }
            WriteAqlArgAt(hidden_arguments, buffer, it.size_, it.offset_);
          } else {
            LogError("Pcie atomics not enabled, hostcall not supported");
            return false;
          }
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenDefaultQueue: {
        uint64_t vqVA = 0;
        amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(dev());
        if (nullptr != defQueue && devKernel->dynamicParallelism()) {
          if (!createVirtualQueue(defQueue->size()) || !createSchedulerParam()) {
            return false;
          }
          vqVA = getVQVirtualAddress();
        }
        WriteAqlArgAt(hidden_arguments, vqVA, it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenCompletionAction: {
        if (devKernel->dynamicParallelism()) {
          auto params = allocKernArg(sizeof(AmdAqlWrap), 64);
          AmdAqlWrap* wrap = reinterpret_cast<AmdAqlWrap*>(params);
          memset(wrap, 0, sizeof(AmdAqlWrap));
          wrap->state = AQL_WRAP_DONE;
          spVA = reinterpret_cast<uint64_t>(wrap);
        }
        WriteAqlArgAt(hidden_arguments, spVA, it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenMultiGridSync: {
        bool multiGridSync = (vcmd != nullptr) ? vcmd->cooperativeMultiDeviceGroups() : false;
        bool singleGridSync = (vcmd != nullptr) ? vcmd->cooperativeGroups() : false;
        Device::MGSyncInfo* syncInfo = nullptr;
        if (multiGridSync) {
          // Find CPU pointer to the right sync info structure. It should be after MGSyncData
          syncInfo = reinterpret_cast<Device::MGSyncInfo*>(
              dev().MGSync() + Device::kMGInfoSizePerDevice * dev().index() +
              Device::kMGSyncDataSize);
          // Update sync data address. Use the offset adjustment to the right location
          syncInfo->mgs = reinterpret_cast<Device::MGSyncData*>(
              dev().MGSync() + Device::kMGInfoSizePerDevice * vcmd->firstDevice());
        } else if (singleGridSync) {
          syncInfo = reinterpret_cast<Device::MGSyncInfo*>(allocKernArg(Device::kSGInfoSize, 64));
          syncInfo->mgs = nullptr;
        }
        if (multiGridSync || singleGridSync) {
          // Update sync data address.
          syncInfo->sgs = {0};
          // Fill rest of sync info fields
          syncInfo->grid_id = vcmd->gridId();
          syncInfo->num_grids = vcmd->numGrids();
          syncInfo->prev_sum = vcmd->prevGridSum();
          syncInfo->all_sum = vcmd->allGridSum();
          syncInfo->num_wg = vcmd->numWorkgroups();
        }
        // Update GPU address for grid sync info. Use the offset adjustment for the right
        // location
        WriteAqlArgAt(hidden_arguments, reinterpret_cast<uint64_t>(syncInfo), it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenHeap:
        // Allocate hidden heap for HIP applications only
        if ((amd::IS_HIP) && (dev().HeapBuffer() == nullptr)) {
          const_cast<Device&>(dev()).HiddenHeapAlloc(*this);
        }
        if (dev().HeapBuffer() != nullptr) {
          // Initialize hidden heap buffer
          if (!isGraphCapture) {
            const_cast<Device&>(dev()).HiddenHeapInit(*this);
            if (!heap_init_fence_emitted_) {
              addSystemScope();
              heap_init_fence_emitted_ = true;
            }
          }
          // Add heap pointer to the code
          size_t heap_ptr = static_cast<size_t>(dev().HeapBuffer()->virtualAddress());
          WriteAqlArgAt(hidden_arguments, heap_ptr, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenBlockCountX:
        WriteAqlArgAt(hidden_arguments, global[0] / local[0], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenBlockCountY:
        WriteAqlArgAt(hidden_arguments, global[1] / local[1], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenBlockCountZ:
        WriteAqlArgAt(hidden_arguments, global[2] / local[2], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGroupSizeX:
        WriteAqlArgAt(hidden_arguments, local[0], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGroupSizeY:
        WriteAqlArgAt(hidden_arguments, local[1], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGroupSizeZ:
        WriteAqlArgAt(hidden_arguments, local[2], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenRemainderX:
        WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(global[0] % local[0]), it.size_,
                      it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenRemainderY:
        if (sizes.dimensions() >= 2) {
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(global[1] % local[1]), it.size_,
                        it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenRemainderZ:
        if (sizes.dimensions() >= 3) {
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(global[2] % local[2]), it.size_,
                        it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenGridDims:
        WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(sizes.dimensions()), it.size_,
                      it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenPrivateBase:
        WriteAqlArgAt(hidden_arguments,
                      reinterpret_cast<amd_queue_t*>(gpu_queue_)->private_segment_aperture_base_hi,
                      it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenSharedBase:
        WriteAqlArgAt(hidden_arguments,
                      reinterpret_cast<amd_queue_t*>(gpu_queue_)->group_segment_aperture_base_hi,
                      it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenQueuePtr:
        WriteAqlArgAt(hidden_arguments, gpu_queue_, it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenDynamicLdsSize:
        WriteAqlArgAt(hidden_arguments, sharedMemBytes, it.size_, it.offset_);
        break;
    }
  }
  address argBuffer = hidden_arguments;
  size_t argSize = std::min(gpuKernel.KernargSegmentByteSize(), signature.paramsSize());

  // Find all parameters for the current kernel
  if (!kernel.parameters().deviceKernelArgs() || gpuKernel.isInternalKernel()) {
    // Allocate buffer to hold kernel arguments
    if (isGraphCapture) {
      argBuffer = command_->getGraphKernArg(gpuKernel.KernargSegmentByteSize(),
                                            gpuKernel.KernargSegmentAlignment(), dev().index());
      command_->SetKernelName(gpuKernel.getDemangledName());
    } else {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
              "Kernel name = %s, argSize = %zu, "
              "KernargSegmentByteSize = %lu "
              "KernargSegmentAlignment = %lu",
              gpuKernel.getDemangledName().c_str(), argSize,
              gpuKernel.KernargSegmentByteSize(), gpuKernel.KernargSegmentAlignment());
      argBuffer = reinterpret_cast<address>(
          allocKernArg(gpuKernel.KernargSegmentByteSize(), gpuKernel.KernargSegmentAlignment()));
    }

    nontemporalMemcpy(argBuffer, parameters, argSize);
    if (roc_device_.info().largeBar_ && !isGraphCapture) {
      const auto kernArgImpl = dev().settings().kernel_arg_impl_;
      if (kernArgImpl == KernelArgImpl::DeviceKernelArgsHDP) {
        *dev().info().hdpMemFlushCntl = 1u;
        auto kSentinel = *reinterpret_cast<volatile int*>(dev().info().hdpMemFlushCntl);
      } else if (kernArgImpl == KernelArgImpl::DeviceKernelArgsReadback && argSize != 0) {
#if defined(ATI_ARCH_X86)
        _mm_sfence();
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
        *(argBuffer + argSize - 1) = *(parameters + argSize - 1);
#if defined(ATI_ARCH_X86)
        _mm_mfence();
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
        auto kSentinel = *reinterpret_cast<volatile unsigned char*>(argBuffer + argSize - 1);
      }
    }
  }
  // Check for group memory overflow
  //! @todo Check should be in HSA - here we should have at most an assert
  assert(dev().info().localMemSizePerCU_ > 0);
  if (ldsUsage > dev().info().localMemSizePerCU_) {
    LogError("No local memory available\n");
    return false;
  }

      // Initialize the dispatch Packet
    static_assert(sizeof(hsa_kernel_dispatch_packet_t)
                  == sizeof(hsa_amd_ext_kernel_dispatch_packet_t));

    union {
      hsa_kernel_dispatch_packet_t kernelDispatch;
      hsa_amd_ext_kernel_dispatch_packet_t extKernelDispatch;
    } dispatchPacketUnion;

    auto& dispatchPacket = dispatchPacketUnion.kernelDispatch;
    memset(&dispatchPacket, 0, sizeof(dispatchPacket));

    uint32_t newGlobalSize[3] = {global[0], global[1], global[2]};

    dispatchPacket.header = kInvalidAql;
    dispatchPacket.kernel_object = gpuKernel.KernelCodeHandle();

    // dispatchPacket.header = aqlHeader_;
    // dispatchPacket.setup |= sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    bool extDispatchPacket =
        (sizes.dimensions() > 0 && sizes.cluster()[0] > 1) ||
        (sizes.dimensions() > 1 && sizes.cluster()[1] > 1) ||
        (sizes.dimensions() > 2 && sizes.cluster()[2] > 1) ||
        dev().settings().ext_dispatch_packet_;

    if (extDispatchPacket) {
      auto& dispatchPacketExt = dispatchPacketUnion.extKernelDispatch;

      dispatchPacketExt.cluster_size_x = sizes.dimensions() > 0 ? sizes.cluster()[0] : 1;
      dispatchPacketExt.cluster_size_y = sizes.dimensions() > 1 ? sizes.cluster()[1] : 1;
      dispatchPacketExt.cluster_size_z = sizes.dimensions() > 2 ? sizes.cluster()[2] : 1;

      // Already validated in HIP Launch Params that newGlobalSize is perfectly divisible by local
      // and it is divisible by cluster size.
      dispatchPacketExt.cluster_count_x = sizes.dimensions() > 0
                                          ? (newGlobalSize[0] / local[0] / sizes.cluster()[0]) : 1;
      dispatchPacketExt.cluster_count_y = sizes.dimensions() > 1
                                          ? (newGlobalSize[1] / local[1] / sizes.cluster()[1]) : 1;
      dispatchPacketExt.cluster_count_z = sizes.dimensions() > 2
                                          ? (newGlobalSize[2] / local[2] / sizes.cluster()[2]) : 1;

    } else {
      dispatchPacket.grid_size_x = sizes.dimensions() > 0 ? newGlobalSize[0] : 1;
      dispatchPacket.grid_size_y = sizes.dimensions() > 1 ? newGlobalSize[1] : 1;
      dispatchPacket.grid_size_z = sizes.dimensions() > 2 ? newGlobalSize[2] : 1;
    }

    if (dev().settings().groupMemCarveout_) {
      uint8_t percent = devKernel->workGroupInfo()->groupMemCarveout_
          ? devKernel->workGroupInfo()->groupMemCarveout_
          : dev().GetGroupMemCarveout();
      auto& dispatchPacketExt = dispatchPacketUnion.extKernelDispatch;
      // Encodings [1, 127] represent a range from 0% (no group memory) to 100% (maximum
      // group memory)
      if (dev().isa().versionMajor() == 12 && dev().isa().versionMinor() == 5) {
        dispatchPacketExt.perf_hint.group_mem_carveout = 127;
      } else {
        dispatchPacketExt.perf_hint.group_mem_carveout = (percent + 1) * 1.26F;
      }
    }

    dispatchPacket.workgroup_size_x = sizes.dimensions() > 0 ? local[0] : 1;
    dispatchPacket.workgroup_size_y = sizes.dimensions() > 1 ? local[1] : 1;
    dispatchPacket.workgroup_size_z = sizes.dimensions() > 2 ? local[2] : 1;

    dispatchPacket.kernarg_address = argBuffer;
    dispatchPacket.group_segment_size = ldsUsage + sharedMemBytes;
    dispatchPacket.private_segment_size = devKernel->workGroupInfo()->privateMemSize_;

    if ((devKernel->workGroupInfo()->usedStackSize_ & 0x1) == 0x1) {
      dispatchPacket.private_segment_size =
              std::max<uint64_t>(dev().StackSize(), dispatchPacket.private_segment_size);
      if (dispatchPacket.private_segment_size > 16 * Ki) {
        dispatchPacket.private_segment_size = 16 * Ki;
      }
    }

    // Pass the header accordingly
    auto aqlHeaderWithOrder = aqlHeader_;

  if (vcmd != nullptr) {
    if (vcmd->getAnyOrderLaunchFlag()) {
      constexpr uint32_t kAqlHeaderMask = ~(1 << HSA_PACKET_HEADER_BARRIER);
      aqlHeaderWithOrder &= kAqlHeaderMask;
    }
    if (vcmd->getCommandEntryScope() == amd::Device::kCacheStateSystem) {
      addSystemScope_ = true;
    }
  }

  // Copy scheduler's AQL packet for possible relaunch from the scheduler itself
    if (aql_packet != nullptr) {
      *aql_packet = dispatchPacket;
      if (extDispatchPacket) {
        aql_packet->header = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                              (1 << HSA_PACKET_HEADER_BARRIER) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        aql_packet->setup = static_cast<uint8_t>(sizes.dimensions()
                              << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
      } else {
        aql_packet->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                              (1 << HSA_PACKET_HEADER_BARRIER) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        aql_packet->setup = static_cast<uint16_t>(sizes.dimensions()
                              << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
      }
    }


    uint16_t rest = 0;
    if (extDispatchPacket) {
      // When launching an AQL packet, the 32 bits has to be written atomically for CP to track,
      // on normal dispatch packet, first 32 bits are header & setup. In ext dispatch packet,
      // the first 32 bits are header, amd_format, setup. Update the "rest" of the 32 bits, so we
      // can commit it atomically in packet_store_release.
      rest = (HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH
              | ((sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) << 8));
    } else {
      rest = (sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
    }

    metadata_preloader_.PrepareDispatch(gpuKernel.MetadataKernelDescriptor(),
                                        gpuKernel.MetadataPreloadLength(),
                                        gpuKernel.MetadataPreloadOffset());

    if (isGraphCapture) {
      // Dispatch the packet
      if (!dispatchAqlPacket(&dispatchPacket, aqlHeaderWithOrder, rest,
                             GPU_FLUSH_ON_EXECUTION, command_->getPktCapturingState(),
                             command_->getAqlPacket())) {
        return false;
      }
    } else {
      if (!dispatchAqlPacket(&dispatchPacket, aqlHeaderWithOrder, rest,
                             GPU_FLUSH_ON_EXECUTION, false, nullptr, attach_signal)) {
        return false;
      }
    }

  if (!isGraphCapture) {
    // Output printf buffer
    if (!printfDbg()->output(*this, printfEnabled, gpuKernel.printfInfo())) {
      LogError("\nCould not print data from the printf buffer!");
      return false;
    }

    if (gpuKernel.dynamicParallelism()) {
      dispatchBarrierPacket(kBarrierPacketHeader, true);
      if (virtualQueue_ != nullptr) {
        static_cast<KernelBlitManager&>(blitMgr()).runScheduler(
            getVQVirtualAddress(), schedulerQueue_, schedulerThreads_, spVA);
      }
    }

    // Check if image buffer write back is required
    if (imageBufferWrtBack) {
      // Make sure the original kernel execution is done
      releaseGpuMemoryFence();
      for (const auto imageBuffer : wrtBackImageBuffer) {
        Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
        amd::Image* image = imageBuffer->owner()->asImage();
        Image* devImage = static_cast<Image*>(dev().getGpuMemory(imageBuffer->owner()));
        Memory* cpyImage = dev().getGpuMemory(devImage->CopyImageBuffer());
        amd::Coord3D offs(0);
        // Copy memory from the the backing store image into original buffer
        bool result = blitMgr().copyImageToBuffer(*cpyImage, *buffer, offs, offs, image->getRegion(),
                                                  true, image->getRowPitch(), image->getSlicePitch());
      }
    }
  }
  return true;
}

/**
 * @brief Api to dispatch a kernel for execution. The implementation
 * parses the input object, an instance of virtual command to obtain
 * the parameters of global size, work group size, offsets of work
 * items, enable/disable profiling, etc.
 *
 * It also parses the kernel arguments buffer to inject into Hsa Runtime
 * the list of kernel parameters.
 */
// ================================================================================================
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  if (vcmd.cooperativeGroups()) {
    // Wait for the execution on the current queue, since the coop groups will use the device queue
    releaseGpuMemoryFence(kSkipCpuWait);

    // Get device queue for exclusive GPU access
    VirtualGPU* queue = dev().xferQueue();
    if (!queue) {
      LogError("Runtime failed to acquire a cooperative queue!");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }

    // Lock the queue, using the blit manager lock
    std::scoped_lock k(*(queue->blitMgr().lockXfer()));

    queue->profilingBegin(vcmd);

    // Add a dependency into the device queue on the current queue
    queue->Barriers().AddExternalSignal(Barriers().GetLastSignal());

    if (dev().settings().gwsInitSupported_ == true) {
      uint32_t workgroups = vcmd.numWorkgroups();
      static_cast<KernelBlitManager&>(queue->blitMgr()).RunGwsInit(workgroups - 1);
    }

    // Sync AQL packets
    queue->setAqlHeader(dispatchPacketHeader_);

    // Submit kernel to HW
    if (!queue->submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
                                     static_cast<void*>(as_cl(&vcmd.event())),
                                     vcmd.sharedMemBytes(), &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
    // Wait for the execution on the device queue. Keep the current queue in-order
    queue->releaseGpuMemoryFence(kSkipCpuWait);

    // Add a dependency into the current queue on the coop queue
    Barriers().AddExternalSignal(queue->Barriers().GetLastSignal());
    hasPendingDispatch_ = true;
    retainExternalSignals_ = true;

    queue->profilingEnd();
  } else if (vcmd.getPktCapturingState()) {
    // Lightweight capture path: no lock, no profiling, no queue acquisition.
    // profilingBegin acquires a HW queue — we need gpu_queue_ for hidden args,
    // so call it but skip the lock since capture only does a memcpy.
    profilingBegin(vcmd);

    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
                              static_cast<void*>(as_cl(&vcmd.event())), vcmd.sharedMemBytes(),
                              &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd();
  } else {
    // Make sure VirtualGPU has an exclusive access to the resources
    std::scoped_lock lock(execution());

    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
                              static_cast<void*>(as_cl(&vcmd.event())), vcmd.sharedMemBytes(),
                              &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd();
  }
}

}  // namespace amd::roc

#endif  // HIP_GRAPH_DISPATCH_OPTIMIZED
