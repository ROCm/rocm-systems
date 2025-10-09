/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

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

#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/kernel.hpp"
#include "platform/commandqueue.hpp"
#include "device/pal/palconstbuf.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/palkernel.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palcounters.hpp"
#include "device/pal/palthreadtrace.hpp"
#include "device/pal/paltimestamp.hpp"
#include "device/pal/palblit.hpp"
#include "device/appprofile.hpp"
#include "device/devhostcall.hpp"
#include "hsa.h"
#include "amd_hsa_kernel_code.h"
#include "amd_hsa_queue.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
#include "palQueue.h"
#include "palFence.h"
#include "palQueueSemaphore.h"

#ifdef _WIN32
#include <d3d10_1.h>
#include "platform/interop_d3d9.hpp"
#include "platform/interop_d3d10.hpp"
#include "platform/interop_d3d11.hpp"
#endif  // _WIN32

namespace amd::pal {

uint32_t VirtualGPU::Queue::AllocedQueues(const VirtualGPU& gpu, Pal::EngineType type) {
  uint32_t allocedQueues = 0;
  for (const auto& queue : gpu.dev().QueuePool()) {
    allocedQueues += (queue.second->engineType_ == type) ? 1 : 0;
  }
  return allocedQueues;
}

// ================================================================================================
VirtualGPU::Queue* VirtualGPU::Queue::Create(VirtualGPU& gpu, Pal::QueueType queueType,
                                             uint engineIdx, Pal::ICmdAllocator* cmdAllocator,
                                             uint rtCU, amd::CommandQueue::Priority priority,
                                             uint64_t residency_limit, uint max_command_buffers) {
  Pal::IDevice* palDev = gpu.dev().iDev();
  Pal::Result result;
  Pal::CmdBufferCreateInfo cmdCreateInfo = {};
  Pal::QueueCreateInfo qCreateInfo = {};
  qCreateInfo.engineIndex =
      (queueType == Pal::QueueTypeCompute) ? gpu.dev().computeEnginesId()[engineIdx] : engineIdx;
  qCreateInfo.aqlQueue = true;
  qCreateInfo.queueType = queueType;
  qCreateInfo.priority = Pal::QueuePriority::Normal;

  if (queueType == Pal::QueueTypeDma) {
    cmdCreateInfo.engineType = qCreateInfo.engineType = Pal::EngineTypeDma;
  } else {
    cmdCreateInfo.engineType = qCreateInfo.engineType = Pal::EngineTypeCompute;
  }
  std::map<ExclusiveQueueType, uint32_t>::const_iterator it;
  if ((priority == amd::CommandQueue::Priority::Medium) &&
      (amd::CommandQueue::RealTimeDisabled == rtCU)) {
    it = gpu.dev().exclusiveComputeEnginesId().find(ExclusiveQueueType::Medium);
    cmdCreateInfo.engineType = qCreateInfo.engineType = Pal::EngineTypeCompute;
    qCreateInfo.priority = Pal::QueuePriority::Medium;
  } else if (amd::CommandQueue::RealTimeDisabled != rtCU) {
    if (gpu.dev().settings().enableWgpMode_) {
      rtCU = rtCU * 2;
    }
    qCreateInfo.numReservedCu = amd::alignDown(
        rtCU,
        gpu.dev().properties().engineProperties[Pal::EngineTypeCompute].dedicatedCuGranularity);
    if (qCreateInfo.numReservedCu == 0) {
      return nullptr;
    }
    if ((priority == amd::CommandQueue::Priority::Medium) &&
        // If Windows HWS is enabled, then the both real time queues are allocated
        // on the same engine
        (gpu.dev().exclusiveComputeEnginesId().find(ExclusiveQueueType::RealTime1) !=
         gpu.dev().exclusiveComputeEnginesId().end())) {
      it = gpu.dev().exclusiveComputeEnginesId().find(ExclusiveQueueType::RealTime1);
    } else {
      it = gpu.dev().exclusiveComputeEnginesId().find(ExclusiveQueueType::RealTime0);
    }
    cmdCreateInfo.engineType = qCreateInfo.engineType = Pal::EngineTypeCompute;
    cmdCreateInfo.flags.realtimeComputeUnits = true;
    qCreateInfo.priority = Pal::QueuePriority::Realtime;

    // If the app creates an exclusive compute, then find the engine id
    if (qCreateInfo.engineType == Pal::EngineTypeCompute) {
      if (it != gpu.dev().exclusiveComputeEnginesId().end()) {
        qCreateInfo.engineIndex = it->second;
      } else {
        return nullptr;
      }
    }
  }

  // Find queue object size
  size_t qSize = palDev->GetQueueSize(qCreateInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  cmdCreateInfo.pCmdAllocator = cmdAllocator;
  cmdCreateInfo.queueType = queueType;

  // Find command buffer object size
  size_t cmdSize = palDev->GetCmdBufferSize(cmdCreateInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  // Find fence object size
  size_t fSize = palDev->GetFenceSize(&result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  size_t allocSize = qSize + max_command_buffers * (cmdSize + fSize);
  VirtualGPU::Queue* queue =
      new (allocSize) VirtualGPU::Queue(gpu, palDev, residency_limit, max_command_buffers);
  if (queue != nullptr) {
    address addrQ = nullptr;
    if (((qCreateInfo.engineType == Pal::EngineTypeCompute) ||
         (qCreateInfo.engineType == Pal::EngineTypeDma)) &&
        (qCreateInfo.priority != Pal::QueuePriority::Realtime)) {
      uint32_t index = AllocedQueues(gpu, qCreateInfo.engineType);
      // Create PAL queue object
      if (index < GPU_MAX_HW_QUEUES) {
        Device::QueueRecycleInfo* info = new (qSize) Device::QueueRecycleInfo();
        if (info == nullptr) {
          LogError("Could not create QueueRecycleInfo!");
          return nullptr;
        }
        addrQ = reinterpret_cast<address>(&info[1]);
        qCreateInfo.aqlPacketList = info->AqlPacketList();
        result = palDev->CreateQueue(qCreateInfo, addrQ, &queue->iQueue_);
        if (result == Pal::Result::Success) {
          const_cast<Device&>(gpu.dev()).QueuePool().insert({queue->iQueue_, info});
          info->engineType_ = qCreateInfo.engineType;
          // Save uniqueue index for scratch buffer access
          info->index_ = index;
        } else {
          delete queue;
          return nullptr;
        }
      } else {
        int usage = std::numeric_limits<int>::max();
        uint indexBase = std::numeric_limits<uint32_t>::max();
        // Loop through all allocated queues and find the lowest usage
        for (const auto& it : gpu.dev().QueuePool()) {
          if ((qCreateInfo.engineType == it.second->engineType_) &&
              (it.second->counter_ <= usage)) {
            if ((it.second->counter_ < usage) ||
                // Preserve the order of allocations, because SDMA engines
                // should be used in round-robin manner
                ((it.second->counter_ == usage) && (it.second->index_ < indexBase))) {
              queue->iQueue_ = it.first;
              usage = it.second->counter_;
              indexBase = it.second->index_;
            }
          }
        }
        // Increment the usage of the current queue
        gpu.dev().QueuePool().find(queue->iQueue_)->second->counter_++;
      }
      Device::QueueRecycleInfo* info = gpu.dev().QueuePool().find(queue->iQueue_)->second;
      queue->aql_mgmt_ = &info->aql_packet_mgmt_;
      queue->lock_ = &info->queue_lock_;
      addrQ = reinterpret_cast<address>(&queue[1]);
    } else {
      Device::QueueRecycleInfo* info = new Device::QueueRecycleInfo();
      if (info == nullptr) {
        LogError("Could not create QueueRecycleInfo!");
        return nullptr;
      }
      queue->info_ = info;
      queue->aql_mgmt_ = &info->aql_packet_mgmt_;
      // Exclusive compute path
      addrQ = reinterpret_cast<address>(&queue[1]);
      qCreateInfo.aqlPacketList = info->AqlPacketList();
      result = palDev->CreateQueue(qCreateInfo, addrQ, &queue->iQueue_);
    }
    if (result != Pal::Result::Success) {
      delete queue;
      return nullptr;
    }
    queue->UpdateAppPowerProfile();
    address addrCmd = addrQ + qSize;
    address addrF = addrCmd + max_command_buffers * cmdSize;
    Pal::CmdBufferBuildInfo cmdBuildInfo = {};

    for (uint i = 0; i < max_command_buffers; ++i) {
      result = palDev->CreateCmdBuffer(cmdCreateInfo, &addrCmd[i * cmdSize], &queue->iCmdBuffs_[i]);
      if (result != Pal::Result::Success) {
        delete queue;
        return nullptr;
      }

      Pal::FenceCreateInfo fenceCreateinfo = {};
      fenceCreateinfo.flags.signaled = false;
      result = palDev->CreateFence(fenceCreateinfo, &addrF[i * fSize], &queue->iCmdFences_[i]);
      if (result != Pal::Result::Success) {
        delete queue;
        return nullptr;
      }
      if (i == StartCmdBufIdx) {
        result = queue->iCmdBuffs_[i]->Begin(cmdBuildInfo);
        if (result != Pal::Result::Success) {
          delete queue;
          return nullptr;
        }
      }
    }
  }
  return queue;
}

VirtualGPU::Queue::~Queue() {
  delete reinterpret_cast<Device::QueueRecycleInfo*>(info_);

  if (nullptr != iQueue_) {
    // Make sure the queues are idle
    // It's unclear why PAL could still have a busy queue
    amd::ScopedLock l(lock_);
    iQueue_->WaitIdle();
  }

  // Remove all memory references
  std::vector<Pal::IGpuMemory*> memRef;
  for (auto it : memReferences_) {
    memRef.push_back(it.first->iMem());
  }
  if (memRef.size() != 0) {
    iDev_->RemoveGpuMemoryReferences(memRef.size(), &memRef[0], iQueue_);
  }
  memReferences_.clear();

  for (uint i = 0; i < max_command_buffers_; ++i) {
    if (nullptr != iCmdBuffs_[i]) {
      iCmdBuffs_[i]->Destroy();
    }
    if (nullptr != iCmdFences_[i]) {
      iCmdFences_[i]->Destroy();
    }
  }

  if (nullptr != iQueue_) {
    // Find if this queue was used in recycling
    if (lock_ != nullptr) {
      // Release the queue if the counter is 0
      if (--gpu_.dev().QueuePool().find(iQueue_)->second->counter_ == 0) {
        iQueue_->Destroy();
        const auto& info = gpu_.dev().QueuePool().find(iQueue_);
        // Readjust HW queue index for scratch buffer access
        for (auto& queue : gpu_.dev().QueuePool()) {
          if ((queue.second->engineType_ == info->second->engineType_) &&
              (queue.second->index_ > info->second->index_)) {
            queue.second->index_--;
          }
        }
        delete gpu_.dev().QueuePool().find(iQueue_)->second;
        const_cast<Device&>(gpu_.dev()).QueuePool().erase(iQueue_);
      }
    } else {
      iQueue_->Destroy();
    }
  }
}

Pal::Result VirtualGPU::Queue::UpdateAppPowerProfile() {
  std::wstring wsAppPathAndFileName = Device::appProfile()->wsAppPathAndFileName();

  const wchar_t* wAppPathAndName = wsAppPathAndFileName.c_str();
  // Find the last occurance of the '\\' character and extract the name of the application as wide
  // char.
  const wchar_t* wAppNamePtr = wcsrchr(wAppPathAndName, '\\');
  const wchar_t* wAppName = wAppNamePtr ? wAppNamePtr + 1 : wAppPathAndName;

  return iQueue_->UpdateAppPowerProfile(wAppName, wAppPathAndName);
}

void VirtualGPU::Queue::addCmdMemRef(GpuMemoryReference* mem) {
  if (gpu_.dev().settings().alwaysResident_) {
    return;
  }
  Pal::IGpuMemory* iMem = mem->iMem();
  auto it = memReferences_.find(mem);
  if (it != memReferences_.end()) {
    it->second = cmdBufIdSlot_;
  } else {
    // Update runtime tracking with TS
    memReferences_[mem] = cmdBufIdSlot_;
    // Update PAL list with the new entry
    Pal::GpuMemoryRef memRef = {};
    memRef.pGpuMemory = iMem;
    palMemRefs_.push_back(memRef);
    // Check SDI memory object
    if (iMem->Desc().flags.isExternPhys && (sdiReferences_.find(iMem) == sdiReferences_.end())) {
      sdiReferences_.insert(iMem);
      palSdiRefs_.push_back(iMem);
    }
    residency_size_ += iMem->Desc().size;
  }
}

void VirtualGPU::Queue::removeCmdMemRef(GpuMemoryReference* mem) {
  Pal::IGpuMemory* iMem = mem->iMem();
  if (0 != memReferences_.erase(mem)) {
    iDev_->RemoveGpuMemoryReferences(1, &iMem, iQueue_);
    residency_size_ -= iMem->Desc().size;
  }
}

void VirtualGPU::Queue::addCmdDoppRef(Pal::IGpuMemory* iMem, bool lastDoppCmd, bool pfpaDoppCmd) {
  for (size_t i = 0; i < palDoppRefs_.size(); i++) {
    if (palDoppRefs_[i].pGpuMemory == iMem) {
      // If both LAST_DOPP_SUBMISSION and PFPA_DOPP_SUBMISSION VCOPs are requested,
      // the LAST_DOPP_SUBMISSION is send as requsted by KMD
      //
      if (palDoppRefs_[i].flags.lastPfpaCmd == 1) {
        return;  // no need to override the last submission command
      }

      if (lastDoppCmd) {
        palDoppRefs_[i].flags.lastPfpaCmd = 1;
        palDoppRefs_[i].flags.pfpa = 0;
      } else if (pfpaDoppCmd) {
        palDoppRefs_[i].flags.pfpa = 1;
      }
      return;
    }
  }

  //  this is the first reference of the DOPP desktop texture, add it in the vector
  Pal::DoppRef doppRef = {};
  doppRef.flags.pfpa = pfpaDoppCmd ? 1 : 0;
  doppRef.flags.lastPfpaCmd = lastDoppCmd ? 1 : 0;
  doppRef.pGpuMemory = iMem;
  palDoppRefs_.push_back(doppRef);
}

// ================================================================================================
bool VirtualGPU::Queue::flush() {
  amd::ScopedLock l(lock_);
  const Settings& settings = gpu_.dev().settings();

  if (!settings.alwaysResident_ && palMemRefs_.size() != 0) {
    Pal::Result result = iDev_->AddGpuMemoryReferences(palMemRefs_.size(), &palMemRefs_[0], iQueue_,
                                                       Pal::GpuMemoryRefCantTrim);
    if (Pal::Result::Success != result) {
      LogPrintfError("PAL failed to make resident resources! result: %d", result);
      return false;
    }
    palMemRefs_.clear();
  }

  // Stop commands building
  Pal::Result result;
  result = iCmdBuffs_[cmdBufIdSlot_]->End();
  if (Pal::Result::Success != result) {
    LogPrintfError("PAL failed to finalize a command buffer! result: %d", result);
    return false;
  }

  // Reset the fence. PAL will reset OS event
  result = iDev_->ResetFences(1, &iCmdFences_[cmdBufIdSlot_]);
  if (Pal::Result::Success != result) {
    LogPrintfError("PAL failed to reset a fence! result:%d", result);
    return false;
  }

  Pal::PerSubQueueSubmitInfo perSubQueueSubmitInfo = {};
  perSubQueueSubmitInfo.cmdBufferCount = 1;
  perSubQueueSubmitInfo.ppCmdBuffers = &iCmdBuffs_[cmdBufIdSlot_];

  Pal::MultiSubmitInfo submitInfo = {};
  submitInfo.perSubQueueInfoCount = 1;
  submitInfo.pPerSubQueueInfo = &perSubQueueSubmitInfo;

  submitInfo.doppRefCount = palDoppRefs_.size();
  submitInfo.pDoppRefs = palDoppRefs_.data();

  submitInfo.externPhysMemCount = palSdiRefs_.size();
  submitInfo.ppExternPhysMem = palSdiRefs_.data();

  submitInfo.fenceCount = 1;
  submitInfo.ppFences = &iCmdFences_[cmdBufIdSlot_];

  if (iQueue_->Type() == Pal::QueueTypeCompute) {
    if (gpu_.dev().settings().kernel_arg_impl_ == KernelArgImpl::DeviceKernelArgs) {
      // If runtime uses device memory for kernel arguments, then perform a CPU read back on
      // submission. That will make sure NBIO puches all previous CPU write requests through PCIE
      gpu_.managedBuffer().CpuReadBack();
    }
    if (amd::IS_HIP) {
      // HIP disables per resource tracking, because the app may embed SVM ptr into other buffers.
      // Force CPU sync if there are pending operations on SDMA, until OS fences will be added
      gpu_.WaitForIdleSdma();
    }
  }
  // Submit command buffer to OS
  if (gpu_.rgpCaptureEna()) {
    result = gpu_.dev().captureMgr()->TimedQueueSubmit(iQueue_, cmdBufIdCurrent_, submitInfo);
  } else {
    result = iQueue_->Submit(submitInfo);
  }
  if (Pal::Result::Success != result) {
    LogPrintfError("PAL failed to submit CMD! result:%d", result);
    if (GPU_ANALYZE_HANG) {
      DumpMemoryReferences();
    }
    return false;
  }
  // Make sure the slot isn't busy
  constexpr bool IbReuse = true;
  if (GPU_FLUSH_ON_EXECUTION) {
    waitForFence<!IbReuse>(cmdBufIdSlot_);
  }

  // Reset the counter of commands
  cmdCnt_ = 0;

  // Find the next command buffer
  cmdBufIdCurrent_++;

  if (cmdBufIdCurrent_ == GpuEvent::InvalidID) {
    // Wait for the last one
    waitForFence<!IbReuse>(cmdBufIdSlot_);
    cmdBufIdCurrent_ = 1;
    cmbBufIdRetired_ = 0;
  }

  // Wrap current slot
  cmdBufIdSlot_ = cmdBufIdCurrent_ % max_command_buffers_;

  waitForFence<IbReuse>(cmdBufIdSlot_);

  // Progress retired TS
  if ((cmdBufIdCurrent_ > max_command_buffers_) &&
      (cmbBufIdRetired_ < (cmdBufIdCurrent_ - max_command_buffers_))) {
    cmbBufIdRetired_ = cmdBufIdCurrent_ - max_command_buffers_;
  }

  // Reset command buffer, so CB chunks could be reused
  result = iCmdBuffs_[cmdBufIdSlot_]->Reset(nullptr, false);
  if (Pal::Result::Success != result) {
    LogPrintfError("PAL failed CB reset! result:%d", result);
    return false;
  }
  // Start command buffer building
  Pal::CmdBufferBuildInfo cmdBuildInfo = {};
  cmdBuildInfo.pMemAllocator = &vlAlloc_;
  result = iCmdBuffs_[cmdBufIdSlot_]->Begin(cmdBuildInfo);
  if (Pal::Result::Success != result) {
    LogPrintfError("PAL failed CB building initialization! result:%d", result);
    return false;
  }

  // Clear dopp references
  palDoppRefs_.clear();
  palSdiRefs_.clear();

  // Remove old memory references
  if ((memReferences_.size() > 2048) || (residency_size_ > residency_limit_)) {
    for (auto it = memReferences_.begin(); it != memReferences_.end();) {
      if (it->second == cmdBufIdSlot_) {
        palMems_.push_back(it->first->iMem());
        residency_size_ -= it->first->iMem()->Desc().size;
        it = memReferences_.erase(it);
      } else {
        ++it;
      }
    }
  }
  if (!settings.alwaysResident_ && palMems_.size() != 0) {
    iDev_->RemoveGpuMemoryReferences(palMems_.size(), &palMems_[0], iQueue_);
    palMems_.clear();
  }

  return true;
}

// ================================================================================================
bool VirtualGPU::Queue::waitForEvent(uint id) {
  amd::ScopedLock l(lock_);
  if (isDone(id)) {
    return true;
  }

  if (id == cmdBufIdCurrent_) {
    // There is an error in the flush() and wait is bogus
    return false;
  }

  uint slotId = id % max_command_buffers_;
  constexpr bool IbReuse = true;
  bool result = waitForFence<!IbReuse>(slotId);
  cmbBufIdRetired_ = id;
  return result;
}

// ================================================================================================
bool VirtualGPU::Queue::isDone(uint id) {
  amd::ScopedLock l(lock_);
  if ((id <= cmbBufIdRetired_) || (id > cmdBufIdCurrent_)) {
    return true;
  }

  if (id == cmdBufIdCurrent_) {
    // Flush the current command buffer
    if (!flush()) {
      // If flush failed, then exit earlier...
      return false;
    }
  }

  if (Pal::Result::Success != iCmdFences_[id % max_command_buffers_]->GetStatus()) {
    return false;
  }
  cmbBufIdRetired_ = id;
  return true;
}

// ================================================================================================
void VirtualGPU::Queue::DumpMemoryReferences() const {
  std::fstream dump;
  std::stringstream file_name("ocl_hang_dump.txt");
  uint64_t start = amd::Os::timeNanos() / 1e9;

  dump.open(file_name.str().c_str(), (std::fstream::out | std::fstream::app));
  // Check if we have OpenCL program
  if (dump.is_open()) {
    dump << start << " Queue: ";
    switch (iQueue_->Type()) {
      case Pal::QueueTypeCompute:
        dump << "Compute";
        break;
      case Pal::QueueTypeDma:
        dump << "SDMA";
        break;
      default:
        dump << "unknown";
        break;
    }
    dump << "\n"
         << "Resident memory resources:\n";
    uint idx = 0;
    for (auto it : memReferences_) {
      dump << " " << idx << "\t[";
      dump.setf(std::ios::hex, std::ios::basefield);
      dump.setf(std::ios::showbase);
      dump << (it.first)->iMem()->Desc().gpuVirtAddr << ", "
           << (it.first)->iMem()->Desc().gpuVirtAddr + (it.first)->iMem()->Desc().size;
      dump.setf(std::ios::dec);
      dump << "] CbId:" << it.second << ", Heap: " << (it.first)->iMem()->Desc().heaps[0] << "\n";
      idx++;
    }

    if (last_kernel_ != nullptr) {
      const amd::KernelSignature& signature = last_kernel_->signature();
      dump << last_kernel_->name() << std::endl;
      for (size_t i = 0; i < signature.numParameters(); ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        // Find if the current argument is a memory object
        if ((desc.type_ == T_POINTER) && (desc.addressQualifier_ != CL_KERNEL_ARG_ADDRESS_LOCAL)) {
          dump << " " << desc.name_ << ": " << std::endl;
        }
      }
    }
    dump.close();
  }
}

bool VirtualGPU::MemoryDependency::create(size_t numMemObj) {
  if (numMemObj > 0) {
    // Allocate the array of memory objects for dependency tracking
    memObjectsInQueue_ = new MemoryState[numMemObj];
    if (nullptr == memObjectsInQueue_) {
      return false;
    }
    memset(memObjectsInQueue_, 0, sizeof(MemoryState) * numMemObj);
    maxMemObjectsInQueue_ = numMemObj;
  }

  return true;
}

void VirtualGPU::MemoryDependency::validate(VirtualGPU& gpu, const Memory* memory, bool readOnly) {
  bool flushL1Cache = false;

  if (maxMemObjectsInQueue_ == 0) {
    // Return earlier if tracking is disabled
    return;
  }

  uint64_t curStart = memory->vmAddress();
  uint64_t curEnd = curStart + memory->size();

  if (memory->isModified(gpu) || !readOnly) {
    // Mark resource as modified
    memory->setModified(gpu, !readOnly);

    // Loop through all memory objects in the queue and find dependency
    // @note don't include objects from the current kernel
    for (size_t j = 0; j < endMemObjectsInQueue_; ++j) {
      // Check if the queue already contains this mem object and
      // GPU operations aren't readonly
      uint64_t busyStart = memObjectsInQueue_[j].start_;
      uint64_t busyEnd = memObjectsInQueue_[j].end_;

      // Check if the start inside the busy region
      if ((((curStart >= busyStart) && (curStart < busyEnd)) ||
           // Check if the end inside the busy region
           ((curEnd > busyStart) && (curEnd <= busyEnd)) ||
           // Check if the start/end cover the busy region
           ((curStart <= busyStart) && (curEnd >= busyEnd))) &&
          // If the buys region was written or the current one is for write
          (!memObjectsInQueue_[j].readOnly_ || !readOnly)) {
        flushL1Cache = true;
        break;
      }
    }
  }

  // Did we reach the limit?
  if (maxMemObjectsInQueue_ <= numMemObjectsInQueue_) {
    flushL1Cache = true;
  }

  if (flushL1Cache) {
    // Flush cache
    gpu.addBarrier(RgpSqqtBarrierReason::MemDependency);

    // Clear memory dependency state
    const static bool All = true;
    clear(!All);
  }

  // Insert current memory object into the queue always,
  // since runtime calls flush before kernel execution and it has to keep
  // current kernel in tracking
  memObjectsInQueue_[numMemObjectsInQueue_].start_ = curStart;
  memObjectsInQueue_[numMemObjectsInQueue_].end_ = curEnd;
  memObjectsInQueue_[numMemObjectsInQueue_].readOnly_ = readOnly;
  numMemObjectsInQueue_++;
}

void VirtualGPU::MemoryDependency::clear(bool all) {
  if (numMemObjectsInQueue_ > 0) {
    if (all) {
      endMemObjectsInQueue_ = numMemObjectsInQueue_;
    }

    // If the current launch didn't start from the beginning, then move the data
    if (0 != endMemObjectsInQueue_) {
      size_t i, j;
      // Preserve all objects from the current kernel
      for (i = 0, j = endMemObjectsInQueue_; j < numMemObjectsInQueue_; i++, j++) {
        memObjectsInQueue_[i].start_ = memObjectsInQueue_[j].start_;
        memObjectsInQueue_[i].end_ = memObjectsInQueue_[j].end_;
        memObjectsInQueue_[i].readOnly_ = memObjectsInQueue_[j].readOnly_;
      }
    } else if (numMemObjectsInQueue_ >= maxMemObjectsInQueue_) {
      // note: The array growth shouldn't occur under the normal conditions,
      // but in a case when SVM path sends the amount of SVM ptrs over
      // the max size of kernel arguments
      MemoryState* ptr = new MemoryState[maxMemObjectsInQueue_ << 1];
      if (nullptr == ptr) {
        numMemObjectsInQueue_ = 0;
        return;
      }
      maxMemObjectsInQueue_ <<= 1;
      memcpy(ptr, memObjectsInQueue_, sizeof(MemoryState) * numMemObjectsInQueue_);
      delete[] memObjectsInQueue_;
      memObjectsInQueue_ = ptr;
    }

    // Adjust the number of active objects
    numMemObjectsInQueue_ -= endMemObjectsInQueue_;
    endMemObjectsInQueue_ = 0;
  }
}

void VirtualGPU::addPinnedMem(amd::Memory* mem) {
  if (nullptr == findPinnedMem(mem->getHostMem(), mem->getSize())) {
    if (pinnedMems_.size() > 7) {
      pinnedMems_.front()->release();
      pinnedMems_.erase(pinnedMems_.begin());
    }

    // Start operation, since we should release mem object
    flushDMA(dev().getGpuMemory(mem)->getGpuEvent(*this)->engineId_);

    // Delay destruction
    pinnedMems_.push_back(mem);
  }
}

void VirtualGPU::releasePinnedMem() {
  for (auto& amdMemory : pinnedMems_) {
    amdMemory->release();
  }
  pinnedMems_.clear();
}

amd::Memory* VirtualGPU::findPinnedMem(void* addr, size_t size) {
  for (auto& amdMemory : pinnedMems_) {
    if ((amdMemory->getHostMem() == addr) && (size <= amdMemory->getSize())) {
      return amdMemory;
    }
  }
  return nullptr;
}

bool VirtualGPU::createVirtualQueue(uint deviceQueueSize) {
  uint MinDeviceQueueSize = 16 * 1024;
  deviceQueueSize = std::max(deviceQueueSize, MinDeviceQueueSize);

  maskGroups_ = deviceQueueSize / (512 * Ki);
  maskGroups_ = (maskGroups_ == 0) ? 1 : maskGroups_;

  // Align the queue size for the multiple dispatch scheduler.
  // Each thread works with 32 entries * maskGroups
  uint extra = deviceQueueSize % (sizeof(AmdAqlWrap) * DeviceQueueMaskSize * maskGroups_);
  if (extra != 0) {
    deviceQueueSize += (sizeof(AmdAqlWrap) * DeviceQueueMaskSize * maskGroups_) - extra;
  }

  if (deviceQueueSize_ == deviceQueueSize) {
    return true;
  } else {
    delete vqHeader_;
    delete virtualQueue_;
    vqHeader_ = nullptr;
    virtualQueue_ = nullptr;
    deviceQueueSize_ = 0;
  }
  uint numSlots = deviceQueueSize / sizeof(AmdAqlWrap);
  uint allocSize = deviceQueueSize;

  // Add the virtual queue header
  allocSize += sizeof(AmdVQueueHeader);
  allocSize = amd::alignUp(allocSize, sizeof(AmdAqlWrap));

  uint argOffs = allocSize;

  // Add the kernel arguments and wait events
  uint singleArgSize = amd::alignUp(
      dev().info().maxParameterSize_ + 64 + dev().settings().numWaitEvents_ * sizeof(uint64_t),
      sizeof(AmdAqlWrap));
  allocSize += singleArgSize * numSlots;

  uint eventsOffs = allocSize;
  // Add the device events
  allocSize += dev().settings().numDeviceEvents_ * sizeof(AmdEvent);

  uint eventMaskOffs = allocSize;
  // Add mask array for events
  allocSize += amd::alignUp(dev().settings().numDeviceEvents_, DeviceQueueMaskSize) / 8;

  uint slotMaskOffs = allocSize;
  // Add mask array for AmdAqlWrap slots
  allocSize += amd::alignUp(numSlots, DeviceQueueMaskSize) / 8;

  // Align size to 64 bytes for more efficient fill operation
  allocSize = amd::alignUp(allocSize, 8 * sizeof(uint64_t));

  virtualQueue_ = new Memory(dev(), allocSize);
  Resource::MemoryType type = (GPU_PRINT_CHILD_KERNEL == 0) ? Resource::Local : Resource::Remote;
  if ((virtualQueue_ == nullptr) || !virtualQueue_->create(type)) {
    return false;
  }

  if (GPU_PRINT_CHILD_KERNEL != 0) {
    address ptr = reinterpret_cast<address>(virtualQueue_->map(this, Resource::WriteOnly));
    if (nullptr == ptr) {
      return false;
    }
  }

  uint64_t pattern = 0;
  amd::Coord3D origin(0, 0, 0);
  amd::Coord3D region(virtualQueue_->size(), 0, 0);
  if (!dev().xferMgr().fillBuffer(*virtualQueue_, &pattern, sizeof(pattern), region, origin,
                                  region)) {
    return false;
  }

  uint64_t vaBase = virtualQueue_->vmAddress();
  AmdVQueueHeader header = {};
  // Initialize the virtual queue header
  header.aql_slot_num = numSlots;
  header.event_slot_num = dev().settings().numDeviceEvents_;
  header.event_slot_mask = vaBase + eventMaskOffs;
  header.event_slots = vaBase + eventsOffs;
  header.aql_slot_mask = vaBase + slotMaskOffs;
  header.wait_size = dev().settings().numWaitEvents_;
  header.arg_size = dev().info().maxParameterSize_ + 64;
  header.mask_groups = maskGroups_;

  vqHeader_ = new AmdVQueueHeader;
  if (nullptr == vqHeader_) {
    return false;
  }
  *vqHeader_ = header;

  virtualQueue_->writeRawData(*this, 0, sizeof(AmdVQueueHeader), &header, false);

  // Go over all slots and perform initialization
  AmdAqlWrap slot = {};
  size_t offset = sizeof(AmdVQueueHeader);
  for (uint i = 0; i < numSlots; ++i) {
    uint64_t argStart = vaBase + argOffs + i * singleArgSize;
    slot.aql.kernarg_address = reinterpret_cast<void*>(argStart);
    slot.wait_list = argStart + dev().info().maxParameterSize_ + 64;
    virtualQueue_->writeRawData(*this, offset, sizeof(AmdAqlWrap), &slot, false);
    offset += sizeof(AmdAqlWrap);
  }

  deviceQueueSize_ = deviceQueueSize;

  return true;
}

// ================================================================================================
VirtualGPU::VirtualGPU(Device& device)
    : device::VirtualDevice(device),
      engineID_(MainEngine),
      gpuDevice_(static_cast<Device&>(device)),
      printfDbgHSA_(nullptr),
      tsCache_(nullptr),
      managedBuffer_(*this, device.settings().stagedXferSize_ + 32 * Ki),
      writeBuffer_(device, managedBuffer_, device.settings().stagedXferSize_),
      hwRing_(0),
      readjustTimeGPU_(0),
      lastTS_(nullptr),
      profileTs_(nullptr),
      vqHeader_(nullptr),
      virtualQueue_(nullptr),
      schedParams_(nullptr),
      deviceQueueSize_(0),
      maskGroups_(1),
      hsaQueueMem_(nullptr),
      cmdAllocator_(nullptr) {
  // Note: Virtual GPU device creation must be a thread safe operation
  index_ = gpuDevice_.numOfVgpus_++;
  gpuDevice_.vgpus_.resize(gpuDevice_.numOfVgpus());
  gpuDevice_.vgpus_[index()] = this;

  queues_[MainEngine] = nullptr;
  queues_[SdmaEngine] = nullptr;

  // The hostcall buffer for this vqueue is initialized on demand.
  hostcallBuffer_ = nullptr;
}

// ================================================================================================
bool VirtualGPU::create(bool profiling, uint deviceQueueSize, uint rtCUs,
                        amd::CommandQueue::Priority priority) {
  device::BlitManager::Setup blitSetup;

  // Resize the list of device resources always,
  // because destructor calls eraseResourceList() even if create() failed
  dev().resizeResoureList(index());

  // Virtual GPU will have profiling enabled
  state_.profiling_ = profiling;

  Pal::CmdAllocatorCreateInfo createInfo = {};
  createInfo.flags.threadSafe = true;
  // \todo forces PAL to reuse CBs, but requires postamble
  createInfo.flags.autoMemoryReuse = false;
  createInfo.allocInfo[Pal::CommandDataAlloc].allocHeap = Pal::GpuHeapGartUswc;
  createInfo.allocInfo[Pal::CommandDataAlloc].suballocSize =
      VirtualGPU::Queue::MaxCommands *
      (320 + ((profiling) ? 96 : 0) + ((dev().captureMgr() != nullptr) ? 512 : 0));
  createInfo.allocInfo[Pal::CommandDataAlloc].allocSize =
      dev().settings().maxCmdBuffers_ * createInfo.allocInfo[Pal::CommandDataAlloc].suballocSize;

  createInfo.allocInfo[Pal::EmbeddedDataAlloc].allocHeap = Pal::GpuHeapGartUswc;
  createInfo.allocInfo[Pal::EmbeddedDataAlloc].allocSize = 256 * Ki;
  createInfo.allocInfo[Pal::EmbeddedDataAlloc].suballocSize = 64 * Ki;

  createInfo.allocInfo[Pal::LargeEmbeddedDataAlloc].allocHeap = Pal::GpuHeapGartUswc;
  createInfo.allocInfo[Pal::LargeEmbeddedDataAlloc].allocSize = 64 * Ki;
  createInfo.allocInfo[Pal::LargeEmbeddedDataAlloc].suballocSize = 32 * Ki;

  createInfo.allocInfo[Pal::GpuScratchMemAlloc].allocHeap = Pal::GpuHeapInvisible;
  createInfo.allocInfo[Pal::GpuScratchMemAlloc].allocSize = 64 * Ki;
  createInfo.allocInfo[Pal::GpuScratchMemAlloc].suballocSize = 4 * Ki;

  Pal::Result result;
  size_t cmdAllocSize = dev().iDev()->GetCmdAllocatorSize(createInfo, &result);
  if (Pal::Result::Success != result) {
    return false;
  }
  char* addr = new char[cmdAllocSize];
  if (Pal::Result::Success != dev().iDev()->CreateCmdAllocator(createInfo, addr, &cmdAllocator_)) {
    return false;
  }

  uint idx = index() % dev().numComputeEngines();
  uint64_t residency_limit = dev().properties().gpuMemoryProperties.flags.supportPerSubmitMemRefs
                                 ? 0
                                 : (dev().properties().gpuMemoryProperties.maxLocalMemSize >> 2);
  uint max_cmd_buffers = dev().settings().maxCmdBuffers_;

  if (dev().numComputeEngines()) {
    queues_[MainEngine] = Queue::Create(*this, Pal::QueueTypeCompute, idx, cmdAllocator_, rtCUs,
                                        priority, residency_limit, max_cmd_buffers);
    if (nullptr == queues_[MainEngine]) {
      return false;
    }
    const auto& info = dev().QueuePool().find(queues_[MainEngine]->iQueue_);
    hwRing_ = (info != dev().QueuePool().end())
                  ? info->second->index_
                  : (index() % dev().numExclusiveComputeEngines()) + GPU_MAX_HW_QUEUES;

    // Check if device has SDMA engines
    if (dev().numDMAEngines() != 0 && !dev().settings().disableSdma_) {
      uint sdma;
      // If only 1 SDMA engine is available then use that one, otherwise it's a round-robin manner
      if ((dev().numDMAEngines() < 2) || ((idx + 1) & 0x1)) {
        sdma = 0;
      } else {
        sdma = 1;
      }
      queues_[SdmaEngine] = Queue::Create(
          *this, Pal::QueueTypeDma, sdma, cmdAllocator_, amd::CommandQueue::RealTimeDisabled,
          amd::CommandQueue::Priority::Normal, residency_limit, max_cmd_buffers);
      if (nullptr == queues_[SdmaEngine]) {
        return false;
      }
    }
  } else {
    LogError("Runtme couldn't find compute queues!");
    return false;
  }

  // Create buffers for kernel arg management
  if (!managedBuffer_.create(dev().settings().kernel_arg_impl_ == KernelArgImpl::DeviceKernelArgs
                                 ? Resource::Persistent
                                 : Resource::RemoteUSWC)) {
    // Try just USWC if persistent memory failed
    if (dev().settings().kernel_arg_impl_ == KernelArgImpl::DeviceKernelArgs) {
      if (!managedBuffer_.create(Resource::RemoteUSWC)) {
        return false;
      }
    } else {
      return false;
    }
  }

  // Diable double copy optimization,
  // since UAV read from nonlocal is fast enough
  blitSetup.disableCopyBufferToImageOpt_ = true;
  if (!allocConstantBuffers()) {
    return false;
  }

  // Create HSAILPrintf class
  printfDbgHSA_ = new PrintfDbgHSA(gpuDevice_);
  if (nullptr == printfDbgHSA_) {
    LogError("Could not create PrintfDbgHSA class!");
    return false;
  }

  tsCache_ = new TimeStampCache(*this);
  if (nullptr == tsCache_) {
    LogError("Could not create TimeStamp cache!");
    return false;
  }

  if (!memoryDependency().create(dev().settings().numMemDependencies_)) {
    LogError("Could not create the array of memory objects!");
    return false;
  }

  if (!allocHsaQueueMem()) {
    LogError("Could not create hsaQueueMem object!");
    return false;
  }

  // Check if the app requested a device queue creation
  if (dev().settings().useDeviceQueue_ && (0 != deviceQueueSize) &&
      !createVirtualQueue(deviceQueueSize)) {
    LogError("Could not create a virtual queue!");
    return false;
  }

  // Choose the appropriate class for blit engine
  switch (dev().settings().blitEngine_) {
    default:
    // Fall through ...
    case Settings::BlitEngineHost:
      blitSetup.disableAll();
    // Fall through ...
    case Settings::BlitEngineCAL:
    case Settings::BlitEngineKernel:
      blitMgr_ = new KernelBlitManager(*this, blitSetup);
      break;
  }
  if ((nullptr == blitMgr_) || !blitMgr_->create(gpuDevice_)) {
    LogError("Could not create BlitManager!");
    return false;
  }

  // If the developer mode manager is available and it's not a device queue,
  // then enable RGP capturing
  if ((index() != 0) && dev().captureMgr() != nullptr) {
    bool dbg_vmid = false;
    state_.rgpCaptureEnabled_ = true;
    dev().captureMgr()->RegisterTimedQueue(2 * index(), queue(MainEngine).iQueue_, &dbg_vmid);
    dev().captureMgr()->RegisterTimedQueue(2 * index() + 1, queue(SdmaEngine).iQueue_, &dbg_vmid);
  }

  return true;
}

// ================================================================================================
bool VirtualGPU::allocHsaQueueMem() {
  // Allocate a dummy HSA queue
  hsaQueueMem_ = new Memory(dev(), sizeof(amd_queue_t));
  if ((hsaQueueMem_ == nullptr) || (!hsaQueueMem_->create(Resource::Local))) {
    delete hsaQueueMem_;
    return false;
  }
  amd_queue_t hsa_queue = {};

  // Provide private and local heap addresses
  constexpr uint addressShift = LP64_SWITCH(0, 32);
  hsa_queue.private_segment_aperture_base_hi = static_cast<uint32_t>(
      dev().properties().gpuMemoryProperties.privateApertureBase >> addressShift);
  hsa_queue.group_segment_aperture_base_hi = static_cast<uint32_t>(
      dev().properties().gpuMemoryProperties.sharedApertureBase >> addressShift);

  hsaQueueMem_->writeRawData(*this, 0, sizeof(amd_queue_t), &hsa_queue, true);

  return true;
}

VirtualGPU::~VirtualGPU() {
  // Not safe to remove a queue. So lock the device
  amd::ScopedLock k(dev().lockAsyncOps());
  amd::ScopedLock lock(dev().vgpusAccess());

  if (queues_[MainEngine] != nullptr) {
    // Clear all timestamps, associated with this virtual GPU
    auto& mgmt = *queues_[MainEngine]->aql_mgmt_;
    for (uint32_t i = 0; i < AqlPacketMgmt::kAqlPacketsListSize; ++i) {
      if (mgmt.aql_vgpus_[i] == this) {
        mgmt.aql_vgpus_[i] = nullptr;
        mgmt.aql_events_[i].invalidate();
      }
    }
  }

  // Destroy RGP trace
  if (rgpCaptureEna()) {
    dev().captureMgr()->FinishRGPTrace(this, true);
  }

  while (!freeCbQueue_.empty()) {
    auto cb = freeCbQueue_.front();
    delete cb;
    freeCbQueue_.pop();
  }

  // Destroy printfHSA object
  delete printfDbgHSA_;

  // Destroy TimeStamp cache
  delete tsCache_;

  // Destroy resource list with the constant buffers
  for (uint i = 0; i < constBufs_.size(); ++i) {
    delete constBufs_[i];
  }

  managedBuffer_.release();

  delete vqHeader_;
  delete virtualQueue_;
  delete hsaQueueMem_;

  // Release scratch buffer memory to reduce memory pressure
  //!@note OCLtst uses single device with multiple tests
  //! Release memory only if it's the last command queue.
  //! The first queue is reserved for the transfers on device
  if (static_cast<int>(gpuDevice_.numOfVgpus_ - 1) <= 1) {
    gpuDevice_.destroyScratchBuffers();
  }

  // Destroy BlitManager object
  delete blitMgr_;

  {
    // Destroy queues
    delete queues_[MainEngine];
    delete queues_[SdmaEngine];

    if (nullptr != cmdAllocator_) {
      cmdAllocator_->Destroy();
      delete[] reinterpret_cast<char*>(cmdAllocator_);
    }
  }

  {
    // Find all available virtual GPUs and lock them
    // from the execution of commands, since the queue index and resource list
    // Will be adjusted
    for (auto it : dev().vgpus()) {
      if (it != this) {
        it->execution().lock();
      }
    }

    // Not safe to add a resource if create/destroy queue is in process, since
    // the size of the TS array can change
    amd::ScopedLock r(dev().lockResources());
    gpuDevice_.numOfVgpus_--;
    gpuDevice_.vgpus_.erase(gpuDevice_.vgpus_.begin() + index());
    for (uint idx = index(); idx < dev().vgpus().size(); ++idx) {
      dev().vgpus()[idx]->index_--;
    }
    dev().eraseResoureList(index());

    // Find all available virtual GPUs and unlock them
    // for the execution of commands
    for (auto it : dev().vgpus()) {
      it->execution().unlock();
    }
  }

  if (hostcallBuffer_ != nullptr) {
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "deleting hostcall buffer %p for virtual queue %p",
            hostcallBuffer_, this);
    amd::disableHostcalls(hostcallBuffer_);
    dev().svmFree(hostcallBuffer_);
  }
}

void VirtualGPU::submitReadMemory(amd::ReadMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  // Translate memory references and ensure cache up-to-date
  pal::Memory* memory = dev().getGpuMemory(&vcmd.source());

  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(vcmd.destination(), &offset);

  profilingBegin(vcmd);

  memory->syncCacheFromHost(*this);
  cl_command_type type = vcmd.type();
  bool result = false;
  amd::Memory* bufferFromImage = nullptr;

  // Force buffer read for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_READ_IMAGE) &&
      (vcmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImage = createBufferFromImage(vcmd.source());
    if (nullptr == bufferFromImage) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_READ_BUFFER;
      memory = dev().getGpuMemory(bufferFromImage);
    }
  }

  // Process different write commands
  switch (type) {
    case CL_COMMAND_READ_BUFFER: {
      amd::Coord3D origin(vcmd.origin()[0]);
      amd::Coord3D size(vcmd.size()[0]);
      if (nullptr != bufferFromImage) {
        size_t elemSize = vcmd.source().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != nullptr) {
        // Accelerated transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result = blitMgr().copyBuffer(*memory, *hostMemory, origin, dstOrigin, size,
                                      vcmd.isEntireMemory(), vcmd.copyMetadata());
      } else {
        // The logic below will perform 2 step copy to make sure memory pinning doesn't
        // occur on the first unaligned page, because in Windows memory manager can
        // have CPU access to the allocation header in another thread
        // and a race condition is possible.
        char* tmpHost =
            amd::alignUp(reinterpret_cast<char*>(vcmd.destination()), PinnedMemoryAlignment);

        // Find the partial size for unaligned copy
        size_t partial = tmpHost - reinterpret_cast<char*>(vcmd.destination());
        result = true;
        // Check if it's staging copy, then ignore unaligned address
        if (size[0] <= dev().settings().pinnedMinXferSize_) {
          partial = size[0];
        }
        // Make first step transfer
        if (partial > 0) {
          result = blitMgr().readBuffer(*memory, vcmd.destination(), origin, partial, false,
                                        vcmd.copyMetadata());
        }
        // Second step transfer if something left to copy
        if (partial < size[0]) {
          result &= blitMgr().readBuffer(*memory, tmpHost, origin[0] + partial, size[0] - partial,
                                         false, vcmd.copyMetadata());
        }
      }
      if (nullptr != bufferFromImage) {
        bufferFromImage->release();
      }
    } break;
    case CL_COMMAND_READ_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(vcmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, vcmd.size().c, vcmd.hostRect().rowPitch_,
                            vcmd.hostRect().slicePitch_);
      if (hostMemory != nullptr) {
        result = blitMgr().copyBufferRect(*memory, *hostMemory, vcmd.bufRect(), hostbufferRect,
                                          vcmd.size(), vcmd.isEntireMemory(), vcmd.copyMetadata());
      } else {
        result =
            blitMgr().readBufferRect(*memory, vcmd.destination(), vcmd.bufRect(), vcmd.hostRect(),
                                     vcmd.size(), vcmd.isEntireMemory(), vcmd.copyMetadata());
      }
    } break;
    case CL_COMMAND_READ_IMAGE:
      if (memory->memoryType() == Resource::ImageBuffer) {
        Image* imageBuffer = static_cast<Image*>(memory);
        // Check if synchronization has to be performed
        if (nullptr != imageBuffer->CopyImageBuffer()) {
          memory = imageBuffer->CopyImageBuffer();
          Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
          amd::Image* image = imageBuffer->owner()->asImage();
          amd::Coord3D offs(0);
          // Copy memory from the original image buffer into the backing store image
          result = blitMgr().copyBufferToImage(*buffer, *imageBuffer->CopyImageBuffer(), offs, offs,
                                               image->getRegion(), true, image->getRowPitch(),
                                               image->getSlicePitch(), vcmd.copyMetadata());
        }
      }
      if (hostMemory != nullptr) {
        // Accelerated image to buffer transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result = blitMgr().copyImageToBuffer(*memory, *hostMemory, vcmd.origin(), dstOrigin,
                                             vcmd.size(), vcmd.isEntireMemory(), vcmd.rowPitch(),
                                             vcmd.slicePitch(), vcmd.copyMetadata());
      } else {
        result = blitMgr().readImage(*memory, vcmd.destination(), vcmd.origin(), vcmd.size(),
                                     vcmd.rowPitch(), vcmd.slicePitch(), vcmd.isEntireMemory(),
                                     vcmd.copyMetadata());
      }
      break;
    default:
      LogError("Unsupported type for the read command");
      break;
  }

  if (!result) {
    LogError("submitReadMemory failed!");
    vcmd.setStatus(CL_INVALID_OPERATION);
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  // Translate memory references and ensure cache up to date
  pal::Memory* memory = dev().getGpuMemory(&vcmd.destination());
  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(vcmd.source(), &offset);

  profilingBegin(vcmd);

  bool entire = vcmd.isEntireMemory();

  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  memory->syncCacheFromHost(*this, syncFlags);

  cl_command_type type = vcmd.type();
  bool result = false;
  amd::Memory* bufferFromImage = nullptr;

  // Force buffer write for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_WRITE_IMAGE) &&
      (vcmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImage = createBufferFromImage(vcmd.destination());
    if (nullptr == bufferFromImage) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_WRITE_BUFFER;
      memory = dev().getGpuMemory(bufferFromImage);
    }
  }

  // Process different write commands
  switch (type) {
    case CL_COMMAND_WRITE_BUFFER: {
      amd::Coord3D origin(vcmd.origin()[0]);
      amd::Coord3D size(vcmd.size()[0]);
      if (nullptr != bufferFromImage) {
        size_t elemSize = vcmd.destination().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if ((hostMemory != nullptr) && (vcmd.size()[0] > dev().settings().prepinnedMinSize_)) {
        // Accelerated transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result = blitMgr().copyBuffer(*hostMemory, *memory, srcOrigin, origin, size,
                                      vcmd.isEntireMemory(), vcmd.copyMetadata());
      } else {
        // The logic below will perform 2 step copy to make sure memory pinning doesn't
        // occur on the first unaligned page, because in Windows memory manager can
        // have CPU access to the allocation header in another thread
        // and a race condition is possible.
        const char* tmpHost =
            amd::alignUp(reinterpret_cast<const char*>(vcmd.source()), PinnedMemoryAlignment);

        // Find the partial size for unaligned copy
        size_t partial = tmpHost - reinterpret_cast<const char*>(vcmd.source());
        result = true;
        // Check if it's staging copy, then ignore unaligned address
        if (size[0] <= dev().settings().pinnedMinXferSize_) {
          partial = size[0];
        }
        // Make first step transfer
        if (partial > 0) {
          result = blitMgr().writeBuffer(vcmd.source(), *memory, origin, partial, false,
                                         vcmd.copyMetadata());
        }
        // Second step transfer if something left to copy
        if (partial < size[0]) {
          result &= blitMgr().writeBuffer(tmpHost, *memory, origin[0] + partial, size[0] - partial,
                                          false, vcmd.copyMetadata());
        }
      }
      if (nullptr != bufferFromImage) {
        bufferFromImage->release();
      }
    } break;
    case CL_COMMAND_WRITE_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(vcmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, vcmd.size().c, vcmd.hostRect().rowPitch_,
                            vcmd.hostRect().slicePitch_);
      if (hostMemory != nullptr) {
        result = blitMgr().copyBufferRect(*hostMemory, *memory, hostbufferRect, vcmd.bufRect(),
                                          vcmd.size(), vcmd.isEntireMemory(), vcmd.copyMetadata());
      } else {
        result = blitMgr().writeBufferRect(vcmd.source(), *memory, vcmd.hostRect(), vcmd.bufRect(),
                                           vcmd.size(), vcmd.isEntireMemory(), vcmd.copyMetadata());
      }
    } break;
    case CL_COMMAND_WRITE_IMAGE:
      if (hostMemory != nullptr) {
        // Accelerated buffer to image transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result = blitMgr().copyBufferToImage(*hostMemory, *memory, srcOrigin, vcmd.origin(),
                                             vcmd.size(), vcmd.isEntireMemory(), vcmd.rowPitch(),
                                             vcmd.slicePitch(), vcmd.copyMetadata());
      } else {
        result = blitMgr().writeImage(vcmd.source(), *memory, vcmd.origin(), vcmd.size(),
                                      vcmd.rowPitch(), vcmd.slicePitch(), vcmd.isEntireMemory(),
                                      vcmd.copyMetadata());
      }
      break;
    default:
      LogError("Unsupported type for the write command");
      break;
  }

  if (!result) {
    LogError("submitWriteMemory failed!");
    vcmd.setStatus(CL_INVALID_OPERATION);
  } else {
    // Mark this as the most-recently written cache of the destination
    vcmd.destination().signalWrite(&gpuDevice_);
  }
  profilingEnd(vcmd);
}

bool VirtualGPU::copyMemory(cl_command_type type, amd::Memory& srcMem, amd::Memory& dstMem,
                            bool entire, const amd::Coord3D& srcOrigin,
                            const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                            const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                            amd::CopyMetadata copyMetadata) {
  // Translate memory references and ensure cache up-to-date
  pal::Memory* dstMemory = dev().getGpuMemory(&dstMem);
  pal::Memory* srcMemory = dev().getGpuMemory(&srcMem);

  if (dstMemory == nullptr || srcMemory == nullptr) {
    LogError("submitcopyMemory Failed!");
    return false;
  }

  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  dstMemory->syncCacheFromHost(*this, syncFlags);
  srcMemory->syncCacheFromHost(*this);

  amd::Memory* bufferFromImageSrc = nullptr;
  amd::Memory* bufferFromImageDst = nullptr;

  // Force buffer read for IMAGE1D_BUFFER
  if (srcMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    bufferFromImageSrc = createBufferFromImage(srcMem);
    if (nullptr == bufferFromImageSrc) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      srcMemory = dev().getGpuMemory(bufferFromImageSrc);
    }
  }
  // Force buffer write for IMAGE1D_BUFFER
  if (dstMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    bufferFromImageDst = createBufferFromImage(dstMem);
    if (nullptr == bufferFromImageDst) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      dstMemory = dev().getGpuMemory(bufferFromImageDst);
    }
  }

  type = getCopyCommandType(type, srcMem.getType(), dstMem.getType());

  bool result = false;

  // Check if HW can be used for memory copy
  switch (type) {
    case CL_COMMAND_MAKE_BUFFERS_RESIDENT_AMD:
    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D realSrcOrigin(srcOrigin[0]);
      amd::Coord3D realDstOrigin(dstOrigin[0]);
      amd::Coord3D realSize(size.c[0], size.c[1], size.c[2]);

      if (nullptr != bufferFromImageSrc) {
        const size_t elemSize = srcMem.asImage()->getImageFormat().getElementSize();
        realSrcOrigin.c[0] *= elemSize;
        if (nullptr != bufferFromImageDst) {
          realDstOrigin.c[0] *= elemSize;
        }
        realSize.c[0] *= elemSize;
      } else if (nullptr != bufferFromImageDst) {
        const size_t elemSize = dstMem.asImage()->getImageFormat().getElementSize();
        realDstOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
      }

      result = blitMgr().copyBuffer(*srcMemory, *dstMemory, realSrcOrigin, realDstOrigin, realSize,
                                    entire, copyMetadata);
    } break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      result = blitMgr().copyBufferRect(*srcMemory, *dstMemory, srcRect, dstRect, size, entire,
                                        copyMetadata);
      break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
      amd::Coord3D realDstOrigin(dstOrigin);
      if (nullptr != bufferFromImageDst) {
        const size_t elemSize = dstMem.asImage()->getImageFormat().getElementSize();
        realDstOrigin.c[0] *= elemSize;
      }
      result =
          blitMgr().copyImageToBuffer(*srcMemory, *dstMemory, srcOrigin, realDstOrigin, size, entire,
                                      dstRect.rowPitch_, dstRect.slicePitch_, copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
      amd::Coord3D realSrcOrigin(srcOrigin);
      if (nullptr != bufferFromImageSrc) {
        const size_t elemSize = srcMem.asImage()->getImageFormat().getElementSize();
        realSrcOrigin.c[0] *= elemSize;
      }
      result =
          blitMgr().copyBufferToImage(*srcMemory, *dstMemory, realSrcOrigin, dstOrigin, size, entire,
                                      srcRect.rowPitch_, srcRect.slicePitch_, copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_IMAGE:
      result = blitMgr().copyImage(*srcMemory, *dstMemory, srcOrigin, dstOrigin, size, entire,
                                   copyMetadata);
      break;
    default:
      LogError("Unsupported command type for memory copy!");
      break;
  }
  if (nullptr != bufferFromImageSrc) {
    bufferFromImageSrc->release();
  }
  if (nullptr != bufferFromImageDst) {
    bufferFromImageDst->release();
  }
  if (!result) {
    LogError("submitCopyMemory failed!");
    return false;
  } else {
    // Mark this as the most-recently written cache of the destination
    dstMem.signalWrite(&gpuDevice_);
  }
  return true;
}

void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  cl_command_type type = vcmd.type();
  bool entire = vcmd.isEntireMemory();

  if (!copyMemory(type, vcmd.source(), vcmd.destination(), entire, vcmd.srcOrigin(),
                  vcmd.dstOrigin(), vcmd.size(), vcmd.srcRect(), vcmd.dstRect(),
                  vcmd.copyMetadata())) {
    vcmd.setStatus(CL_INVALID_OPERATION);
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);

  cl_command_type type = vcmd.type();
  // no op for FGS supported device
  if (!dev().isFineGrainedSystem()) {
    amd::Coord3D srcOrigin(0, 0, 0);
    amd::Coord3D dstOrigin(0, 0, 0);
    amd::Coord3D size(vcmd.srcSize(), 1, 1);
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    bool result = false;
    amd::Memory* srcMem = amd::MemObjMap::FindMemObj(vcmd.src());
    amd::Memory* dstMem = amd::MemObjMap::FindMemObj(vcmd.dst());

    device::Memory::SyncFlags syncFlags;
    if (nullptr != srcMem) {
      srcMem->commitSvmMemory();
      srcOrigin.c[0] =
          static_cast<const_address>(vcmd.src()) - static_cast<address>(srcMem->getSvmPtr());
      if (!(srcMem->validateRegion(srcOrigin, size))) {
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
    if (nullptr != dstMem) {
      dstMem->commitSvmMemory();
      dstOrigin.c[0] =
          static_cast<const_address>(vcmd.dst()) - static_cast<address>(dstMem->getSvmPtr());
      if (!(dstMem->validateRegion(dstOrigin, size))) {
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }

    if (nullptr == srcMem && nullptr == dstMem) {  // both not in svm space
      std::memcpy(vcmd.dst(), vcmd.src(), vcmd.srcSize());
      result = true;
    } else if (nullptr == srcMem && nullptr != dstMem) {  // src not in svm space
      Memory* memory = dev().getGpuMemory(dstMem);
      // Synchronize source and destination memory
      syncFlags.skipEntire_ = dstMem->isEntirelyCovered(dstOrigin, size);
      memory->syncCacheFromHost(*this, syncFlags);

      result = blitMgr().writeBuffer(vcmd.src(), *memory, dstOrigin, size,
                                     dstMem->isEntirelyCovered(dstOrigin, size));
      // Mark this as the most-recently written cache of the destination
      dstMem->signalWrite(&gpuDevice_);
    } else if (nullptr != srcMem && nullptr == dstMem) {  // dst not in svm space
      Memory* memory = dev().getGpuMemory(srcMem);
      // Synchronize source and destination memory
      memory->syncCacheFromHost(*this);

      result = blitMgr().readBuffer(*memory, vcmd.dst(), srcOrigin, size,
                                    srcMem->isEntirelyCovered(srcOrigin, size));
    } else if (nullptr != srcMem && nullptr != dstMem) {  // both in svm space
      bool entire =
          srcMem->isEntirelyCovered(srcOrigin, size) && dstMem->isEntirelyCovered(dstOrigin, size);
      result =
          copyMemory(type, *srcMem, *dstMem, entire, srcOrigin, dstOrigin, size, srcRect, dstRect);
    }

    if (!result) {
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // direct memcpy for FGS enabled system
    amd::SvmBuffer::memFill(vcmd.dst(), vcmd.src(), vcmd.srcSize(), 1);
  }
  profilingEnd(vcmd);
}

void VirtualGPU::submitMapMemory(amd::MapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  pal::Memory* memory = dev().getGpuMemory(&vcmd.memory());

  // Save map info for unmap operation
  memory->saveMapInfo(vcmd.mapPtr(), vcmd.origin(), vcmd.size(), vcmd.mapFlags(),
                      vcmd.isEntireMemory());

  // If we have host memory, use it
  if ((memory->owner()->getHostMem() != nullptr) && memory->isDirectMap()) {
    if (!memory->isHostMemDirectAccess()) {
      // Make sure GPU finished operation before
      // synchronization with the backing store
      memory->wait(*this);
    }

    // Target is the backing store, so just ensure that owner is up-to-date
    memory->owner()->cacheWriteBack(this);

    // Add memory to VA cache, so rutnime can detect direct access to VA
    dev().addVACache(memory);
  } else if (memory->isPersistentMapped()) {
    // Nothing to do here
  } else if (memory->mapMemory() != nullptr) {
    // Target is a remote resource, so copy
    assert(memory->mapMemory() != nullptr);
    if (vcmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
      amd::Coord3D dstOrigin(0, 0, 0);
      if (memory->desc().buffer_) {
        if (!blitMgr().copyBuffer(*memory, *memory->mapMemory(), vcmd.origin(), vcmd.origin(),
                                  vcmd.size(), vcmd.isEntireMemory())) {
          LogError("submitMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
      } else if ((vcmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        Memory* memoryBuf = memory;
        amd::Coord3D origin(vcmd.origin()[0]);
        amd::Coord3D size(vcmd.size()[0]);
        size_t elemSize = vcmd.memory().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;

        amd::Memory* bufferFromImage = createBufferFromImage(vcmd.memory());
        if (nullptr == bufferFromImage) {
          LogError("We should not fail buffer creation from image_buffer!");
        } else {
          memoryBuf = dev().getGpuMemory(bufferFromImage);
        }
        if (!blitMgr().copyBuffer(*memoryBuf, *memory->mapMemory(), origin, dstOrigin, size,
                                  vcmd.isEntireMemory())) {
          LogError("submitMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
        if (nullptr != bufferFromImage) {
          bufferFromImage->release();
        }
      } else {
        // Validate if it's a view for a map of mip level
        if (vcmd.memory().parent() != nullptr) {
          amd::Image* amdImage = vcmd.memory().parent()->asImage();
          if ((amdImage != nullptr) && (amdImage->getMipLevels() > 1)) {
            // Save map write info in the parent object
            dev().getGpuMemory(amdImage)->saveMapInfo(vcmd.mapPtr(), vcmd.origin(), vcmd.size(),
                                                      vcmd.mapFlags(), vcmd.isEntireMemory(),
                                                      vcmd.memory().asImage());
          }
        }
        if (!blitMgr().copyImageToBuffer(*memory, *memory->mapMemory(), vcmd.origin(), dstOrigin,
                                         vcmd.size(), vcmd.isEntireMemory())) {
          LogError("submitMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
      }
    }
  } else {
    LogError("Unhandled map!");
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand& vcmd) {
  bool unmapMip = false;
  amd::Image* amdImage;
  {
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    pal::Memory* memory = dev().getGpuMemory(&vcmd.memory());
    amd::Memory* owner = memory->owner();
    const device::Memory::WriteMapInfo* writeMapInfo = memory->writeMapInfo(vcmd.mapPtr());
    if (nullptr == writeMapInfo) {
      LogError("Unmap without map call");
      return;
    }
    profilingBegin(vcmd);

    // Check if image is a mipmap and assign a saved view
    amdImage = owner->asImage();
    if ((amdImage != nullptr) && (amdImage->getMipLevels() > 1) &&
        (writeMapInfo->baseMip_ != nullptr)) {
      // Assign mip level view
      amdImage = writeMapInfo->baseMip_;
      // Clear unmap flags from the parent image
      memory->clearUnmapInfo(vcmd.mapPtr());
      memory = dev().getGpuMemory(amdImage);
      unmapMip = true;
      writeMapInfo = memory->writeMapInfo(vcmd.mapPtr());
    }

    // We used host memory
    if ((owner->getHostMem() != nullptr) && memory->isDirectMap()) {
      if (writeMapInfo->isUnmapWrite()) {
        // Target is the backing store, so sync
        owner->signalWrite(nullptr);
        memory->syncCacheFromHost(*this);
      }
      // Remove memory from VA cache
      dev().removeVACache(memory);
    }
    // data check was added for persistent memory that failed to get aperture
    // and therefore are treated like a remote resource
    else if (memory->isPersistentMapped()) {
      // Map/unmap must be serialized
      amd::ScopedLock lock(owner->lockMemoryOps());
      memory->unmap(this);
      if (memory->getMapCount() == 0) {
        memory->setPersistentMapFlag(false);
      }
    } else if (memory->mapMemory() != nullptr) {
      if (writeMapInfo->isUnmapWrite()) {
        amd::Coord3D srcOrigin(0, 0, 0);
        // Target is a remote resource, so copy
        assert(memory->mapMemory() != nullptr);
        if (memory->desc().buffer_) {
          if (!blitMgr().copyBuffer(*memory->mapMemory(), *memory, writeMapInfo->origin_,
                                    writeMapInfo->origin_, writeMapInfo->region_,
                                    writeMapInfo->isEntire())) {
            LogError("submitUnmapMemory() - copy failed");
            vcmd.setStatus(CL_OUT_OF_RESOURCES);
          }
        } else if ((vcmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
          Memory* memoryBuf = memory;
          amd::Coord3D origin(writeMapInfo->origin_[0]);
          amd::Coord3D size(writeMapInfo->region_[0]);
          size_t elemSize = vcmd.memory().asImage()->getImageFormat().getElementSize();
          origin.c[0] *= elemSize;
          size.c[0] *= elemSize;

          amd::Memory* bufferFromImage = createBufferFromImage(vcmd.memory());
          if (nullptr == bufferFromImage) {
            LogError("We should not fail buffer creation from image_buffer!");
          } else {
            memoryBuf = dev().getGpuMemory(bufferFromImage);
          }
          if (!blitMgr().copyBuffer(*memory->mapMemory(), *memoryBuf, srcOrigin, origin, size,
                                    writeMapInfo->isEntire())) {
            LogError("submitUnmapMemory() - copy failed");
            vcmd.setStatus(CL_OUT_OF_RESOURCES);
          }
          if (nullptr != bufferFromImage) {
            bufferFromImage->release();
          }
        } else {
          if (!blitMgr().copyBufferToImage(*memory->mapMemory(), *memory, srcOrigin,
                                           writeMapInfo->origin_, writeMapInfo->region_,
                                           writeMapInfo->isEntire())) {
            LogError("submitUnmapMemory() - copy failed");
            vcmd.setStatus(CL_OUT_OF_RESOURCES);
          }
        }
      }
    } else {
      LogError("Unhandled unmap!");
      vcmd.setStatus(CL_INVALID_VALUE);
    }

    // Clear unmap flags
    memory->clearUnmapInfo(vcmd.mapPtr());

    profilingEnd(vcmd);
  }
  // Release a view for a mipmap map
  if (unmapMip) {
    // Memory release should be outside of the execution lock,
    // because mapMemory_ isn't marked for a specifc GPU
    amdImage->release();
  }
}

bool VirtualGPU::fillMemory(cl_command_type type, amd::Memory* amdMemory, const void* pattern,
                            size_t patternSize, const amd::Coord3D& origin,
                            const amd::Coord3D& size, bool forceBlit) {
  pal::Memory* memory = dev().getGpuMemory(amdMemory);
  bool entire = amdMemory->isEntirelyCovered(origin, size);

  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  memory->syncCacheFromHost(*this, syncFlags);

  bool result = false;
  amd::Memory* bufferFromImage = nullptr;
  float fillValue[4];

  // Force fill buffer for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_FILL_IMAGE) && (amdMemory->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImage = createBufferFromImage(*amdMemory);
    if (nullptr == bufferFromImage) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_FILL_BUFFER;
      memory = dev().getGpuMemory(bufferFromImage);
    }
  }

  // Find the the right fill operation
  switch (type) {
    case CL_COMMAND_FILL_BUFFER:
    case CL_COMMAND_SVM_MEMFILL: {
      amd::Coord3D realOrigin(origin[0]);
      amd::Coord3D realSize(size[0]);
      // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
      if (nullptr != bufferFromImage) {
        size_t elemSize = amdMemory->asImage()->getImageFormat().getElementSize();
        realOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
        memset(fillValue, 0, sizeof(fillValue));
        amdMemory->asImage()->getImageFormat().formatColor(pattern, fillValue);
        pattern = fillValue;
        patternSize = elemSize;
      }
      result = blitMgr().fillBuffer(*memory, pattern, patternSize, realSize, realOrigin, realSize,
                                    amdMemory->isEntirelyCovered(origin, size), forceBlit);
      if (nullptr != bufferFromImage) {
        bufferFromImage->release();
      }
    } break;
    case CL_COMMAND_FILL_IMAGE:
      result = blitMgr().fillImage(*memory, pattern, origin, size,
                                   amdMemory->isEntirelyCovered(origin, size));
      break;
    default:
      LogError("Unsupported command type for FillMemory!");
      break;
  }

  if (!result) {
    LogError("fillMemory failed!");
    return false;
  }

  // Mark this as the most-recently written cache of the destination
  amdMemory->signalWrite(&gpuDevice_);
  return true;
}

void VirtualGPU::submitFillMemory(amd::FillMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);
  if (cmd.type() == CL_COMMAND_FILL_IMAGE) {
    if (!fillMemory(cmd.type(), &cmd.memory(), cmd.pattern(), cmd.patternSize(), cmd.origin(),
                    cmd.size())) {
      cmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    size_t width = cmd.size().c[0];
    size_t height = cmd.size().c[1];
    size_t depth = cmd.size().c[2];
    size_t pitch = cmd.surface().c[0];
    amd::Coord3D origin = cmd.origin();
    amd::Coord3D region{cmd.surface().c[1], cmd.surface().c[2], depth};
    amd::BufferRect rect;
    rect.create(static_cast<size_t*>(origin), static_cast<size_t*>(region), pitch, 0);

    bool force_blit = false;
    if (amd::IS_HIP) {
      constexpr uint32_t kManagedAlloc = (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR);
      // In case of HMM, use blit kernel instead of CPU memcpy
      if ((cmd.memory().getMemFlags() & kManagedAlloc) == kManagedAlloc) {
        force_blit = true;
      }
    }

    for (size_t slice = 0; slice < depth; slice++) {
      for (size_t row = 0; row < height; row++) {
        const size_t rowOffset = rect.offset(0, row, slice);
        if (!fillMemory(cmd.type(), &cmd.memory(), cmd.pattern(), cmd.patternSize(),
                        amd::Coord3D{rowOffset, 0, 0}, amd::Coord3D{width, 1, 1}, force_blit)) {
          cmd.setStatus(CL_INVALID_OPERATION);
        }
      }
    }
  }
  profilingEnd(cmd);
}

void VirtualGPU::submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  // Get the device memory objects for the current device
  Memory* srcDevMem = dev().getGpuMemory(&cmd.source());
  Memory* dstDevMem = dev().getGpuMemory(&cmd.destination());

  bool p2pAllowed = true;

  // If any device object is null, then no HW P2P and runtime has to use staging
  if (srcDevMem == nullptr) {
    srcDevMem = static_cast<pal::Memory*>(
        cmd.source().getDeviceMemory(*cmd.source().getContext().devices()[0]));
    p2pAllowed = false;
  } else if (dstDevMem == nullptr) {
    dstDevMem = static_cast<pal::Memory*>(
        cmd.destination().getDeviceMemory(*cmd.destination().getContext().devices()[0]));
    p2pAllowed = false;
  }

  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = cmd.isEntireMemory();
  amd::Coord3D size = cmd.size();

  bool result = false;
  switch (cmd.type()) {
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D srcOrigin(cmd.srcOrigin()[0]);
      amd::Coord3D dstOrigin(cmd.dstOrigin()[0]);

      if (p2pAllowed) {
        result = blitMgr().copyBuffer(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size,
                                      cmd.isEntireMemory());
      } else {
        amd::ScopedLock lock(dev().P2PStageOps());
        Memory* dstStgMem = static_cast<pal::Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.source().getContext().devices()[0]));
        Memory* srcStgMem = static_cast<pal::Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.destination().getContext().devices()[0]));

        size_t copy_size = Device::kP2PStagingSize;
        size_t left_size = size[0];
        amd::Coord3D stageOffset(0);
        result = true;
        do {
          if (left_size <= copy_size) {
            copy_size = left_size;
          }
          left_size -= copy_size;
          amd::Coord3D cpSize(copy_size);

          // Perform 2 step transfer with staging buffer
          result &= srcDevMem->dev().xferMgr().copyBuffer(*srcDevMem, *dstStgMem, srcOrigin,
                                                          stageOffset, cpSize);
          srcOrigin.c[0] += copy_size;
          result &= dstDevMem->dev().xferMgr().copyBuffer(*srcStgMem, *dstDevMem, stageOffset,
                                                          dstOrigin, cpSize);
          dstOrigin.c[0] += copy_size;
        } while (left_size > 0);
      }
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT: {
      if (p2pAllowed) {
        result = blitMgr().copyBufferRect(*srcDevMem, *dstDevMem, cmd.srcRect(), cmd.dstRect(),
                                          size, cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        amd::ScopedLock lock(dev().P2PStageOps());
        Memory* dstStgMem = static_cast<pal::Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.source().getContext().devices()[0]));
        Memory* srcStgMem = static_cast<pal::Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.destination().getContext().devices()[0]));

        if ((cmd.srcRect().slicePitch_ * size[2]) <= Device::kP2PStagingSize) {
          result = true;
          // Perform 2 step transfer with staging buffer
          result &= srcDevMem->dev().xferMgr().copyBufferRect(*srcDevMem, *dstStgMem, cmd.srcRect(),
                                                              cmd.srcRect(), size, false,
                                                              cmd.copyMetadata());

          result &= dstDevMem->dev().xferMgr().copyBufferRect(*srcStgMem, *dstDevMem, cmd.srcRect(),
                                                              cmd.dstRect(), size, false,
                                                              cmd.copyMetadata());
        } else {
          size_t srcOffset;
          size_t dstOffset;
          result = true;

          for (size_t z = 0; z < size[2]; ++z) {
            for (size_t y = 0; y < size[1]; ++y) {
              srcOffset = cmd.srcRect().offset(0, y, z);
              dstOffset = cmd.dstRect().offset(0, y, z);

              amd::Coord3D srcOrigin(srcOffset);
              amd::Coord3D dstOrigin(dstOffset);
              size_t copy_size = Device::kP2PStagingSize;
              size_t left_size = size[0];
              amd::Coord3D stageOffset(0);
              do {
                if (left_size <= copy_size) {
                  copy_size = left_size;
                }
                left_size -= copy_size;

                // Perform 2 step transfer with staging buffer
                result &= srcDevMem->partialMemCopyTo(*(srcDevMem->dev().xferQueue()), srcOrigin,
                                                      stageOffset, copy_size, *dstStgMem);
                srcDevMem->dev().xferQueue()->waitAllEngines();

                result &= srcStgMem->partialMemCopyTo(*(dstDevMem->dev().xferQueue()), stageOffset,
                                                      dstOrigin, copy_size, *dstDevMem);
                srcStgMem->dev().xferQueue()->waitAllEngines();

                srcOrigin.c[0] += copy_size;
                dstOrigin.c[0] += copy_size;
              } while (left_size > 0);
            }
          }
        }
      }
      break;
    }
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      LogError("Unsupported P2P type!");
      break;
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitCopyMemoryP2P failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  }

  cmd.destination().signalWrite(&dstDevMem->dev());

  profilingEnd(cmd);
}

void VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem()) {
    // Make sure we have memory for the command execution
    pal::Memory* memory = dev().getGpuMemory(vcmd.getSvmMem());

    memory->saveMapInfo(vcmd.svmPtr(), vcmd.origin(), vcmd.size(), vcmd.mapFlags(),
                        vcmd.isEntireMemory());

    if (memory->mapMemory() != nullptr) {
      if (vcmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
        assert(memory->desc().buffer_ && "SVM memory can't be an image");
        if (!blitMgr().copyBuffer(*memory, *memory->mapMemory(), vcmd.origin(), vcmd.origin(),
                                  vcmd.size(), vcmd.isEntireMemory())) {
          LogError("submitSVMMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
      }
    } else if ((memory->owner()->getHostMem() != nullptr) && memory->isDirectMap()) {
      if (!memory->isHostMemDirectAccess()) {
        // Make sure GPU finished operation before
        // synchronization with the backing store
        memory->wait(*this);
      }

      // Target is the backing store, so just ensure that owner is up-to-date
      memory->owner()->cacheWriteBack(this);
    } else {
      LogError("Unhandled svm map!");
    }
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem()) {
    pal::Memory* memory = dev().getGpuMemory(vcmd.getSvmMem());
    const device::Memory::WriteMapInfo* writeMapInfo = memory->writeMapInfo(vcmd.svmPtr());

    if (memory->mapMemory() != nullptr) {
      if (writeMapInfo->isUnmapWrite()) {
        amd::Coord3D srcOrigin(0, 0, 0);
        // Target is a remote resource, so copy
        assert(memory->desc().buffer_ && "SVM memory can't be an image");
        if (!blitMgr().copyBuffer(*memory->mapMemory(), *memory, writeMapInfo->origin_,
                                  writeMapInfo->origin_, writeMapInfo->region_,
                                  writeMapInfo->isEntire())) {
          LogError("submitSvmUnmapMemory() - copy failed");
          vcmd.setStatus(CL_OUT_OF_RESOURCES);
        }
      }
    } else if ((memory->owner()->getHostMem() != nullptr) && memory->isDirectMap()) {
      if (writeMapInfo->isUnmapWrite()) {
        // Target is the backing store, so sync
        memory->owner()->signalWrite(nullptr);
        memory->syncCacheFromHost(*this);
      }
    }
    memory->clearUnmapInfo(vcmd.svmPtr());
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  if (!dev().isFineGrainedSystem()) {
    size_t patternSize = vcmd.patternSize();
    size_t fillSize = patternSize * vcmd.times();
    amd::Memory* dstMemory = amd::MemObjMap::FindMemObj(vcmd.dst());
    assert(dstMemory && "No svm Buffer to fill with!");
    size_t offset = reinterpret_cast<uintptr_t>(vcmd.dst()) -
                    reinterpret_cast<uintptr_t>(dstMemory->getSvmPtr());

    pal::Memory* memory = dev().getGpuMemory(dstMemory);

    amd::Coord3D origin(offset, 0, 0);
    amd::Coord3D size(fillSize, 1, 1);

    assert((dstMemory->validateRegion(origin, size)) && "The incorrect fill size!");

    if (!fillMemory(vcmd.type(), dstMemory, vcmd.pattern(), vcmd.patternSize(), origin, size)) {
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // for FGS capable device, fill CPU memory directly
    amd::SvmBuffer::memFill(vcmd.dst(), vcmd.pattern(), vcmd.patternSize(), vcmd.times());
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (const auto& it : vcmd.memObjects()) {
    // Find device memory
    pal::Memory* memory = dev().getGpuMemory(it);

    if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
      memory->mgpuCacheWriteBack(*this);
    } else if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
      // Synchronize memory from host if necessary.
      // The sync function will perform memory migration from
      // another device if necessary
      device::Memory::SyncFlags syncFlags;
      memory->syncCacheFromHost(*this, syncFlags);
    } else {
      LogWarning("Unknown operation for memory migration!");
    }
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& vcmd) {
  // in-order semantics: previous commands need to be done before we start
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);
  std::vector<void*>& svmPointers = vcmd.svmPointers();
  if (vcmd.pfnFreeFunc() == nullptr) {
    // pointers allocated using clSVMAlloc
    for (uint32_t i = 0; i < svmPointers.size(); ++i) {
      dev().svmFree(svmPointers[i]);
    }
  } else {
    vcmd.pfnFreeFunc()(as_cl(vcmd.queue()->asCommandQueue()), svmPointers.size(),
                       static_cast<void**>(&(svmPointers[0])), vcmd.userData());
  }
  profilingEnd(vcmd);
}

void VirtualGPU::submitStreamOperation(amd::StreamOperationCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(cmd);

  const cl_command_type type = cmd.type();
  const uint64_t value = cmd.value();
  const uint64_t mask = cmd.mask();
  const unsigned int flags = cmd.flags();
  const size_t sizeBytes = cmd.sizeBytes();
  const size_t offset = cmd.offset();

  amd::Memory* amdMemory = &cmd.memory();
  Memory* memory = dev().getGpuMemory(amdMemory);

  if (type == ROCCLR_COMMAND_STREAM_WAIT_VALUE) {
    // Use a blit kernel to perform the wait operation
    // mask is applied on value before performing
    // the comparision defined by 'condition'
    bool result = static_cast<KernelBlitManager&>(blitMgr()).streamOpsWait(*memory, value, offset,
                                                                           sizeBytes, flags, mask);
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
            "Waiting for value: 0x%lx."
            " Flags: 0x%lx mask: 0x%lx",
            value, flags, mask);
    if (!result) {
      LogError("submitStreamOperation: Wait failed!");
    }
  } else if (type == ROCCLR_COMMAND_STREAM_WRITE_VALUE) {
    bool result = static_cast<KernelBlitManager&>(blitMgr()).streamOpsWrite(*memory, value, offset,
                                                                            sizeBytes);
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Writing value: 0x%lx", value);
    if (!result) {
      LogError("submitStreamOperation: Write failed!");
    }
  } else {
    ShouldNotReachHere();
  }
  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitVirtualMap(amd::VirtualMapCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);
  amd::Memory* phys_mem_obj = vcmd.memory();
  amd::Memory* vaddr_base_obj = amd::MemObjMap::FindVirtualMemObj(vcmd.ptr());
  if (vaddr_base_obj == nullptr || !(vaddr_base_obj->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
    profilingEnd(vcmd);
    return;
  }

  // Create a view, since original base obj will map the whole memory and multimap cases wont work.
  amd::Memory* vaddr_sub_obj = nullptr;
  size_t vaddr_offset = 0;
  if (phys_mem_obj != nullptr) {
    constexpr bool kParent = false;
    vaddr_sub_obj = phys_mem_obj->getContext().devices()[0]->CreateVirtualBuffer(
        phys_mem_obj->getContext(), const_cast<void*>(vcmd.ptr()), vcmd.size(),
        phys_mem_obj->getUserData().deviceId, phys_mem_obj->getUserData().locationType, kParent);

    // Calculate the offset from the original pointer.
    vaddr_offset = (reinterpret_cast<address>(vaddr_sub_obj->getSvmPtr()) -
                    reinterpret_cast<address>(vaddr_base_obj->getSvmPtr()));
  }

  // The imem() in the backend is shared between base and sub/view object.
  pal::Memory* vaddr_pal_mem = dev().getGpuMemory(vaddr_base_obj);
  Pal::IGpuMemory* phymem_igpu_mem =
      (phys_mem_obj == nullptr) ? nullptr : dev().getGpuMemory(phys_mem_obj)->iMem();

  Pal::VirtualMemoryRemapRange range{vaddr_pal_mem->iMem(), vaddr_offset,
                                     phymem_igpu_mem,       0,
                                     vcmd.size(),           Pal::VirtualGpuMemAccessMode::NoAccess};

  // Wait for previous operations before unmap
  if (phys_mem_obj == nullptr) {
    // @note: Need to verify if compute requires a wait or IB flush is enough
    WaitForIdleCompute();
    WaitForIdleSdma();
  }

  eventBegin(MainEngine);
  auto result = queue(MainEngine).iQueue_->RemapVirtualMemoryPages(1, &range, false, nullptr);
  // Capture GPU event for the paging operation
  GpuEvent event;
  eventEnd(MainEngine, event);
  setGpuEvent(event);
  if (result == Pal::Result::Success) {
    if (phys_mem_obj != nullptr) {
      // assert the vaddr_mem_obj wasn't mapped already
      assert(amd::MemObjMap::FindMemObj(vcmd.ptr()) == nullptr);
      amd::MemObjMap::AddMemObj(vcmd.ptr(), vaddr_sub_obj);
      vaddr_sub_obj->getUserData().phys_mem_obj = phys_mem_obj;
      phys_mem_obj->getUserData().vaddr_mem_obj = vaddr_sub_obj;
    } else {
      // assert the vaddr_mem_obj is mapped and needs to be removed
      amd::Memory* vaddr_sub_obj = amd::MemObjMap::FindMemObj(vcmd.ptr());
      assert(vaddr_sub_obj != nullptr);
      assert(vcmd.ptr() == vaddr_sub_obj->getSvmPtr());

      amd::MemObjMap::RemoveMemObj(vcmd.ptr());
      if (vaddr_sub_obj->getUserData().phys_mem_obj != nullptr) {
        vaddr_sub_obj->getUserData().phys_mem_obj->getUserData().vaddr_mem_obj = nullptr;
        vaddr_sub_obj->getUserData().phys_mem_obj = nullptr;
      }
    }
  }
  profilingEnd(vcmd);
}

// ================================================================================================
void VirtualGPU::PrintChildren(const HSAILKernel& hsaKernel, VirtualGPU* gpuDefQueue) {
  AmdAqlWrap* wraps = (AmdAqlWrap*)(&((AmdVQueueHeader*)gpuDefQueue->virtualQueue_->data())[1]);
  uint p = 0;
  for (uint i = 0; i < gpuDefQueue->vqHeader_->aql_slot_num; ++i) {
    if (wraps[i].state != 0) {
      uint j;
      if (p == GPU_PRINT_CHILD_KERNEL) {
        break;
      }
      p++;
      std::stringstream print;
      print.flags(std::ios::right | std::ios_base::hex | std::ios_base::uppercase);
      print << "Slot#: " << i << "\n";
      print << "\tenqueue_flags: " << wraps[i].enqueue_flags << "\n";
      print << "\tcommand_id: " << wraps[i].command_id << "\n";
      print << "\tchild_counter: " << wraps[i].child_counter << "\n";
      print << "\tcompletion: " << wraps[i].completion << "\n";
      print << "\tparent_wrap: " << wraps[i].parent_wrap << "\n";
      print << "\twait_list: " << wraps[i].wait_list << "\n";
      print << "\twait_num: " << wraps[i].wait_num << "\n";
      uint offsEvents = wraps[i].wait_list - gpuDefQueue->virtualQueue_->vmAddress();
      size_t* events = reinterpret_cast<size_t*>(gpuDefQueue->virtualQueue_->data() + offsEvents);
      for (j = 0; j < wraps[i].wait_num; ++j) {
        uint offs = static_cast<uint64_t>(events[j]) - gpuDefQueue->virtualQueue_->vmAddress();
        AmdEvent* eventD = (AmdEvent*)(gpuDefQueue->virtualQueue_->data() + offs);
        print << "Wait Event#: " << j << "\n";
        print << "\tState: " << eventD->state << "; Counter: " << eventD->counter << "\n";
      }
      print << "WorkGroupSize[ " << wraps[i].aql.workgroup_size_x << ", ";
      print << wraps[i].aql.workgroup_size_y << ", ";
      print << wraps[i].aql.workgroup_size_z << "]\n";
      print << "GridSize[ " << wraps[i].aql.grid_size_x << ", ";
      print << wraps[i].aql.grid_size_y << ", ";
      print << wraps[i].aql.grid_size_z << "]\n";

      HSAILKernel* child = nullptr;
      for (auto it = hsaKernel.prog().kernels().begin(); it != hsaKernel.prog().kernels().end();
           ++it) {
        if (wraps[i].aql.kernel_object == static_cast<HSAILKernel*>(it->second)->gpuAqlCode()) {
          child = static_cast<HSAILKernel*>(it->second);
        }
      }
      if (child == nullptr) {
        printf("Error: couldn't find child kernel!\n");
        continue;
      }
      const uint64_t kernarg_address =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wraps[i].aql.kernarg_address));
      uint offsArg = kernarg_address - gpuDefQueue->virtualQueue_->vmAddress();
      address argum = gpuDefQueue->virtualQueue_->data() + offsArg;
      print << "Kernel: " << child->name() << "\n";
      const amd::KernelSignature& signature = child->signature();

      // Check if runtime has to setup hidden arguments
      for (const auto it : signature.parameters()) {
        const char* extraArgName = nullptr;
        switch (it.info_.oclObject_) {
          case amd::KernelParameterDescriptor::HiddenNone:
            // void* zero = 0;
            // WriteAqlArgAt(const_cast<address>(parameters), zero, it.size_, it.offset_);
            break;
          case amd::KernelParameterDescriptor::HiddenGlobalOffsetX:
            extraArgName = "Offset0: ";
            break;
          case amd::KernelParameterDescriptor::HiddenGlobalOffsetY:
            extraArgName = "Offset1: ";
            break;
          case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ:
            extraArgName = "Offset2: ";
            break;
          case amd::KernelParameterDescriptor::HiddenPrintfBuffer:
            extraArgName = "PrintfBuf: ";
            break;
          case amd::KernelParameterDescriptor::HiddenDefaultQueue:
            extraArgName = "VqueuePtr: ";
            break;
          case amd::KernelParameterDescriptor::HiddenCompletionAction:
            extraArgName = "AqlWrap: ";
            break;
          default:
            break;
        }
        if (extraArgName) {
          print << "\t" << extraArgName << *reinterpret_cast<size_t*>(argum);
          print << "\n";
          argum += sizeof(size_t);
          continue;
        }
        print << "\t" << it.name_ << ": ";
        for (int s = it.size_ - 1; s >= 0; --s) {
          print.width(2);
          print.fill('0');
          print << static_cast<uint32_t>(argum[s]);
        }
        argum += it.offset_;
        print << "\n";
      }
      printf("%s", print.str().c_str());
    }
  }
}

// ================================================================================================
bool VirtualGPU::PreDeviceEnqueue(const amd::Kernel& kernel, const HSAILKernel& hsaKernel,
                                  VirtualGPU** gpuDefQueue, uint64_t* vmDefQueue) {
  amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(dev());
  if (nullptr == defQueue) {
    LogError("Default device queue wasn't allocated");
    return false;
  } else {
    if (dev().settings().useDeviceQueue_) {
      *gpuDefQueue = static_cast<VirtualGPU*>(defQueue->vDev());
      if ((*gpuDefQueue)->hwRing() == hwRing()) {
        LogError("Can't submit the child kernels to the same HW ring as the host queue!");
        return false;
      }
    } else {
      createVirtualQueue(defQueue->size());
      *gpuDefQueue = this;
    }
  }
  *vmDefQueue = (*gpuDefQueue)->virtualQueue_->vmAddress();

  (*gpuDefQueue)->writeVQueueHeader(*this, hsaKernel.prog().kernelTable());

  // Acquire USWC memory for the scheduler parameters
  (*gpuDefQueue)->schedParams_ = &xferWrite().Acquire(sizeof(SchedulerParam));

  // Add memory handles before the actual dispatch
  addVmMemory((*gpuDefQueue)->virtualQueue_);
  addVmMemory((*gpuDefQueue)->schedParams_);

  return true;
}

// ================================================================================================
void VirtualGPU::PostDeviceEnqueue(const amd::Kernel& kernel, const HSAILKernel& hsaKernel,
                                   VirtualGPU* gpuDefQueue, uint64_t vmDefQueue,
                                   uint64_t vmParentWrap, GpuEvent* gpuEvent) {
  uint32_t id = gpuEvent->id_;
  amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(dev());

  // Make sure exculsive access to the device queue
  amd::ScopedLock(defQueue->lock());
  Memory& schedParams = xferWrite().Acquire(sizeof(SchedulerParam));

  if (GPU_PRINT_CHILD_KERNEL != 0) {
    waitForEvent(gpuEvent);
    PrintChildren(hsaKernel, gpuDefQueue);
  }

  if (!dev().settings().useDeviceQueue_) {
    // Add the termination handshake to the host queue
    eventBegin(MainEngine);
    iCmd()->CmdVirtualQueueHandshake(vmParentWrap + offsetof(AmdAqlWrap, state), AQL_WRAP_DONE,
                                     vmParentWrap + offsetof(AmdAqlWrap, child_counter), 0,
                                     dev().settings().useDeviceQueue_);
    eventEnd(MainEngine, *gpuEvent);
  }

  // Get the global loop start before the scheduler
  Pal::gpusize loopStart = gpuDefQueue->iCmd()->CmdVirtualQueueDispatcherStart();
  static_cast<KernelBlitManager&>(gpuDefQueue->blitMgr())
      .runScheduler(*gpuDefQueue->virtualQueue_, *gpuDefQueue->schedParams_, 0,
                    gpuDefQueue->vqHeader_->aql_slot_num / (DeviceQueueMaskSize * maskGroups_));
  gpuDefQueue->addBarrier(RgpSqqtBarrierReason::PostDeviceEnqueue, BarrierType::FlushL2);

  // Get the address of PM4 template and add write it to params
  //! @note DMA flush must not occur between patch and the scheduler
  Pal::gpusize patchStart = gpuDefQueue->iCmd()->CmdVirtualQueueDispatcherStart();
  // Program parameters for the scheduler
  SchedulerParam* param = reinterpret_cast<SchedulerParam*>(gpuDefQueue->schedParams_->data());
  param->signal = 1;
  // Scale clock to 1024 to avoid 64 bit div in the scheduler
  param->eng_clk = (1000 * 1024) / dev().info().maxEngineClockFrequency_;
  param->hw_queue = patchStart + sizeof(uint32_t) /* Rewind packet*/;
  param->hsa_queue = gpuDefQueue->hsaQueueMem()->vmAddress();
  param->releaseHostCP = 0;
  param->parentAQL = vmParentWrap;
  param->dedicatedQueue = dev().settings().useDeviceQueue_;

  // Fill the scratch buffer information
  if (hsaKernel.prog().maxScratchRegs() > 0) {
    pal::Memory* scratchBuf = dev().scratch(gpuDefQueue->hwRing())->memObj_;
    param->scratchSize = scratchBuf->size();
    param->scratch = scratchBuf->vmAddress();
    param->numMaxWaves = 32 * dev().info().maxComputeUnits_;
    param->scratchOffset = dev().scratch(gpuDefQueue->hwRing())->offset_;
    addVmMemory(scratchBuf);
  } else {
    param->numMaxWaves = 0;
    param->scratchSize = 0;
    param->scratch = 0;
    param->scratchOffset = 0;
  }

  // Add all kernels in the program to the mem list.
  //! \note Runtime doesn't know which one will be called
  hsaKernel.prog().fillResListWithKernels(*this);

  Pal::gpusize signalAddr = gpuDefQueue->schedParams_->vmAddress();
  gpuDefQueue->eventBegin(MainEngine);
  gpuDefQueue->iCmd()->CmdVirtualQueueDispatcherEnd(
      signalAddr, loopStart,
      gpuDefQueue->vqHeader_->aql_slot_num / (DeviceQueueMaskSize * maskGroups_));
  // Note: Device enqueue can't have extra commands after INDIRECT_BUFFER call.
  // Thus TS command for profiling has to follow in the next CB.
  constexpr bool ForceSubmitFirst = true;
  gpuDefQueue->eventEnd(MainEngine, *gpuEvent, ForceSubmitFirst);

  if (dev().settings().useDeviceQueue_) {
    // Add the termination handshake to the host queue
    eventBegin(MainEngine);
    iCmd()->CmdVirtualQueueHandshake(vmParentWrap + offsetof(AmdAqlWrap, state), AQL_WRAP_DONE,
                                     vmParentWrap + offsetof(AmdAqlWrap, child_counter), signalAddr,
                                     dev().settings().useDeviceQueue_);
    if (id != gpuEvent->id_) {
      LogError("Something is wrong. ID mismatch!\n");
    }
    eventEnd(MainEngine, *gpuEvent);
  }

  xferWrite().Release(*gpuDefQueue->schedParams_);
  gpuDefQueue->schedParams_ = nullptr;
}

// ================================================================================================
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  if (vcmd.cooperativeGroups()) {
    uint32_t workgroups = 1;
    for (uint i = 0; i < vcmd.sizes().dimensions(); i++) {
      if (vcmd.sizes().local()[i] != 0) {
        workgroups *= (vcmd.sizes().global()[i] / vcmd.sizes().local()[i]);
      }
    }

    bool test = true;
    VirtualGPU* queue = (test) ? this : dev().xferQueue();

    // Wait for the execution on the current queue, since the coop groups will use the device queue
    waitAllEngines();

    amd::ScopedLock lock(queue->blitMgr().lockXfer());

    queue->profilingBegin(vcmd);

    static_cast<KernelBlitManager&>(queue->blitMgr()).RunGwsInit(workgroups);
    queue->addBarrier(RgpSqqtBarrierReason::PostDeviceEnqueue);

    // Submit kernel to HW
    if (!queue->submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(), false,
                                     vcmd.sharedMemBytes())) {
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    queue->profilingEnd(vcmd);

    // Wait for the execution on the device queue. Keep the current queue in-order
    queue->waitAllEngines();
  } else {
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(), false,
                              vcmd.sharedMemBytes(), vcmd.getAnyOrderLaunchFlag())) {
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
  }
}

// ================================================================================================
bool VirtualGPU::submitKernelInternal(const amd::NDRangeContainer& sizes, const amd::Kernel& kernel,
                                      const_address parameters, bool nativeMem,
                                      uint32_t sharedMemBytes, bool anyOrder) {
  state_.anyOrder_ = anyOrder;

  // Get the HSA kernel object
  const HSAILKernel& hsaKernel = static_cast<const HSAILKernel&>(*(kernel.getDeviceKernel(dev())));

  // If RGP capturing is enabled, then start SQTT trace
  if (rgpCaptureEna()) {
    size_t newLocalSize[3] = {1, 1, 1};
    size_t newGlobalSize[3] = {0, 0, 0};
    for (uint i = 0; i < sizes.dimensions(); i++) {
      newGlobalSize[i] = sizes.global()[i];
      if (sizes.local()[i] != 0) {
        newLocalSize[i] = sizes.local()[i];
      }
    }
    dev().captureMgr()->PreDispatch(
        this, hsaKernel,
        // Report global size in workgroups, since that's the RGP trace semantics
        newGlobalSize[0] / newLocalSize[0], newGlobalSize[1] / newLocalSize[1],
        newGlobalSize[2] / newLocalSize[2]);
  }

  bool printfEnabled = (hsaKernel.printfInfo().size() > 0) ? true : false;
  if (printfEnabled && !printfDbgHSA().init(*this, printfEnabled)) {
    LogError("Printf debug buffer initialization failed!");
    return false;
  }

  uint64_t vmDefQueue = 0;
  VirtualGPU* gpuDefQueue = nullptr;
  if (hsaKernel.dynamicParallelism()) {
    // Initialize GPU device queue for execution (gpuDefQueue)
    if (!PreDeviceEnqueue(kernel, hsaKernel, &gpuDefQueue, &vmDefQueue)) {
      return false;
    }
  }
  size_t ldsSize;

  ClPrint(amd::LOG_INFO, amd::LOG_KERN, "!\tkernel : %s\n", hsaKernel.name().c_str());

  if (PAL_EMBED_KERNEL_MD) {
    char buf[256];
    sprintf(buf, "kernel: %s\n private mem size: %x\n group mem size: %x\n",
            hsaKernel.name().c_str(), hsaKernel.spillSegSize(), hsaKernel.ldsSize());
    iCmd()->CmdCommentString(buf);
  }

  bool imageBufferWrtBack = false;         // Image buffer write back is required
  std::vector<Image*> wrtBackImageBuffer;  // Array of images for write back

  // Check memory dependency and SVM objects
  if (!processMemObjectsHSA(kernel, parameters, nativeMem, ldsSize, imageBufferWrtBack,
                            wrtBackImageBuffer)) {
    LogError("Wrong memory objects!");
    return false;
  }

  // Add ISA memory object to the resource tracking list
  AddKernel(kernel);

  GpuEvent gpuEvent(queues_[MainEngine]->cmdBufId());
  uint32_t id = gpuEvent.id_;
  uint64_t vmParentWrap = 0;
  uint32_t aql_index = 0;
  // Program the kernel arguments for the GPU execution
  hsa_kernel_dispatch_packet_t* aqlPkt =
      hsaKernel.loadArguments(*this, kernel, sizes, parameters, ldsSize + sharedMemBytes,
                              vmDefQueue, &vmParentWrap, &aql_index);
  assert((nullptr != aqlPkt) && "Couldn't load kernel arguments");

  // Dynamic call stack size is considered to calculate private segment size and scratch regs
  // in LightningKernel::postLoad(). As it is not called during hipModuleLaunchKernel unlike
  // hipLaunchKernel/hipLaunchKernelGGL, Updated value is passed to dispatch packet.
  size_t privateMemSize = hsaKernel.spillSegSize();
  if ((hsaKernel.workGroupInfo()->usedStackSize_ & 0x1) == 0x1) {
    privateMemSize = std::max<uint32_t>(static_cast<uint32_t>(device().StackSize()),
                                        hsaKernel.workGroupInfo()->scratchRegs_ * sizeof(uint32_t));
    // Validate privateMemSize is more than max allowed.
    size_t maxStackSize = device().MaxStackSize();
    if (privateMemSize > maxStackSize) {
      ClPrint(amd::LOG_INFO, amd::LOG_KERN,
              "Scratch size (%zu) exceeds max allowed (%zu) for kernel : %s", privateMemSize,
              maxStackSize, hsaKernel.name().c_str());
      LogError("Scratch size exceeds max allowed.");
      return false;
    }
  }

  // Set up the dispatch information
  Pal::DispatchAqlParams dispatchParam = {};
  dispatchParam.pAqlPacket = aqlPkt;
  if (privateMemSize > 0) {
    const Device::ScratchBuffer* scratch = dev().scratch(hwRing());
    dispatchParam.scratchAddr = scratch->memObj_->vmAddress();
    dispatchParam.scratchSize = scratch->size_;
    dispatchParam.scratchOffset = scratch->offset_;
    dispatchParam.workitemPrivateSegmentSize = privateMemSize;
  }
  dispatchParam.pCpuAqlCode = hsaKernel.cpuAqlKd();
  dispatchParam.hsaQueueVa = hsaQueueMem_->vmAddress();
  if (!hsaKernel.prog().isLC() && hsaKernel.workGroupInfo()->wavesPerSimdHint_ != 0) {
    constexpr uint32_t kWavesPerSimdLimit = 4;
    dispatchParam.wavesPerSh =
        kWavesPerSimdLimit * dev().info().cuPerShaderArray_ * dev().info().simdPerCU_;
  } else {
    dispatchParam.wavesPerSh = 0;
  }
  dispatchParam.useAtc = dev().settings().svmFineGrainSystem_ ? true : false;
  dispatchParam.kernargSegmentSize = hsaKernel.argsBufferSize();
  dispatchParam.aqlPacketIndex = aql_index;
  // Run AQL dispatch in HW
  eventBegin(MainEngine);
  iCmd()->CmdDispatchAql(dispatchParam);

  if (id != gpuEvent.id_) {
    LogError("Something is wrong. ID mismatch!\n");
  }
  eventEnd(MainEngine, gpuEvent);
  AqlPacketUpdateTs(aql_index, gpuEvent);

  // Execute scheduler for device enqueue
  if (hsaKernel.dynamicParallelism()) {
    PostDeviceEnqueue(kernel, hsaKernel, gpuDefQueue, vmDefQueue, vmParentWrap, &gpuEvent);
  }

  // Update the global GPU event
  constexpr bool kNeedFLush = false;
  setGpuEvent(gpuEvent, kNeedFLush);

  if (printfEnabled && !printfDbgHSA().output(*this, printfEnabled, hsaKernel.printfInfo())) {
    LogError("Couldn't read printf data from the buffer!\n");
    return false;
  }

  // Check if image buffer write back is required
  if (imageBufferWrtBack) {
    // Make sure the original kernel execution is done
    addBarrier(RgpSqqtBarrierReason::MemDependency);
    for (const auto imageBuffer : wrtBackImageBuffer) {
      Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
      amd::Image* image = imageBuffer->owner()->asImage();
      amd::Coord3D offs(0);
      // Copy memory from the the backing store image into original buffer
      bool result = blitMgr().copyImageToBuffer(*imageBuffer->CopyImageBuffer(), *buffer, offs,
                                                offs, image->getRegion(), true,
                                                image->getRowPitch(), image->getSlicePitch());
    }
  }

  // Perform post dispatch logic for RGP traces
  if (rgpCaptureEna()) {
    dev().captureMgr()->PostDispatch(this);
  }

  return true;
}

// ================================================================================================
void VirtualGPU::submitNativeFn(amd::NativeFnCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  Unimplemented();  //!< @todo: Unimplemented
}

// ================================================================================================
void VirtualGPU::submitMarker(amd::Marker& vcmd) {
  //!@note runtime doesn't need to lock this command on execution

  if (vcmd.waitingEvent() != nullptr) {
    bool foundEvent = false;

    // Loop through all outstanding command batches
    while (!cbQueue_.empty()) {
      auto cb = cbQueue_.front();
      // Wait for completion
      foundEvent = awaitCompletion(cb, vcmd.waitingEvent());
      // Release a command batch
      freeCbQueue_.push(cb);
      // Remove command batch from the list
      cbQueue_.pop();
      // Early exit if we found a command
      if (foundEvent) break;
    }

    // Event should be in the current command batch
    if (!foundEvent) {
      state_.forceWait_ = true;
    }
  } else if (amd::IS_HIP) {
    // Use GPU based timing for HIP events

    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());
    GpuEvent event;
    profilingBegin(vcmd);
    eventBegin(MainEngine);
    eventEnd(MainEngine, event);
    setGpuEvent(event);
    profilingEnd(vcmd);
  }
}

// ================================================================================================
void VirtualGPU::submitAccumulate(amd::AccumulateCommand& vcmd) {}

// ================================================================================================
void VirtualGPU::submitExternalSemaphoreCmd(amd::ExternalSemaphoreCmd& cmd) {
  const Pal::IQueueSemaphore* sem = reinterpret_cast<const Pal::IQueueSemaphore*>(cmd.sem_ptr());

  if (cmd.semaphoreCmd() == amd::ExternalSemaphoreCmd::COMMAND_SIGNAL_EXTSEMAPHORE) {
    flushDMA(MainEngine);
    if (Pal::Result::Success != queues_[MainEngine]->iQueue_->SignalQueueSemaphore(
                                    const_cast<Pal::IQueueSemaphore*>(sem), cmd.fence())) {
      LogError("Failed to signal external semaphore");
    }
  } else {
    if (Pal::Result::Success != queues_[MainEngine]->iQueue_->WaitQueueSemaphore(
                                    const_cast<Pal::IQueueSemaphore*>(sem), cmd.fence())) {
      LogError("Failed to wait on external semaphore");
    }
  }
}

void VirtualGPU::releaseMemory(GpuMemoryReference* mem) {
  queues_[MainEngine]->removeCmdMemRef(mem);
  if (!dev().settings().disableSdma_) {
    queues_[SdmaEngine]->removeCmdMemRef(mem);
  }
}

void VirtualGPU::submitPerfCounter(amd::PerfCounterCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  const amd::PerfCounterCommand::PerfCounterList counters = vcmd.getCounters();

  PalCounterReference* palRef = PalCounterReference::Create(*this);
  if (palRef == nullptr) {
    LogError("We failed to allocate memory for the GPU perfcounter");
    vcmd.setStatus(CL_INVALID_OPERATION);
    return;
  }

  bool newExperiment = false;

  for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
    amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[i]);
    const PerfCounter* counter = static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

    // Make sure we have a valid gpu performance counter
    if (nullptr == counter) {
      amd::PerfCounter::Properties prop = amdCounter->properties();
      PerfCounter* gpuCounter = new PerfCounter(
          gpuDevice_, palRef, prop[CL_PERFCOUNTER_GPU_BLOCK_INDEX],
          prop[CL_PERFCOUNTER_GPU_COUNTER_INDEX], prop[CL_PERFCOUNTER_GPU_EVENT_INDEX]);
      if (nullptr == gpuCounter) {
        LogError("We failed to allocate memory for the GPU perfcounter");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      } else if (gpuCounter->create()) {
        newExperiment = true;
      } else {
        LogPrintfError(
            "We failed to allocate a perfcounter in PAL.\
                    Block: %d, counter: #d, event: %d",
            gpuCounter->info()->blockIndex_, gpuCounter->info()->counterIndex_,
            gpuCounter->info()->eventIndex_);
      }
      amdCounter->setDeviceCounter(gpuCounter);
    }
  }

  if (newExperiment) {
    palRef->finalize();
  }

  palRef->release();

  Pal::IPerfExperiment* palPerf = nullptr;
  for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
    amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[i]);
    const PerfCounter* counter = static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

    if (palPerf != counter->iPerf()) {
      palPerf = counter->iPerf();
      // Find the state and sends the command to PAL
      if (vcmd.getState() == amd::PerfCounterCommand::Begin) {
        state_.perfCounterEnabled_ = true;
        GpuEvent event;
        eventBegin(MainEngine);
        iCmd()->CmdBeginPerfExperiment(palPerf);
        eventEnd(MainEngine, event);
        setGpuEvent(event);
      } else if (vcmd.getState() == amd::PerfCounterCommand::End) {
        GpuEvent event;
        eventBegin(MainEngine);
        iCmd()->CmdEndPerfExperiment(palPerf);
        eventEnd(MainEngine, event);
        setGpuEvent(event);
        state_.perfCounterEnabled_ = false;
      } else {
        LogError("Unsupported performance counter state");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
  }
}

void VirtualGPU::submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  switch (cmd.type()) {
    case CL_COMMAND_THREAD_TRACE_MEM: {
      amd::ThreadTrace* amdThreadTrace = &cmd.getThreadTrace();
      ThreadTrace* threadTrace = static_cast<ThreadTrace*>(amdThreadTrace->getDeviceThreadTrace());

      if (threadTrace == nullptr) {
        PalThreadTraceReference* palRef = PalThreadTraceReference::Create(*this);
        if (palRef == nullptr) {
          LogError("Failure in memory allocation for the GPU threadtrace");
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }

        size_t numSe = amdThreadTrace->deviceSeNumThreadTrace();

        ThreadTrace* gpuThreadTrace = new ThreadTrace(gpuDevice_, palRef, cmd.getMemList(), numSe);
        if (nullptr == gpuThreadTrace) {
          LogError("Failure in memory allocation for the GPU threadtrace");
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }

        if (gpuThreadTrace->create()) {
          amdThreadTrace->setDeviceThreadTrace(gpuThreadTrace);
        } else {
          LogError("Failure in memory allocation for the GPU threadtrace");
          delete gpuThreadTrace;
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }

        palRef->finalize();
        palRef->release();
      }

      break;
    }
    default:
      LogError("Unsupported command type for ThreadTraceMemObjects!");
      break;
  }

  profilingEnd(cmd);
}

void VirtualGPU::submitThreadTrace(amd::ThreadTraceCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  switch (cmd.type()) {
    case CL_COMMAND_THREAD_TRACE: {
      amd::ThreadTrace* amdThreadTrace = static_cast<amd::ThreadTrace*>(&cmd.getThreadTrace());
      ThreadTrace* threadTrace = static_cast<ThreadTrace*>(amdThreadTrace->getDeviceThreadTrace());

      // gpu thread trace object had to be generated prior to begin/end/pause/resume due
      // to ThreadTraceMemObjectsCommand execution
      if (threadTrace == nullptr) {
        return;
      } else {
        Pal::IPerfExperiment* palPerf = threadTrace->iPerf();
        if (cmd.getState() == amd::ThreadTraceCommand::Begin) {
          amd::ThreadTrace::ThreadTraceConfig* traceCfg =
              static_cast<amd::ThreadTrace::ThreadTraceConfig*>(cmd.threadTraceConfig());
          iCmd()->CmdBeginPerfExperiment(palPerf);
        } else if (cmd.getState() == amd::ThreadTraceCommand::End) {
          GpuEvent event;
          eventBegin(MainEngine);
          iCmd()->CmdEndPerfExperiment(palPerf);
          threadTrace->populateUserMemory();
          eventEnd(MainEngine, event);
          setGpuEvent(event);
        } else if (cmd.getState() == amd::ThreadTraceCommand::Pause) {
          // There's no Pause from the PerfExperiment interface
        } else if (cmd.getState() == amd::ThreadTraceCommand::Resume) {
          // There's no Resume from the PerfExperiment interface
        }
      }
      break;
    }
    default:
      LogError("Unsupported command type for ThreadTrace!");
      break;
  }

  profilingEnd(cmd);
}

void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (const auto& it : vcmd.getMemList()) {
    // amd::Memory object should never be nullptr
    assert(it && "Memory object for interop is nullptr");
    pal::Memory* memory = dev().getGpuMemory(it);

    // If resource is a shared copy of original resource, then
    // runtime needs to copy data from original resource
    it->getInteropObj()->copyOrigToShared();
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (const auto& it : vcmd.getMemList()) {
    // amd::Memory object should never be nullptr
    assert(it && "Memory object for interop is nullptr");
    pal::Memory* memory = dev().getGpuMemory(it);

    // If resource is a shared copy of original resource, then
    // runtime needs to copy data back to original resource
    it->getInteropObj()->copySharedToOrig();
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSignal(amd::SignalCommand& vcmd) {
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);
  pal::Memory* pGpuMemory = dev().getGpuMemory(&vcmd.memory());

  GpuEvent gpuEvent;
  uint32_t value = vcmd.markerValue();

  if (vcmd.type() == CL_COMMAND_WAIT_SIGNAL_AMD) {
    eventBegin(MainEngine);
    addVmMemory(pGpuMemory);

    iCmd()->CmdWaitBusAddressableMemoryMarker(*(pGpuMemory->iMem()), value, 0xFFFFFFFF,
                                              Pal::CompareFunc::GreaterEqual);
    eventEnd(MainEngine, gpuEvent);

  } else if (vcmd.type() == CL_COMMAND_WRITE_SIGNAL_AMD) {
    EngineType activeEngineID = engineID_;
    engineID_ = static_cast<EngineType>(pGpuMemory->getGpuEvent(*this)->engineId_);

    // Make sure GPU finished operation and data reached memory before the marker write
    addBarrier(RgpSqqtBarrierReason::SignalSubmit, BarrierType::FlushL2);
    // Workarounds: We had systems where an extra delay was necessary.
    {
      // Flush CB associated with the DGMA buffer
      isDone(pGpuMemory->getGpuEvent(*this));
    }

    eventBegin(engineID_);
    queues_[engineID_]->addCmdMemRef(pGpuMemory->memRef());

    queues_[engineID_]->iCmd()->
#if (PAL_CLIENT_INTERFACE_MAJOR_VERSION < 396)
        CmdUpdateBusAddressableMemoryMarker(*(pGpuMemory->iMem()), value);
#else
        CmdUpdateBusAddressableMemoryMarker(*(pGpuMemory->iMem()), vcmd.markerOffset(), value);
#endif
    eventEnd(engineID_, gpuEvent);

    // Restore the original engine
    engineID_ = activeEngineID;
  }

  // Update the global GPU event
  setGpuEvent(gpuEvent);

  profilingEnd(vcmd);
}

void VirtualGPU::submitMakeBuffersResident(amd::MakeBuffersResidentCommand& vcmd) {
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);

  std::vector<amd::Memory*> memObjects = vcmd.memObjects();
  uint32_t numObjects = memObjects.size();
  Pal::GpuMemoryRef* pGpuMemRef = new Pal::GpuMemoryRef[numObjects];
  Pal::IGpuMemory** pGpuMems = new Pal::IGpuMemory*[numObjects];

  for (uint i = 0; i < numObjects; i++) {
    pal::Memory* pGpuMemory = dev().getGpuMemory(memObjects[i]);
    pGpuMemory->syncCacheFromHost(*this);

    pGpuMemRef[i].pGpuMemory = pGpuMemory->iMem();
    pGpuMems[i] = pGpuMemory->iMem();
  }

  dev().iDev()->AddGpuMemoryReferences(numObjects, pGpuMemRef, queues_[MainEngine]->iQueue_,
                                       Pal::GpuMemoryRefCantTrim);
  {
    amd::ScopedLock l(queues_[MainEngine]->lock_);
    dev().iDev()->InitBusAddressableGpuMemory(queues_[MainEngine]->iQueue_, numObjects, pGpuMems);
  }

  if (numObjects != 0) {
    dev().iDev()->RemoveGpuMemoryReferences(numObjects, &pGpuMems[0], queues_[MainEngine]->iQueue_);
  }

  for (uint i = 0; i < numObjects; i++) {
    vcmd.busAddress()[i].surface_bus_address = pGpuMems[i]->Desc().surfaceBusAddr;
    vcmd.busAddress()[i].marker_bus_address = pGpuMems[i]->Desc().markerBusAddr;
  }
  profilingEnd(vcmd);
}


bool VirtualGPU::awaitCompletion(CommandBatch* cb, const amd::Event* waitingEvent) {
  bool found = false;
  amd::Command* current;
  amd::Command* head = cb->head_;

  // Make sure that profiling is enabled
  if (state_.profileEnabled_) {
    return profilingCollectResults(cb, waitingEvent);
  }
  // Mark the first command in the batch as running
  if (head != nullptr) {
    head->setStatus(CL_RUNNING);
  } else {
    return found;
  }

  // Wait for the last known GPU event
  waitEventLock(cb);

  while (nullptr != head) {
    current = head->getNext();
    if (head->status() == CL_SUBMITTED) {
      head->setStatus(CL_RUNNING);
      head->setStatus(CL_COMPLETE);
    } else if (head->status() == CL_RUNNING) {
      head->setStatus(CL_COMPLETE);
    } else if ((head->status() != CL_COMPLETE) && (current != nullptr)) {
      LogPrintfError("Unexpected command status - %d!", head->status());
    }

    // Check if it's a waiting command
    if (head == waitingEvent) {
      found = true;
    }

    head->release();
    head = current;
  }

  return found;
}

void VirtualGPU::flush(amd::Command* list, bool wait) {
  CommandBatch* cb = nullptr;
  bool gpuCommand = false;

  for (uint i = 0; i < AllEngines; ++i) {
    if (events_[i].isValid()) {
      gpuCommand = true;
    }
  }

  // If the batch doesn't have any GPU command and the list is empty
  if (!gpuCommand && cbQueue_.empty()) {
    state_.forceWait_ = true;
  }

  // Insert the current batch into a list
  if (nullptr != list) {
    if (!freeCbQueue_.empty()) {
      cb = freeCbQueue_.front();
    }

    if (nullptr == cb) {
      cb = new CommandBatch(list, events_, lastTS_);
    } else {
      freeCbQueue_.pop();
      cb->init(list, events_, lastTS_);
    }
  }

  {
    //! @todo: Check if really need a lock
    amd::ScopedLock lock(execution());
    for (uint i = 0; i < AllEngines; ++i) {
      flushDMA(i);
      // Reset event so we won't try to wait again,
      // if runtime didn't submit any commands
      //! @note: it's safe to invalidate events, since
      //! we already saved them with the batch creation step above
      events_[i].invalidate();
    }
  }

  // Mark last TS as nullptr, so runtime won't process empty batches with the old TS
  lastTS_ = nullptr;
  if (nullptr != cb) {
    cbQueue_.push(cb);
  }

  wait |= state_.forceWait_;
  // Loop through all outstanding command batches
  while (!cbQueue_.empty()) {
    cb = cbQueue_.front();
    // Check if command batch finished without a wait
    bool finished = true;
    for (uint i = 0; i < AllEngines; ++i) {
      finished &= isDone(&cb->events_[i]);
    }
    if (finished || wait) {
      // Wait for completion
      awaitCompletion(cb);
      // Release a command batch
      freeCbQueue_.push(cb);
      // Remove command batch from the list
      cbQueue_.pop();
    } else {
      // Early exit if no finished
      break;
    }
  }
  state_.forceWait_ = false;
}

void VirtualGPU::enableSyncedBlit() const { return blitMgr_->enableSynchronization(); }

void VirtualGPU::setGpuEvent(GpuEvent gpuEvent, bool flush) {
  events_[engineID_] = gpuEvent;

  // Flush current DMA buffer if requested
  if (flush) {
    flushDMA(engineID_);
  }
}

void VirtualGPU::flushDMA(uint engineID) {
  if (engineID == MainEngine) {
    // Clear memory dependency state, since runtime flushes compute
    // memoryDependency().clear();
    //!@todo Keep memory dependency alive even if we flush DMA,
    //! since only L2 cache is flushed in KMD frame,
    //! but L1 still has to be invalidated.
  }

  isDone(&events_[engineID]);
}

bool VirtualGPU::waitAllEngines(CommandBatch* cb) {
  uint i;
  GpuEvent* events;  //!< GPU events for the batch

  // If command batch is nullptr then wait for the current
  if (nullptr == cb) {
    events = events_;
  } else {
    events = cb->events_;
  }

  bool earlyDone = true;
  // The first loop is to flush all engines and/or check if
  // engines are idle already
  for (i = 0; i < AllEngines; ++i) {
    earlyDone &= isDone(&events[i]);
  }

  // Rlease all pinned memory
  releasePinnedMem();

  // The second loop is to wait all engines
  for (i = 0; i < AllEngines; ++i) {
    waitForEvent(&events[i]);
  }

  return earlyDone;
}

void VirtualGPU::waitEventLock(CommandBatch* cb) {
  bool earlyDone = false;
  {
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());
    earlyDone = waitAllEngines(cb);
  }
  // Get timestamp, incase readjustTimeGPU_ needs to be updated
  uint64_t endTimeStampCPU = amd::Os::timeNanos();

  // Free resource cache if we have too many entries
  //! \note we do it here, when all engines are idle,
  // because Vista/Win7 idles GPU on a resource destruction
  static const size_t MinCacheEntries = 4096;
  dev().resourceCache().free(MinCacheEntries);

  // Find the timestamp object of the last command in the batch
  if (cb->lastTS_ != nullptr) {
    // If earlyDone is TRUE, then CPU didn't wait for GPU.
    // Thus the sync point between CPU and GPU is unclear and runtime
    // will use an older adjustment value to maintain the same timeline
    if (!earlyDone ||
        //! \note Workaround for APU(s).
        //! GPU-CPU timelines may go off too much, thus always
        //! force calibration with the last batch in the list
        (cbQueue_.size() <= 1) || (readjustTimeGPU_ == 0)) {
      uint64_t startTimeStampGPU = 0;
      uint64_t endTimeStampGPU = 0;

      // Get the timestamp value of the last command in the batch
      cb->lastTS_->value(&startTimeStampGPU, &endTimeStampGPU);

      // Adjust the base time by the execution time
      readjustTimeGPU_ = endTimeStampGPU - endTimeStampCPU;
    }
  }
}

bool VirtualGPU::allocConstantBuffers() {
  // Allocate constant buffers.
  // Use double size, reported to the app to account for internal arguments
  const uint32_t MinCbSize = 2 * dev().info().maxParameterSize_;
  uint i;

  // Create/reallocate constant buffer resources
  for (i = 0; i < MaxConstBuffersArguments; ++i) {
    ConstantBuffer* constBuf = new ConstantBuffer(managedBuffer_, MinCbSize);

    if ((constBuf != nullptr) && constBuf->Create()) {
      addConstBuffer(constBuf);
    } else {
      // We failed to create a constant buffer
      delete constBuf;
      return false;
    }
  }

  return true;
}

void VirtualGPU::profilingBegin(amd::Command& command) {
  // Is profiling enabled?
  if (command.profilingInfo().enabled_) {
    // Allocate a timestamp object from the cache
    TimeStamp* ts = tsCache_->allocTimeStamp();
    if (nullptr == ts) {
      return;
    }
    // Save the TimeStamp object in the current OCL event
    command.data().emplace_back(ts);
    profileTs_ = ts;
    state_.profileEnabled_ = true;
  }
}

void VirtualGPU::profilingEnd(amd::Command& command) {
  // Get the TimeStamp object associated witht the current command
  TimeStamp* ts =
      !command.data().empty() ? reinterpret_cast<TimeStamp*>(command.data().back()) : nullptr;
  if (ts != nullptr) {
    // Check if the command actually did any GPU submission
    if (ts->isValid()) {
      lastTS_ = ts;
    } else {
      // Destroy the TimeStamp object
      tsCache_->freeTimeStamp(ts);
      command.data().clear();
    }
  }
}

bool VirtualGPU::profilingCollectResults(CommandBatch* cb, const amd::Event* waitingEvent) {
  bool found = false;
  amd::Command* current;
  amd::Command* first = cb->head_;

  // If the command list is, empty then exit
  if (nullptr == first) {
    return found;
  }

  // Wait for the last known GPU events on all engines
  waitEventLock(cb);

  // Find the CPU base time of the entire command batch execution
  uint64_t endTimeStamp = amd::Os::timeNanos();
  uint64_t startTimeStamp = endTimeStamp;

  // First step, walk the command list to find the first valid command
  //! \note The batch may have empty markers at the beginning.
  //! So the start/end of the empty commands is equal to
  //! the start of the first valid command in the batch.
  first = cb->head_;
  while (nullptr != first) {
    // Get the TimeStamp object associated witht the current command
    TimeStamp* ts =
        !first->data().empty() ? reinterpret_cast<TimeStamp*>(first->data().back()) : nullptr;

    if (ts != nullptr) {
      ts->value(&startTimeStamp, &endTimeStamp);
      endTimeStamp -= readjustTimeGPU_;
      startTimeStamp -= readjustTimeGPU_;
      // Assign to endTimeStamp the start of the first valid command
      endTimeStamp = startTimeStamp;
      break;
    }
    first = first->getNext();
  }

  // Second step, walk the command list to construct the time line
  first = cb->head_;
  while (nullptr != first) {
    // Get the TimeStamp object associated witht the current command
    TimeStamp* ts =
        !first->data().empty() ? reinterpret_cast<TimeStamp*>(first->data().back()) : nullptr;

    current = first->getNext();

    if (ts != nullptr) {
      ts->value(&startTimeStamp, &endTimeStamp);
      endTimeStamp -= readjustTimeGPU_;
      startTimeStamp -= readjustTimeGPU_;
      // Destroy the TimeStamp object
      tsCache_->freeTimeStamp(ts);
      first->data().clear();
    } else {
      // For empty commands start/end is equal to
      // the end of the last valid command
      startTimeStamp = endTimeStamp;
    }

    // Update the command status with the proper timestamps
    if (first->status() == CL_SUBMITTED) {
      first->setStatus(CL_RUNNING, startTimeStamp);
      first->setStatus(CL_COMPLETE, endTimeStamp);
    } else if (first->status() == CL_RUNNING) {
      first->setStatus(CL_COMPLETE, endTimeStamp);
    } else if ((first->status() != CL_COMPLETE) && (current != nullptr)) {
      LogPrintfError("Unexpected command status - %d!", first->status());
    }

    // Do we wait this event?
    if (first == waitingEvent) {
      found = true;
    }

    first->release();
    first = current;
  }

  return found;
}

void VirtualGPU::addDoppRef(const Memory* memory, bool lastDoppCmd, bool pfpaDoppCmd) {
  queues_[MainEngine]->addCmdDoppRef(memory->iMem(), lastDoppCmd, pfpaDoppCmd);
}

void VirtualGPU::profileEvent(EngineType engine, bool type) const {
  if (nullptr == profileTs_) {
    return;
  }
  if (type) {
    profileTs_->begin();
  } else {
    profileTs_->end();
  }
}

bool VirtualGPU::processMemObjectsHSA(const amd::Kernel& kernel, const_address params,
                                      bool nativeMem, size_t& ldsAddress, bool& imageBufferWrtBack,
                                      std::vector<Image*>& wrtBackImageBuffer) {
  const amd::KernelParameters& kernelParams = kernel.parameters();

  // Mark the tracker with a new kernel,
  // so we can avoid checks of the aliased objects
  memoryDependency().newKernel();

  size_t count = kernelParams.getNumberOfSvmPtr();
  if (count > 0) {
    bool supportFineGrainedSystem = dev().isFineGrainedSystem(true);
    FGSStatus status = kernelParams.getSvmSystemPointersSupport();
    switch (status) {
      case FGS_YES:
        if (!supportFineGrainedSystem) {
          return false;
        }
        break;
      case FGS_NO:
        supportFineGrainedSystem = false;
        break;
      case FGS_DEFAULT:
      default:
        break;
    }
    // get svm non arugment information
    void* const* svmPtrArray =
        reinterpret_cast<void* const*>(params + kernelParams.getExecInfoOffset());
    for (size_t i = 0; i < count; i++) {
      amd::Memory* memory = amd::MemObjMap::FindMemObj(svmPtrArray[i]);
      if (nullptr == memory) {
        if (!supportFineGrainedSystem) {
          return false;
        } else {
          addBarrier(RgpSqqtBarrierReason::MemDependency);
          // Clear memory dependency state
          const static bool All = true;
          memoryDependency().clear(!All);
          continue;
        }
      } else {
        // Validate Mem Access in case of VMM Memory
        if (!memory->ValidateMemAccess(dev(), true)) {
          return false;
        }

        Memory* gpuMemory = dev().getGpuMemory(memory);
        if (nullptr != gpuMemory) {
          // Synchronize data with other memory instances if necessary
          gpuMemory->syncCacheFromHost(*this);

          const static bool IsReadOnly = false;
          // Validate SVM passed in the non argument list
          memoryDependency().validate(*this, gpuMemory, IsReadOnly);

          // Wait for resource if it was used on an inactive engine
          //! \note syncCache may call DRM transfer
          constexpr bool WaitOnBusyEngine = true;
          gpuMemory->wait(*this, WaitOnBusyEngine);

          // Mark signal write for cache coherency,
          // since this object isn't a part of kernel arg setup
          if ((memory->getMemFlags() & CL_MEM_READ_ONLY) == 0) {
            memory->signalWrite(&dev());
          }
          addVmMemory(gpuMemory);
        } else {
          return false;
        }
      }
    }
  }

  bool srdResource = false;
  amd::Memory* const* memories =
      reinterpret_cast<amd::Memory* const*>(params + kernelParams.memoryObjOffset());
  const HSAILKernel& hsaKernel = static_cast<const HSAILKernel&>(*(kernel.getDeviceKernel(dev())));
  const amd::KernelSignature& signature = kernel.signature();
  ldsAddress = hsaKernel.ldsSize();

  if (!nativeMem) {
    // Process cache coherency first, since the extra transfers may affect
    // other mem dependency tracking logic: TS and signalWrite()
    for (uint i = 0; i < signature.numMemories(); ++i) {
      amd::Memory* mem = memories[i];
      if (mem != nullptr) {
        // Synchronize data with other memory instances if necessary
        dev().getGpuMemory(mem)->syncCacheFromHost(*this);
      }
    }
  }

  // Check all parameters for the current kernel
  for (size_t i = 0; i < signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    const amd::KernelParameterDescriptor::InfoData& info = desc.info_;

    // Find if current argument is a buffer
    if (desc.type_ == T_POINTER) {
      // If it is a local pointer
      if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
        ldsAddress = amd::alignUp(ldsAddress, desc.info_.arrayIndex_);
        if (desc.size_ == 8) {
          // Save the original LDS size
          uint64_t ldsSize = *reinterpret_cast<const uint64_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          WriteAqlArgAt(const_cast<address>(params), ldsAddress, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        } else {
          // Save the original LDS size
          uint32_t ldsSize = *reinterpret_cast<const uint32_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          uint32_t ldsAddr = ldsAddress;
          WriteAqlArgAt(const_cast<address>(params), ldsAddr, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        }
      } else {
        Memory* gpuMem = nullptr;
        amd::Memory* mem = nullptr;
        uint32_t index = info.arrayIndex_;
        if (nativeMem) {
          gpuMem = reinterpret_cast<Memory* const*>(memories)[index];
          if (nullptr != gpuMem) {
            mem = gpuMem->owner();
          }
        } else {
          mem = memories[index];
          if (mem != nullptr) {
            gpuMem = dev().getGpuMemory(mem);
          }
        }
        //! This condition is for SVM fine-grain
        if ((gpuMem == nullptr) && dev().isFineGrainedSystem(true)) {
          addBarrier(RgpSqqtBarrierReason::MemDependency);
          // Clear memory dependency state
          const static bool All = true;
          memoryDependency().clear(!All);
          continue;
        } else if (gpuMem != nullptr) {
          // Validate memory for a dependency in the queue
          memoryDependency().validate(*this, gpuMem, info.readOnly_);
          // Wait for resource if it was used on an inactive engine
          //! \note syncCache may call DRM transfer
          constexpr bool WaitOnBusyEngine = true;
          gpuMem->wait(*this, WaitOnBusyEngine);

          addVmMemory(gpuMem);
          const void* globalAddress = *reinterpret_cast<const void* const*>(params + desc.offset_);
          logVmMemory(desc.name_, gpuMem);

          //! Check if compiler expects read/write.
          //! Note: SVM with subbuffers has an issue with tracking.
          //! Conformance can send read only subbuffer, but update the region
          //! in the kernel.
          if ((mem != nullptr) && ((!info.readOnly_ && (mem->getSvmPtr() == nullptr)) ||
                                   ((mem->getMemFlags() & CL_MEM_READ_ONLY) == 0))) {
            mem->signalWrite(&dev());
          }
          if (info.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
            if (gpuMem->memoryType() == Resource::ImageBuffer) {
              Image* imageBuffer = static_cast<Image*>(gpuMem);
              // Check if synchronization has to be performed
              if (imageBuffer->CopyImageBuffer() != nullptr) {
                Memory* buffer = dev().getGpuMemory(mem->parent());
                amd::Image* image = mem->asImage();
                amd::Coord3D offs(0);
                // Copy memory from the original image buffer into the backing store image
                bool result = blitMgr().copyBufferToImage(
                    *buffer, *imageBuffer->CopyImageBuffer(), offs, offs, image->getRegion(), true,
                    image->getRowPitch(), image->getSlicePitch());
                // Make sure the copy operation is done
                addBarrier(RgpSqqtBarrierReason::MemDependency);
                // Use backing store SRD as the replacment
                uint64_t srd = imageBuffer->CopyImageBuffer()->hwSrd();
                WriteAqlArgAt(const_cast<address>(params), srd, sizeof(srd), desc.offset_);
                // Add backing store image to the list of memory handles
                addVmMemory(imageBuffer->CopyImageBuffer());
                // If it's not a read only resource, then runtime has to write back
                if (!info.readOnly_) {
                  wrtBackImageBuffer.push_back(imageBuffer);
                  imageBufferWrtBack = true;
                }
              }
            }

            //! \note Special case for the image views.
            //! Copy SRD to CB1, so blit manager will be able to release
            //! this view without a wait for SRD resource.
            if (gpuMem->memoryType() == Resource::ImageView) {
              // Copy the current image SRD into CB1
              uint64_t srd = cb(1)->UploadDataToHw(gpuMem->hwState(), HsaImageObjectSize);
              // Then use a pointer in aqlArgBuffer to CB1
              // Patch the GPU VA address in the original arguments
              WriteAqlArgAt(const_cast<address>(params), srd, sizeof(srd), desc.offset_);
              addVmMemory(cb(1)->ActiveMemory());
            } else {
              srdResource = true;
            }
            if (gpuMem->desc().isDoppTexture_) {
              addDoppRef(gpuMem, kernel.parameters().getExecNewVcop(),
                         kernel.parameters().getExecPfpaVcop());
            }
          }
        }
      }
    } else if (desc.type_ == T_VOID) {
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ReferenceObject) {
        // Copy the current structure into CB1
        size_t gpuPtr =
            static_cast<size_t>(cb(1)->UploadDataToHw(params + desc.offset_, desc.size_));
        // Then use a pointer in aqlArgBuffer to CB1
        const auto it = hsaKernel.patch().find(desc.offset_);
        // Patch the GPU VA address in the original arguments
        WriteAqlArgAt(const_cast<address>(params), gpuPtr, sizeof(size_t), it->second);
        addVmMemory(cb(1)->ActiveMemory());
      }
    } else if (desc.type_ == T_SAMPLER) {
      srdResource = true;
    } else if (desc.type_ == T_QUEUE) {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::DeviceQueue* queue =
          reinterpret_cast<amd::DeviceQueue* const*>(params + kernelParams.queueObjOffset())[index];
      VirtualGPU* gpuQueue = static_cast<VirtualGPU*>(queue->vDev());
      uint64_t vmQueue;
      if (dev().settings().useDeviceQueue_) {
        vmQueue = gpuQueue->vQueue()->vmAddress();
      } else {
        if (!createVirtualQueue(queue->size())) {
          LogError("Virtual queue creation failed!");
          return false;
        }
        vmQueue = vQueue()->vmAddress();
      }
      // Patch the GPU VA address in the original arguments
      WriteAqlArgAt(const_cast<address>(params), vmQueue, sizeof(vmQueue), desc.offset_);
      break;
    }
  }

  if (ldsAddress > dev().info().localMemSize_) {
    LogError("No local memory available\n");
    return false;
  }

  if (srdResource || hsaKernel.prog().isStaticSampler()) {
    dev().srds().fillResourceList(*this);
  }

  const static bool IsReadOnly = false;
  for (const pal::Memory* mem : hsaKernel.prog().globalStores()) {
    // Validate global store for a dependency in the queue
    memoryDependency().validate(*this, mem, IsReadOnly);
    addVmMemory(mem);
  }
  if (hsaKernel.prog().hasGlobalStores()) {
    // Validate code object for a dependency in the queue
    memoryDependency().validate(*this, &hsaKernel.prog().codeSegGpu(), IsReadOnly);
  }

  addVmMemory(&hsaKernel.prog().codeSegGpu());

  if (hsaKernel.workGroupInfo()->scratchRegs_ > 0) {
    const Device::ScratchBuffer* scratch = dev().scratch(hwRing());
    // Validate scratch buffer to force sync mode, because
    // the current scratch logic is optimized for size and performance
    // Note: runtime can skip sync if the same kernel is used,
    // since the number of scratch regs remains the same
    if (!IsSameKernel(kernel)) {
      memoryDependency().validate(*this, scratch->memObj_, IsReadOnly);
    }
    addVmMemory(scratch->memObj_);
    logVmMemory("scratch", scratch->memObj_);
  }

  // Synchronize dispatches unconditionally in case memory tracking is disabled
  memoryDependency().sync(*this);

  return true;
}

void VirtualGPU::writeVQueueHeader(VirtualGPU& hostQ, const Memory* kernelTable) {
  if (nullptr == kernelTable) {
    vqHeader_->kernel_table = 0;
  } else {
    vqHeader_->kernel_table = kernelTable->vmAddress();
    addVmMemory(kernelTable);
  }

  virtualQueue_->writeRawData(hostQ, 0, sizeof(AmdVQueueHeader), vqHeader_, true);
}

bool VirtualGPU::validateSdmaOverlap(const Resource& src, const Resource& dst) {
  uint64_t srcVmEnd = src.vmAddress() + src.vmSize();
  if (((src.vmAddress() >= sdmaRange_.start_) && (src.vmAddress() <= sdmaRange_.end_)) ||
      ((srcVmEnd >= sdmaRange_.start_) && (srcVmEnd <= sdmaRange_.end_)) ||
      ((src.vmAddress() <= sdmaRange_.start_) && (srcVmEnd >= sdmaRange_.end_))) {
    sdmaRange_.start_ = dst.vmAddress();
    sdmaRange_.end_ = dst.vmAddress() + dst.vmSize();
    return true;
  }

  sdmaRange_.start_ = std::min(sdmaRange_.start_, dst.vmAddress());
  sdmaRange_.end_ = std::max(sdmaRange_.end_, dst.vmAddress() + dst.vmSize());
  return false;
}

// ================================================================================================
void* VirtualGPU::getOrCreateHostcallBuffer() {
  if (hostcallBuffer_ != nullptr) {
    return hostcallBuffer_;
  }

  // The number of packets required in each buffer is at least equal to the
  // maximum number of waves supported by the device.
  auto wavesPerCu = dev().info().maxThreadsPerCU_ / dev().info().wavefrontWidth_;
  auto numPackets = dev().info().maxComputeUnits_ * wavesPerCu;

  auto size = amd::getHostcallBufferSize(numPackets);
  auto align = amd::getHostcallBufferAlignment();

  hostcallBuffer_ = dev().svmAlloc(dev().context(), size, align,
                                   CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, nullptr);
  if (!hostcallBuffer_) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE, "Failed to create hostcall buffer");
    return nullptr;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "Created hostcall buffer %p (numPackets == %d, size == %d, align == %d) for virtual "
          "queue %p\n",
          hostcallBuffer_, numPackets, size, align, this);

  if (!amd::enableHostcalls(dev(), hostcallBuffer_, numPackets)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE, "Failed to register hostcall buffer %p with listener",
            hostcallBuffer_);
    return nullptr;
  }
  return hostcallBuffer_;
}

}  // namespace amd::pal
