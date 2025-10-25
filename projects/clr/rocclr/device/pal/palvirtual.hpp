/* Copyright (c) 2015 - 2025 Advanced Micro Devices, Inc.

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

#include <queue>
#include <inttypes.h>
#include "device/pal/paldefs.hpp"
#include "device/pal/palconstbuf.hpp"
#include "device/pal/palprintf.hpp"
#include "device/pal/paltimestamp.hpp"
#include "device/pal/palsched.hpp"
#include "device/pal/palgpuopen.hpp"
#include "platform/commandqueue.hpp"
#include "device/blit.hpp"
#include "palUtil.h"
#include "palCmdBuffer.h"
#include "palCmdAllocator.h"
#include "palQueue.h"
#include "palFence.h"
#include "palLinearAllocator.h"

/*! \addtogroup PAL PAL Resource Implementation
 *  @{
 */

//! PAL Device Implementation
namespace amd::pal {

class Device;
class Kernel;
class Memory;
class CalCounterReference;
class VirtualGPU;
class Program;
class BlitManager;
class ThreadTrace;
class HSAILKernel;

struct AqlPacketMgmt : public amd::EmbeddedObject {
  static constexpr uint32_t kAqlPacketsListSize = 4 * Ki;
  AqlPacketMgmt() : packet_index_(0) { memset(aql_vgpus_, 0, sizeof(aql_vgpus_)); }

  hsa_kernel_dispatch_packet_t aql_packets_[kAqlPacketsListSize];  //!< The list of AQL packets
  GpuEvent aql_events_[kAqlPacketsListSize];    //!< The list of gpu for each AQL packet
  VirtualGPU* aql_vgpus_[kAqlPacketsListSize];  //!< The list of vgpus which had submissions
  std::atomic<uint64_t> packet_index_;          //!< The active packet slot index
};

enum class BarrierType : uint8_t {
  KernelToKernel = 0,
  KernelToCopy,
  CopyToKernel,
  CopyToCopy,
  FlushL2
};

//! Virtual GPU
class VirtualGPU : public device::VirtualDevice {
 public:
  class Queue : public amd::HeapObject {
   public:
    static constexpr uint MaxCommands = 256;
    static constexpr uint StartCmdBufIdx = 1;
    static constexpr uint FirstMemoryReference = 0x80000000;
    static constexpr uint64_t WaitTimeoutInNsec = 6000000000;
    static constexpr uint64_t PollIntervalInNsec = 200000;

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    static Queue* Create(VirtualGPU& gpu,                       //!< ROCCLR virtual GPU object
                         Pal::QueueType queueType,              //!< PAL queue type
                         uint engineIdx,                        //!< Select particular engine index
                         Pal::ICmdAllocator* cmdAlloc,          //!< PAL CMD buffer allocator
                         uint rtCU,                             //!< The number of reserved CUs
                         amd::CommandQueue::Priority priority,  //!< Queue priority
                         uint64_t residency_limit,              //!< Enables residency limit
                         uint max_command_buffers  //!< Number of allocated command buffers
    );

    Queue(VirtualGPU& gpu, Pal::IDevice* iDev, uint64_t residency_limit, uint max_command_buffers)
        : lock_(nullptr),
          iQueue_(nullptr),
          iCmdBuffs_(max_command_buffers, nullptr),
          iCmdFences_(max_command_buffers, nullptr),
          last_kernel_(nullptr),
          gpu_(gpu),
          iDev_(iDev),
          cmdBufIdSlot_(StartCmdBufIdx),
          cmdBufIdCurrent_(StartCmdBufIdx),
          cmbBufIdRetired_(0),
          cmdCnt_(0),
          vlAlloc_(64 * Ki),
          residency_size_(0),
          residency_limit_(residency_limit),
          max_command_buffers_(max_command_buffers) {
      vlAlloc_.Init();
    }

    ~Queue();

    void addCmdMemRef(GpuMemoryReference* mem);
    void removeCmdMemRef(GpuMemoryReference* mem);

    void addCmdDoppRef(Pal::IGpuMemory* iMem, bool lastDoppCmd, bool pfpaDoppCmd);

    void addMemRef(Pal::IGpuMemory* iMem) const {
      Pal::GpuMemoryRef memRef = {};
      memRef.pGpuMemory = iMem;
      iDev_->AddGpuMemoryReferences(1, &memRef, nullptr, Pal::GpuMemoryRefCantTrim);
    }
    void removeMemRef(Pal::IGpuMemory* iMem) const {
      iDev_->RemoveGpuMemoryReferences(1, &iMem, nullptr);
    }

    // Notice KMD to update applicaiton profile
    Pal::Result UpdateAppPowerProfile();

    // ibReuse forces event wait without polling, to make sure event occured
    template <bool ibReuse> bool waitForFence(uint cbId) const {
      Pal::Result result = Pal::Result::Success;
      uint64_t start;
      uint64_t end;
      if (!ibReuse) {
        start = amd::Os::timeNanos();
      }
      while ((Pal::Result::Success != (result = iCmdFences_[cbId]->GetStatus())) || ibReuse) {
        if (result == Pal::Result::ErrorFenceNeverSubmitted) {
          result = Pal::Result::Success;
          break;
        }
        if (!ibReuse) {
          end = amd::Os::timeNanos();
        }
        if (!ibReuse && ((end - start) < PollIntervalInNsec)) {
          amd::Os::yield();
          continue;
        }
        result = iDev_->WaitForFences(1, &iCmdFences_[cbId], true,
                                      std::chrono::nanoseconds{WaitTimeoutInNsec});
        if (Pal::Result::Success == result) {
          break;
        } else if ((Pal::Result::NotReady == result) || (Pal::Result::Timeout == result)) {
          LogPrintfWarning("PAL fence isn't ready! result:%d", result);
          if (GPU_ANALYZE_HANG) {
            DumpMemoryReferences();
          }
        } else {
          LogError("PAL wait for a fence failed!");
          break;
        }
      }
      return (result == Pal::Result::Success) ? true : false;
    }

    //! Flushes the current command buffer to HW
    //! Returns ID associated with the submission
    template <bool avoidBarrierSubmit = false> uint submit(bool forceFlush);

    bool flush();

    bool waitForEvent(uint id);

    bool isDone(uint id);

    Pal::ICmdBuffer* iCmd() const { return iCmdBuffs_[cmdBufIdSlot_]; }

    uint cmdBufId() const { return cmdBufIdCurrent_; }

    static uint32_t AllocedQueues(const VirtualGPU& gpu, Pal::EngineType type);

    amd::Monitor* lock_;                       //!< Lock PAL queue for access
    Pal::IQueue* iQueue_;                      //!< PAL queue object
    std::vector<Pal::ICmdBuffer*> iCmdBuffs_;  //!< PAL command buffers
    std::vector<Pal::IFence*> iCmdFences_;     //!< PAL fences, associated with CMD
    const amd::Kernel* last_kernel_;           //!< Last submitted kernel
    AqlPacketMgmt* aql_mgmt_;                  //!< AQL packet emulation managment
    void* info_ = nullptr;                     //!< Queue info for RT queues

   private:
    void DumpMemoryReferences() const;
    VirtualGPU& gpu_;       //!< ROCCLR virtual GPU object
    Pal::IDevice* iDev_;    //!< PAL device
    uint cmdBufIdSlot_;     //!< Command buffer ID slot for submissions
    uint cmdBufIdCurrent_;  //!< Current global command buffer ID
    uint cmbBufIdRetired_;  //!< The last retired command buffer ID
    uint cmdCnt_;           //!< Counter of commands
    std::unordered_map<GpuMemoryReference*, uint> memReferences_;
    Util::VirtualLinearAllocator vlAlloc_;
    std::vector<Pal::GpuMemoryRef> palMemRefs_;
    std::vector<Pal::IGpuMemory*> palMems_;
    std::vector<Pal::DoppRef> palDoppRefs_;
    std::set<Pal::IGpuMemory*> sdiReferences_;
    std::vector<const Pal::IGpuMemory*> palSdiRefs_;
    uint64_t residency_size_;   //!< Resource residency size
    uint64_t residency_limit_;  //!< Enables residency limit
    uint max_command_buffers_;
  };

  struct CommandBatch : public amd::HeapObject {
    amd::Command* head_;           //!< Command batch head
    GpuEvent events_[AllEngines];  //!< Last known GPU events
    TimeStamp* lastTS_;            //!< TS associated with command batch

    //! Constructor
    CommandBatch(amd::Command* head,      //!< Command batch head
                 const GpuEvent* events,  //!< HW events on all engines
                 TimeStamp* lastTS        //!< Last TS in command batch
    ) {
      init(head, events, lastTS);
    }

    void init(amd::Command* head,      //!< Command batch head
              const GpuEvent* events,  //!< HW events on all engines
              TimeStamp* lastTS        //!< Last TS in command batch
    ) {
      head_ = head;
      lastTS_ = lastTS;
      memcpy(&events_, events, AllEngines * sizeof(GpuEvent));
    }
  };

  //! The virtual GPU states
  union State {
    struct {
      uint profiling_ : 1;           //!< Profiling is enabled
      uint forceWait_ : 1;           //!< Forces wait in flush()
      uint profileEnabled_ : 1;      //!< Profiling is enabled for WaveLimiter
      uint perfCounterEnabled_ : 1;  //!< PerfCounter is enabled
      uint rgpCaptureEnabled_ : 1;   //!< RGP capture is enabled in the runtime
      uint imageBufferWrtBack_ : 1;  //!< Enable image buffer write back
      uint anyOrder_ : 1;            //!< Kernel launches don't need a barrier
    };
    uint value_;
    State() : value_(0) {}
  };

  typedef std::vector<ConstantBuffer*> constbufs_t;

  class MemoryDependency : public amd::EmbeddedObject {
   public:
    //! Default constructor
    MemoryDependency()
        : memObjectsInQueue_(nullptr), numMemObjectsInQueue_(0), maxMemObjectsInQueue_(0) {}

    ~MemoryDependency() { delete[] memObjectsInQueue_; }

    //! Creates memory dependecy structure
    bool create(size_t numMemObj);

    //! Notify the tracker about new kernel
    void newKernel() { endMemObjectsInQueue_ = numMemObjectsInQueue_; }

    //! Validates memory object on dependency
    void validate(VirtualGPU& gpu, const Memory* memory, bool readOnly);

    //! Invalidates GPU caches if memory dependency tracking is disabled
    void sync(VirtualGPU& gpu) const {
      if (maxMemObjectsInQueue_ == 0) {
        // Ignore the barrier in any order mode. The app is responsible for synchronization.
        // HW will execute the kernels asynchronously
        if (!gpu.anyOrder()) {
          // Wait for GPU and invalidate L1 cache
          gpu.addBarrier(RgpSqqtBarrierReason::MemDependency);
        }
      }
    }

    //! Clear memory dependency
    void clear(bool all = true);

   private:
    struct MemoryState {
      uint64_t start_;  //! Busy memory start address
      uint64_t end_;    //! Busy memory end address
      bool readOnly_;   //! Current GPU state in the queue
    };

    MemoryState* memObjectsInQueue_;  //!< Memory object state in the queue
    size_t endMemObjectsInQueue_;     //!< End of mem objects in the queue
    size_t numMemObjectsInQueue_;     //!< Number of mem objects in the queue
    size_t maxMemObjectsInQueue_;     //!< Maximum number of mem objects in the queue
  };

 public:
  VirtualGPU(Device& device);
  //! Creates virtual gpu object
  bool create(bool profiling,            //!< Enables profilng on the queue
              uint deviceQueueSize = 0,  //!< Device queue size, 0 if host queue
              uint rtCUs = amd::CommandQueue::RealTimeDisabled,
              amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal);
  ~VirtualGPU();

  uint64_t getQueueID() { return hwRing_; }
  void submitReadMemory(amd::ReadMemoryCommand& vcmd);
  void submitWriteMemory(amd::WriteMemoryCommand& vcmd);
  void submitCopyMemory(amd::CopyMemoryCommand& vcmd);
  void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& vcmd);
  void submitMapMemory(amd::MapMemoryCommand& vcmd);
  void submitUnmapMemory(amd::UnmapMemoryCommand& vcmd);
  void submitKernel(amd::NDRangeKernelCommand& vcmd);
  bool submitKernelInternal(const amd::NDRangeContainer& sizes,  //!< Workload sizes
                            const amd::Kernel& kernel,           //!< Kernel for execution
                            const_address parameters,            //!< Parameters for the kernel
                            bool nativeMem = true,               //!< Native memory objects
                            uint32_t sharedMemBytes = 0,         //!< Shared memory size
                            bool anyOrder = false  //!< TRUE if any order launch mode is enabled
  );
  void submitNativeFn(amd::NativeFnCommand& vcmd);
  void submitFillMemory(amd::FillMemoryCommand& vcmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);
  void submitMarker(amd::Marker& vcmd);
  void submitAccumulate(amd::AccumulateCommand& vcmd);
  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd);
  void submitPerfCounter(amd::PerfCounterCommand& vcmd);
  void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd);
  void submitThreadTrace(amd::ThreadTraceCommand& vcmd);
  void submitSignal(amd::SignalCommand& vcmd);
  void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& vcmd);
  virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
  virtual void submitVirtualMap(amd::VirtualMapCommand& cmd);
  virtual void submitStreamOperation(amd::StreamOperationCommand& cmd);
  void submitExternalSemaphoreCmd(amd::ExternalSemaphoreCmd& cmd);

  void releaseMemory(GpuMemoryReference* mem);

  void flush(amd::Command* list = nullptr, bool wait = false);

  void profilerAttach(bool enable = false) {}

  bool isFenceDirty() const { return false; }

  void HiddenHeapInit() {}

  //! Dispatches multiple AQL packets in a single batch operation
  bool dispatchAqlPacketBatch(const std::vector<uint8_t*>& packets,
                              const std::vector<std::string>& kernelNames,
                              amd::AccumulateCommand* vcmd = nullptr) {
    return false;
  }

  void resetFenceDirty() {}

  //! Returns GPU device object associated with this kernel
  const Device& dev() const { return gpuDevice_; }

  //! Set the last known GPU event
  void setGpuEvent(GpuEvent gpuEvent,  //!< GPU event for tracking
                   bool flush = false  //!< TRUE if flush is required
  );

  //! Flush DMA buffer on the specified engine
  void flushDMA(uint engineID  //!< Engine ID for DMA flush
  );

  //! Wait for all engines on this Virtual GPU
  //! Returns TRUE if CPU didn't wait for GPU
  bool waitAllEngines(CommandBatch* cb = nullptr  //!< Command batch
  );

  //! Waits for the latest GPU event with a lock to prevent multiple entries
  void waitEventLock(CommandBatch* cb  //!< Command batch
  );

  //! Returns a resource associated with the constant buffer
  const ConstantBuffer* cb(uint idx) const { return constBufs_[idx]; }

  //! Adds CAL objects into the constant buffer vector
  void addConstBuffer(ConstantBuffer* cb) { constBufs_.push_back(cb); }

  //! Start the command profiling
  void profilingBegin(amd::Command& command);  //!< Command queue object

  //! End the command profiling
  void profilingEnd(amd::Command& command);

  //! Collect the profiling results
  bool profilingCollectResults(CommandBatch* cb,               //!< Command batch
                               const amd::Event* waitingEvent  //!< Waiting event
  );

  //! Embeds memory handle info into the CB associated with this VGPU
  inline void logVmMemory(const std::string name,  //!< Brief description of the memory object
                          const Memory* memory     //!< GPU memory object
  );

  //! Adds a memory handle into the PAL memory array for Virtual Heap
  inline void addVmMemory(const Memory* memory  //!< GPU memory object
  );

  //! Adds the last submitted kernel to the queue for tracking a possible hang
  inline void AddKernel(const amd::Kernel& kernel  //!< AMD kernel object
  ) const;

  //! Checks if runtime dispatches the same kernel as previously
  inline bool IsSameKernel(const amd::Kernel& kernel  //!< AMD kernel object
  ) const;

  //! Adds a dopp desktop texture reference
  void addDoppRef(const Memory* memory,  //!< GPU memory object
                  bool lastDoopCmd,      //!< is the last submission for the pre-present primary
                  bool pfpaDoppCmd       //!< is a submission for the pre-present primary
  );

  //! Return xfer buffer for staging operations
  XferBuffer& xferWrite() { return writeBuffer_; }

  //! Return managed buffer for staging operations
  ManagedBuffer& managedBuffer() { return managedBuffer_; }

  //! Adds a pinned memory object into a map
  void addPinnedMem(amd::Memory* mem);

  //! Release pinned memory objects
  void releasePinnedMem();

  //! Finds if pinned memory is cached
  amd::Memory* findPinnedMem(void* addr, size_t size);

  //! Get the PrintfDbgHSA object
  PrintfDbgHSA& printfDbgHSA() const { return *printfDbgHSA_; }

  //! Enables synchronized transfers
  void enableSyncedBlit() const;

  //! Checks if profiling is enabled
  bool profiling() const { return state_.profiling_; }

  //! Checks if the queue is in any order mode
  bool anyOrder() const { return state_.anyOrder_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Returns hsaQueueMem_
  const Memory* hsaQueueMem() const { return hsaQueueMem_; }

  //! Returns the HW ring used on this virtual device
  uint hwRing() const { return hwRing_; }

  //! Returns virtual queue object for device enqueuing
  Memory* vQueue() const { return virtualQueue_; }

  //! Update virtual queue header
  void writeVQueueHeader(VirtualGPU& hostQ, const Memory* kernelTable);

  //! Returns TRUE if virtual queue was successfully allocatted
  bool createVirtualQueue(uint deviceQueueSize  //!< Device queue size
  );

  EngineType engineID_;  //!< Engine ID for this VirtualGPU

  //! Returns PAL command buffer interface
  Pal::ICmdBuffer* iCmd() const {
    Queue* queue = queues_[engineID_];
    return queue->iCmd();
  }

  //! Returns true if the provided command buffer is the active one
  bool isActiveCmd(Pal::ICmdBuffer* iCmd) const {
    return (queues_[engineID_] != nullptr) && (iCmd == queues_[engineID_]->iCmd()) ? true : false;
  }

  //! Returns queue, associated with VirtualGPU
  Queue& queue(EngineType id) const { return *queues_[id]; }

  void addBarrier(RgpSqqtBarrierReason reason = RgpSqqtBarrierReason::MemDependency,
                  BarrierType type = BarrierType::KernelToKernel) const {
    Pal::AcquireReleaseInfo barrier = {.srcGlobalStageMask = Pal::PipelineStageCs,
                                       .dstGlobalStageMask = Pal::PipelineStageCs,
                                       .srcGlobalAccessMask = Pal::CoherShader,
                                       .dstGlobalAccessMask = Pal::CoherShader,
                                       .memoryBarrierCount = 0,
                                       .pMemoryBarriers = nullptr,
                                       .imageBarrierCount = 0,
                                       .pImageBarriers = nullptr,
                                       .reason = static_cast<uint32_t>(reason)};

    if (type == BarrierType::KernelToCopy) {
      barrier.dstGlobalStageMask = Pal::PipelineStageBlt;
      barrier.dstGlobalAccessMask = Pal::CoherCopy;
    } else if (type == BarrierType::CopyToKernel) {
      barrier.srcGlobalStageMask = Pal::PipelineStageBlt;
      barrier.srcGlobalAccessMask = Pal::CoherCopy;
    } else if (type == BarrierType::CopyToCopy) {
      barrier.srcGlobalStageMask = barrier.dstGlobalStageMask = Pal::PipelineStageBlt;
      barrier.srcGlobalAccessMask = barrier.dstGlobalAccessMask = Pal::CoherCopy;
    } else if (type == BarrierType::FlushL2) {
      barrier.srcGlobalStageMask |= Pal::PipelineStageBlt;
      barrier.dstGlobalStageMask |= Pal::PipelineStageBlt;
      barrier.srcGlobalAccessMask = barrier.dstGlobalAccessMask = Pal::CoherCopy | Pal::CoherCpu;
    }

    iCmd()->CmdReleaseThenAcquire(barrier);
    queues_[engineID_]->submit<true>(false);
  }

  void eventBegin(EngineType engId) const {
    const static bool Begin = true;
    profileEvent(engId, Begin);
  }

  void eventEnd(EngineType engId, GpuEvent& event, bool forceExec = false) const {
    constexpr bool End = false;
    if (forceExec) {
      constexpr bool ForceFlush = true;
      event.id_ = queues_[engId]->submit(ForceFlush);
      profileEvent(engId, End);
    } else {
      profileEvent(engId, End);
      event.id_ = queues_[engId]->submit(GPU_FLUSH_ON_EXECUTION);
    }
    event.engineId_ = engId;
  }

  void waitForEvent(GpuEvent* event) const {
    if (event->isValid()) {
      assert(event->engineId_ < AllEngines);
      queues_[event->engineId_]->waitForEvent(event->id_);
      event->invalidate();
    }
  }

  bool isDone(GpuEvent* event) {
    if (event->isValid()) {
      assert(event->engineId_ < AllEngines);
      if (queues_[event->engineId_]->isDone(event->id_)) {
        event->invalidate();
        return true;
      }
      return false;
    }
    return true;
  }

  //! Returns TRUE if SDMA requires overlap synchronizaiton
  bool validateSdmaOverlap(const Resource& src,  //!< Source resource for SDMA transfer
                           const Resource& dst   //!< Destination resource for SDMA transfer
  );

  //! Checks if RGP capture is enabled
  bool rgpCaptureEna() const { return state_.rgpCaptureEnabled_; }

  //! Waits for idle on compute engine
  void WaitForIdleCompute() {
    if (events_[MainEngine].isValid()) {
      queues_[events_[MainEngine].engineId_]->waitForEvent(events_[MainEngine].id_);
      events_[MainEngine].invalidate();
    }
  }

  //! Waits for idle on SDMA engine
  void WaitForIdleSdma() {
    if (events_[SdmaEngine].isValid()) {
      queues_[events_[SdmaEngine].engineId_]->waitForEvent(events_[SdmaEngine].id_);
      events_[SdmaEngine].invalidate();
    }
  }

  void* getOrCreateHostcallBuffer();

  //! Waits on an outstanding kernel.
  void releaseGpuMemoryFence() {
    if (amd::IS_HIP) {
      WaitForIdleCompute();
    }
  }

  //! Updates timestamp for AQL packet index
  void AqlPacketUpdateTs(uint32_t index, GpuEvent gpu_event) {
    // Save the new CB ID for this slot
    queues_[MainEngine]->aql_mgmt_->aql_events_[index] = gpu_event;
    queues_[MainEngine]->aql_mgmt_->aql_vgpus_[index] = this;
  }

  //! Returns the current active slot for AQL packet
  hsa_kernel_dispatch_packet_t* GetAqlPacketSlot(uint32_t* index) {
    auto& mgmt = *queues_[MainEngine]->aql_mgmt_;
    // Atomic increment global AQL index and wrap around max AQL list size
    *index = ++mgmt.packet_index_ % AqlPacketMgmt::kAqlPacketsListSize;
    if (mgmt.aql_events_[*index].isValid()) {
      // Make sure GPU doesn't process this slot
      mgmt.aql_vgpus_[*index]->waitForEvent(&mgmt.aql_events_[*index]);
    }
    return &mgmt.aql_packets_[*index];
  }

 protected:
  void profileEvent(EngineType engine, bool type) const;

  //! Creates buffer object from image
  inline amd::Memory* createBufferFromImage(
      amd::Memory& amdImage  //! The parent image object(untiled images only)
  ) {
    amd::Memory* mem = new (amdImage.getContext()) amd::Buffer(amdImage, 0, 0, amdImage.getSize());
    mem->setVirtualDevice(this);
    if ((mem != nullptr) && !mem->create()) {
      mem->release();
    }
    return mem;
  }

  //! Get copy command type from original copy command type and memory object types
  inline cl_command_type getCopyCommandType(cl_command_type type, const cl_mem_object_type srcType,
                                 const cl_mem_object_type dstType) {
    if (srcType == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
      if (dstType == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        type = CL_COMMAND_COPY_BUFFER;
      } else if (dstType == CL_MEM_OBJECT_BUFFER) {
        type = CL_COMMAND_COPY_BUFFER;
      } else if (type == CL_COMMAND_COPY_IMAGE) {
        type = CL_COMMAND_COPY_BUFFER_TO_IMAGE;
      }
    } else if (dstType == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
      if (srcType == CL_MEM_OBJECT_BUFFER) {
        type = CL_COMMAND_COPY_BUFFER;
      } else if (type == CL_COMMAND_COPY_IMAGE) {
        type = CL_COMMAND_COPY_IMAGE_TO_BUFFER;
      }
    }
    return type;
  }

 private:
  struct MemoryRange {
    uint64_t start_;  //!< Memory range start address
    uint64_t end_;    //!< Memory range end address
    MemoryRange() : start_(0), end_(0) {}
  };

  //! Allocates constant buffers
  bool allocConstantBuffers();

  //! Allocate hsaQueueMem_
  bool allocHsaQueueMem();

  //! Awaits a command batch with a waiting event
  bool awaitCompletion(CommandBatch* cb,                         //!< Command batch for to wait
                       const amd::Event* waitingEvent = nullptr  //!< A waiting event
  );

  //! Detects memory dependency for HSAIL kernels and flushes caches
  bool processMemObjectsHSA(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                            const_address params,       //!< Pointer to the param's store
                            bool nativeMem,             //!< Native memory objects
                            size_t& ldsAddess,          //!< Returns LDS size, used in the kernel
                            bool& imageBufferWrtBack,   //!< Image buffer write back is required
                            std::vector<Image*>& wrtBackImageBuffer  //!< images for write back
  );

  //! Common function for fill memory used by both svm Fill and non-svm fill
  bool fillMemory(cl_command_type type,        //!< the command type
                  amd::Memory* amdMemory,      //!< memory object to fill
                  const void* pattern,         //!< pattern to fill the memory
                  size_t patternSize,          //!< pattern size
                  const amd::Coord3D& origin,  //!< memory origin
                  const amd::Coord3D& size,    //!< memory size for filling
                  bool forceBlit = false       //!< force shader blit path
  );

  bool copyMemory(cl_command_type type,            //!< the command type
                  amd::Memory& srcMem,             //!< source memory object
                  amd::Memory& dstMem,             //!< destination memory object
                  bool entire,                     //!< flag of entire memory copy
                  const amd::Coord3D& srcOrigin,   //!< source memory origin
                  const amd::Coord3D& dstOrigin,   //!< destination memory object
                  const amd::Coord3D& size,        //!< copy size
                  const amd::BufferRect& srcRect,  //!< region of source for copy
                  const amd::BufferRect& dstRect,  //!< region of destination for copy
                  amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  );

  void PrintChildren(const HSAILKernel& hsaKernel,  //!< The parent HSAIL kernel
                     VirtualGPU* gpuDefQueue        //!< Device queue for children execution
  );

  bool PreDeviceEnqueue(const amd::Kernel& kernel,     //!< Parent amd kernel object
                        const HSAILKernel& hsaKernel,  //!< Parent HSAIL object
                        VirtualGPU** gpuDefQueue,      //!< [Return] GPU default queue
                        uint64_t* vmDefQueue           //!< [Return] VM handle to the virtual queue
  );

  void PostDeviceEnqueue(
      const amd::Kernel& kernel,     //!< Parent amd kernel object
      const HSAILKernel& hsaKernel,  //!< Parent HSAIL object
      VirtualGPU* gpuDefQueue,       //!< GPU default queue
      uint64_t vmDefQueue,           //!< VM handle to the virtual queue
      uint64_t vmParentWrap,         //!< VM handle to the wrapped AQL packet location
      GpuEvent* gpuEvent             //!< [Return] GPU event associated with the device enqueue
  );

  Device& gpuDevice_;  //!< physical GPU device

  PrintfDbgHSA* printfDbgHSA_;  //!< HSAIL printf implemenation

  TimeStampCache* tsCache_;            //!< TimeStamp cache
  MemoryDependency memoryDependency_;  //!< Memory dependency class

  std::vector<amd::Memory*> pinnedMems_;  //!< Pinned memory list

  ManagedBuffer managedBuffer_;  //!< Managed write buffer
  constbufs_t constBufs_;        //!< constant buffers
  XferBuffer writeBuffer_;       //!< Transfer/staging buffer for uploads

  typedef std::queue<CommandBatch*> CommandBatchQueue;
  CommandBatchQueue cbQueue_;      //!< Queue of command batches
  CommandBatchQueue freeCbQueue_;  //!< Queue of free command batches

  uint hwRing_;  //!< HW ring used on this virtual device

  State state_;                  //!< virtual GPU current state
  GpuEvent events_[AllEngines];  //!< Last known GPU events

  uint64_t readjustTimeGPU_;  //!< Readjust time between GPU and CPU timestamps
  TimeStamp* lastTS_;         //!< Last timestamp executed on Virtual GPU
  TimeStamp* profileTs_;      //!< current profiling timestamp for command

  AmdVQueueHeader* vqHeader_;  //!< Sysmem copy for virtual queue header
  Memory* virtualQueue_;       //!< Virtual device queue
  Memory* schedParams_;        //!< The scheduler parameters
  uint deviceQueueSize_;       //!< Device queue size
  uint maskGroups_;  //!< The number of mask groups processed in the scheduler by one thread

  Memory* hsaQueueMem_;               //!< Memory for the amd_queue_t object
  Pal::ICmdAllocator* cmdAllocator_;  //!< Command buffer allocator
  Queue* queues_[AllEngines];         //!< HW queues for all engines
  MemoryRange sdmaRange_;             //!< SDMA memory range for write access

  void* hostcallBuffer_;  //!< Hostcall buffer

  using KernelArgImpl = device::Settings::KernelArgImpl;
};

inline void VirtualGPU::logVmMemory(const std::string name, const Memory* memory) {
  if (PAL_EMBED_KERNEL_MD || (AMD_LOG_LEVEL >= amd::LOG_INFO)) {
    char buf[256];
    sprintf(buf, "%s = ptr:[%p-%p] size:[%" PRIu64 "] heap[%d]", name.c_str(),
            reinterpret_cast<void*>(memory->vmAddress()),
            reinterpret_cast<void*>(memory->vmAddress() + memory->size()),
            memory->iMem()->Desc().size, memory->iMem()->Desc().heaps[0]);
    if (PAL_EMBED_KERNEL_MD) {
      iCmd()->CmdCommentString(buf);
    }
    ClPrint(amd::LOG_INFO, amd::LOG_MEM, "%s", buf);
  }
}

inline void VirtualGPU::addVmMemory(const Memory* memory) {
  queues_[MainEngine]->addCmdMemRef(memory->memRef());
  memory->setBusy(*this, queues_[MainEngine]->cmdBufId());
}

inline void VirtualGPU::AddKernel(const amd::Kernel& kernel) const {
  queues_[MainEngine]->last_kernel_ = &kernel;
}

inline bool VirtualGPU::IsSameKernel(const amd::Kernel& kernel) const {
  return (queues_[MainEngine]->last_kernel_ == &kernel) ? true : false;
}

template <bool avoidBarrierSubmit> uint VirtualGPU::Queue::submit(bool forceFlush) {
  cmdCnt_++;
  uint id = cmdBufIdCurrent_;
  bool flushCmd = ((cmdCnt_ > MaxCommands) || forceFlush) && !avoidBarrierSubmit;
  if (flushCmd) {
    if (!flush()) {
      return GpuEvent::InvalidID;
    }
  }
  return id;
}
/*@}*/  // namespace amd::pal
}  // namespace amd::pal
