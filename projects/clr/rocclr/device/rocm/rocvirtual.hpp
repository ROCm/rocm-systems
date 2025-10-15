/* Copyright (c) 2008 - 2025 Advanced Micro Devices, Inc.

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

#include "platform/commandqueue.hpp"
#include "rocdefs.hpp"
#include "rocdevice.hpp"
#include "utils/flags.hpp"
#include "utils/util.hpp"
#include "rocprintf.hpp"
#include "rocsched.hpp"
#include "device/device.hpp"
#include "os/os.hpp"
#include <stack>

namespace amd::roc {
class Device;
class Memory;
struct ProfilingSignal;
class Timestamp;

// Initial HSA signal value
constexpr static hsa_signal_value_t kInitSignalValueOne = 1;

// Timeouts for HSA signal wait
constexpr static uint64_t kTimeout100us = 100 * K;
constexpr static uint64_t kUnlimitedWait = std::numeric_limits<uint64_t>::max();

constexpr static uint64_t kTimeout4Secs = 4 * M;

inline bool WaitForSignal(hsa_signal_t signal, bool active_wait = false, bool yield = false) {
  hsa_wait_state_t wait_state = HSA_WAIT_STATE_BLOCKED;
  if (active_wait) {
    wait_state = HSA_WAIT_STATE_ACTIVE;
  }

  if (Hsa::signal_load_relaxed(signal) > 0) {
    // When it is blocked wait, we wait in active state for 100 us before proceeding to wait in
    // blocked state indefinitely.
    if (!active_wait) {
      ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Host active wait for Signal = (0x%lx) for %d ns",
              signal.handle, kTimeout100us);
      if (Hsa::signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                    kTimeout100us, HSA_WAIT_STATE_ACTIVE) != 0) {
        if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
          ClPrint(amd::LOG_ERROR, amd::LOG_SIG,
                  "Device not Stable, while waiting for Signal ="
                  "(0x%lx) for %d ns",
                  signal.handle, kTimeout100us);
          return true;
        }
      }
    }

    // This is unlimited wait, but we wait for 4 secs and check if the device is
    // unstable, if so we return, otherwise we continue to wait in the while loop.
    while (Hsa::signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                     kTimeout4Secs, wait_state) != 0) {
      if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
        ClPrint(amd::LOG_ERROR, amd::LOG_SIG,
                "Device not Stable, while waiting for Signal ="
                "(0x%lx) for %d ns",
                signal.handle, kTimeout4Secs);
        return true;
      }
      if (yield && wait_state == HSA_WAIT_STATE_ACTIVE) {
        amd::Os::yield();
      }
    }
  }

  return true;
}

inline void fetchSignalTime(hsa_signal_t signal, hsa_agent_t gpu_device, uint64_t* start,
                            uint64_t* end) {
  if (start != nullptr && end != nullptr) {
    hsa_amd_profiling_dispatch_time_t time = {};
    Hsa::profiling_get_dispatch_time(gpu_device, signal, &time);
    *start = time.start;
    *end = time.end;
  }
}

// Timestamp for keeping track of some profiling information for various commands
// including EnqueueNDRangeKernel and clEnqueueCopyBuffer.
class Timestamp : public amd::ReferenceCountedObject {
 private:
  static double ticksToTime_;

  uint64_t start_;
  uint64_t end_;
  VirtualGPU* gpu_;                        //!< Virtual GPU, associated with this timestamp
  amd::Command& command_;                  ///!< Command, associated with this timestamp
  amd::Command* parsedCommand_;            //!< Command down the list, considering command_ as head
  std::vector<ProfilingSignal*> signals_;  //!< The list of all signals, associated with the TS
  hsa_signal_t callback_signal_;  //!< Signal associated with a callback for possible later update
  amd::Monitor lock_;             //!< Serialize timestamp update
  bool accum_ena_ = false;        //!< If TRUE then the accumulation of execution times has started
  bool hasHwProfiling_ = false;   //!< If TRUE then HwProfiling is enabled for the command
  bool blocking_ = true;          //!< If TRUE callback is blocking

  Timestamp(const Timestamp&) = delete;
  Timestamp& operator=(const Timestamp&) = delete;

 public:
  Timestamp(VirtualGPU* gpu, amd::Command& command)
      : start_(std::numeric_limits<uint64_t>::max()),
        end_(0),
        gpu_(gpu),
        command_(command),
        parsedCommand_(nullptr),
        callback_signal_(hsa_signal_t{}),
        lock_(true) /* Timestamp lock */ {}

  ~Timestamp() {}

  void getTime(uint64_t* start, uint64_t* end) {
    checkGpuTime();
    *start = start_;
    *end = end_;
  }

  void AddProfilingSignal(ProfilingSignal* signal) {
    signals_.push_back(signal);
    hasHwProfiling_ = true;
  }

  const std::vector<ProfilingSignal*>& Signals() const { return signals_; }

  const bool HwProfiling() const { return hasHwProfiling_; }

  //! Finds execution ticks on GPU
  void checkGpuTime();

  // Start a timestamp (get timestamp from OS)
  void start() { start_ = amd::Os::timeNanos(); }

  // End a timestamp (get timestamp from OS)
  void end() {
    // Timestamp value can be updated by HW profiling if current command had a stall.
    // Although CPU TS should be still valid in this situation, there are cases in VM mode
    // when CPU timeline is out of sync with GPU timeline and shifted time can be reported
    if (end_ == 0) {
      end_ = amd::Os::timeNanos();
    }
  }

  static void setGpuTicksToTime(double ticksToTime) { ticksToTime_ = ticksToTime; }
  static double getGpuTicksToTime() { return ticksToTime_; }

  //! Returns amd::command assigned to this timestamp
  amd::Command& command() const { return command_; }

  //! Sets the parsed command
  void setParsedCommand(amd::Command* command) { parsedCommand_ = command; }

  //! Gets the parsed command
  amd::Command* getParsedCommand() const { return parsedCommand_; }

  //! Returns virtual GPU device, used with this timestamp
  VirtualGPU* gpu() const { return gpu_; }

  //! Updates the callback signal
  void SetCallbackSignal(hsa_signal_t callback_signal, bool blocking = true) {
    callback_signal_ = callback_signal;
    blocking_ = blocking;
  }
  //! Returns the callback signal
  hsa_signal_t GetCallbackSignal() const { return callback_signal_; }

  //! Return if callback is blocking/non-blocking
  bool GetBlocking() { return blocking_; }
};

class VirtualGPU : public device::VirtualDevice {
 public:
  class ManagedBuffer : public amd::EmbeddedObject {
   public:
    //! The number of chunks the arg pool will be divided
    static constexpr uint32_t kPoolNumSignals = 4;
    ManagedBuffer(VirtualGPU& gpu, uint32_t pool_size)
        : gpu_(gpu), pool_size_(pool_size), pool_signal_(kPoolNumSignals) {}
    ~ManagedBuffer();

    //! Allocates all necessary resources to manage memory
    bool Create(amd::Device::MemorySegment mem_segment);

    //! Acquires memory for use on the gpu
    address Acquire(uint32_t size);

    //! Acquires custom aligned memory for use on the gpu
    address Acquire(uint32_t size, uint32_t alignment);

    //! Reset mem pool
    void ResetPool();

   private:
    VirtualGPU& gpu_;                        //!< Queue object for ROCm device
    address pool_base_ = nullptr;            //!< Memory pool base address
    uint32_t pool_size_;                     //!< Memory pool base size
    uint32_t pool_chunk_end_ = 0;            //!< The end offset of the current chunk
    uint32_t active_chunk_ = 0;              //!< The index of the current active chunk
    uint32_t pool_cur_offset_ = 0;           //!< Current active offset for update
    std::vector<hsa_signal_t> pool_signal_;  //!< Pool of HSA signals to manage multiple chunks
  };
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

    //! Clear memory dependency
    void clear(bool all = true);

    //! Max number of mem objects in the queue
    size_t maxMemObjectsInQueue() const { return maxMemObjectsInQueue_; }

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

  class HwQueueTracker : public amd::EmbeddedObject {
   public:
    HwQueueTracker(const VirtualGPU& gpu) : gpu_(gpu) {}

    ~HwQueueTracker();

    //! Creates a pool of signals for tracking of HW operations on the queue
    bool Create();

    //! Finds a free signal for the upcomming operation
    hsa_signal_t ActiveSignal(hsa_signal_value_t init_val = kInitSignalValueOne,
                              Timestamp* ts = nullptr, bool attach_signal = true);

    //! Wait for the curent active signal. Can idle the queue
    bool WaitCurrent() {
      ProfilingSignal* signal = signal_list_[current_id_];
      return CpuWaitForSignal(signal);
    }

    //! Update current active engine
    void SetActiveEngine(HwQueueEngine engine = HwQueueEngine::Compute) { engine_ = engine; }
    HwQueueEngine GetActiveEngine() const { return engine_; }

    //! Returns the last submitted signal for a wait
    std::vector<hsa_signal_t>& WaitingSignal(HwQueueEngine engine = HwQueueEngine::Compute);

    //! Resets current signal back to the previous one. It's necessary in a case of ROCr failure.
    void ResetCurrentSignal();

    //! Adds an external signal(submission in another queue) for dependency tracking
    void AddExternalSignal(ProfilingSignal* signal) { external_signals_.push_back(signal); }

    //! Get the last active signal on the queue
    ProfilingSignal* GetLastSignal() const { return signal_list_[current_id_]; }

    //! Clear external signals
    void ClearExternalSignals() { external_signals_.clear(); }

    //! Empty check for external signals
    bool IsExternalSignalListEmpty() const { return external_signals_.empty(); }

    //! Get/Set SDMA profiling
    bool GetSDMAProfiling() { return sdma_profiling_; }
    void SetSDMAProfiling(bool profile) {
      sdma_profiling_ = profile;
      Hsa::profiling_async_copy_enable(profile);
    }

   private:
    //! Creates HSA signal with the specified scope
    bool CreateSignal(ProfilingSignal* signal, bool interrupt = false) const;

    //! Wait for the next active signal
    void WaitNext() {
      size_t next = (current_id_ + 1) % signal_list_.size();
      ProfilingSignal* signal = signal_list_[next];
      CpuWaitForSignal(signal);
    }

    //! Wait for the provided signal
    bool CpuWaitForSignal(ProfilingSignal* signal);

    HwQueueEngine engine_ = HwQueueEngine::Unknown;  //!< Engine used in the current operations
    std::stack<ProfilingSignal*> signal_pool_irq_;   //!< The pool of free signals with interrupts
    std::stack<ProfilingSignal*> signal_pool_;       //!< The pool of free signals without interrupt
    std::vector<ProfilingSignal*> signal_list_;      //!< The pool of all signals for processing
    size_t current_id_ = 0;                          //!< Last submitted signal
    bool sdma_profiling_ = false;                    //!< If TRUE, then SDMA profiling is enabled
    const VirtualGPU& gpu_;                          //!< VirtualGPU, associated with this tracker
    std::vector<ProfilingSignal*> external_signals_;  //!< External signals for a wait in this queue
    std::vector<hsa_signal_t> waiting_signals_;       //!< Current waiting signals in this queue
  };

  VirtualGPU(Device& device, bool profiling = false, bool cooperative = false,
             const std::vector<uint32_t>& cuMask = {},
             amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal);
  ~VirtualGPU();

  bool create();
  const Device& dev() const { return roc_device_; }

  void profilingBegin(amd::Command& command, bool sdmaProfiling = false);
  void profilingEnd(bool clearHwEvent = false);

  void updateCommandsState(amd::Command* list) const;

  void submitReadMemory(amd::ReadMemoryCommand& cmd);
  void submitWriteMemory(amd::WriteMemoryCommand& cmd);
  void submitCopyMemory(amd::CopyMemoryCommand& cmd);
  void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd);
  void submitMapMemory(amd::MapMemoryCommand& cmd);
  void submitUnmapMemory(amd::UnmapMemoryCommand& cmd);
  void submitKernel(amd::NDRangeKernelCommand& cmd);
  bool submitKernelInternal(
      const amd::NDRangeContainer& sizes,                  //!< Workload sizes
      const amd::Kernel& kernel,                           //!< Kernel for execution
      const_address parameters,                            //!< Parameters for the kernel
      void* event_handle,                                  //!< Handle to OCL event for debugging
      uint32_t sharedMemBytes = 0,                         //!< Shared memory size
      amd::NDRangeKernelCommand* vcmd = nullptr,           //!< Original launch command
      hsa_kernel_dispatch_packet_t* aql_packet = nullptr,  //!< Scheduler launch
      bool attach_signal = false);
  void submitNativeFn(amd::NativeFnCommand& cmd);
  void submitMarker(amd::Marker& cmd);
  void submitAccumulate(amd::AccumulateCommand& cmd);
  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
  void submitPerfCounter(amd::PerfCounterCommand& cmd);

  void flush(amd::Command* list = nullptr, bool wait = false);
  void submitFillMemory(amd::FillMemoryCommand& cmd);
  void submitStreamOperation(amd::StreamOperationCommand& cmd);
  void submitBatchMemoryOperation(amd::BatchMemoryOperationCommand& cmd);
  void submitVirtualMap(amd::VirtualMapCommand& cmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);

  void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
  void submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd);

  virtual void submitSignal(amd::SignalCommand& cmd) {}
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd) {}

  void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) {}
  void submitThreadTrace(amd::ThreadTraceCommand& vcmd) {}

  virtual void submitExternalSemaphoreCmd(amd::ExternalSemaphoreCmd& cmd) {}

  virtual address allocKernelArguments(size_t size, size_t alignment) final;
  virtual void ReleaseAllHwQueues() final;
  virtual void ReleaseHwQueue() final;

  /**
   * @brief Waits on an outstanding kernel without regard to how
   * it was dispatched - with or without a signal
   *
   * @return bool true if Wait returned successfully, false otherwise
   */
  bool releaseGpuMemoryFence(bool skip_copy_wait = false);

  hsa_agent_t gpu_device() const { return gpu_device_; }
  hsa_queue_t* gpu_queue() { return gpu_queue_; }
  void set_gpu_queue(hsa_queue_t* gpu_queue) { gpu_queue_ = gpu_queue; }

  // Return pointer to PrintfDbg
  PrintfDbg* printfDbg() const { return printfdbg_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Detects memory dependency for HSAIL kernels and uses appropriate AQL header
  bool processMemObjects(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                         const_address params,       //!< Pointer to the param's store
                         size_t& ldsAddress,         //!< LDS usage
                         bool cooperativeGroups,     //!< Dispatch with cooperative groups
                         bool& imageBufferWrtBack,   //!< Image buffer write back is required
                         std::vector<device::Memory*>& wrtBackImageBuffer  //!< Images for writeback
  );

  //! Returns a managed buffer for staging copies
  ManagedBuffer& Staging() { return managed_buffer_; }

  //! Adds a pinned memory object into a map
  void addPinnedMem(amd::Memory* mem);

  //! Release pinned memory objects
  void releasePinnedMem();

  //! Finds if pinned memory is cached
  amd::Memory* findPinnedMem(void* addr, size_t size);

  void enableSyncBlit() const;

  void hasPendingDispatch() { hasPendingDispatch_ = true; }
  bool IsPendingDispatch() const { return (hasPendingDispatch_) ? true : false; }
  void addSystemScope() {
    addSystemScope_ = true;
    fence_state_ = amd::Device::CacheState::kCacheStateInvalid;
  }
  void SetCopyCommandType(cl_command_type type) { copy_command_type_ = type; }

  HwQueueTracker& Barriers() { return barriers_; }

  Timestamp* timestamp() const { return timestamp_; }
  amd::Command* command() const { return command_; }

  void* allocKernArg(size_t size, size_t alignment);
  bool isFenceDirty() const { return fence_dirty_; }
  void setFenceDirty(bool state) { fence_dirty_ = state; }
  void WaitCompleteSignal(hsa_signal_t signal);

  void HiddenHeapInit();
  void setLastUsedSdmaEngine(uint32_t mask) { lastUsedSdmaEngineMask_ = mask; }
  uint32_t getLastUsedSdmaEngine() const { return lastUsedSdmaEngineMask_.load(); }
  uint64_t getQueueID();

  //! Analyzes a crashed AQL queue to find a broken AQL packet
  void AnalyzeAqlQueue() const;

 private:
  //! Dispatches a barrier with blocking HSA signals
  void dispatchBlockingWait();

  bool dispatchAqlPacket(hsa_kernel_dispatch_packet_t* packet, uint16_t header, uint16_t rest,
                         bool blocking = true, bool capturing = false,
                         const uint8_t* aqlPacket = nullptr, bool attach_signal = false);
  bool dispatchAqlPacket(hsa_barrier_and_packet_t* packet, uint16_t header, uint16_t rest,
                         bool blocking = true, bool attach_signal = false);

  //! Dispatches multiple AQL packets in a single batch operation
  bool dispatchAqlPacketBatch(const std::vector<uint8_t*>& packets,
                              const std::vector<std::string>& kernelNames,
                              amd::AccumulateCommand* vcmd = nullptr);
  template <typename AqlPacket> bool dispatchGenericAqlPacket(AqlPacket* packet, uint16_t header,
                                                              uint16_t rest, bool blocking,
                                                              bool attach_signal = false);
  //! Dispatches multiple AQL packets with a single doorbell ring
  template <typename AqlPacket> bool dispatchGenericAqlPacketBatch(const std::vector<AqlPacket*>& packets,
                                                                   bool blocking, bool attach_signal = false,
                                                                   const std::vector<std::string>* kernelNames = nullptr);

  bool dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet, const uint32_t gfxVersion,
                                bool blocking, const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi);
  void dispatchBarrierPacket(uint16_t packetHeader, bool skipSignal = false,
                             hsa_signal_t signal = hsa_signal_t{0});
  void dispatchBarrierValuePacket(uint16_t packetHeader, bool resolveDepSignal = false,
                                  hsa_signal_t signal = hsa_signal_t{0},
                                  hsa_signal_value_t value = 0, hsa_signal_value_t mask = 0,
                                  hsa_signal_condition32_t cond = HSA_SIGNAL_CONDITION_EQ,
                                  bool skipTs = false,
                                  hsa_signal_t completionSignal = hsa_signal_t{0});
  void initializeDispatchPacket(hsa_kernel_dispatch_packet_t* packet, amd::NDRangeContainer& sizes);

  void resetKernArgPool() { managed_kernarg_buffer_.ResetPool(); }

  uint64_t getVQVirtualAddress();

  bool createSchedulerParam();

  //! Returns TRUE if virtual queue was successfully allocatted
  bool createVirtualQueue(uint deviceQueueSize);

  //! Common function for fill memory used by both svm Fill and non-svm fill
  bool fillMemory(cl_command_type type,         //!< the command type
                  amd::Memory* amdMemory,       //!< memory object to fill
                  const void* pattern,          //!< pattern to fill the memory
                  size_t patternSize,           //!< pattern size
                  const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                  const amd::Coord3D& origin,   //!< memory origin
                  const amd::Coord3D& size,     //!< memory size for filling
                  bool forceBlit = false        //!< force shader blit path
  );

  //! Common function for memory copy used by both svm Copy and non-svm Copy
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

  //! Updates AQL header for the upcomming dispatch
  void setAqlHeader(uint16_t header) { aqlHeader_ = header; }

  //! Resets the current queue state. Note: should be called after AQL queue becomes idle
  void ResetQueueStates();

  //! Track the progress of the queue based on the last write index and completion signal
  template <typename AqlPacket>
  inline void TrackQueueProgress(const AqlPacket& packet, uint64_t index) {
    // Track the progress of the current virtual queue
    last_write_index_ = index;
    // Update the last completion signal if the packet has one
    if (packet.completion_signal.handle != 0) {
      last_barrier_index_ = index;
      last_completion_signal_ = packet.completion_signal;
    }
  }

  //! Returns true if the queue is considered as idle. That means all submitted packets are
  //! complete. Note: it doesn't track the state of caches
  bool IsQueueIdle() const {
    bool result = false;
    // Make sure the last packet contained a completion signal
    if (last_barrier_index_ == last_write_index_) {
      if ((last_write_index_ == 0) && (last_completion_signal_.handle == 0)) {
        result = true;
      } else {
        result = (Hsa::signal_load_relaxed(last_completion_signal_) == 0);
      }
    }
    return result;
  }

  std::vector<amd::Memory*> pinnedMems_;  //!< Pinned memory list

  //! Queue state flags
  union {
    struct {
      uint32_t hasPendingDispatch_ : 1;     //!< A kernel dispatch is outstanding
      uint32_t profiling_ : 1;              //!< Profiling is enabled
      uint32_t cooperative_ : 1;            //!< Cooperative launch is enabled
      uint32_t addSystemScope_ : 1;         //!< Insert a system scope to the next aql
      uint32_t tracking_created_ : 1;       //!< Enabled if tracking object was properly initialized
      uint32_t retainExternalSignals_ : 1;  //!< Indicate to retain external signal array
    };
    uint32_t state_;
  };

  Timestamp* timestamp_;
  amd::Command* command_;   //!< Current command
  hsa_agent_t gpu_device_;  //!< Physical device
  hsa_queue_t* gpu_queue_;  //!< Active queue associated with a vgpu
  hsa_barrier_and_packet_t barrier_packet_ {};
  hsa_amd_barrier_value_packet_t barrier_value_packet_ {};

  uint32_t dispatch_id_;  //!< This variable must be updated atomically.
  Device& roc_device_;    //!< roc device object
  PrintfDbg* printfdbg_;
  MemoryDependency memoryDependency_;  //!< Memory dependency class
  uint16_t aqlHeader_;                 //!< AQL header for dispatch

  amd::Memory* virtualQueue_;  //!< Virtual device queue
  uint deviceQueueSize_;       //!< Device queue size
  uint maskGroups_;            //!< The number of mask groups processed in the scheduler by
                               //!< one thread
  uint schedulerThreads_;      //!< The number of scheduler threads

  hsa_queue_t* schedulerQueue_;

  HwQueueTracker barriers_;  //!< Tracks active barriers in ROCr

  ManagedBuffer managed_buffer_;          //!< Memory manager for staging copies
  ManagedBuffer managed_kernarg_buffer_;  //!< Managed memory for kernel args

  friend class Timestamp;

  //  PM4 packet for gfx8 performance counter
  enum {
    SLOT_PM4_SIZE_DW = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(uint32_t),
    SLOT_PM4_SIZE_AQLP = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / 64
  };

  uint16_t dispatchPacketHeaderNoSync_;
  uint16_t dispatchPacketHeader_;

  //!< bit-vector representing the CU mask. Each active bit represents using one CU
  const std::vector<uint32_t> cuMask_;
  amd::CommandQueue::Priority priority_;  //!< The priority for the hsa queue

  cl_command_type copy_command_type_;  //!< Type of the copy command, used for ROC profiler
                                       //!< OCL doesn't distinguish diffrent copy types,
                                       //!< but ROC profiler expects D2H or H2D detection
  int fence_state_;                    //!< Fence scope
                                       //!< kUnknown/kFlushedToDevice/kFlushedToSystem
  bool fence_dirty_;                   //!< Fence modified flag

  std::atomic<uint> lastUsedSdmaEngineMask_;  //!< Last Used SDMA Engine mask
  uint64_t last_write_index_ = 0;             //!< The last HW queue write index for any packet
  uint64_t last_barrier_index_ = 0;           //!< The last HW queue write index for a packet
                                              //!< with a complition signal
  hsa_signal_t last_completion_signal_{};     //!< The last completion signal

  using KernelArgImpl = device::Settings::KernelArgImpl;
};
}  // namespace amd::roc
