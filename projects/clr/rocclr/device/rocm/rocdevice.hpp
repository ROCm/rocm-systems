/* Copyright (c) 2009 - 2025 Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "CL/cl.h"
#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/perfctr.hpp"
#include "platform/memory.hpp"
#include "utils/concurrent.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "utils/versions.hpp"

#include "device/rocm/rocrctx.hpp"
#include "device/rocm/rocsettings.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rocdefs.hpp"
#include "device/rocm/rocprintf.hpp"
#include "device/rocm/rocglinterop.hpp"

#include <atomic>
#include <iostream>
#include <vector>
#include <memory>

/*! \addtogroup HSA
 *  @{
 */

//! HSA Device Implementation
namespace amd::roc {

/**
 * @brief List of environment variables that could be used to
 * configure the behavior of Hsa Runtime
 */
#define ENVVAR_HSA_POLL_KERNEL_COMPLETION "HSA_POLL_COMPLETION"

//! Forward declarations
class Command;
class Device;
class GpuCommand;
class Heap;
class HeapBlock;
class Program;
class Kernel;
class Memory;
class Resource;
class VirtualDevice;
class PrintfDbg;

class ProfilingSignal : public amd::ReferenceCountedObject {
 public:
  hsa_signal_t signal_;   //!< HSA signal to track profiling information
  Timestamp* ts_;         //!< Timestamp object associated with the signal
  HwQueueEngine engine_;  //!< Engine used with this signal
  amd::Monitor lock_;     //!< Signal lock for update

  typedef union {
    struct {
      uint32_t done_ : 1;              //!< True if signal is done
      uint32_t isPacketDispatch_ : 1;  //!< True if the packet, used with the signal, is dispatch
      uint32_t interrupt_ : 1;         //!< True if the signal will trigger an interrupt
      uint32_t reserved_ : 29;
    };
    uint32_t data_;
  } Flags;

  Flags flags_;

  //! Cached timing data - populated when signal completes, avoids repeated HSA calls
  struct CachedTiming {
    uint64_t start_ = 0;   //!< Cached start timestamp from HSA
    uint64_t end_ = 0;     //!< Cached end timestamp from HSA
    bool valid_ = false;   //!< True if timing data has been cached
  };
  CachedTiming cached_timing_;

  ProfilingSignal()
      : ts_(nullptr),
        engine_(HwQueueEngine::Compute),
        lock_(true) /* Signal Ops Lock */
  {
    signal_.handle = 0;
    flags_.data_ = 0;
    flags_.done_ = true;
  }

  virtual ~ProfilingSignal();
  amd::Monitor& LockSignalOps() { return lock_; }

  //! Cache timing data from HSA for this signal (called once when signal completes)
  void CacheTimingData(hsa_agent_t gpu_device);

  //! Reset cached timing for signal reuse
  void ResetCachedTiming() {
    amd::ScopedLock lock(lock_);
    cached_timing_.start_ = 0;
    cached_timing_.end_ = 0;
    cached_timing_.valid_ = false;
  }

  //! Check if timing is already cached
  bool IsTimingCached() const { return cached_timing_.valid_; }

  //! Get cached timing values
  void GetCachedTiming(uint64_t& start, uint64_t& end) {
    amd::ScopedLock lock(lock_);
    start = cached_timing_.start_;
    end = cached_timing_.end_;
  }
};

class Sampler : public device::Sampler {
 public:
  //! Constructor
  Sampler(const Device& dev) : dev_(dev) {}

  //! Default destructor for the device memory object
  virtual ~Sampler();

  //! Creates a device sampler from the OCL sampler state
  bool create(const amd::Sampler& owner  //!< AMD sampler object
  );

 private:
  void fillSampleDescriptor(hsa_ext_sampler_descriptor_v2_t& samplerDescriptor,
                            const amd::Sampler& sampler) const;
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);

  const Device& dev_;  //!< Device object associated with the sampler

  hsa_ext_sampler_t hsa_sampler;
};

// A NULL Device type used only for offline compilation
// Only functions that are used for compilation will be in this device
class NullDevice : public amd::Device {
 public:
  //! constructor
  NullDevice(){};

  //! create the device
  bool create(const amd::Isa& isa);

  //! Initialise all the offline devices that can be used for compilation
  static bool init();
  //! Teardown for offline devices
  static void tearDown();

  //! Destructor for the Null device
  virtual ~NullDevice();

  const Settings& settings() const { return static_cast<Settings&>(*settings_); }

  //! Construct an device program object from the ELF assuming it is valid
  device::Program* createProgram(amd::Program& owner,
                                 amd::option::Options* options = nullptr) override;

  // List of dummy functions which are disabled for NullDevice

  //! Create a new virtual device environment.
  device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr) override {
    ShouldNotReachHere();
    return nullptr;
  }

  virtual bool registerSvmMemory(void* ptr, size_t size) const {
    ShouldNotReachHere();
    return false;
  }

  virtual void deregisterSvmMemory(void* ptr) const { ShouldNotReachHere(); }

  //! Just returns nullptr for the dummy device
  device::Memory* createMemory(amd::Memory& owner) const override {
    ShouldNotReachHere();
    return nullptr;
  }
  device::Memory* createMemory(size_t size, size_t alignment = 0) const override {
    ShouldNotReachHere();
    return nullptr;
  }
  //! Sampler object allocation
  bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                     device::Sampler** sampler   //!< device sampler object
  ) const override {
    ShouldNotReachHere();
    return true;
  }

  //! Just returns nullptr for the dummy device
  device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const override {
    ShouldNotReachHere();
    return nullptr;
  }

  device::Signal* createSignal() const override {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Just returns nullptr for the dummy device
  void* svmAlloc(amd::Context& context,   //!< The context used to create a buffer
                 size_t size,             //!< size of svm spaces
                 size_t alignment,        //!< alignment requirement of svm spaces
                 cl_svm_mem_flags flags,  //!< flags of creation svm spaces
                 void* svmPtr             //!< existing svm pointer for mGPU case
  ) const override {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Just returns nullptr for the dummy device
  void svmFree(void* ptr  //!< svm pointer needed to be freed
  ) const override {
    ShouldNotReachHere();
    return;
  }

  void* virtualAlloc(void* req_addr, size_t size, size_t alignment) override {
    ShouldNotReachHere();
    return nullptr;
  }

  bool virtualFree(void* addr) override {
    ShouldNotReachHere();
    return true;
  }

  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice) override {
    ShouldNotReachHere();
    return false;
  }

  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const override {
    ShouldNotReachHere();
    return false;
  }

  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const override {
    ShouldNotReachHere();
    return true;
  }

  //! Determine if we can use device memory for SVM
  const bool forceFineGrain(amd::Memory* memory) const {
    return (memory->getContext().devices().size() > 1);
  }

  virtual bool importExtSemaphore(void** extSemahore, const amd::Os::FileDesc& handle,
                                  amd::ExternalSemaphoreHandleType sem_handle_type) override {
    ShouldNotReachHere();
    return false;
  }

  void DestroyExtSemaphore(void* extSemaphore) override { ShouldNotReachHere(); }

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device

  bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                          bool validateOnly) override {
    ShouldNotReachHere();
    return false;
  }

  bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                            bool validateOnly) override {
    ShouldNotReachHere();
    return false;
  }

  //! Releases non-blocking map target memory
  virtual void freeMapTarget(amd::Memory& mem, void* target) { ShouldNotReachHere(); }

  //! Empty implementation on Null device
  bool globalFreeMemory(size_t* freeMemory) const override {
    ShouldNotReachHere();
    return false;
  }

  //! Empty implementation on Null device
  bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                uint64_t* size_copied, int32_t* status) override {
    ShouldNotReachHere();
    return false;
  }

  //! Empty implementation on Null device
  bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                 uint64_t* size_copied, int32_t* status) override {
    ShouldNotReachHere();
    return false;
  }

  bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                    cl_set_device_clock_mode_output_amd* pSetClockModeOutput) override {
    return true;
  }

  bool IsHwEventReady(const amd::Event& event, bool wait = false,
                      amd::SyncPolicy policy = amd::SyncPolicy::Auto) const override {
    return false;
  }

  void getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const override {};
  void ReleaseGlobalSignal(void* signal) const override {}

#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUriLocator() const {
    ShouldNotReachHere();
    return nullptr;
  }
#endif
#endif

 private:
  static constexpr bool offlineDevice_ = true;
};

struct AgentInfo {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t fine_grain_pool;
  hsa_amd_memory_pool_t coarse_grain_pool;
  hsa_amd_memory_pool_t kern_arg_pool;
  hsa_amd_memory_pool_t ext_fine_grain_pool;
};

//! A HSA device ordinal (physical HSA device)
class Device : public NullDevice {
 public:
  //! Initialise the whole HSA device subsystem (init, device enumeration, etc).
  static bool init();
  static void tearDown();

  //! Lookup all AMD HSA devices and memory regions.
  static hsa_status_t iterateAgentCallback(hsa_agent_t agent, void* data);
  static hsa_status_t iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t region, void* data);
  static hsa_status_t iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t region, void* data);
  static hsa_status_t loaderQueryHostAddress(const void* device, const void** host);

  static bool loadHsaModules();

  hsa_agent_t getBackendDevice() const { return bkendDevice_; }
  //! Get the CPU agent with the least NUMA distance to this GPU
  const hsa_agent_t& getCpuAgent() const { return cpu_agent_info_->agent; }

  //! Get the CPU agent that is in a 'index' NUMA node
  const hsa_agent_t getCpuAgent(int index) const {
    if ((index < 0) || (index >= cpu_agents_.size())) {
      // Return default CPU agent
      return cpu_agent_info_->agent;
    }
    return cpu_agents_[index].agent;
  }

  void setupCpuAgent();  // Setup the CPU agent which has the least NUMA distance to this GPU

  void checkAtomicSupport();  //!< Check the support for pcie atomics

  //! Destructor for the physical HSA device
  virtual ~Device();

  // Temporary, delete it later when HSA Runtime and KFD is fully fucntional.
  void fake_device();

  ///////////////////////////////////////////////////////////////////////////////
  // TODO: Below are all mocked up virtual functions from amd::Device, they may
  // need real implementation.
  ///////////////////////////////////////////////////////////////////////////////

  //! Instantiate a new virtual device
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr) override;

  //! Construct an device program object from the ELF assuming it is valid
  virtual device::Program* createProgram(amd::Program& owner,
                                         amd::option::Options* options = nullptr) override;

  virtual device::Memory* createMemory(amd::Memory& owner) const override;
  virtual device::Memory* createMemory(size_t size, size_t alignment = 0) const override;
  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
  ) const override;

  //! Just returns nullptr for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const override {
    return nullptr;
  }

  virtual device::Signal* createSignal() const override;

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device
  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) override;

  /**
   * @brief Removes the external device as an available device.
   *
   * @note: The current implementation is to avoid build break
   * and does not represent actual / correct implementation. This
   * needs to be done.
   */
  bool unbindExternalDevice(
      uint flags,               //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const gfxDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* gfxContext,         //!< HGLRC/GLXContext handle
      bool validateOnly         //!< Only validate if the device can inter-operate with
                                //!< pDevice/pContext, do not bind.
  ) override;

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory) const override;
  virtual void* hostAlloc(size_t size, size_t alignment,
                          MemorySegment mem_seg = MemorySegment::kNoAtomics,
                          const void* agentInfo = nullptr) const override;  // nullptr uses default CPU agent
  virtual void hostFree(void* ptr, size_t size = 0) const override;

  virtual bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                        uint64_t* size_copied, int32_t* status) override;
  virtual bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                         uint64_t* size_copied, int32_t* status) override;

  bool deviceAllowAccess(void* dst) const override;

  bool allowPeerAccess(device::Memory* memory) const override;
  void deviceVmemRelease(uint64_t mem_handle) const;
  uint64_t deviceVmemAlloc(size_t size, uint64_t flags) const;

  void* deviceLocalAlloc(size_t size,
                        const AllocationFlags& flags = AllocationFlags{}) const override;
  void* reserveMemory(size_t size, size_t alignment) const;
  void releaseMemory(void* ptr, size_t size) const;
  void memFree(void* ptr, size_t size) const;

  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags = CL_MEM_READ_WRITE, void* svmPtr = nullptr) const override;

  virtual void svmFree(void* ptr) const override;

  virtual bool SetSvmAttributes(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                                bool use_cpu = false, int numa_id = kDefaultNumaNode) const override;
  virtual bool GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                                size_t num_attributes, const void* dev_ptr, size_t count) const override;
  virtual size_t ScratchLimitCurrent() const final;
  virtual bool UpdateScratchLimitCurrent(size_t limit) const final;
  virtual void* virtualAlloc(void* req_addr, size_t size, size_t alignment) override;
  virtual bool virtualFree(void* addr) override;

  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice) override;
  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const override;
  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const override { return true; }

  virtual bool ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags, void* shareableHandle) override;

  bool ImportShareableHSAHandle(void* osHandle, uint64_t* hsa_handle_ptr) const;

  virtual amd::Memory* ImportShareableVMMHandle(void* osHandle) override;

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput) override;

  virtual bool IsHwEventReady(const amd::Event& event, bool wait = false,
                              amd::SyncPolicy policy = amd::SyncPolicy::Auto) const override;
  virtual void getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const override;
  virtual void ReleaseGlobalSignal(void* signal) const override;
  virtual bool CreateUserEvent(amd::UserEvent* event) const override;
  virtual void SetUserEvent(amd::UserEvent* event) const override;

  //! Allocate host memory in terms of numa policy set by user
  void* hostNumaAlloc(size_t size, size_t alignment, MemorySegment mem_seg) const;

  //! Pin a host pointer allocated by C/C++ or OS allocator (i.e. ordinary system DRAM) and
  //! return a new device pointer accessible by the GPU agent.
  void* hostLock(void* hostMem, size_t size, MemorySegment memSegment) const;

  //! Returns transfer engine object
  const device::BlitManager& xferMgr() const { return xferQueue()->blitMgr(); }

  const size_t alloc_granularity() const { return alloc_granularity_; }

  const hsa_profile_t agent_profile() const { return agent_profile_; }

  //! Finds an appropriate map target
  amd::Memory* findMapTarget(size_t size) const;

  //! Adds a map target to the cache
  bool addMapTarget(amd::Memory* memory) const;

  //! Returns a ROC memory object from AMD memory object
  roc::Memory* getRocMemory(amd::Memory* mem  //!< Pointer to AMD memory object
  ) const;

  //! Create internal blit program
  bool createBlitProgram();

  // P2P agents avaialble for this device
  const std::vector<hsa_agent_t>& p2pAgents() const { return p2p_agents_; }

  //! Returns the list of HSA agents used for IPC memory attach
  const hsa_agent_t* IpcAgents() const { return p2p_agents_list_; }

  // User enabled peer devices
  const bool isP2pEnabled() const { return (enabled_p2p_devices_.size() > 0) ? true : false; }

  // Update the global free memory size
  void updateFreeMemory(size_t size, bool free);

  //! Returns the lock object for the virtual gpus list
  amd::Monitor& vgpusAccess() const { return vgpusAccess_; }

  typedef std::vector<VirtualGPU*> VirtualGPUs;
  //! Returns the list of all virtual GPUs running on this device
  const VirtualGPUs& vgpus() const { return vgpus_; }
  VirtualGPUs vgpus_;  //!< The list of all running virtual gpus (lock protected)

  VirtualGPU* xferQueue() const;

  //! Acquire HSA queue. This method can create a new HSA queue or
  hsa_queue_t* acquireQueue(
      uint32_t queue_size_hint, bool coop_queue = false, const std::vector<uint32_t>& cuMask = {},
      amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal,
      bool managed = false, bool dedicated_queue = false);

  //! Release HSA queue
  void releaseQueue(hsa_queue_t*, const std::vector<uint32_t>& cuMask = {}, bool coop_queue = false,
                    bool managed = false);

  hsa_queue_t* AcquireActiveQueue(amd::CommandQueue::Priority priority);
  bool ReleaseActiveQueue(hsa_queue_t* queue, amd::CommandQueue::Priority priority);

  //! For the given HSA queue, return an existing hostcall buffer or create a
  //! new one. queuePool_ keeps a mapping from HSA queue to hostcall buffer.
  void* getOrCreateHostcallBuffer(hsa_queue_t* queue, bool coop_queue = false,
                                  const std::vector<uint32_t>& cuMask = {});

  //! Return multi GPU grid launch sync buffer
  address MGSync() const { return mg_sync_; }

  //! Returns value for corresponding Link Attributes in a vector, given other device
  virtual bool findLinkInfo(const amd::Device& other_device, std::vector<LinkAttrType>* link_attr) override;

  //! Returns a GPU memory object from AMD memory object
  roc::Memory* getGpuMemory(amd::Memory* mem  //!< Pointer to AMD memory object
  ) const;

  //! Initialize memory in AMD HMM on the current device or keeps it in the host memory
  bool SvmAllocInit(void* memory, size_t size) const;

  void getGlobalCUMask(std::string cuMaskStr);

  static hsa_status_t BackendErrorCallBackHandler(const hsa_amd_event_t* event, void* data);

  static void RegisterBackendErrorCb();

  virtual amd::Memory* GetArenaMemObj(const void* ptr, size_t& offset, size_t size = 0) override;

  virtual uint32_t getPreferredNumaNode() const final { return preferred_numa_node_; }

  const bool isFineGrainSupported() const override;

  //! Returns True if memory pointer is known to ROCr (excludes HMM allocations)
  bool IsValidAllocation(const void* dev_ptr, size_t size, hsa_amd_pointer_info_t* ptr_info);

  //! Allocates hidden heap for device memory allocations
  void HiddenHeapAlloc(const VirtualGPU& gpu);
  //! Init hidden heap for device memory allocations
  void HiddenHeapInit(const VirtualGPU& gpu);
  bool isXgmi() const override { return isXgmi_; }

  //! SDMA engine allocation for per-stream affinity
  uint32_t AllocateSdmaEngine(VirtualGPU* vgpu, HwQueueEngine engine_type,
                              hsa_agent_t dstAgent, hsa_agent_t srcAgent) const {
    return sdma_engine_allocator_.AllocateEngine(vgpu, engine_type, dstAgent, srcAgent);
  }
  void ReleaseSdmaEngine(VirtualGPU* vgpu) const {
    sdma_engine_allocator_.ReleaseEngine(vgpu);
  }
  //! Returns the map of code objects to kernels
  const auto& KernelMap() const { return kernel_map_; }
  //! Adds a kernel to the kernel map
  void AddKernel(Kernel& gpuKernel) const;
  //! Removes a kernel from the kernel map
  void RemoveKernel(Kernel& gpuKernel) const;

  // Returns the number of allocated queues for a given priority on this device
  uint32_t NumQueues(uint qIndex) const { return num_queues_[qIndex].load(); }

  //! enum for keeping the total and available queue priorities
  enum QueuePriority : uint { Low = 0, Normal = 1, High = 2, Total = 3 };

  //! Returns true if PM4 emulation is enabled
  bool IsPm4Emulation() const { return pm4_emulation_; }

 private:
  bool create();

  //! Construct a new physical HSA device
  Device(hsa_agent_t bkendDevice);

  static constexpr int kDefaultNumaNode = -1;

  bool SetSvmAttributesInt(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                           bool first_alloc = false, bool use_cpu = false,
                           int numa_id = kDefaultNumaNode) const;
  static constexpr hsa_signal_value_t InitSignalValue = 1;

  static hsa_ven_amd_loader_1_00_pfn_t amd_loader_ext_table;

  amd::Monitor* mapCacheOps_;            //!< Lock to serialise cache for the map resources
  std::vector<amd::Memory*>* mapCache_;  //!< Map cache info structure

  bool populateOCLDeviceConstants();
  static bool isHsaInitialized_;
  static std::vector<hsa_agent_t> gpu_agents_;
  static std::vector<AgentInfo> cpu_agents_;
  uint32_t preferred_numa_node_;
  std::vector<hsa_agent_t> p2p_agents_;   //!< List of P2P agents available for this device
  mutable std::mutex lock_allow_access_;  //!< To serialize allow_access calls
  hsa_agent_t bkendDevice_;
  uint32_t pciDeviceId_;
  hsa_agent_t* p2p_agents_list_ = nullptr;
  hsa_profile_t agent_profile_;
  hsa_amd_memory_pool_t group_segment_;

  AgentInfo* cpu_agent_info_;

  hsa_amd_memory_pool_t gpuvm_segment_;
  hsa_amd_memory_pool_t gpu_fine_grained_segment_;
  hsa_amd_memory_pool_t gpu_ext_fine_grained_segment_;
  hsa_signal_t prefetch_signal_;  //!< Prefetch signal, used to explicitly prefetch SVM on device
  std::atomic<int> cache_state_;  //!< State of cache, kUnknown/kFlushedToDevice/kFlushedToSystem

  size_t gpuvm_segment_max_alloc_;
  size_t alloc_granularity_;
  static constexpr bool offlineDevice_ = false;
  VirtualGPU* xferQueue_;  //!< Transfer queue, created on demand

  std::atomic<size_t> freeMem_;       //!< Total of free memory available
  mutable amd::Monitor vgpusAccess_;  //!< Lock to serialise virtual gpu list access
  bool hsa_exclusive_gpu_access_;  //!< TRUE if current device was moved into exclusive GPU access
                                   //!< mode
  static address mg_sync_;         //!< MGPU grid launch sync memory (SVM location)

  struct QueueInfo {
    int refCount;           //! Reference counter. Shows how many time the queue was shared
    void* hostcallBuffer_;  //! Host call buffer for the HSA queue
    bool hasDedicatedQueue_;  //! True if this queue is a dedicated queue (e.g., null stream)

    // Constructor
    QueueInfo() : refCount(0), hostcallBuffer_(nullptr), hasDedicatedQueue_(false) {}

    //! Get the current hardware queue depth (wptr - rptr)
    static uint64_t GetHwQueueDepth(hsa_queue_t* queue) {
      uint64_t wptr = Hsa::queue_load_write_index_relaxed(queue);
      uint64_t rptr = Hsa::queue_load_read_index_relaxed(queue);
      return wptr - rptr;
    }

    //! Get a combined metric for queue selection (lower is better)
    uint64_t GetLoadMetric(hsa_queue_t* queue, uint32_t mode = 1) const {
      auto depth = GetHwQueueDepth(queue);

      // Dedicated queue penalty: prefer regular queues, but use dedicated if regular queues
      // have depth > ~128 packets. Penalty = 128 << 4 = 2048.
      uint64_t dedicated_queue_penalty = hasDedicatedQueue_ ? 2048 : 0;

      // Advanced weighted metric: Give queue depth significantly more weight than refCount
      uint64_t metric = dedicated_queue_penalty + (depth << 4) + static_cast<uint64_t>(refCount);
      return metric;
    }
  };

  struct QueueCompare {
    const Device* device_;

    QueueCompare(const Device* dev = nullptr) : device_(dev) {}

    // Customized queue compare operator to make sure the queues are sorted in the creation order
    bool operator()(hsa_queue_t* lhs, hsa_queue_t* rhs) const {
      if (device_ != nullptr && device_->settings().dynamic_queues_ > 0) {
        return (lhs->id < rhs->id) ? true : false;
      } else {
        return (lhs < rhs) ? true : false;
      }
    }
  };
  //! a vector for keeping Pool of HSA queues with low, normal and high priorities for recycling
  std::vector<std::map<hsa_queue_t*, QueueInfo, QueueCompare>> queuePool_;
  amd::Monitor active_queue_access_;            //!< Lock to serialise virtual gpu list access
  std::atomic<uint32_t> num_queues_[QueuePriority::Total] = {};  //!< Per-priority queue counters

  //! Use dynamic queues mode to get a queue from pool
  hsa_queue_t* getQueueFromPool(const uint qIndex, bool force_reuse = false);

  void* coopHostcallBuffer_;
  //! returns value for corresponding LinkAttrbutes in a vector given Memory pool.
  virtual bool findLinkInfo(const hsa_amd_memory_pool_t& pool,
                            std::vector<LinkAttrType>* link_attr);

  //! Pool of HSA queues with custom CU masks
  std::vector<std::map<hsa_queue_t*, QueueInfo, QueueCompare>> queueWithCUMaskPool_;
  hsa_amd_memory_pool_t getHostMemoryPool(MemorySegment mem_seg,
                                          const AgentInfo* agentInfo = nullptr) const;
  //! Read and Write mask for device<->host
  uint32_t maxSdmaReadMask_;
  uint32_t maxSdmaWriteMask_;
  bool isXgmi_;  //!< Flag to indicate if there is XGMI between CPU<->GPU
  bool pm4_emulation_ = false;  //!< Flag to indicate if PM4 emulation is enabled
  uint32_t numHwPipes_;  //!< Number of hardware pipes

  //! SDMA engine allocator for per-stream affinity
  struct SdmaEngineAllocator {
    amd::Monitor lock_;  //!< Protects the allocation state
    std::unordered_map<VirtualGPU*, uint32_t> vgpu_to_engine_;  //!< VirtualGPU -> engine mask
    std::atomic<uint32_t> next_rr_engine_{0};  //!< Simple RR counter for future use
    const Device& device_;  //!< Reference to parent device for accessing masks

    SdmaEngineAllocator(const Device& device)
        : lock_(true), device_(device) {}

    //! Allocate an SDMA engine for a VirtualGPU
    //! Queries HSA for engine status and preferred engines, then allocates
    //! For inter-GPU copies, strongly prefers recommended engines even if already allocated
    uint32_t AllocateEngine(VirtualGPU* vgpu, HwQueueEngine engine_type,
                           hsa_agent_t dstAgent, hsa_agent_t srcAgent);

    //! Release engine allocation for a VirtualGPU
    void ReleaseEngine(VirtualGPU* vgpu);
  };
  mutable SdmaEngineAllocator sdma_engine_allocator_;

  //! Code object to kernel info map (used in the crash dump analysis)
  mutable std::map<uint64_t, Kernel&> kernel_map_;

  //! Friend function callbackQueue can access and set device class variables.
  friend void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data);

 public:
  std::atomic<uint> numOfVgpus_;  //!< Virtual gpu unique index

  //! Returns the valid SDMA engine bitmask for the given operation type.
  uint32_t GetSdmaValidMask(HwQueueEngine engine_type) const {
    return (engine_type == HwQueueEngine::SdmaRead) ? maxSdmaReadMask_ : maxSdmaWriteMask_;
  }

#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUriLocator() const;
#endif
#endif
};  // class roc::Device

void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data);

}  // namespace amd::roc

/**
 * @}
 */
