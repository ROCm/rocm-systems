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

  //! Construct an HSAIL program object from the ELF assuming it is valid
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

  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const {
    ShouldNotReachHere();
    return false;
  }

  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const {
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
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr);

  //! Construct an HSAIL program object from the ELF assuming it is valid
  virtual device::Program* createProgram(amd::Program& owner,
                                         amd::option::Options* options = nullptr);

  virtual device::Memory* createMemory(amd::Memory& owner) const;
  virtual device::Memory* createMemory(size_t size, size_t alignment = 0) const;
  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
  ) const;

  //! Just returns nullptr for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const {
    return nullptr;
  }

  virtual device::Signal* createSignal() const;

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device
  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly);

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
  );

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory) const;
  virtual void* hostAlloc(size_t size, size_t alignment,
                          MemorySegment mem_seg = MemorySegment::kNoAtomics,
                          const void* agentInfo = nullptr) const override;  // nullptr uses default CPU agent
  virtual void hostFree(void* ptr, size_t size = 0) const;

  virtual bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                        uint64_t* size_copied, int32_t* status) override;
  virtual bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                         uint64_t* size_copied, int32_t* status) override;

  bool deviceAllowAccess(void* dst) const;

  bool allowPeerAccess(device::Memory* memory) const;
  void deviceVmemRelease(uint64_t mem_handle) const;
  uint64_t deviceVmemAlloc(size_t size, uint64_t flags) const;

  void* deviceLocalAlloc(size_t size,
                        const AllocationFlags& flags = AllocationFlags{}) const;
  void* reserveMemory(size_t size, size_t alignment) const;
  void releaseMemory(void* ptr, size_t size) const;
  void memFree(void* ptr, size_t size) const;

  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags = CL_MEM_READ_WRITE, void* svmPtr = nullptr) const;

  virtual void svmFree(void* ptr) const;

  virtual bool SetSvmAttributes(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                                bool use_cpu = false, int numa_id = kDefaultNumaNode) const;
  virtual bool GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                                size_t num_attributes, const void* dev_ptr, size_t count) const;
  virtual size_t ScratchLimitCurrent() const final;
  virtual bool UpdateScratchLimitCurrent(size_t limit) const final;
  virtual void* virtualAlloc(void* req_addr, size_t size, size_t alignment);
  virtual bool virtualFree(void* addr);

  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice);
  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const;
  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const { return true; }

  virtual bool ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags, void* shareableHandle);

  bool ImportShareableHSAHandle(void* osHandle, uint64_t* hsa_handle_ptr) const;

  virtual amd::Memory* ImportShareableVMMHandle(void* osHandle);

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput);

  virtual bool IsHwEventReady(const amd::Event& event, bool wait = false,
                              amd::SyncPolicy policy = amd::SyncPolicy::Auto) const;
  virtual void getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const;
  virtual void ReleaseGlobalSignal(void* signal) const;
  virtual bool CreateUserEvent(amd::UserEvent* event) const;
  virtual void SetUserEvent(amd::UserEvent* event) const;

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
  //! share previously created
  hsa_queue_t* acquireQueue(
      uint32_t queue_size_hint, bool coop_queue = false, const std::vector<uint32_t>& cuMask = {},
      amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal,
      bool managed = false);

  //! Release HSA queue
  void releaseQueue(hsa_queue_t*, const std::vector<uint32_t>& cuMask = {}, bool coop_queue = false,
                    bool managed = false);

  hsa_queue_t* AcquireActiveNormalQueue();
  bool ReleaseActiveNormalQueue(hsa_queue_t* queue);

  //! For the given HSA queue, return an existing hostcall buffer or create a
  //! new one. queuePool_ keeps a mapping from HSA queue to hostcall buffer.
  void* getOrCreateHostcallBuffer(hsa_queue_t* queue, bool coop_queue = false,
                                  const std::vector<uint32_t>& cuMask = {});

  //! Return multi GPU grid launch sync buffer
  address MGSync() const { return mg_sync_; }

  //! Returns value for corresponding Link Attributes in a vector, given other device
  virtual bool findLinkInfo(const amd::Device& other_device, std::vector<LinkAttrType>* link_attr);

  //! Returns a GPU memory object from AMD memory object
  roc::Memory* getGpuMemory(amd::Memory* mem  //!< Pointer to AMD memory object
  ) const;

  //! Initialize memory in AMD HMM on the current device or keeps it in the host memory
  bool SvmAllocInit(void* memory, size_t size) const;

  void getGlobalCUMask(std::string cuMaskStr);

  static hsa_status_t BackendErrorCallBackHandler(const hsa_amd_event_t* event, void* data);

  static void RegisterBackendErrorCb();

  virtual amd::Memory* GetArenaMemObj(const void* ptr, size_t& offset, size_t size = 0);

  const uint32_t getPreferredNumaNode() const { return preferred_numa_node_; }

  const bool isFineGrainSupported() const;

  //! Returns True if memory pointer is known to ROCr (excludes HMM allocations)
  bool IsValidAllocation(const void* dev_ptr, size_t size, hsa_amd_pointer_info_t* ptr_info);

  //! Allocates hidden heap for device memory allocations
  void HiddenHeapAlloc(const VirtualGPU& gpu);
  //! Init hidden heap for device memory allocations
  void HiddenHeapInit(const VirtualGPU& gpu);
  void getSdmaRWMasks(uint32_t* readMask, uint32_t* writeMask) const;
  bool isXgmi() const { return isXgmi_; }

  //! Returns the map of code objects to kernels
  const auto& KernelMap() const { return kernel_map_; }
  //! Adds a kernel to the kernel map
  void AddKernel(Kernel& gpuKernel) const;
  //! Removes a kernel from the kernel map
  void RemoveKernel(Kernel& gpuKernel) const;

  // Returns the number of allocated normal queues on this device
  uint32_t NumNormalQueues() const { return num_normal_queues_.load(); }

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
  };

  struct QueueCompare {
    // Customized queue compare operator to make sure the queues are sorted in the creation order
    bool operator()(hsa_queue_t* lhs, hsa_queue_t* rhs) const {
      if (DEBUG_HIP_DYNAMIC_QUEUES) {
        return (lhs->id < rhs->id) ? true : false;
      } else {
        return (lhs < rhs) ? true : false;
      }
    }
  };
  //! a vector for keeping Pool of HSA queues with low, normal and high priorities for recycling
  std::vector<std::map<hsa_queue_t*, QueueInfo, QueueCompare>> queuePool_;
  amd::Monitor active_queue_access_;            //!< Lock to serialise virtual gpu list access
  std::atomic<uint32_t> num_normal_queues_{0};  //!< The total number of allocated normal queues

  //! returns a hsa queue from queuePool with least refCount and updates the refCount as well
  hsa_queue_t* getQueueFromPool(const uint qIndex);

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

  //! Code object to kernel info map (used in the crash dump analysis)
  mutable std::map<uint64_t, Kernel&> kernel_map_;

  //! Friend function callbackQueue can access and set device class variables.
  friend void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data);

 public:
  std::atomic<uint> numOfVgpus_;  //!< Virtual gpu unique index

  //! enum for keeping the total and available queue priorities
  enum QueuePriority : uint { Low = 0, Normal = 1, High = 2, Total = 3 };

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
