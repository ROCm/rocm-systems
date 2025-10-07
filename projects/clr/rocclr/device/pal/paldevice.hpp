/* Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.

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
#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/memory.hpp"
#include "platform/runtime.hpp"
#include "utils/concurrent.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/palsettings.hpp"
#include "device/pal/palappprofile.hpp"
#include "device/pal/palcapturemgr.hpp"
#include "device/pal/palsignal.hpp"
#include "acl.h"
#include "memory"

#include <atomic>
#include <unordered_set>

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "device/devurilocator.hpp"
#endif
#endif
/*! \addtogroup PAL
 *  @{
 */

//! PAL Device Implementation
namespace amd::pal {

//! A nil device object
class NullDevice : public amd::Device {
 protected:
#if defined(WITH_COMPILER_LIB)
  static Compiler* compiler_;
#endif

 public:
#if defined(WITH_COMPILER_LIB)
  Compiler* compiler() const { return compiler_; }
#endif

 public:
  static bool init(void);

  //! Construct a new identifier
  NullDevice();

  //! Creates an offline device with the specified target
  bool create(const char* palName,            //!< Device name
              const amd::Isa& isa,            //!< Device ISA
              Pal::GfxIpLevel ipLevel,        //!< GPU ip level
              Pal::AsicRevision asicRevision  //!< PAL ASIC revision
  );

  //! Instantiate a new virtual device
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr) {
    return nullptr;
  }

  //! Compile the given source code.
  virtual device::Program* createProgram(amd::Program& owner,
                                         amd::option::Options* options = nullptr);

  //! Just returns NULL for the dummy device
  virtual device::Memory* createMemory(amd::Memory& owner) const { return nullptr; }
  //! Just returns NULL for the dummy device
  virtual device::Memory* createMemory(size_t size, size_t alignment = 0) const { return nullptr; }
  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
  ) const {
    ShouldNotReachHere();
    return true;
  }

  //! Just returns NULL for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const {
    return nullptr;
  }

  //! Signal object allocation
  virtual device::Signal* createSignal() const { return nullptr; }

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device

  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) {
    return true;
  }

  virtual bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                    bool validateOnly) {
    return true;
  }

  //! Releases non-blocking map target memory
  virtual void freeMapTarget(amd::Memory& mem, void* target) {}

  Pal::GfxIpLevel ipLevel() const { return ipLevel_; }
  Pal::AsicRevision asicRevision() const { return asicRevision_; }

  //! Empty implementation on Null device
  virtual bool globalFreeMemory(size_t* freeMemory) const { return false; }

  //! Empty implementation on Null device
  virtual bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                        uint64_t* size_copied, int32_t* status) {
    ShouldNotReachHere();
    return false;
  }

  //! Empty implementation on Null device
  virtual bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                         uint64_t* size_copied, int32_t* status) {
    ShouldNotReachHere();
    return false;
  }

  //! Get GPU device settings
  const pal::Settings& settings() const { return reinterpret_cast<pal::Settings&>(*settings_); }
  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags, void* svmPtr) const {
    return NULL;
  }
  virtual void svmFree(void* ptr) const { return; }
  virtual void* virtualAlloc(void* addr, size_t size, size_t alignment) { return nullptr; };
  virtual bool virtualFree(void* addr) { return true; }

  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice) {
    return true;
  }

  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const { return true; }

  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const { return true; }

  virtual bool ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags,
                                        void* shareableHandle) {
    return false;
  }

  virtual amd::Memory* ImportShareableVMMHandle(void* osHandle) { return nullptr; }

  virtual bool importExtSemaphore(void** extSemaphore, const amd::Os::FileDesc& handle,
                                  amd::ExternalSemaphoreHandleType sem_handle_type) override {
    return false;
  }
  virtual void DestroyExtSemaphore(void* extSemaphore) {}

  void* Alloc(const Util::AllocInfo& allocInfo) { return allocator_.Alloc(allocInfo); }
  void Free(const Util::FreeInfo& freeInfo) { allocator_.Free(freeInfo); }
  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
    return true;
  }
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUriLocator() const { return nullptr; }
#endif
#endif
 protected:
  static Util::GenericAllocator allocator_;  //!< Generic memory allocator in PAL

  Pal::AsicRevision asicRevision_;  //!< ASIC revision
  Pal::GfxIpLevel ipLevel_;         //!< Device IP level
  const char* palName_;             //!< Device name

  //! Fills OpenCL device info structure
  void fillDeviceInfo(const Pal::DeviceProperties& palProp,  //!< PAL device properties
                      const Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount],
                      size_t maxTextureSize,          //!< Maximum texture size supported in HW
                      uint numComputeRings,           //!< Number of compute rings
                      uint numExclusiveComputeRings,  //!< Number of exclusive compute rings
                      Pal::IDevice* pal_device        //!< PAL device for which info is filled
  );
};

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
class GpuMemoryReference;
class VirtualDevice;
class PrintfDbg;
class ThreadTrace;

#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif
enum class ExclusiveQueueType : uint32_t { RealTime0 = 0, RealTime1, Medium };
class Sampler : public device::Sampler {
 public:
  //! Constructor
  Sampler(const Device& dev) : dev_(dev) {}

  //! Default destructor for the device memory object
  virtual ~Sampler();

  //! Creates a device sampler from the OCL sampler state
  bool create(uint32_t oclSamplerState,  //!< OCL sampler state
              const uint addressMode[3]  //!< Address modes in X, Y and Z
  );

  //! Creates a device sampler from the OCL sampler state
  bool create(const amd::Sampler& owner  //!< AMD sampler object
  );

 private:
  //! Disable default copy constructor
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);

  const Device& dev_;  //!< Device object associated with the sampler
};

//! A GPU device ordinal (physical GPU device)
class Device : public NullDevice {
 public:
  struct QueueRecycleInfo : public amd::HeapObject {
    int counter_;                    //!< Lock usage counter
    Pal::EngineType engineType_;     //!< Engine type
    uint32_t index_;                 //!< HW queue index for scratch buffer access
    amd::Monitor queue_lock_;        //!< Queue lock for access
    AqlPacketMgmt aql_packet_mgmt_;  //!< AQL packets management class for debugger support
    QueueRecycleInfo()
        : counter_(1),
          engineType_(Pal::EngineTypeCompute),
          index_(0),
          queue_lock_(true) /* Queue lock for sharing */ {}

    //! Returns the aql packet list
    uintptr_t AqlPacketList() const {
      return reinterpret_cast<uintptr_t>(&aql_packet_mgmt_.aql_packets_);
    }
  };

  //! Locks any access to the virtual GPUs
  class ScopedLockVgpus : public amd::StackObject {
   public:
    //! Default constructor
    ScopedLockVgpus(const Device& dev);

    //! Destructor
    ~ScopedLockVgpus();

   private:
    const Device& dev_;  //! Device object
  };

  //! Transfer buffers
  class XferBuffers : public amd::HeapObject {
   public:
    static constexpr size_t MaxXferBufListSize = 8;

    //! Default constructor
    XferBuffers(const Device& device, Resource::MemoryType type, size_t bufSize)
        : type_(type), bufSize_(bufSize), acquiredCnt_(0), gpuDevice_(device) {}

    //! Default destructor
    ~XferBuffers();

    //! Creates the xfer buffers object
    bool create();

    //! Acquires an instance of the transfer buffers
    Memory& acquire();

    //! Releases transfer buffer
    void release(VirtualGPU& gpu,  //!< Virual GPU object used with the buffer
                 Memory& buffer    //!< Transfer buffer for release
    );

    //! Returns the buffer's size for transfer
    size_t bufSize() const { return bufSize_; }

   private:
    //! Disable copy constructor
    XferBuffers(const XferBuffers&);

    //! Disable assignment operator
    XferBuffers& operator=(const XferBuffers&);

    //! Get device object
    const Device& dev() const { return gpuDevice_; }

    Resource::MemoryType type_;       //!< The buffer's type
    size_t bufSize_;                  //!< Staged buffer size
    std::list<Memory*> freeBuffers_;  //!< The list of free buffers
    std::atomic<uint> acquiredCnt_;   //!< The total number of acquired buffers
    amd::Monitor lock_;               //!< Stgaed buffer acquire/release lock
    const Device& gpuDevice_;         //!< GPU device object
  };

  struct ScratchBuffer : public amd::HeapObject {
    Memory* memObj_;   //!< Memory objects for scratch buffers
    uint64_t offset_;  //!< Offset from the global scratch store
    uint64_t size_;    //!< Scratch buffer size on this queue

    //! Default constructor
    ScratchBuffer() : memObj_(nullptr), offset_(0), size_(0) {}

    //! Default constructor
    ~ScratchBuffer();

    //! Destroys memory objects
    void destroyMemory();
  };


  class SrdManager : public amd::HeapObject {
   public:
    SrdManager(const Device& dev, uint srdSize, uint bufSize)
        : dev_(dev),
          numFlags_(bufSize / (srdSize * MaskBits)),
          srdSize_(srdSize),
          bufSize_(bufSize) {}
    ~SrdManager();

    //! Allocates a new SRD slot for a resource
    uint64_t allocSrdSlot(address* cpuAddr);

    //! Frees a SRD slot
    void freeSrdSlot(uint64_t addr);

    // Fills the memory list for VidMM KMD
    void fillResourceList(VirtualGPU& gpu);

   private:
    //! Disable copy constructor
    SrdManager(const SrdManager&);

    //! Disable assignment operator
    SrdManager& operator=(const SrdManager&);

    struct Chunk {
      Memory* buf_;
      uint* flags_;
      Chunk() : buf_(NULL), flags_(NULL) {}
    };

    static constexpr uint MaskBits = 32;
    const Device& dev_;        //!< GPU device for the chunk manager
    amd::Monitor ml_;          //!< Global lock for the SRD manager
    std::vector<Chunk> pool_;  //!< Pool of SRD buffers
    uint numFlags_;            //!< Total number of flags in array
    uint srdSize_;             //!< SRD size
    uint bufSize_;             //!< Buffer size that holds SRDs
  };

  //! Initialise the whole GPU device subsystem
  static bool init();

  //! Shutdown the whole GPU device subsystem
  static void tearDown();

  //! Construct a new physical GPU device
  Device();

  //! Initialise a device (i.e. all parts of the constructor that could
  //! potentially fail)
  bool create(Pal::IDevice* device  //!< PAL device interface object
  );

  //! Destructor for the physical GPU device
  virtual ~Device();

  //! Instantiate a new virtual device
  device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = NULL);

  //! Memory allocation
  virtual device::Memory* createMemory(amd::Memory& owner  //!< abstraction layer memory object
  ) const;
  virtual device::Memory* createMemory(size_t size, size_t alignment = 0) const;
  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
  ) const;

  //! Allocates a view object from the device memory
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const;

  //! Signal object allocation
  virtual device::Signal* createSignal() const { return new pal::Signal(); }

  //! Create the device program.
  virtual device::Program* createProgram(amd::Program& owner, amd::option::Options* options = NULL);

  //! Attempt to bind with external graphics API's device/context
  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly);

  //! Attempt to unbind with external graphics API's device/context
  virtual bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                    bool validateOnly);

  //! Free resource cache on device if OCL context was destroyed.
  //! @note: Backend device doesn't track resources per context and releases all resources,
  //! regardless the number of still active contexts
  virtual void ContextDestroy() {
    // The if condition is a best effort to avoid crash if the function is called after DLL detached
    if (!amd::Runtime::isLibraryDetached()) {
      resourceCache().free();
    }
  }
  //! Validates kernel before execution
  virtual bool validateKernel(const amd::Kernel& kernel,  //!< AMD kernel object
                              const device::VirtualDevice* vdev, bool coop_group = false);

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput);

  //! Retrieves information about free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory) const;

  /**
   * @brief Read data from a file to device memory.
   * @param[IN] handle: file descriptor of the file to read.
   * @param[IN] devicePtr: VRAM buffer pointer.
   * @param[IN] size: size of read.
   * @param[IN] file_offset: offset into fd where data has to be read.
   * @param[IN/OUT] size_copied: actual size read.
   * @param[IN/OUT] status: additional status.
   */
  virtual bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                        uint64_t* size_copied, int32_t* status);

  /**
   * Write data from device memory to a file.
   * @param[IN] handle: file descriptor of the file to write.
   * @param[IN] devicePtr: VRAM buffer pointer.
   * @param[IN] size: size of write.
   * @param[IN] file_offset: offset into fd where data has to written.
   * @param[IN/OUT] size_copied: actual size copied.
   * @param[IN/OUT] status: additional status.
   */
  virtual bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                         uint64_t* size_copied, int32_t* status);

  //! Returns a GPU memory object from AMD memory object
  pal::Memory* getGpuMemory(amd::Memory* mem  //!< Pointer to AMD memory object
  ) const;

  amd::Monitor& lockAsyncOps() const { return lockAsyncOps_; }

  //! Returns the lock object for the virtual gpus list
  amd::Monitor& vgpusAccess() const { return vgpusAccess_; }

  //! Returns the monitor object for PAL
  amd::Monitor& lockPAL() const { return lockPAL_; }

  //! Returns the monitor object for PAL
  amd::Monitor& lockResources() const { return lockResourceOps_; }

  //! Returns the number of virtual GPUs allocated on this device
  uint numOfVgpus() const { return numOfVgpus_; }
  uint numOfVgpus_;  //!< The number of virtual GPUs (lock protected)

  typedef std::vector<VirtualGPU*> VirtualGPUs;

  //! Returns the list of all virtual GPUs running on this device
  const VirtualGPUs& vgpus() const { return vgpus_; }
  VirtualGPUs vgpus_;  //!< The list of all running virtual gpus (lock protected)

  //! Scratch buffer allocation
  pal::Memory* createScratchBuffer(size_t size  //!< Size of buffer
  ) const;

  //! Returns transfer buffer object
  XferBuffers& xferRead() const { return *xferRead_; }

  //! Finds an appropriate map target
  amd::Memory* findMapTarget(size_t size) const;

  //! Adds a map target to the cache
  bool addMapTarget(amd::Memory* memory) const;

  //! Returns resource cache object
  ResourceCache& resourceCache() const { return *resourceCache_; }

  //! Returns the number of available compute rings
  uint numComputeEngines() const { return computeEnginesId_.size(); }

  //! Returns the vector of available compute rings with the engine index
  const std::vector<uint32_t>& computeEnginesId() const { return computeEnginesId_; }

  //! Returns the number of available compute rings
  uint numExclusiveComputeEngines() const {
    return exclusiveComputeEnginesId_.size() +
           ((exclusiveComputeEnginesId().find(ExclusiveQueueType::RealTime1) ==
             exclusiveComputeEnginesId().end())
                ? 1
                : 0);
  }

  //! Returns the map of available exclusive compute rings with the engine index
  const std::map<ExclusiveQueueType, uint32_t>& exclusiveComputeEnginesId() const {
    return exclusiveComputeEnginesId_;
  }

  //! Returns the number of available DMA engines
  uint numDMAEngines() const { return numDmaEngines_; }

  //! Returns engines object
  const device::BlitManager& xferMgr() const;

  VirtualGPU* xferQueue() const { return xferQueue_; }

  //! Retrieves the internal format from the OCL format
  Pal::ChNumFormat getPalFormat(const amd::Image::Format& format,  //! OCL image format
                                Pal::ChannelMapping* channel) const;

  const ScratchBuffer* scratch(uint idx) const { return scratch_[idx]; }

  //! Returns the global scratch buffer
  Memory* globalScratchBuf() const { return globalScratchBuf_; };

  //! Destroys scratch buffer memory
  void destroyScratchBuffers();

  //! Initialize heap resources if uninitialized
  bool initializeHeapResources();

  //! Set HW sampler to the specified state
  void fillHwSampler(uint32_t state,                       //!< Sampler's OpenCL state
                     void* hwState,                        //!< Sampler's HW state
                     uint32_t hwStateSize,                 //!< Size of sampler's HW state
                     const uint* addressMode,              //!< Address modes in X, Y and Z
                     uint32_t mipFilter = CL_FILTER_NONE,  //!< Mip filter
                     float minLod = 0.f,                   //!< Min level of detail
                     float maxLod = CL_MAXFLOAT            //!< Max level of detail
  ) const;

  //! host memory alloc
  virtual void* hostAlloc(size_t size, size_t alignment, MemorySegment mem_seg = kNoAtomics,
                          const void* agentInfo = nullptr) const override;

  //! SVM allocation
  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags, void* svmPtr) const;

  bool allowPeerAccess(device::Memory* memory) const;

  //! Free host SVM memory
  void hostFree(void* ptr, size_t size) const;

  //! SVM free
  virtual void svmFree(void* ptr) const;

  //! Virtual address space allocation(reservation)
  virtual void* virtualAlloc(void* addr, size_t size, size_t alignment);
  virtual bool virtualFree(void* addr);

  //! Set/Get memory access set by the app
  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice);
  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const;
  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const;

  virtual bool ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags, void* shareableHandle);

  virtual amd::Memory* ImportShareableVMMHandle(void* osHandle);

  //! Returns SRD manger object
  SrdManager& srds() const { return *srdManager_; }

  //! Returns PAL device properties
  const Pal::DeviceProperties& properties() const { return properties_; }

  //! Returns PAL platform interface
  Pal::IPlatform* iPlat() const { return platform_; }

  //! Returns PAL device interface
  Pal::IDevice* iDev() const { return device_; }

  //! Allow access for peer device
  bool deviceAllowAccess(void* dst) const;

  //! Returns a handle to the capture manager (RGP or UberTrace)
  ICaptureMgr* captureMgr() const { return captureMgr_; }

  //! Update free memory for OCL extension
  void updateAllocedMemory(Pal::GpuHeap heap,  //!< PAL GPU heap for update
                           Pal::gpusize size,  //!< Size of alocated/destroyed memory
                           bool free           //!< TRUE if runtime frees memory
  ) const;

  //! Create internal blit program
  bool createBlitProgram();

  //! Interop for GL device
  bool initGLInteropPrivateExt(void* GLplatformContext, void* GLdeviceContext) const;
  bool glCanInterop(void* GLplatformContext, void* GLdeviceContext) const;
  bool resGLAssociate(void* GLContext, uint name, uint type, Pal::OsExternalHandle* handle,
                      void** mbResHandle, size_t* offset, cl_image_format& newClFormat
#ifdef ATI_OS_WIN
                      ,
                      Pal::DoppDesktopInfo& doppDesktopInfo
#endif
  ) const;
  bool resGLAcquire(void* GLplatformContext, void* mbResHandle, uint type) const;
  bool resGLRelease(void* GLplatformContext, void* mbResHandle, uint type) const;
  bool resGLFree(void* GLplatformContext, void* mbResHandle, uint type) const;

  //! Adds a resource to the global list
  void addResource(Resource* res) const {
    amd::ScopedLock lock(lockResources());
    auto findIt = resourceList_->find(res);
    res->resizeGpuEvents(numOfVgpus() - 1);
    if (resourceList_->end() == findIt) {
      resourceList_->insert(res);
    }
  }

  //! Removes a resource from the global list
  void removeResource(Resource* res) const {
    amd::ScopedLock lock(lockResources());
    resourceList_->erase(res);
  }

  //! Resizes global resource list to accumulate a new queue
  void resizeResoureList(uint index) const {
    // Not safe to resize the list when runtime creates/destroys a queue at the same time
    // or other queues process a command, since the size of the TS array can change
    Device::ScopedLockVgpus v(*this);
    amd::ScopedLock r(lockResources());
    for (const auto& it : *resourceList_) {
      it->resizeGpuEvents(index);
    }
  }

  //! Erases an old queue from the list
  void eraseResoureList(uint index) const {
    amd::ScopedLock lock(lockResources());
    for (const auto& it : *resourceList_) {
      it->eraseGpuEvents(index);
    }
  }

  bool AcquireExclusiveGpuAccess();
  void ReleaseExclusiveGpuAccess(VirtualGPU& vgpu) const;

  //! Returns PAL Queue pool for recycling
  std::map<Pal::IQueue*, QueueRecycleInfo*>& QueuePool() { return queue_pool_; }
  const std::map<Pal::IQueue*, QueueRecycleInfo*>& QueuePool() const { return queue_pool_; }

  virtual bool findLinkInfo(const amd::Device& other_device, std::vector<LinkAttrType>* link_attr) {
    // Unsupported in PAL
    LogError("The function is unsupported on Windows");
    return false;
  }

  virtual bool importExtSemaphore(void** extSemaphore, const amd::Os::FileDesc& handle,
                                  amd::ExternalSemaphoreHandleType sem_handle_type) override;

  virtual void DestroyExtSemaphore(void* extSemaphore);
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUrilocator() const { return nullptr; }
#endif
#endif
  //! Allocates hidden heap for device memory allocations
  void HiddenHeapAlloc(const VirtualGPU& gpu);

  Pal::gpusize GetMaxFrameBuffer() const { return maxFrameBufferAllocation_; }

  Pal::gpusize TotalAlloc() const {
    Pal::gpusize local = allocedMem[Pal::GpuHeapLocal] - resourceCache().persistentCacheSize();
    Pal::gpusize invisible = allocedMem[Pal::GpuHeapInvisible] - resourceCache().lclCacheSize();
    Pal::gpusize total_alloced = local + invisible;
    return total_alloced;
  }

 private:
  static void PAL_STDCALL PalDeveloperCallback(void* pPrivateData, const Pal::uint32 deviceIndex,
                                               Pal::Developer::CallbackType type, void* pCbData);

  //! Disable copy constructor
  Device(const Device&);

  //! Disable assignment
  Device& operator=(const Device&);

  //! Sends the stall command to all queues
  bool stallQueues();

  //! Buffer allocation
  pal::Memory* createBuffer(amd::Memory& owner,  //!< Abstraction layer memory object
                            bool directAccess    //!< Use direct host memory access
  ) const;

  //! Image allocation
  pal::Memory* createImage(amd::Memory& owner,  //!< Abstraction layer memory object
                           bool directAccess    //!< Use direct host memory access
  ) const;

  //! Allocates/reallocates the scratch buffer, according to the usage
  bool allocScratch(uint regNum,             //!< Number of the scratch registers
                    const VirtualGPU* vgpu,  //!< Virtual GPU for the allocation
                    uint vgprs               //!< Used VGPRs in the kernel
  );

  //! Interop for D3D devices
  bool associateD3D11Device(void* d3d11Device  //!< void* is of type ID3D11Device*
  );
  bool associateD3D10Device(void* d3d10Device  //!< void* is of type ID3D10Device*
  );
  bool associateD3D9Device(void* d3d9Device  //!< void* is of type IDirect3DDevice9*
  );
  //! Interop for GL device
  bool glAssociate(void* GLplatformContext, void* GLdeviceContext) const;
  bool glDissociate(void* GLplatformContext, void* GLdeviceContext) const;

  static char* platformObj_;         //!< Memory allocated for PAL platform object
  static Pal::IPlatform* platform_;  //!< Pointer to the PAL platform object

  mutable amd::Monitor lockAsyncOps_;  //!< Lock to serialise all async ops on this device
  //! Lock to serialise all async ops on initialization heap operation
  mutable amd::Monitor lockForInitHeap_;
  mutable amd::Monitor lockPAL_;          //!< Lock to serialise PAL access
  mutable amd::Monitor vgpusAccess_;      //!< Lock to serialise virtual gpu list access
  mutable amd::Monitor scratchAlloc_;     //!< Lock to serialise scratch allocation
  mutable amd::Monitor mapCacheOps_;      //!< Lock to serialise cache for the map resources
  mutable amd::Monitor lockResourceOps_;  //!< Lock to serialise resource access
  mutable std::mutex lockAllowAccess_;    //!< To serialize allow_access calls
  XferBuffers* xferRead_;                 //!< Transfer buffers read
  std::vector<amd::Memory*>* mapCache_;   //!< Map cache info structure
  ResourceCache* resourceCache_;          //!< Resource cache
  std::map<ExclusiveQueueType, uint32_t>
      exclusiveComputeEnginesId_;           //!< The number of available compute engines
  std::vector<uint32_t> computeEnginesId_;  //!< PAL index for compute engine
  uint numDmaEngines_;                      //!< The number of available compute engines
  bool heapInitComplete_;                //!< Keep track of initialization status of heap resources
  VirtualGPU* xferQueue_;                //!< Transfer queue
  std::vector<ScratchBuffer*> scratch_;  //!< Scratch buffers for kernels
  Memory* globalScratchBuf_;             //!< Global scratch buffer
  SrdManager* srdManager_;               //!< SRD manager object
  static AppProfile appProfile_;         //!< application profile
  mutable bool freeCPUMem_ = false;      //!< flag to mark GPU free SVM CPU mem
  Pal::DeviceProperties properties_;     //!< PAL device properties
  Pal::IDevice* device_;                 //!< PAL device object
  mutable std::atomic<Pal::gpusize>
      allocedMem[Pal::GpuHeap::GpuHeapCount];    //!< Free memory counter
  std::unordered_set<Resource*>* resourceList_;  //!< Active resource list
  ICaptureMgr* captureMgr_;                      //!< RGP/UberTrace capture manager
  Pal::GpuMemoryHeapProperties
      heaps_[Pal::GpuHeapCount];           //!< Information about heaps, returned from PAL
  Pal::gpusize maxFrameBufferAllocation_;  //!< To reserve some memory in frame buffer
  std::map<Pal::IQueue*, QueueRecycleInfo*> queue_pool_;  //!< Pool of PAL queues for recycling
  amd::Program* trap_handler_ = nullptr;  //!< Trap handler program for debugger setup
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
