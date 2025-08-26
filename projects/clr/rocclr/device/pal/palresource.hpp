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

#include "platform/command.hpp"
#include "platform/program.hpp"
#include "device/pal/paldefs.hpp"
#include "util/palBuddyAllocatorImpl.h"

#include <atomic>
#include <unordered_map>

//! \namespace amd::pal PAL Resource Implementation
namespace amd::pal {

class Device;
class VirtualGPU;

/*! \addtogroup PAL PAL Resource Implementation
 *  @{
 */
class GpuMemoryReference : public amd::ReferenceCountedObject {
 public:
  static GpuMemoryReference* Create(const Device& dev, const Pal::GpuMemoryCreateInfo& createInfo);

  static GpuMemoryReference* Create(const Device& dev,
                                    const Pal::PinnedGpuMemoryCreateInfo& createInfo);

  static GpuMemoryReference* Create(const Device& dev,
                                    const Pal::SvmGpuMemoryCreateInfo& createInfo);

  static GpuMemoryReference* Create(const Device& dev,
                                    const Pal::ExternalGpuMemoryOpenInfo& openInfo);

  static GpuMemoryReference* Create(const Device& dev, const Pal::ExternalImageOpenInfo& openInfo,
                                    Pal::ImageCreateInfo* imgCreateInfo, Pal::IImage** image);

  static GpuMemoryReference* Create(const Device& dev, const Pal::PeerGpuMemoryOpenInfo& openInfo);

  static GpuMemoryReference* Create(const Device& dev, const Pal::PeerImageOpenInfo& openInfo,
                                    Pal::IImage** image);
  //! Default constructor
  GpuMemoryReference(const Device& dev);

  //! Get PAL memory object
  Pal::IGpuMemory* iMem() const { return gpuMem_; }

  Pal::Result MakeResident() const;

  Pal::IGpuMemory* gpuMem_;  //!< PAL GPU memory object
  void* cpuAddress_;         //!< CPU address of this memory
  const Device& device_;     //!< GPU device
  //! @note: This field is necessary for the thread safe release only
  VirtualGPU* gpu_;        //!< Resource will be used only on this queue
  bool va_range_ = false;  //!< Resource is a VA range(@note: PAL doesn't provide this info)

 protected:
  //! Default destructor
  ~GpuMemoryReference();

 private:
  //! Disable copy constructor
  GpuMemoryReference(const GpuMemoryReference&) = delete;

  //! Disable operator=
  GpuMemoryReference& operator=(const GpuMemoryReference&) = delete;
};

static constexpr Pal::gpusize MaxGpuAlignment = 4 * Ki;

//! GPU resource
class Resource : public amd::HeapObject {
 public:
  enum InteropType {
    InteropTypeless = 0,
    InteropVertexBuffer,
    InteropIndexBuffer,
    InteropRenderBuffer,
    InteropTexture,
    InteropTextureViewLevel,
    InteropTextureViewCube,
    InteropSurface
  };

  struct CreateParams : public amd::StackObject {
    amd::Memory* owner_;       //!< Resource's owner
    VirtualGPU* gpu_;          //!< Resource won't be shared between multiple queues
    const Resource* svmBase_;  //!< SVM base for MGPU allocations
    bool interprocess_;        //!< Ressource can be used in the interprocess communication
    size_t alignment_;         //!< allocation address alignment
    CreateParams()
        : owner_(nullptr), gpu_(nullptr), svmBase_(nullptr), interprocess_(false), alignment_(0) {}
  };

  struct PinnedParams : public CreateParams {
    const amd::HostMemoryReference* hostMemRef_;  //!< System memory pointer for pinning
    size_t size_;                                 //!< System memory size
  };

  struct ViewParams : public CreateParams {
    size_t offset_;             //!< Alias resource offset
    size_t size_;               //!< Alias resource size
    const Resource* resource_;  //!< Parent resource for the view creation
    const void* memory_;
  };

  struct ImageViewParams : public CreateParams {
    size_t level_;              //!< Image mip level for a new view
    size_t layer_;              //!< Image layer for a new view
    const Resource* resource_;  //!< Parent resource for the view creation
    const void* memory_;
  };

  struct ImageBufferParams : public CreateParams {
    const Resource* resource_;  //!< Parent resource for the image creation
    const void* memory_;
  };

  struct OGLInteropParams : public CreateParams {
    InteropType type_;  //!< OGL resource type
    uint handle_;       //!< OGL resource handle
    uint mipLevel_;     //!< Texture mip level
    uint layer_;        //!< Texture layer
    void* glPlatformContext_;
  };

  struct VkInteropParams : public CreateParams {
    InteropType type_;  //!< Vulkan resource type
    amd::Os::FileDesc handle_;
    const void* name_;
    bool nt_handle_;
  };

#ifdef _WIN32
  struct D3DInteropParams : public CreateParams {
    InteropType type_;  //!< D3D resource type
    void* iDirect3D_;   //!< D3D resource interface object
    void* handle_;      //!< D3D resource handle
    uint mipLevel_;     //!< Texture mip level
    int layer_;         //!< Texture layer
    uint misc;          //!< miscellaneous cases
  };
#endif  // _WIN32

  //! Resource memory
  enum MemoryType {
    Empty = 0x0,         //!< resource is empty
    Local,               //!< resource in local memory
    Persistent,          //!< resource in persistent memory
    Remote,              //!< resource in nonlocal memory
    RemoteUSWC,          //!< resource in nonlocal memory
    Pinned,              //!< resource in pinned system memory
    View,                //!< resource is an alias
    OGLInterop,          //!< resource is an OGL memory object
    D3D10Interop,        //!< resource is a D3D10 memory object
    D3D11Interop,        //!< resource is a D3D11 memory object
    ImageView,           //!< resource is a view to some image
    ImageBuffer,         //!< resource is an image view of a buffer
    BusAddressable,      //!< resource is a bus addressable memory
    ExternalPhysical,    //!< resource is an external physical memory
    D3D9Interop,         //!< resource is a D3D9 memory object
    Scratch,             //!< resource is scratch memory
    Shader,              //!< resource is a shader
    P2PAccess,           //!< resource is a shared resource for P2P access
    VkInterop,           //!< resource is a Vulkan memory object
    VaRange,             //!< reousrce is a virtual address range
    IpcMemory,           //!< reousrce is a IPC memory object
    ImageExternalBuffer  //!< resource is an image view of an external buffer
  };

  //! Resource map flags
  enum MapFlags {
    NoOverwrite = 0x00000002,  //!< lock with no overwrite
    ReadOnly = 0x00000004,     //!< lock for read only operation
    WriteOnly = 0x00000008,    //!< lock for write only operation
    NoWait = 0x00000010,       //!< lock with no wait
  };

  //! Resource descriptor
  struct Descriptor : public amd::HeapObject {
    MemoryType type_;              //!< Memory type
    size_t width_;                 //!< Resource width
    size_t height_;                //!< Resource height
    size_t depth_;                 //!< Resource depth
    uint baseLevel_;               //!< The base level for the view
    uint mipLevels_;               //!< Number of mip levels
    uint flags_;                   //!< Resource flags, used in creation
    size_t pitch_;                 //!< Resource pitch, valid if locked
    size_t slice_;                 //!< Resource slice, valid if locked
    cl_image_format format_;       //!< CL image format
    cl_mem_object_type topology_;  //!< CL mem object type
    union {
      struct {
        uint dimSize_ : 2;           //!< Dimension size
        uint cardMemory_ : 1;        //!< PAL resource is in video memory
        uint imageArray_ : 1;        //!< PAL resource is an array of images
        uint buffer_ : 1;            //!< PAL resource is a buffer
        uint tiled_ : 1;             //!< PAL resource is tiled
        uint SVMRes_ : 1;            //!< SVM flag to the pal resource
        uint scratch_ : 1;           //!< Scratch buffer
        uint isAllocExecute_ : 1;    //!< SVM resource allocation attribute for shader\cmdbuf
        uint isDoppTexture_ : 1;     //!< PAL resource is for a DOPP desktop texture
        uint gl2CacheDisabled_ : 1;  //!< PAL resource is allocated with GPU L2 cache disabled.
        uint reserved_va_ : 1;       //!< PAL resource was allocated for a reserved VA
        uint interprocess_ : 1;      //!< PAL resource can be shared between processes
      };
      uint state_;
    };
  };

  //! Constructor of 1D Resource object
  Resource(const Device& gpuDev,  //!< GPU device object
           size_t size            //!< Resource size
  );

  //! Constructor of Image Resource object
  Resource(const Device& gpuDev,          //!< GPU device object
           size_t width,                  //!< resource width
           size_t height,                 //!< resource height
           size_t depth,                  //!< resource depth
           cl_image_format format,        //!< resource format
           cl_mem_object_type imageType,  //!< CL image type
           uint mipLevels = 1             //!< Number of mip levels
  );

  //! Destructor of the resource
  virtual ~Resource();

  /*! \brief Creates a PAL object, associated with the resource
   *
   *  \return True if we succesfully created a PAL resource
   */
  virtual bool create(MemoryType memType,        //!< memory type
                      CreateParams* params = 0,  //!< special parameters for resource allocation
                      bool forceLinear = false   //!< Forces linear tiling for images
  );

  /*! \brief Copies a subregion of memory from one resource to another
   *
   *  This is a general copy from anything to anything (as long as it fits).
   *  All positions and sizes are given in bytes. Note, however, that only
   *  a subset of this general interface is currently implemented.
   *
   *  \return true if successful
   */
  bool partialMemCopyTo(VirtualGPU& gpu,                //!< Virtual GPU device object
                        const amd::Coord3D& srcOrigin,  //!< Origin of the source region
                        const amd::Coord3D& dstOrigin,  //!< Origin of the destination region
                        const amd::Coord3D& size,       //!< Size of the region to copy
                        Resource& dstResource,          //!< Destination resource
                        bool enableRectCopy = false,    //!< Rectangular DMA support
                        bool flushDMA = false,          //!< Flush DMA if requested
                        uint bytesPerElement = 1        //!< Bytes Per Element
  ) const;

  /*! \brief Copies size/4 DWORD of memory to a surface
   *
   *  This is a raw copy to any surface using a CP packet.
   *  Size needs to be atleast a DWORD or multiple
   *
   */
  void writeRawData(VirtualGPU& gpu,   //!< Virtual GPU device object
                    size_t offset,     //!< Offset for in the buffer for data
                    size_t size,       //!< Size in bytes of data to be copied(multiple of DWORDS)
                    const void* data,  //!< Data to be copied
                    bool waitForEvent  //!< Wait for event complete
  ) const;

  //! Returns the offset in GPU memory for aliases
  size_t offset() const { return offset_; }

  //! Returns the GPU device that owns this resource
  const Device& dev() const { return gpuDevice_; }

  //! Returns the descriptor for resource
  const Descriptor& desc() const { return desc_; }

  //! Returns the PAL memory object
  Pal::IGpuMemory* iMem() const { return memRef_->iMem(); }

  //! Returns a pointer to the memory reference
  GpuMemoryReference* memRef() const { return memRef_; }

  //! Returns global memory offset
  uint64_t vmAddress() const { return iMem()->Desc().gpuVirtAddr + offset_; }

  //! Returns global memory offset
  uint64_t vmSize() const { return desc_.width_ * desc_.height_ * desc_.depth_ * elementSize(); }

  //! Returns global memory offset
  bool mipMapped() const { return (desc().mipLevels_ > 1) ? true : false; }

  //! Checks if persistent memory can have a direct map
  bool isPersistentDirectMap(bool writeMap = true) const;
  int getMapCount() const { return mapCount_; }

  /*! \brief Locks the resource and returns a physical pointer
   *
   *  \note This operation stalls HW pipeline!
   *
   *  \return Pointer to the physical memory
   */
  void* map(VirtualGPU* gpu,  //!< Virtual GPU device object
            uint flags = 0,   //!< flags for the map operation
            // Optimization for multilayer map/unmap
            uint startLayer = 0,  //!< Start layer for multilayer map
            uint numLayers = 0    //!< End layer for multilayer map
  );

  //! Unlocks the resource if it was locked
  void unmap(VirtualGPU* gpu  //!< Virtual GPU device object
  );

  //! Marks the resource as busy
  void setBusy(VirtualGPU& gpu,   //!< Virtual GPU device object
               GpuEvent calEvent  //!< PAL event
  ) const;

  //! Wait for the resource
  void wait(VirtualGPU& gpu,               //!< Virtual GPU device object
            bool waitOnBusyEngine = false  //!< Wait only if engine has changed
  ) const;

  //! Performs host write to the resource GPU memory
  bool hostWrite(VirtualGPU* gpu,             //!< Virtual GPU device object
                 const void* hostPtr,         //!< Host pointer to the SRC data
                 const amd::Coord3D& origin,  //!< Offsets for the update
                 const amd::Coord3D& size,    //!< The number of bytes to write
                 uint flags = 0,              //!< Map flags
                 size_t rowPitch = 0,         //!< Raw data row pitch
                 size_t slicePitch = 0        //!< Raw data slice pitch
  );

  //! Performs host read from the resource GPU memory
  bool hostRead(VirtualGPU* gpu,             //!< Virtual GPU device object
                void* hostPtr,               //!< Host pointer to the DST data
                const amd::Coord3D& origin,  //!< Offsets for the update
                const amd::Coord3D& size,    //!< The number of bytes to write
                size_t rowPitch = 0,         //!< Raw data row pitch
                size_t slicePitch = 0        //!< Raw data slice pitch
  );

  //! Gets the resource element size
  uint elementSize() const { return elementSize_; }

  //! Get the mapped address of this resource
  address data() const { return reinterpret_cast<address>(address_); }

  //! Frees all allocated PAL memories and resources, associated with this objects.
  //! Note: doesn't destroy the object itself
  void free();

  //! Return memory type
  MemoryType memoryType() const { return desc().type_; }

  //! Retunrs true if memory type matches specified
  bool isMemoryType(MemoryType memType) const;

  //! Returns TRUE if resource was allocated as CPU accessible and cacheable
  bool isCacheable() const { return (isMemoryType(Remote) || isMemoryType(Pinned)) ? true : false; }

  //! Returns TRUE if resource was allocated as CPU visible device memory
  bool IsPersistent() const { return isMemoryType(Persistent) ? true : false; }

  bool glAcquire();
  bool glRelease();

  //! Returns HW state for the resource (used for images only)
  const void* hwState() const { return hwState_; }

  //! Returns CPU HW SRD for the resource (used for images only)
  uint64_t hwSrd() const { return hwSrd_; }

  //! Returns the number of components in the image format
  uint numComponents() const {
    return Pal::Formats::NumComponents(image_->GetImageCreateInfo().swizzledFormat.format);
  }

  //! Adds GPU event, associated with this resource
  void addGpuEvent(const VirtualGPU& gpu, GpuEvent event) const;

  //! Returns GPU event associated with this resource and specified queue
  GpuEvent* getGpuEvent(const VirtualGPU& gpu) const;

  //! Resizes the events array to account the new queue
  void resizeGpuEvents(uint index) { events_.resize(index + 1); }

  //! Erase an entry in the array for provided queue index
  void eraseGpuEvents(uint index) { events_.erase(events_.begin() + index); }

  //! Quick view update for managed buffers. It should avoid expensive object allocations
  //! If the base resource is null, then the view is released
  void updateView(Resource* base, size_t offset, size_t size) {
    if (base == nullptr) {
      desc_.type_ = Empty;
      memRef_->release();
      memRef_ = nullptr;
      viewOwner_ = nullptr;
    } else {
      desc_.type_ = View;
      viewOwner_ = base;
      offset_ = offset + viewOwner_->offset();
      assert(viewOwner_->data() != nullptr && "CPU access must be provide for this call!");
      address_ = viewOwner_->data() + offset;
      desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
      memRef_ = viewOwner_->memRef_;
      memRef_->retain();
      desc_.width_ = amd::alignUp(size, Pal::Formats::BytesPerPixel(Pal::ChNumFormat::X32_Uint)) /
                     Pal::Formats::BytesPerPixel(Pal::ChNumFormat::X32_Uint);
      setBusy(*memRef()->gpu_, GpuEvent::InvalidID);
    }
  }

  //! Update the modified field of the event, meaning the resource was updated
  void setModified(VirtualGPU& gpu, bool modified) const;

  //! Update the modified field of the event, meaning the resource was updated
  bool isModified(VirtualGPU& gpu) const;

  /*! \brief Creates a PAL P2P object, associated with the resource
   *
   *  \return True if we succesfully created a PAL P2P resource
   */
  bool CreateP2PAccess(CreateParams* params  //!< special parameters for resource allocation
  );

 protected:
  /*! \brief Creates a PAL memory object, from IPC handle
   *
   *  \return True if we succesfully created a PAL resource
   */
  bool CreateIpc(CreateParams* params);

  /*! \brief Creates a PAL iamge object, associated with the resource
   *
   *  \return True if we succesfully created a PAL resource
   */
  bool CreateImage(CreateParams* params,     //!< special parameters for resource allocation
                   bool forceLinear = false  //!< forces linear tiling for images
  );

  /*! \brief Creates a PAL interop object, associated with the resource
   *
   *  \return True if we succesfully created a PAL interop resource
   */
  bool CreateInterop(CreateParams* params  //!< special parameters for resource allocation
  );

  /*! \brief Creates a PAL pinned object, associated with the resource
   *
   *  \return True if we succesfully created a PAL pinned resource
   */
  bool CreatePinned(CreateParams* params  //!< special parameters for resource allocation
  );

  /*! \brief Creates a PAL SVM object, associated with the resource
   *
   *  \return True if we succesfully created a PAL SVM resource
   */
  bool CreateSvm(CreateParams* params,  //!< special parameters for resource allocation
                 Pal::gpusize svmPtr);

  uint elementSize_;  //!< Size of a single element in bytes

  //! Returns PAL image object
  Pal::IImage* image() const { return image_; }

 private:
  //! Disable copy constructor
  Resource(const Resource&);

  //! Disable operator=
  Resource& operator=(const Resource&);

  //! Calls PAL to map a resource
  void* gpuMemoryMap(size_t* pitch,             //!< Pitch value for the image
                     uint flags,                //!< Map flags
                     Pal::IGpuMemory* resource  //!< PAL memory object
  ) const;

  //! Uses PAL to unmap a resource
  void gpuMemoryUnmap(Pal::IGpuMemory* resource  //!< PAL memory object
  ) const;

  //! Fress all PAL resources associated with OCL resource
  void palFree() const;

  //! Converts Resource memory type to the PAL heaps
  void memTypeToHeap(Pal::GpuMemoryCreateInfo* createInfo  //!< Memory create info
  );

  const Device& gpuDevice_;     //!< GPU device
  Descriptor desc_;             //!< Descriptor for this resource
  std::atomic<int> mapCount_;   //!< Total number of maps
  void* address_;               //!< Physical address of this resource
  size_t offset_;               //!< Resource offset
  GpuMemoryReference* memRef_;  //!< PAL resource reference
  Pal::gpusize subOffset_;      //!< GPU memory offset in the oririnal resource
  const Resource* viewOwner_;   //!< GPU resource, which owns this view
  void* glInteropMbRes_;        //!< Mb Res handle
  uint32_t glType_;             //!< GL interop type
  void* glPlatformContext_;

  // Optimization for multilayer map/unmap
  uint startLayer_;  //!< Start layer for map/unmapLayer
  uint numLayers_;   //!< Number of layers for map/unmapLayer
  uint mapFlags_;    //!< Map flags for map/umapLayer

  Pal::IImage* image_;  //!< PAL image object

  uint32_t* hwState_;  //!< HW state for image object
  uint64_t hwSrd_;     //!< GPU pointer to HW SRD

  //! Note: Access to the events are thread safe.
  mutable std::vector<GpuEvent> events_;  //!< GPU events associated with the resource
};

typedef Util::BuddyAllocator<Device> MemBuddyAllocator;

class MemorySubAllocator : public amd::HeapObject {
 public:
  MemorySubAllocator(Device* device, bool retain_final_chunk = false)
      : device_(device), retain_final_chunk_(retain_final_chunk) {}

  ~MemorySubAllocator();

  //! Create suballocation
  GpuMemoryReference* Allocate(Pal::gpusize size, Pal::gpusize alignment,
                               const Pal::IGpuMemory* reserved_va, Pal::gpusize* offset);
  //! Free suballocation
  bool Free(amd::Monitor* monitor, GpuMemoryReference* mem_ref, Pal::gpusize offset);

 protected:
  //! Allocate new chunk of memory
  virtual bool CreateChunk(const Pal::IGpuMemory* reserved_va);
  bool InitAllocator(GpuMemoryReference* mem_ref);
  void forceResident(GpuMemoryReference* mem_ref);

  Device* device_;
  std::unordered_map<GpuMemoryReference*, MemBuddyAllocator*> heaps_;
  bool retain_final_chunk_;
};

class CoarseMemorySubAllocator : public MemorySubAllocator {
 public:
  CoarseMemorySubAllocator(Device* device) : MemorySubAllocator(device) {}

  bool CreateChunk(const Pal::IGpuMemory* reservedVa) override;
};

class FineMemorySubAllocator : public MemorySubAllocator {
 public:
  FineMemorySubAllocator(Device* device)
      : MemorySubAllocator(device, true /*retain_final_chunk*/) {}

  bool CreateChunk(const Pal::IGpuMemory* reserved_va) override;
};

class FineUncachedMemorySubAllocator : public MemorySubAllocator {
 public:
  FineUncachedMemorySubAllocator(Device* device) : MemorySubAllocator(device) {}
  bool CreateChunk(const Pal::IGpuMemory* reserved_va) override;
};

class ResourceCache : public amd::HeapObject {
 public:
  //! Default constructor
  ResourceCache(Device* device, size_t cacheSizeLimit)
      : lockCacheOps_(true), /* PAL resource cache */
        cacheSize_(0),
        lclCacheSize_(0),
        persistentCacheSize_(0),
        cacheSizeLimit_(cacheSizeLimit),
        mem_sub_alloc_local_(device),
        mem_sub_alloc_coarse_(device),
        mem_sub_alloc_fine_(device),
        mem_sub_alloc_fine_uncached_(device) {}

  //! Default destructor
  ~ResourceCache();

  //! Adds a PAL resource to the cache
  bool addGpuMemory(Resource::Descriptor* desc,  //!< Resource descriptor - cache key
                    GpuMemoryReference* ref,     //!< Resource reference
                    Pal::gpusize offset          //!< Original resource offset
  );

  //! Finds a PAL resource from the cache
  GpuMemoryReference* findGpuMemory(
      Resource::Descriptor* desc,  //!< Resource descriptor - cache key
      Pal::gpusize size, Pal::gpusize alignment,
      const Pal::IGpuMemory* reserved_va,  //!< Reserved VA for SVM suballocations
      Pal::gpusize* offset);

  //! Destroys cache
  //! Returns true if cache was freed and false if cache is already empty.
  bool free(size_t minCacheEntries = 0);

  //! Returns the size of all memory, stored in the cache
  size_t cacheSize() const { return cacheSize_; }

  //! Returns the size of local memory, stored in the cache
  size_t lclCacheSize() const { return lclCacheSize_; }

  //! Returns the size of persistent memory, stored in the cache
  size_t persistentCacheSize() const { return persistentCacheSize_; }

 private:
  //! Disable copy constructor
  ResourceCache(const ResourceCache&);

  //! Disable operator=
  ResourceCache& operator=(const ResourceCache&);

  //! Removes one last entry from the cache
  void removeLast();

  amd::Monitor lockCacheOps_;  //!< Lock to serialise cache access

  size_t cacheSize_;             //!< Current cache size in bytes
  size_t lclCacheSize_;          //!< Local memory stored in the cache
  size_t persistentCacheSize_;   //!< Persistent memory stored in the cache
  const size_t cacheSizeLimit_;  //!< Cache size limit in bytes

  //! PAL resource cache
  std::list<std::pair<Resource::Descriptor*, GpuMemoryReference*> > resCache_;

  MemorySubAllocator mem_sub_alloc_local_;         //!< Allocator for suballocations in Local
  CoarseMemorySubAllocator mem_sub_alloc_coarse_;  //!< Allocator for suballocations in Coarse SVM
  FineMemorySubAllocator mem_sub_alloc_fine_;      //!< Allocator for suballocations in Fine SVM
  FineUncachedMemorySubAllocator
      mem_sub_alloc_fine_uncached_;  //!< Allocator for suballocations in Fine uncached SVM
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
