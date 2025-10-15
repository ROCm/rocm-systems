/* Copyright (c) 2010 - 2025 Advanced Micro Devices, Inc.

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
#include "utils/flags.hpp"
#include "thread/monitor.hpp"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/interop.hpp"
#include "device/device.hpp"

#include <atomic>
#include <utility>
#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <memory>
#include <limits>
#define CL_MEM_FOLLOW_USER_NUMA_POLICY (1u << 31)
#define ROCCLR_MEM_HSA_SIGNAL_MEMORY (1u << 30)
#define ROCCLR_MEM_INTERNAL_MEMORY (1u << 29)
#define CL_MEM_VA_RANGE_AMD (1u << 28)
#define ROCCLR_MEM_HSA_UNCACHED (1u << 27)
#define ROCCLR_MEM_INTERPROCESS (1u << 26)
#define ROCCLR_MEM_PHYMEM (1u << 25)
#define ROCCLR_MEM_HSA_CONTIGUOUS (1u << 24)

namespace amd::device {
class Memory;
class VirtualDevice;
}  // namespace amd::device

namespace amd {

// Forward declaration of the amd::Image and amd::Buffer classes.
class Image;
class Buffer;
class Pipe;

struct BufferRect : public amd::EmbeddedObject {
  //! Default constructor
  BufferRect() : rowPitch_(0), slicePitch_(0), start_(0), end_(0) {}

  //! Creates BufferRect object
  bool create(const size_t* bufferOrigin,  //!< Start locaiton in the buffer
              const size_t* region,        //!< Copy region
              size_t bufferRowPitch,       //!< Provided buffer's row pitch
              size_t bufferSlicePitch      //!< Provided buffer's slice pitch
  );

  //! Returns the plain offset for the (X, Y, Z) location
  size_t offset(size_t x,  //!< Coordinate in X dimension
                size_t y,  //!< Coordinate in Y dimension
                size_t z   //!< Coordinate in Z dimension
  ) const {
    return start_ + x + y * rowPitch_ + z * slicePitch_;
  }

  size_t rowPitch_;    //!< Calculated row pitch for the buffer rect
  size_t slicePitch_;  //!< Calculated slice pitch for the buffer rect
  size_t start_;       //!< Start offset for the copy region
  size_t end_;         //!< Relative end offset from start for the copy region
};

class HostMemoryReference {
 public:
  //! Default constructor
  HostMemoryReference(void* hostMem = NULL) : alloced_(false), hostMem_(hostMem), size_(0) {}

  //! Default destructor
  ~HostMemoryReference() { assert(!alloced_ && "Host buffer not deallocated"); }

  //! Creates host memory reference object
  bool allocateMemory(size_t size, const Context& context);

  // Frees system memory if it was allocated
  void deallocateMemory(const Context& context);

  //! Get the host memory pointer
  void* hostMem() const { return hostMem_; }

  //! Get the host memory size
  size_t size() const { return size_; }

  //! Set the host memory pointer
  void setHostMem(void* hostMem, const Context& context) {
    deallocateMemory(context);
    hostMem_ = hostMem;
  }

  //! Returns true if the host memory has been allocated by this object, false
  // if it has been allocated elsewhere.
  bool alloced() const { return alloced_; }

 private:
  //! Disable copy constructor
  HostMemoryReference(const HostMemoryReference&);

  //! Disable operator=
  HostMemoryReference& operator=(const HostMemoryReference&);

  bool alloced_;   //!< TRUE if memory was allocated
  void* hostMem_;  //!< Host memory pointer
  size_t size_;    //!< The host memory size
};

class Memory : public amd::RuntimeObject {
  typedef void(CL_CALLBACK* DestructorCallBackFunction)(cl_mem memobj, void* user_data);

  enum AllocState { AllocInit = 0, AllocCreate = 1, AllocComplete = 2, AllocRealloced = 3 };

  struct DestructorCallBackEntry {
    struct DestructorCallBackEntry* next_;

    DestructorCallBackFunction callback_;
    void* data_;

    DestructorCallBackEntry(DestructorCallBackFunction callback, void* data)
        : callback_(callback), data_(data) {}
  };

 public:
  enum MemoryType {
    kSvmMemoryPtr = 0x1,
    kArenaMemoryPtr = 0x100,
  };

  struct UserData {
    int deviceId = 0;  //!< Device ID memory is allocated on
    int locationType =
        0;  //!< The type of the location (i.e. device or host) memory is allocated on
    void* data = nullptr;                  //!< Opaque user data from CL or HIP or etc.
    amd::Memory* phys_mem_obj = nullptr;   //<! Physical mem obj, only set on virtual mem
    amd::Memory* vaddr_mem_obj = nullptr;  //<! Virtual address mem obj, only set on virtual mem
    uint64_t hsa_handle = 0;               //!< Opaque hsa handle saved for Virtual memories
    unsigned int flags = 0;                //!< HIP memory flags
    //! hipMallocPitch allocates buffer using width & height and returns pitch & device pointer.
    //! Since device pointer is void*, It looses the values of width & height used for allocation.
    //! Memory object has total size however it's pitch * height * depth and its not straight
    //! forward to deduce these numbers from size and thus, below metadata are added to memory class
    //! so that can be used to validate in hipMemset2D's inputs.
    size_t pitch_ = 0;   //!< Pitch value in bytes
    size_t width_ = 0;   //!< Width value in bytes
    size_t height_ = 0;  //!< Height value
    size_t depth_ = 0;   //!< Depth value

    bool sync_mem_ops_ = false;  //!< Memops sync, when set synchronize all mem operations.
  };

 protected:
  typedef cl_mem_object_type Type;
  typedef cl_mem_flags Flags;
  typedef DeviceMap<const Device*, device::Memory*> DeviceMemory;

  //! Returns the number of devices this memory object is associated, including P2P access
  uint32_t NumDevicesWithP2P();

  size_t numDevices_;  //!< Number of devices

  //! The device memory objects included in this memory
  DeviceMemory* deviceMemories_;

  //! The device alloced state
  std::unordered_map<const Device*, std::atomic<AllocState>> deviceAlloced_;

  //! Linked list of destructor callbacks.
  std::atomic<DestructorCallBackEntry*> destructorCallbacks_;

  SharedReference<Context> context_;  //!< Owning context
  Memory* parent_;
  const Type type_;                 //!< Object type (Buffer, Image2D, Image3D)
  HostMemoryReference hostMemRef_;  //!< Host-side memory reference(or NULL if none)
  size_t origin_;
  size_t size_;                  //!< Size in bytes
  Flags flags_;                  //!< Construction flags
  size_t version_;               //!< Update count, used for coherency
  const Device* lastWriter_;     //!< Which device wrote most recently (NULL if host)
  InteropObject* interopObj_;    //!< Interop object
  device::VirtualDevice* vDev_;  //!< Memory object belongs to a virtual device only
  std::atomic_uint mapCount_;    //!< Keep track of number of mappings for a memory object
  void* svmHostAddress_;         //!< svm host address;
  size_t resOffset_;             //!< resource offset
  union {
    struct {
      uint32_t isParent_ : 1;          //!< This object is a parent
      uint32_t forceSysMemAlloc_ : 1;  //!< Forces system memory allocation
      uint32_t svmPtrCommited_ : 1;    //!< svm host address committed flag
      uint32_t canBeCached_ : 1;       //!< flag to if the object can be cached
      uint32_t p2pAccess_ : 1;         //!< Memory object allows P2P access
      uint32_t ipcShared_ : 1;         //!< Memory shared between processes
      uint32_t largeBarSystem_ : 1;    //!< VRAM is visiable for host
      uint32_t image_view_ : 1;        //!< Memory object is an image view
    };
    uint32_t flagsEx_;
  };
  //! unique buffer id for each memory allocation
  uint32_t uniqueId_ = 0;
  //! used to save the user data during memory allocation.
  UserData userData_;

 private:
  //! Disable default assignment operator
  Memory& operator=(const Memory&);

  //! Disable default copy operator
  Memory(const Memory&);

  Monitor lockMemoryOps_;         //!< Lock to serialize memory operations
  std::set<Memory*> subBuffers_;  //!< List of all subbuffers for this memory object
  device::Memory* svmBase_;       //!< svmBase allocation for MGPU case
  size_t alignment_ = 0;          //!< alignment for allocation address

 protected:
  //! The constructor creates a memory object but does not allocate either host memory
  //! or device memory. Default parameters are appropriate for Buffer creation.
  Memory(Context& context,     //!< Context object
         Type type,            //!< Memory type
         Flags flags,          //!< Object's flags
         size_t size,          //!< Memory size
         void* svmPtr = NULL,  //!< svm host memory address, NULL if no SVM mem object
         size_t alignment = 0  //!< allocation addr alignment
  );
  Memory(Memory& parent,             //!< Parent Mem obj
         Flags flags,                //!< Object's flags
         size_t offset,              //!< Memory offset
         size_t size,                //!< Memory size
         Type type = 0,              //!< Memory type
         Context* context = nullptr  //!< Input context
  );

  //! Memory object destructor
  virtual ~Memory();

  //! Copies initialization data to the backing store
  virtual void copyToBackingStore(void* initFrom  //!< Pointer to the initialization memory
  );

  //! Initializes the device memory array
  virtual void initDeviceMemory();

  void setSize(size_t size) { size_ = size; }
  void setInteropObj(InteropObject* obj) { interopObj_ = obj; }

  void resetAllocationState();

 public:
  //! Placement new operator.
  void* operator new(size_t size,            //!< Original allocation size
                     const Context& context  //!< Context this memory object is allocated in.
  );
  // Provide a "matching" placement delete operator.
  void operator delete(void*,                  //!< Pointer to deallocate
                       const Context& context  //!< Context this memory object is allocated in.
  );
  // and a regular delete operator to satisfy synthesized methods.
  void operator delete(void*  //!< Pointer to deallocate
  );

  //! Returns the memory lock object
  amd::Monitor& lockMemoryOps() { return lockMemoryOps_; }

  //! Adds a view into the list
  void addSubBuffer(Memory* item);

  //! virtual function used to distinguish memory objects from other CL objects
  virtual ObjectType objectType() const { return ObjectTypeMemory; }

  //! Removes a subbuffer from the list
  void removeSubBuffer(Memory* item);

  //! Returns the list of all subbuffers
  std::set<Memory*>& subBuffers() { return subBuffers_; }

  //! Returns the number of devices
  size_t numDevices() const { return numDevices_; }

  //! static_cast to Buffer with sanity check
  virtual Buffer* asBuffer() { return NULL; }
  //! static_cast to Image with sanity check
  virtual Image* asImage() { return NULL; }
  //! static_cast to Pipe with sanity check
  virtual Pipe* asPipe() { return NULL; }

  //! Creates and initializes device (cache) memory for all devices
  virtual bool create(void* initFrom = NULL,     //!< Pointer to the initialization data
                      bool sysMemAlloc = false,  //!< Allocate device memory in system memory
                      bool skipAlloc = false,    //!< Skip device memory allocation
                      bool forceAlloc = false    //!< Force device memory allocation
  );

  //! Allocates device (cache) memory for a specific device
  bool addDeviceMemory(const Device* dev  //!< Device object
  );

  //! Replaces device (cache) memory for a specific device
  void replaceDeviceMemory(const Device* dev,  //!< Device object
                           device::Memory* dm  //!< New device memory object for replacement
  );

  //! Find the section for the given device. Return NULL if not found.
  device::Memory* getDeviceMemory(const Device& dev,  //!< Device object
                                  bool alloc = true   //!< Allocates memory
  );

  //! Get origianl device memory
  device::Memory* getOriginalDeviceMemory() const { return deviceMemories_[0].value_; };

  //! Allocate host memory (as required)
  bool allocHostMemory(void* initFrom,         //!< Host memory provided by the application
                       bool allocHostMem,      //!< Force system memory allocation
                       bool forceCopy = false  //!< Force system memory allocation
  );

  // Accessors
  Memory* parent() const { return parent_; }
  void SetParent(amd::Memory* parent) {
    parent_ = parent;
    if (parent != nullptr) {
      parent_->isParent_ = true;
      parent_->retain();
    }
  }
  bool isParent() const { return isParent_; }
  bool ImageView() const { return image_view_; }

  size_t getOrigin() const { return origin_; }
  void setOrigin(size_t origin) { origin_ = origin; }
  size_t getSize() const { return size_; }
  Flags getMemFlags() const { return flags_; }
  Type getType() const { return type_; }

  const Device* getLastWriter() { return lastWriter_; }
  const HostMemoryReference* getHostMemRef() const { return &hostMemRef_; }
  void* getHostMem() const { return hostMemRef_.hostMem(); }
  void setHostMem(void* mem) { hostMemRef_.setHostMem(mem, context_()); }

  size_t getVersion() const { return version_; }

  Context& getContext() const { return context_(); }
  bool isInterop() const { return (getInteropObj() != NULL) ? true : false; }

  InteropObject* getInteropObj() const { return interopObj_; }

  bool setDestructorCallback(DestructorCallBackFunction callback, void* data);

  //! Signal that a write has occurred to a cached version
  void signalWrite(const Device* writer);
  //! Force an asynchronous writeback from the most-recent dirty cache to host
  void cacheWriteBack(device::VirtualDevice* vDev);

  //! Returns true if the specified area covers memory intirely
  virtual bool isEntirelyCovered(const Coord3D& origin,  //!< Origin location of the covered region
                                 const Coord3D& region   //!< Covered region dimensions
  ) const = 0;

  //! Returns true if the specified area is not degenerate and is inside of allocated memory
  virtual bool validateRegion(const Coord3D& origin,  //!< Origin location of the covered region
                              const Coord3D& region   //!< Covered region dimensions
  ) const = 0;

  void setVirtualDevice(device::VirtualDevice* vDev) { vDev_ = vDev; }
  device::VirtualDevice* getVirtualDevice() const { return vDev_; }
  bool forceSysMemAlloc() const { return forceSysMemAlloc_; }

  void incMapCount() { ++mapCount_; }
  void decMapCount() { --mapCount_; }
  uint mapCount() const { return mapCount_; }

  bool usesSvmPointer() const;

  void* getSvmPtr() const { return svmHostAddress_; }   //!< svm pointer accessor;
  void setSvmPtr(void* ptr) { svmHostAddress_ = ptr; }  //!< svm pointer setter;

  size_t getOffset() const { return resOffset_; }         //!< resource offset accessor;
  void setOffset(size_t offset) { resOffset_ = offset; }  //!< resource offset setter;

  bool isSvmPtrCommited() const {
    return svmPtrCommited_;
  }  //!< svm host address committed accessor;
  void commitSvmMemory();  //!< svm host address committed accessor;
  void uncommitSvmMemory();
  void setCacheStatus(bool canBeCached) {
    canBeCached_ = canBeCached;
  }  //!< set the memobject cached status
  bool canBeCached() const { return canBeCached_; }  //!< get the memobject cached status

  //! Check if this objects allows P2P access
  bool P2PAccess() const { return p2pAccess_; }

  // Set ipcShared status
  void setIpcShared(bool ipcShared) { ipcShared_ = ipcShared; }
  //! Check if this object allows IPC
  bool ipcShared() const { return ipcShared_; }

  //! Returns the base device memory object for possible P2P access
  device::Memory* BaseP2PMemory() const { return deviceMemories_[0].value_; }
  device::Memory* svmBase() const { return svmBase_; }  //!< Returns SVM base for MGPU case

  uint32_t getUniqueId() { return uniqueId_; }
  //! save the user data during memory allocation
  UserData& getUserData() { return userData_; }

  //! find if memory object is Arena memory
  virtual bool isArena() { return false; }

  //! get device by id when glb ctx is used.
  Device* GetDeviceById();

  //! Validate memory access for vmm memory
  bool ValidateMemAccess(const Device& dev, bool read_write);

  //! Get alignment_
  size_t getAlignment() const { return alignment_; }
};

//! Buffers are a specialization of memory. Just a wrapper, really,
//! but this gives us flexibility for later changes.

class Buffer : public Memory {
 protected:
  cl_bus_address_amd busAddress_;

  //! Initializes the device memory array which is nested
  // after'Image1DD3D10' object in memory layout.
  virtual void initDeviceMemory();

  Buffer(Context& context, Type type, Flags flags, size_t size)
      : Memory(context, type, flags, size) {}

 public:
  Buffer(Context& context, Flags flags, size_t size, void* svmPtr = NULL, size_t alignment = 0)
      : Memory(context, CL_MEM_OBJECT_BUFFER, flags, size, svmPtr, alignment) {}
  Buffer(Memory& parent, Flags flags, size_t origin, size_t size, Context* context = nullptr)
      : Memory(parent, flags, origin, size, 0, context) {}

  bool create(void* initFrom = NULL,     //!< Pointer to the initialization data
              bool sysMemAlloc = false,  //!< Allocate device memory in system memory
              bool skipAlloc = false,    //!< Skip device memory allocation
              bool forceAlloc = false    //!< Force device memory allocation
  );

  //! static_cast to Buffer with sanity check
  virtual Buffer* asBuffer() { return this; }

  //! Returns true if the specified area covers buffer entirely
  bool isEntirelyCovered(const Coord3D& origin,  //!< Origin location of the covered region
                         const Coord3D& region   //!< Covered region dimensions
  ) const;

  //! Returns true if the specified area is not degenerate and is inside of allocated memory
  bool validateRegion(const Coord3D& origin,  //!< Origin location of the covered region
                      const Coord3D& region   //!< Covered region dimensions
  ) const;

  cl_bus_address_amd busAddress() const { return busAddress_; }
};

//! Pipes are a specialization of Buffers.
class Pipe : public Buffer {
 protected:
  size_t packetSize_;  //!< Size in bytes of pipe packet
  size_t maxPackets_;  //!< Number of max pipe packets
  bool initialized_;   //!< Mark if the pipe is initialized

  virtual void initDeviceMemory();

 public:
  Pipe(Context& context, Flags flags, size_t size, size_t pipe_packet_size, size_t pipe_max_packets)
      : Buffer(context, CL_MEM_OBJECT_PIPE, flags, size), initialized_(false) {
    packetSize_ = pipe_packet_size;
    maxPackets_ = pipe_max_packets;
  }

  //! static_cast to Pipe with sanity check
  virtual Pipe* asPipe() { return this; }

  //! Returns pipe size pitch in bytes
  size_t getPacketSize() const { return packetSize_; }

  //! return max number of pipe packets
  size_t getMaxNumPackets() const { return maxPackets_; }
};

//! Images are a specialization of memory
class Image : public Memory {
 public:
  // declaration of list of supported formats
  static const cl_image_format supportedFormats[];
  static const cl_image_format supportedFormatsRA[];
  static const cl_image_format supportedDepthStencilFormats[];
  static uint32_t numSupportedFormats(const Context& context, cl_mem_object_type image_type,
                                      cl_mem_flags flags = 0);
  static uint32_t getSupportedFormats(const Context& context, cl_mem_object_type image_type,
                                      const uint32_t num_entries, cl_image_format* image_formats,
                                      cl_mem_flags flags = 0);

  //! Helper struct to manipulate image formats.
  struct Format : public cl_image_format {
    //! Construct a new ImageFormat wrapper.
    Format(const cl_image_format& format) {
      image_channel_order = format.image_channel_order;
      image_channel_data_type = format.image_channel_data_type;
    }

    //! Return true if this is a valid image format, false otherwise.
    bool isValid() const;

    //! Returns true if this format is supported by runtime, false otherwise
    bool isSupported(const Context& context, cl_mem_object_type image_type = 0,
                     cl_mem_flags flags = 0) const;

    //! Compare 2 image formats.
    bool operator==(const Format& rhs) const {
      return image_channel_order == rhs.image_channel_order &&
             image_channel_data_type == rhs.image_channel_data_type;
    }
    bool operator!=(const Format& rhs) const { return !(*this == rhs); }

    //! Return the number of channels.
    size_t getNumChannels() const;

    //! Return the element size in bytes.
    size_t getElementSize() const;

    //! Get the channel order by indices. R = 0, G = 1, B = 2, A = 3.
    void getChannelOrder(uint8_t* channelOrder) const;

    //! Adjust colorRGBA according to format, and set it in colorFormat.
    void formatColor(const void* colorRGBA, void* colorFormat) const;
  };

  struct Impl {
    amd::Coord3D region_;  //!< Image size
    size_t rp_;            //!< Image row pitch
    size_t sp_;            //!< Image slice pitch
    const Format format_;  //!< Image format
    void* reserved_;
    size_t bp_;

    Impl(const Format& format, Coord3D region, size_t rp, size_t sp = 0, size_t bp = 0)
        : region_(region), rp_(rp), sp_(sp), format_(format), bp_(bp) {
      DEBUG_ONLY(reserved_ = NULL);
    }
  };

 private:
  Impl impl_;          //!< Image object description
  size_t dim_;         //!< Image dimension
  uint mipLevels_;     //!< The number of mip levels
  uint baseMipLevel_;  //!< The base mip level for a view

 protected:
  Image(const Format& format, Image& parent, uint baseMipLevel = 0, cl_mem_flags flags = 0,
        bool isMipmapView = false);

  ///! Initializes the device memory array which is nested
  // after'Image' object in memory layout.
  virtual void initDeviceMemory();

  //! Copies initialization data to the backing store
  virtual void copyToBackingStore(void* initFrom  //!< Pointer to the initialization memory
  );

  void initDimension();

 public:
  Image(Context& context, Type type, Flags flags, const Format& format, size_t width, size_t height,
        size_t depth, size_t rowPitch, size_t slicePitch, uint mipLevels = 1);

  Image(Buffer& buffer, Type type, Flags flags, const Format& format, size_t width, size_t height,
        size_t depth, size_t rowPitch, size_t slicePitch, uint mipLevels = 1, size_t offset = 0);

  //! Validate image dimensions with supported sizes
  static bool validateDimensions(
      const std::vector<amd::Device*>& devices,  //!< List of devices for validation
      cl_mem_object_type type,                   //!< Image type
      size_t width,                              //!< Image width
      size_t height,                             //!< Image height
      size_t depth,                              //!< Image depth
      size_t arraySize                           //!< Image array size
  );

  const Format& getImageFormat() const { return impl_.format_; }

  //! static_cast to Buffer with sanity check
  virtual Image* asImage() { return this; }

  //! Returns true if specified area covers image entirely
  bool isEntirelyCovered(const Coord3D& origin,  //!< Origin location of the covered region
                         const Coord3D& region   //!< Covered region dimensions
  ) const;

  //! Returns true if the specified area is not degenerate and is inside of allocated memory
  bool validateRegion(const Coord3D& origin,  //!< Origin location of the covered region
                      const Coord3D& region   //!< Covered region dimensions
  ) const;

  //! Returns true if the slice value for the image is valid
  bool isRowSliceValid(size_t rowPitch,    //!< The row pitch value
                       size_t slicePitch,  //!< The slice pitch value
                       size_t width,       //!< The width of the copy region
                       size_t height       //!< The height of the copy region
  ) const;

  //! Creates a view memory object
  virtual Image* createView(
      const Context& context,         //!< Context for a view creation
      const Format& format,           //!< The new format for a view
      device::VirtualDevice* vDev,    //!< Virtual device object
      uint baseMipLevel = 0,          //!< Base mip level for a view
      cl_mem_flags flags = 0,         //!< Memory allocation flags
      bool createMipmapView = false,  //!< To create mipmap view based on this image
      bool forceAlloc = false         //!< To bypass deffered alloc
  );

  //! Returns the impl for this image.
  Impl& getImpl() { return impl_; }

  //! Returns the number of dimensions.
  size_t getDims() const { return dim_; }

  //! Base virtual methods to be overridden in derived image classes
  //!
  //! Returns width of image in pixels
  size_t getWidth() const { return impl_.region_[0]; }

  //! Returns height of image in pixels
  size_t getHeight() const { return impl_.region_[1]; }

  //! Returns image's row pitch in bytes
  size_t getRowPitch() const { return impl_.rp_; }

  //! Returns image's byte pitch
  size_t getBytePitch() const { return impl_.bp_; }

  //! Returns depth of the image in pixels/slices
  size_t getDepth() const { return impl_.region_[2]; }

  //! Returns image's slice pitch in bytes
  size_t getSlicePitch() const { return impl_.sp_; }

  //! Returns image's slice pitch in bytes
  uint getMipLevels() const { return mipLevels_; }

  //! Returns image's slice pitch in bytes
  uint getBaseMipLevel() const { return baseMipLevel_; }

  //! Get the image covered region
  const Coord3D& getRegion() const { return impl_.region_; }

  //! Sets the byte pitch obtained from HWL
  void setBytePitch(size_t bytePitch) { impl_.bp_ = bytePitch; }
};

//! SVM-related functionality.
class SvmBuffer : AllStatic {
 public:
  //! Allocate a shared buffer that is accessible by all devices in the context
  static void* malloc(Context& context, cl_svm_mem_flags flags, size_t size, size_t alignment,
                      const amd::Device* curDev = nullptr, void* hostptr = nullptr);
  //! Release shared buffer
  static void free(const Context& context, void* ptr);

  //! Fill the destination buffer \a dst with the contents of the source
  //! buffer \a src \times times.
  static void memFill(void* dst, const void* src, size_t srcSize, size_t times);

  //! Return true if \a ptr is a pointer allocated using SvmBuffer::malloc
  //! that has not been deallocated afterwards
  static bool malloced(const void* ptr);

 private:
  static void Add(uintptr_t k, uintptr_t v);
  static void Remove(uintptr_t k);
  static bool Contains(uintptr_t ptr);

  static std::map<uintptr_t, uintptr_t> Allocated_;  // !< Allocated buffers
  static Monitor AllocatedLock_;
};

class ArenaMemory : public Buffer {
 public:
  ArenaMemory(Context& context)
      : Buffer(context, 0, std::numeric_limits<size_t>::max(),
               reinterpret_cast<void*>(kArenaMemoryPtr)) {}
  bool isArena() { return true; }
};

class IpcBuffer : public Buffer {
 public:
  IpcBuffer(Context& context, Flags flags, size_t offset, size_t size, const void* handle)
      : Buffer(context, flags, size), handle_(handle) {
    setIpcShared(true);
    setOffset(offset);
  }

  virtual void initDeviceMemory();

  const void* Handle() const { return handle_; }

 private:
  const void* handle_;  //!< Ipc handle, associated with this memory object
};

}  // namespace amd
