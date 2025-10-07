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

#ifndef DEVICE_HPP_
#define DEVICE_HPP_

#include "top.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "utils/util.hpp"
#include "amdocl/cl_kernel.h"
#include "elf/elf.hpp"
#include "appprofile.hpp"
#include "devprogram.hpp"
#include "devkernel.hpp"
#include "amdocl/cl_profile_amd.h"
#if defined(WITH_COMPILER_LIB)
#include "hsailctx.hpp"
#endif
#include "devsignal.hpp"

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "devurilocator.hpp"
#endif
#endif

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <list>
#include <set>
#include <unordered_set>
#include <utility>
#include <shared_mutex>

namespace amd {
class Command;
class CommandQueue;
class ReadMemoryCommand;
class WriteMemoryCommand;
class FillMemoryCommand;
class CopyMemoryCommand;
class CopyMemoryP2PCommand;
class MapMemoryCommand;
class UnmapMemoryCommand;
class MigrateMemObjectsCommand;
class NDRangeKernelCommand;
class NativeFnCommand;
class FlushCommand;
class FinishCommand;
class AcquireExtObjectsCommand;
class ReleaseExtObjectsCommand;
class PerfCounterCommand;
class ReleaseObjectCommand;
class StallQueueCommand;
class Marker;
class AccumulateCommand;
class ThreadTraceCommand;
class ThreadTraceMemObjectsCommand;
class SignalCommand;
class MakeBuffersResidentCommand;
class SvmFreeMemoryCommand;
class SvmCopyMemoryCommand;
class SvmFillMemoryCommand;
class SvmMapMemoryCommand;
class SvmUnmapMemoryCommand;
class SvmPrefetchAsyncCommand;
class StreamOperationCommand;
class BatchMemoryOperationCommand;
class VirtualMapCommand;
class ExternalSemaphoreCmd;
class UserEvent;
class Isa;
class Device;
struct KernelParameterDescriptor;
struct Coord3D;

//! @note: the defines match hip values
enum MemoryAdvice : uint32_t {
  SetReadMostly = 1,           ///< Data will mostly be read and only occassionally be written to
  UnsetReadMostly = 2,         ///< Undo the effect of hipMemAdviseSetReadMostly
  SetPreferredLocation = 3,    ///< Set the preferred location for the data as the specified device
  UnsetPreferredLocation = 4,  ///< Clear the preferred location for the data
  SetAccessedBy = 5,           ///< Data will be accessed by the specified device, reducing
                               ///< the amount of page faults
  UnsetAccessedBy = 6,         ///< HMM decides on the page faulting policy for the specified device
  SetCoarseGrain = 100,        ///< Change cache policy to improve performance (disables coherency)
  UnsetCoarseGrain = 101       ///< Restore coherent cache policy at the cost of some performance
};

enum MemRangeAttribute : uint32_t {
  ReadMostly = 1,            ///< Whether the range will mostly be read and only
                             ///< occassionally be written to
  PreferredLocation = 2,     ///< The preferred location of the range
  AccessedBy = 3,            ///< Memory range has hipMemAdviseSetAccessedBy
                             ///< set for specified device
  LastPrefetchLocation = 4,  ///< The last location to which the range was prefetched
  CoherencyMode = 100,       ///< Current coherency mode for the specified range
};

constexpr int CpuDeviceId = static_cast<int>(-1);
constexpr int InvalidDeviceId = static_cast<int>(-2);

// Max scratch size is device dependent.
constexpr size_t kWave32 = 32;
constexpr size_t kWave64 = 64;
constexpr size_t kScratchBits12X = 18;
constexpr size_t kScratchBits9X = 15;
constexpr size_t kCompilerRequired = 64;
constexpr size_t kMaxStackSize12X =
    (((1 << kScratchBits12X) - 1) * 256 / kWave32) - kCompilerRequired;
constexpr size_t kMaxStackSize11X =
    (((1 << kScratchBits9X) - 1) * 256 / kWave32) - kCompilerRequired;
constexpr size_t kMaxStackSize9X =
    (((1 << kScratchBits9X) - 1) * 256 / kWave64) - kCompilerRequired;

enum class ExternalSemaphoreHandleType : uint32_t {
  OpaqueFd = 1,                 // Handle is an opaque file descriptor
  OpaqueWin32 = 2,              // Handle is an opaque shared NT handle
  OpaqueWin32Kmt = 3,           // Handle is an opaque, globally shared handle
  D3D12Fence = 4,               // Handle is a shared NT handle referencing a
                                // D3D12 fence object
  D3D11Fence = 5,               // Handle is a shared NT handle referencing a
                                // D3D11 fence object
  NvSciSync = 6,                // Opaque handle to NvSciSync Object
  KeyedMutex = 7,               // Handle is a shared NT handle referencing a
                                // D3D11 keyed mutex object
  KeyedMutexKmt = 8,            // Handle is a shared KMT handle referencing a
                                // D3D11 keyed mutex object
  TimelineSemaphoreFd = 9,      // Handle is an opaque handle file
                                // descriptor referencing a timeline
                                // semaphore
  TimelineSemaphoreWin32 = 10,  // Handle is an opaque handle file
                                // descriptor referencing a timeline
                                // semaphore
};
}  // namespace amd

enum OclExtensions {
  ClKhrFp64 = 0,
  ClAmdFp64,
  ClKhrSelectFpRoundingMode,
  ClKhrGlobalInt32BaseAtomics,
  ClKhrGlobalInt32ExtendedAtomics,
  ClKhrLocalInt32BaseAtomics,
  ClKhrLocalInt32ExtendedAtomics,
  ClKhrInt64BaseAtomics,
  ClKhrInt64ExtendedAtomics,
  ClKhr3DImageWrites,
  ClKhrByteAddressableStore,
  ClKhrFp16,
  ClKhrGlSharing,
  ClKhrGLDepthImages,
  ClExtDeviceFission,
  ClAmdDeviceAttributeQuery,
  ClAmdVec3,
  ClAmdPrintf,
  ClAmdMediaOps,
  ClAmdMediaOps2,
  ClAmdPopcnt,
#if defined(_WIN32)
  ClKhrD3d10Sharing,
  ClKhrD3d11Sharing,
  ClKhrD3d9Sharing,
#endif
  ClKhrImage2dFromBuffer,
  ClAMDBusAddressableMemory,
  ClAMDC11Atomics,
  ClKhrSpir,
  ClKhrSubGroups,
  ClKhrGlEvent,
  ClKhrDepthImages,
  ClKhrMipMapImage,
  ClKhrMipMapImageWrites,
  ClAmdCopyBufferP2P,
  ClAmdAssemblyProgram,
#if defined(_WIN32)
  ClAmdPlanarYuv,
#endif
  ClExtTotal
};

static constexpr const char* OclExtensionsString[] = {"cl_khr_fp64 ",
                                                      "cl_amd_fp64 ",
                                                      "cl_khr_select_fprounding_mode ",
                                                      "cl_khr_global_int32_base_atomics ",
                                                      "cl_khr_global_int32_extended_atomics ",
                                                      "cl_khr_local_int32_base_atomics ",
                                                      "cl_khr_local_int32_extended_atomics ",
                                                      "cl_khr_int64_base_atomics ",
                                                      "cl_khr_int64_extended_atomics ",
                                                      "cl_khr_3d_image_writes ",
                                                      "cl_khr_byte_addressable_store ",
                                                      "cl_khr_fp16 ",
                                                      "cl_khr_gl_sharing ",
                                                      "cl_khr_gl_depth_images ",
                                                      "cl_ext_device_fission ",
                                                      "cl_amd_device_attribute_query ",
                                                      "cl_amd_vec3 ",
                                                      "cl_amd_printf ",
                                                      "cl_amd_media_ops ",
                                                      "cl_amd_media_ops2 ",
                                                      "cl_amd_popcnt ",
#if defined(_WIN32)
                                                      "cl_khr_d3d10_sharing ",
                                                      "cl_khr_d3d11_sharing ",
                                                      "cl_khr_dx9_media_sharing ",
#endif
                                                      "cl_khr_image2d_from_buffer ",
                                                      "cl_amd_bus_addressable_memory ",
                                                      "cl_amd_c11_atomics ",
                                                      "cl_khr_spir ",
                                                      "cl_khr_subgroups ",
                                                      "cl_khr_gl_event ",
                                                      "cl_khr_depth_images ",
                                                      "cl_khr_mipmap_image ",
                                                      "cl_khr_mipmap_image_writes ",
                                                      "cl_amd_copy_buffer_p2p ",
                                                      "cl_amd_assembly_program ",
#if defined(_WIN32)
                                                      "cl_amd_planar_yuv",
#endif
                                                      NULL};

static constexpr int AmdVendor = 0x1002;

template <typename T>
inline void WriteAqlArgAt(unsigned char* dst,  //!< The write pointer to the buffer
                          T src,               //!< The source pointer
                          uint size,           //!< The size in bytes to copy
                          size_t offset  //!< The alignment to follow while writing to the buffer
) {
  assert(sizeof(T) <= size && "Argument's size mismatches ABI!");
  *(reinterpret_cast<T*>(dst + offset)) = src;
}

namespace amd::device {
class ClBinary;
class BlitManager;
class Program;
class Kernel;

//! Physical device properties.
struct Info : public amd::EmbeddedObject {
  //! The OpenCL device type.
  cl_device_type type_;

  //! A unique device vendor identifier.
  uint32_t vendorId_;

  //! The available number of parallel compute cores on the compute device.
  uint32_t maxComputeUnits_;

  //! The max number of parallel compute cores on the compute device.
  uint32_t maxPhysicalComputeUnits_;

  //! Maximum dimensions that specify the global and local work-item IDs
  //  used by the data-parallel execution model.
  uint32_t maxWorkItemDimensions_;

  //! Maximum number of work-items that can be specified in each dimension
  //  to clEnqueueNDRangeKernel.
  size_t maxWorkItemSizes_[3];

  //! Maximum number of work-items in a work-group executing a kernel
  //  using the data-parallel execution model.
  size_t maxWorkGroupSize_;

  //! Preferred number of work-items in a work-group executing a kernel
  //  using the data-parallel execution model.
  size_t preferredWorkGroupSize_;

  //! Number of shader engines in physical GPU
  size_t numberOfShaderEngines;

  //! uint32_t Preferred native vector width size for built-in scalar types
  //  that can be put into vectors.
  uint32_t preferredVectorWidthChar_;
  uint32_t preferredVectorWidthShort_;
  uint32_t preferredVectorWidthInt_;
  uint32_t preferredVectorWidthLong_;
  uint32_t preferredVectorWidthFloat_;
  uint32_t preferredVectorWidthDouble_;
  uint32_t preferredVectorWidthHalf_;

  //! Returns the native ISA vector width. The vector width is defined as the
  //  number of scalar elements that can be stored in the vector.
  uint32_t nativeVectorWidthChar_;
  uint32_t nativeVectorWidthShort_;
  uint32_t nativeVectorWidthInt_;
  uint32_t nativeVectorWidthLong_;
  uint32_t nativeVectorWidthFloat_;
  uint32_t nativeVectorWidthDouble_;
  uint32_t nativeVectorWidthHalf_;

  //! Maximum configured engine clock frequency of the device in MHz.
  uint32_t maxEngineClockFrequency_;

  //! Maximum configured memory clock frequency of the device in MHz.
  uint32_t maxMemoryClockFrequency_;

  //! The constant frequency of wall clock in KHz
  uint32_t wallClockFrequency_;

  //! Memory bus width in bits.
  uint32_t vramBusBitWidth_;

  //! Size of L2 Cache in bytes.
  uint32_t l2CacheSize_;

  //! Timestamp frequency in Hz.
  uint32_t timeStampFrequency_;

  //! Describes the address spaces supported  by the device.
  uint32_t addressBits_;

  //! Max number of simultaneous image objects that can be read by a
  //  kernel.
  uint32_t maxReadImageArgs_;

  //! Max number of simultaneous image objects that can be written to
  //  by a kernel.
  uint32_t maxWriteImageArgs_;

  //! Max number of simultaneous image objects that can be read/written to
  //  by a kernel.
  uint32_t maxReadWriteImageArgs_;

  //! Max size of memory object allocation in bytes.
  uint64_t maxMemAllocSize_;

  //! Max size of system memory allocation in bytes.
  size_t maxPhysicalMemAllocSize_;

  //! Max width of 2D image in pixels.
  size_t image2DMaxWidth_;

  //! Max width of 2DA image in pixels.
  size_t image2DAMaxWidth_[2];

  //! Max height of 2D image in pixels.
  size_t image2DMaxHeight_;

  //! Max width of 3D image in pixels.
  size_t image3DMaxWidth_;

  //! Max height of 3D image in pixels.
  size_t image3DMaxHeight_;

  //! Max depth of 3D image in pixels.
  size_t image3DMaxDepth_;

  //! Describes whether images are supported
  uint32_t imageSupport_;

  //! Max size in bytes of the arguments that can be passed to a kernel.
  size_t maxParameterSize_;

  //! Maximum number of samplers that can be used in a kernel.
  uint32_t maxSamplers_;

  //! Describes the alignment in bits of the base address of any
  //  allocated memory object.
  uint32_t memBaseAddrAlign_;

  //! The smallest alignment in bytes which can be used for any data type.
  uint32_t minDataTypeAlignSize_;

  //! Describes single precision floating point capability of the device.
  cl_device_fp_config halfFPConfig_;
  cl_device_fp_config singleFPConfig_;
  cl_device_fp_config doubleFPConfig_;

  //! Type of global memory cache supported.
  cl_device_mem_cache_type globalMemCacheType_;

  //! Size of global memory cache line in bytes.
  uint32_t globalMemCacheLineSize_;

  //! Size of global memory cache in bytes.
  uint64_t globalMemCacheSize_;

  //! Size of global device memory in bytes.
  uint64_t globalMemSize_;

  //! Max size in bytes of a constant buffer allocation.
  uint64_t maxConstantBufferSize_;

  //! Preferred size in bytes of a constant buffer allocation.
  uint64_t preferredConstantBufferSize_;

  //! Max number of arguments declared
  uint32_t maxConstantArgs_;

  //! This is used to determine the type of local memory that is available
  cl_device_local_mem_type localMemType_;

  //! Size of local memory arena in bytes.
  uint64_t localMemSize_;

  //! If enabled, implies that all the memories, caches, registers etc. in
  //  the device implement error correction.
  uint32_t errorCorrectionSupport_;

  //! true if the device and the host have a unified memory and is false otherwise.
  uint32_t hostUnifiedMemory_;

  //! true if the device and the host have a unified memory management subsystem and
  //  is false otherwise.
  uint32_t iommuv2_;

  //! Describes the resolution of device timer.
  size_t profilingTimerResolution_;

  //! Timer starting point offset to Epoch.
  uint64_t profilingTimerOffset_;

  //! true if device is a little endian device.
  uint32_t littleEndian_;

  //! If enabled, implies that commands can be submitted to command-queues
  //  created on this device.
  uint32_t available_;

  //! if the implementation does not have a compiler available to compile
  //  the program source.
  uint32_t compilerAvailable_;

  //! Describes the execution capabilities of the device.
  cl_device_exec_capabilities executionCapabilities_;

  //! Describes the SVM capabilities of the device.
  cl_device_svm_capabilities svmCapabilities_;

  //! Preferred alignment for OpenCL fine-grained SVM atomic types.
  uint32_t preferredPlatformAtomicAlignment_;

  //! Preferred alignment for OpenCL global atomic types.
  uint32_t preferredGlobalAtomicAlignment_;

  //! Preferred alignment for OpenCL local atomic types.
  uint32_t preferredLocalAtomicAlignment_;

  //! Describes the command-queue properties supported of the host queue.
  cl_command_queue_properties queueProperties_;

  //! The platform associated with this device
  cl_platform_id platform_;

  //! Device name string
  char name_[0x40];

  //! Vendor name string
  char vendor_[0x20];

  //! OpenCL software driver version string in the form major.minor
  char driverVersion_[0x20];

  //! Returns the profile name supported by the device.
  const char* profile_;

  //! Returns the OpenCL version supported by the device.
  const char* version_;

  //! The highest OpenCL C version supported by the compiler for this device.
  const char* oclcVersion_;

  //! Returns a space separated list of extension names.
  const char* extensions_;

  //! Returns if device linker is available
  uint32_t linkerAvailable_;

  //! Returns the list of built-in kernels, supported by the device
  const char* builtInKernels_;

  //! Returns max number of pixels for a 1D image
  size_t image1DMaxWidth_;

  //! Returns max number of pixels for a 1DA image
  size_t image1DAMaxWidth_;

  //! Returns max number of pixels for a 1D image created from a buffer object
  size_t imageMaxBufferSize_;

  //! Returns max number of images in a 1D or 2D image array
  size_t imageMaxArraySize_;

  //! Returns true if the devices preference is for the user to be
  //! responsible for synchronization
  uint32_t preferredInteropUserSync_;

  //! Returns maximum size of the internal buffer that holds the output
  //! of printf calls from a kernel
  size_t printfBufferSize_;

  //! Indicates maximum number of supported global atomic counters
  uint32_t maxAtomicCounters_;

  //! Returns the topology for the device
  cl_device_topology_amd deviceTopology_;

  //! Returns PCI Bus Domain ID
  uint32_t pciDomainID;

  //! Returns sddress of HDP_MEM_COHERENCY_FLUSH_CNTL register
  uint32_t* hdpMemFlushCntl;

  //! Returns sddress of HDP_REG_COHERENCY_FLUSH_CNTL register
  uint32_t* hdpRegFlushCntl;

  //! Semaphore information
  uint32_t maxSemaphores_;
  uint32_t maxSemaphoreSize_;

  //! Returns the SKU board name for the device
  char boardName_[128];

  //! Number of SIMD (Single Instruction Multiple Data) units per compute unit
  //! that execute in parallel. All work items from the same work group must be
  //! executed by SIMDs in the same compute unit.
  uint32_t simdPerCU_;
  uint32_t cuPerShaderArray_;  //!< Number of CUs per shader array
  //! The maximum number of work items from the same work group that can be
  //! executed by a SIMD in parallel
  uint32_t simdWidth_;
  //! The number of instructions that a SIMD can execute in parallel
  uint32_t simdInstructionWidth_;
  //! The number of workitems per wavefront
  uint32_t wavefrontWidth_;
  //! Available number of SGPRs
  uint32_t availableSGPRs_;

  uint32_t availableVGPRs_;        //!< Number of addressable VGPRs per thread in DWORDs
  uint32_t vgprsPerSimd_;          //!< Number of VGPRs per SIMD in DWORDs
  uint32_t vgprAllocGranularity_;  //!< Number of VGPRs allocation granularity per thread in DWORDs
  uint32_t availableRegistersPerCU_;  //!< Number of VGPRs per CU in DWORDs

  //! Number of global memory channels
  uint32_t globalMemChannels_;
  //! Number of banks in each global memory channel
  uint32_t globalMemChannelBanks_;
  //! Width in bytes of each of global memory bank
  uint32_t globalMemChannelBankWidth_;
  //! Local memory size per CU
  uint32_t localMemSizePerCU_;
  //! Number of banks of local memory
  uint32_t localMemBanks_;
  //! Number of available async queues
  uint32_t numAsyncQueues_;
  //! Number of available real time queues
  uint32_t numRTQueues_;
  //! Number of available real time compute units
  uint32_t numRTCUs_;
  //! The granularity at which compute units can be dedicated to a queue
  uint32_t granularityRTCUs_;
  //! Thread trace enable
  uint32_t threadTraceEnable_;

  //! Image pitch alignment for image2d_from_buffer
  uint32_t imagePitchAlignment_;
  //! Image base address alignment for image2d_from_buffer
  uint32_t imageBaseAddressAlignment_;

  //! Describes whether buffers from images are supported
  uint32_t bufferFromImageSupport_;

  //! Returns the supported SPIR versions for the device
  const char* spirVersions_;

  //! OpenCL20 device info fields:

  //! The max number of pipe objects that can be passed as arguments to a kernel
  uint32_t maxPipeArgs_;
  //! The max number of reservations that can be active for a pipe per work-item in a kernel
  uint32_t maxPipeActiveReservations_;
  //! The max size of pipe packet in bytes
  uint32_t maxPipePacketSize_;

  //! The command-queue properties supported of the device queue.
  cl_command_queue_properties queueOnDeviceProperties_;
  //! The preferred size of the device queue in bytes
  uint32_t queueOnDevicePreferredSize_;
  //! The max size of the device queue in bytes
  uint32_t queueOnDeviceMaxSize_;
  //! The maximum number of device queues
  uint32_t maxOnDeviceQueues_;
  //! The maximum number of events in use on a device queue
  uint32_t maxOnDeviceEvents_;

  //! The maximum size of global scope variables
  size_t maxGlobalVariableSize_;
  size_t globalVariablePreferredTotalSize_;
  //! Driver store location
  char driverStore_[200];
  //! Device ID
  uint32_t pcieDeviceId_;
  //! PCI Revision ID
  uint32_t pcieRevisionId_;
  //! ASIC Revision
  uint32_t asicRevision_;
  //! Returns the unique identifier for the device
  char uuid_[16];
  //! Max numbers of threads per CU
  uint32_t maxThreadsPerCU_;

  //! GPU device supports a launch of cooperative groups
  uint32_t cooperativeGroups_;
  //! GPU device supports a launch of cooperative groups on multiple devices
  uint32_t cooperativeMultiDeviceGroups_;

  //! large bar support.
  bool largeBar_;

  uint32_t hmmSupported_;            //!< ROCr supports HMM interfaces
  uint32_t hmmCpuMemoryAccessible_;  //!< CPU memory is accessible by GPU without pinning/register
  uint32_t hmmDirectHostAccess_;     //!< HMM memory is accessible from the host without migration

  //! global CU mask which will be applied to all queues created on this device
  std::vector<uint32_t> globalCUMask_;

  //! AQL Barrier Value Packet support
  bool aqlBarrierValue_;

  bool pcie_atomics_;  //!< Pcie atomics support flag

  bool virtualMemoryManagement_;       //!< Virtual memory management support
  size_t virtualMemAllocGranularity_;  //!< virtual memory allocation size/addr granularity

  uint32_t driverNodeId_;
  //! Number of Physical SGPRs per SIMD
  uint32_t sgprsPerSimd_;

  uint32_t numSDMAengines_;  //!< Number of available SDMA engines

  uint32_t luidLowPart_;         //!< Luid low 4 bytes, available in Windows only
  uint32_t luidHighPart_;        //!< Luid high 4 bytes, available in Windows only
  uint32_t luidDeviceNodeMask_;  //!< Luid node mask

  size_t scratchLimitMin;  //! Minimum size of scratch limit of this device memory in bytes.
  size_t scratchLimitMax;  //! Maximum size of scratch limit of this device memory in bytes.

  uint32_t numberOfXccs_;  //! The number of XCC(s) on the device
};

//! Device settings
class Settings : public amd::HeapObject {
 public:
  enum KernelArgImpl {
    HostKernelArgs = 0,        //!< Kernel Arguments are put into host memory
    DeviceKernelArgs,          //!< Device memory kernel arguments with no memory
                               //!< ordering workaround (e.g. XGMI)
    DeviceKernelArgsReadback,  //!< Device memory kernel arguments with kernel
                               //!< argument readback workaround
    DeviceKernelArgsHDP        //!< Device memory kernel arguments with kernel
                               //!< argument readback plus HDP flush workaround.
  };

  uint64_t extensions_;  //!< Supported OCL extensions
  union {
    struct {
      uint apuSystem_ : 1;                    //!< Device is APU system with shared memory
      uint supportRA_ : 1;                    //!< Support RA channel order format
      uint waitCommand_ : 1;                  //!< Enables a wait for every submitted command
      uint customHostAllocator_ : 1;          //!< True if device has custom host allocator
                                              //  that replaces generic OS allocation routines
      uint supportDepthsRGB_ : 1;             //!< Support DEPTH and sRGB channel order format
      uint singleFpDenorm_ : 1;               //!< Support Single FP Denorm
      uint hsailExplicitXnack_ : 1;           //!< Xnack in hsail path for this device
      uint useLightning_ : 1;                 //!< Enable LC path for this device
      uint enableWgpMode_ : 1;                //!< Enable WGP mode for this device
      uint enableWave32Mode_ : 1;             //!< Enable Wave32 mode for this device
      uint lcWavefrontSize64_ : 1;            //!< Enable Wave64 mode for this device
      uint enableXNACK_ : 1;                  //!< Enable XNACK feature
      uint enableCoopGroups_ : 1;             //!< Enable cooperative groups feature
      uint enableCoopMultiDeviceGroups_ : 1;  //!< Enable cooperative groups multi device
      uint fenceScopeAgent_ : 1;              //!< Enable fence scope agent in AQL dispatch packet
      uint rocr_backend_ : 1;                 //!< Device uses ROCr backend for submissions
      uint gwsInitSupported_ : 1;             //!< Check if GWS is supported on this machine.
      uint kernel_arg_opt_ : 1;               //!< Enables kernel arg optimization for blit kernels
      uint kernel_arg_impl_ : 2;              //!< Kernel argument implementation
      uint reserved_ : 12;
    };
    uint value_;
  };

  //! Default constructor
  Settings();

  //! Virtual destructor as this class is used as a base class and is also used
  //! to delete the derived classes.
  virtual ~Settings(){};

  //! Check the specified extension
  bool checkExtension(uint name) const {
    return (extensions_ & (static_cast<uint64_t>(1) << name)) ? true : false;
  }

  //! Enable the specified extension
  void enableExtension(uint name) { extensions_ |= static_cast<uint64_t>(1) << name; }

  size_t stagedXferSize_ = 0;  //!< Staged buffer size

 private:
  //! Disable copy constructor
  Settings(const Settings&);

  //! Disable assignment
  Settings& operator=(const Settings&);
};

//! Device-independent cache memory, base class for the device-specific
//! memories. One Memory instance refers to one or more of these.
class Memory : public amd::HeapObject {
 public:
  //! Resource map flags
  enum CpuMapFlags {
    CpuReadWrite = 0x00000000,  //!< Lock for CPU read/Write
    CpuReadOnly = 0x00000001,   //!< Lock for CPU read only operation
    CpuWriteOnly = 0x00000002,  //!< Lock for CPU write only operation
  };

  //! Memory Access flags at device level
  enum MemAccess {
    kMemAccessNone = 0,      //! No Access
    kMemAccessRead = 1,      //! Read Access
    kMemAccessReadWrite = 3  //! Read and Write Access
  };

  union SyncFlags {
    struct {
      uint skipParent_ : 1;  //!< Skip parent synchronization
      uint skipViews_ : 1;   //!< Skip views synchronization
      uint skipEntire_ : 1;  //!< Skip entire synchronization
    };
    uint value_;
    SyncFlags() : value_(0) {}
  };

  struct WriteMapInfo : public amd::HeapObject {
    amd::Coord3D origin_;  //!< Origin of the map location
    amd::Coord3D region_;  //!< Mapped region
    amd::Image* baseMip_;  //!< The base mip level for images
    union {
      struct {
        uint32_t count_ : 8;       //!< The same map region counter
        uint32_t unmapWrite_ : 1;  //!< Unmap write operation
        uint32_t unmapRead_ : 1;   //!< Unmap read operation
        uint32_t entire_ : 1;      //!< Process the entire memory
      };
      uint32_t flags_;
    };

    //! Returns the state of entire map
    bool isEntire() const { return (entire_) ? true : false; }

    //! Returns the state of map write flag
    bool isUnmapWrite() const { return (unmapWrite_) ? true : false; }

    //! Returns the state of map read flag
    bool isUnmapRead() const { return (unmapRead_) ? true : false; }

    WriteMapInfo() : origin_(0, 0, 0), region_(0, 0, 0), baseMip_(NULL), flags_(0) {}
  };

  //! Constructor (from an amd::Memory object).
  Memory(amd::Memory& owner)
      : flags_(0), owner_(&owner), version_(0), mapMemory_(NULL), indirectMapCount_(0) {
    size_ = owner.getSize();
    memAccess_ = MemAccess::kMemAccessNone;
  }

  //! Constructor (no owner), always eager allocation.
  Memory(size_t size)
      : flags_(0), owner_(NULL), version_(0), mapMemory_(NULL), indirectMapCount_(0), size_(size) {
    memAccess_ = MemAccess::kMemAccessNone;
  }

  enum GLResourceOP {
    GLDecompressResource = 0,  // orders the GL driver to decompress any depth-stencil or MSAA
                               // resource to be sampled by a CL kernel.
    GLInvalidateFBO  // orders the GL driver to invalidate any FBO the resource may be bound to,
                     // since the resource internal state changed.
  };

  //! Default destructor for the device memory object
  virtual ~Memory(){};

  //! Releases virtual objects associated with this memory
  void releaseVirtual();

  //! Read the size
  size_t size() const { return size_; }

  //! Gets the owner Memory instance
  amd::Memory* owner() const { return owner_; }

  //! Immediate blocking write from device cache to owners's backing store.
  //! Marks owner as "current" by resetting the last writer to NULL.
  virtual void syncHostFromCache(device::VirtualDevice* vDev, SyncFlags syncFlags = SyncFlags()) {}

  //! Allocate memory for API-level maps
  virtual void* allocMapTarget(const amd::Coord3D& origin,  //!< The map location in memory
                               const amd::Coord3D& region,  //!< The map region in memory
                               uint mapFlags,               //!< Map flags
                               size_t* rowPitch = NULL,     //!< Row pitch for the mapped memory
                               size_t* slicePitch = NULL    //!< Slice for the mapped memory
  ) {
    return NULL;
  }

  bool isPersistentMapped() const { return (flags_ & PersistentMap) ? true : false; }
  void setPersistentMapFlag(bool persistentMapped) {
    if (persistentMapped == true) {
      flags_ |= PersistentMap;
    } else {
      flags_ &= ~PersistentMap;
    }
  }

  virtual bool pinSystemMemory(void* hostPtr,  //!< System memory address
                               size_t size     //!< Size of allocated system memory
  ) {
    return true;
  }

  //! Releases indirect map surface
  virtual void releaseIndirectMap() {}
  //! decompress any MSAA/depth-stencil interop surfaces.
  //! notify GL to invalidate any surfaces touched by a CL kernel
  virtual bool processGLResource(GLResourceOP operation) { return false; }

  //! Map the device memory to CPU visible
  virtual void* cpuMap(VirtualDevice& vDev,  //!< Virtual device for map operaiton
                       uint flags = 0,       //!< flags for the map operation
                       // Optimization for multilayer map/unmap
                       uint startLayer = 0,       //!< Start layer for multilayer map
                       uint numLayers = 0,        //!< End layer for multilayer map
                       size_t* rowPitch = NULL,   //!< Row pitch for the device memory
                       size_t* slicePitch = NULL  //!< Slice pitch for the device memory
  ) {
    amd::Image* image = owner()->asImage();
    if (image != NULL) {
      *rowPitch = image->getRowPitch();
      *slicePitch = image->getSlicePitch();
    }
    // Default behavior uses preallocated host mem for CPU
    return owner()->getHostMem();
  }

  //! Unmap the device memory
  virtual void cpuUnmap(VirtualDevice& vDev  //!< Virtual device for unmap operaiton
  ) {}

  //! Saves map info for this object
  //! @note: It's not a thread safe operation, the app must implement
  //! synchronization for the multiple write maps if necessary
  void saveMapInfo(const void* mapAddress,        //!< Map cpu address
                   const amd::Coord3D origin,     //!< Origin of the map location
                   const amd::Coord3D region,     //!< Mapped region
                   uint mapFlags,                 //!< Map flags
                   bool entire,                   //!< True if the enitre memory was mapped
                   amd::Image* baseMip = nullptr  //!< The base mip level for map
  );

  const WriteMapInfo* writeMapInfo(const void* mapAddress) const {
    // Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());

    auto it = writeMapInfo_.find(mapAddress);
    if (it == writeMapInfo_.end()) {
      if (writeMapInfo_.size() == 0) {
        LogError("Unmap is a NOP!");
        return nullptr;
      }
      LogWarning("Unknown unmap signature!");
      // Get the first map info
      it = writeMapInfo_.begin();
    }
    return &it->second;
  }

  //! Clear memory object as mapped read only
  void clearUnmapInfo(const void* mapAddress) {
    // Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());
    auto it = writeMapInfo_.find(mapAddress);
    if (it == writeMapInfo_.end()) {
      // Get the first map info
      it = writeMapInfo_.begin();
    }
    if (--it->second.count_ == 0) {
      writeMapInfo_.erase(it);
    }
  }

  //! Returns the state of memory direct access flag
  bool isHostMemDirectAccess() const { return (flags_ & HostMemoryDirectAccess) ? true : false; }

  //! Returns the state of host memory registration flag
  bool isHostMemoryRegistered() const { return (flags_ & HostMemoryRegistered) ? true : false; }

  //! Returns the state of CPU uncached access
  bool isCpuUncached() const { return (flags_ & MemoryCpuUncached) ? true : false; }

  virtual uint64_t virtualAddress() const { return 0; }

  virtual uint64_t originalDeviceAddress() const { return 0; }

  //! Returns CPU pointer to HW state
  virtual const address cpuSrd() const { return nullptr; }

  //! Returns an export handle for the interprocess communication
  virtual bool ExportHandle(void* handle) const { return false; }

  bool getAllowedPeerAccess() const { return (flags_ & AllowedPeerAccess) ? true : false; }
  void setAllowedPeerAccess(bool flag) {
    if (flag == true) {
      flags_ |= AllowedPeerAccess;
    } else {
      flags_ &= ~AllowedPeerAccess;
    }
  }

  //! Set access to the memory in this device.
  void SetAccess(MemAccess memAccess) { memAccess_ = memAccess; }

  //! Get current access of the memory in device.
  MemAccess GetAccess() const { return memAccess_; }

  //! Retrieves shareable handle for hipMalloc'ed address range.
  virtual bool GetFDHandleForMem(void* dev_ptr, size_t size, bool vmm, void* handle) {
    return false;
  }

 protected:
  enum Flags {
    HostMemoryDirectAccess = 0x00000001,  //!< GPU has direct access to the host memory
    MapResourceAlloced = 0x00000002,      //!< Map resource was allocated
    PinnedMemoryAlloced = 0x00000004,     //!< An extra pinned resource was allocated
    SubMemoryObject = 0x00000008,         //!< Memory is sub-memory
    HostMemoryRegistered = 0x00000010,    //!< Host memory was registered
    MemoryCpuUncached = 0x00000020,       //!< Memory is uncached on CPU access(slow read)
    AllowedPeerAccess = 0x00000040,       //!< Memory can be accessed from peer
    PersistentMap = 0x00000080            //!< Map Peristent memory
  };
  uint flags_;  //!< Memory object flags

  MemAccess memAccess_;  //!< Memory Access flag

  amd::Memory* owner_;  //!< The Memory instance that we cache,
                        //!< or NULL if we're device-private workspace.

  volatile size_t version_;  //!< The version we're currently shadowing

  //! NB, the map data below is for an API-level map (from clEnqueueMapBuffer),
  //! not a physical map. When a memory object does not use USE_HOST_PTR we
  //! can use a remote resource and DMA, avoiding the additional CPU memcpy.
  amd::Memory* mapMemory_;                //!< Memory used as map target buffer
  std::atomic<size_t> indirectMapCount_;  //!< Number of maps
  std::unordered_map<const void*, WriteMapInfo>
      writeMapInfo_;  //!< Saved write map info for partial unmap

  //! Increment map count
  void incIndMapCount() { ++indirectMapCount_; }

  //! Decrement map count
  virtual void decIndMapCount() {}

  size_t size_;  //!< Memory size

 private:
  //! Disable default copy constructor
  Memory& operator=(const Memory&) = delete;

  //! Disable operator=
  Memory(const Memory&) = delete;
};

class Sampler : public amd::HeapObject {
 public:
  //! Constructor
  Sampler() : hwSrd_(0), hwState_(nullptr) {}

  //! Default destructor for the device memory object
  virtual ~Sampler(){};

  //! Returns device specific HW state for the sampler
  uint64_t hwSrd() const { return hwSrd_; }

  //! Returns CPU pointer to HW state
  const address hwState() const { return hwState_; }

 protected:
  uint64_t hwSrd_;   //!< Device specific HW state for the sampler
  address hwState_;  //!< CPU pointer to HW state

 private:
  //! Disable default copy constructor
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);
};

class ClBinary : public amd::HeapObject {
 public:
  enum BinaryImageFormat {
    BIF_VERSION2 = 0,  //!< Binary Image Format version 2.0 (ELF)
    BIF_VERSION3       //!< Binary Image Format version 3.0 (ELF)
  };

  //! Constructor
  ClBinary(const amd::Device& dev, BinaryImageFormat bifVer = BIF_VERSION3);

  //! Destructor
  virtual ~ClBinary();

  void init(amd::option::Options* optionsObj);

  /** called only in loading image routines, never storing routines */
  bool setBinary(const char* theBinary, size_t theBinarySize, bool allocated = false,
                 amd::Os::FileDesc fd = amd::Os::FDescInit(), size_t foffset = 0,
                 std::string uri = std::string());

  //! setin elfIn_
  bool setElfIn();
  void resetElfIn();

  //! set out elf
  bool setElfOut(unsigned char eclass, const char* outFile, bool tempFile);
  void resetElfOut();

  //! Set elf header information
  virtual bool setElfTarget();

  // class used in for loading images in new format
  amd::Elf* elfIn() { return elfIn_; }

  // classes used storing and loading images in new format
  amd::Elf* elfOut() { return elfOut_; }
  void elfOut(amd::Elf* v) { elfOut_ = v; }

  //! Create and save ELF binary image
  bool createElfBinary(bool doencrypt, Program::type_t type);

  // save BIF binary image
  void saveBIFBinary(const char* binaryIn, size_t size);

  bool decryptElf(const char* binaryIn, size_t size, char** decryptBin, size_t* decryptSize,
                  int* encryptCode);

  //! Returns the binary pair, fdesc pair, uri for the abstraction layer
  Program::binary_t data() const;
  Program::finfo_t Datafd() const;
  std::string DataURI() const;

  //! Loads llvmir binary from OCL binary file
  bool loadLlvmBinary(std::string& llvmBinary,               //!< LLVMIR binary code
                      amd::Elf::ElfSections& elfSectionType  //!< LLVMIR binary is in SPIR format
  ) const;

  //! Loads compile options from OCL binary file
  bool loadCompileOptions(std::string& compileOptions  //!< return the compile options loaded
  ) const;

  //! Loads link options from OCL binary file
  bool loadLinkOptions(std::string& linkOptions  //!< return the link options loaded
  ) const;

  //! Store compile options into OCL binary file
  void storeCompileOptions(const std::string& compileOptions  //!< the compile options to be stored
  );

  //! Store link options into OCL binary file
  void storeLinkOptions(const std::string& linkOptions  //!< the link options to be stored
  );

  //! Check if the binary is recompilable
  bool isRecompilable(std::string& llvmBinary, amd::Elf::ElfPlatform thePlatform);

  void saveOrigBinary(const char* origBinary, size_t origSize) {
    origBinary_ = origBinary;
    origSize_ = origSize;
  }

  void restoreOrigBinary() {
    if (origBinary_ != NULL) {
      (void)setBinary(origBinary_, origSize_, false);
    }
  }

  //! Set Binary flags
  void setFlags(int encryptCode);

  bool saveSOURCE() { return ((flags_ & BinarySourceMask) == BinarySaveSource); }
  bool saveLLVMIR() { return ((flags_ & BinaryLlvmirMask) == BinarySaveLlvmir); }
  bool saveISA() { return ((flags_ & BinaryIsaMask) == BinarySaveIsa); }

  bool saveAS() { return ((flags_ & BinaryASMask) == BinarySaveAS); }

  // Return the encrypt code for this input binary ( "> 0" means encrypted)
  int getEncryptCode() { return encryptCode_; }

  // Returns TRUE of binary file is SPIR
  bool isSPIR() const;
  // Returns TRUE of binary file is SPIRV
  bool isSPIRV() const;

 protected:
  enum Flags {
    BinaryAllocated = 0x1,  //!< Binary was created

    // Source control
    BinaryNoSaveSource = 0x0,  // 0: default
    BinaryRemoveSource = 0x2,  // for encrypted binary
    BinarySaveSource = 0x4,
    BinarySourceMask = 0x6,

    // LLVMIR control
    BinarySaveLlvmir = 0x0,    // 0: default
    BinaryRemoveLlvmir = 0x8,  // for encrypted binary
    BinaryNoSaveLlvmir = 0x10,
    BinaryLlvmirMask = 0x18,

    // ISA control
    BinarySaveIsa = 0x0,     // 0: default
    BinaryRemoveIsa = 0x80,  // for encrypted binary
    BinaryNoSaveIsa = 0x100,
    BinaryIsaMask = 0x180,

    // AS control
    BinaryNoSaveAS = 0x0,    // 0: default
    BinaryRemoveAS = 0x200,  // for encrypted binary
    BinarySaveAS = 0x400,
    BinaryASMask = 0x600
  };

  //! Returns TRUE if binary file was allocated
  bool isBinaryAllocated() const { return (flags_ & BinaryAllocated) ? true : false; }

#if defined(WITH_COMPILER_LIB)
  //! Returns BIF symbol name by symbolID,
  //! returns empty string if not found or if BIF version is unsupported
  std::string getBIFSymbol(unsigned int symbolID) const;
#endif

 protected:
  const amd::Device& dev_;  //!< Device object

 private:
  //! Disable default copy constructor
  ClBinary(const ClBinary&);

  //! Disable default operator=
  ClBinary& operator=(const ClBinary&);

  //! Releases the binary data store
  void release();

  const char* binary_;  //!< binary data
  size_t size_;         //!< binary size
  uint flags_;          //!< CL binary object flags

  amd::Os::FileDesc fdesc_;  //!< file descriptor
  size_t foffset_;           //!< file offset
  std::string uri_;          //!< memory URI

  const char* origBinary_;  //!< original binary data
  size_t origSize_;         //!< original binary size

  int encryptCode_;  //!< Encryption Code for input binary (0 for not encrypted)

  std::string fname_;  //!< name of elf dump file
  bool tempFile_;      //!< Is the elf dump file temporary

 protected:
  amd::Elf* elfIn_;           //!< ELF object for input ELF binary
  amd::Elf* elfOut_;          //!< ELF object for output ELF binary
  BinaryImageFormat format_;  //!< which binary image format to use
};

inline const Program::binary_t Program::binary() const {
  if (clBinary() == NULL) {
    return {(const void*)0, 0};
  }
  return clBinary()->data();
}

inline std::string Program::BinaryURI() const {
  if (clBinary() == NULL) {
    return std::string();
  }
  return clBinary()->DataURI();
}

inline Program::finfo_t Program::BinaryFd() const {
  if (clBinary() == NULL) {
    return {amd::Os::FDescInit(), 0};
  }
  return clBinary()->Datafd();
}

inline Program::binary_t Program::binary() {
  if (clBinary() == NULL) {
    return {(const void*)0, 0};
  }
  return clBinary()->data();
}

/*! \class PerfCounter
 *
 *  \brief The device interface class for the performance counters
 */
class PerfCounter : public amd::HeapObject {
 public:
  //! Constructor for the device performance
  PerfCounter() {}

  //! Get the performance counter info
  virtual uint64_t getInfo(uint64_t infoType) const = 0;

  //! Destructor for PerfCounter class
  virtual ~PerfCounter() {}

 private:
  //! Disable default copy constructor
  PerfCounter(const PerfCounter&);

  //! Disable default operator=
  PerfCounter& operator=(const PerfCounter&);
};
/*! \class ThreadTrace
 *
 *  \brief The device interface class for the performance counters
 */
class ThreadTrace : public amd::HeapObject {
 public:
  //! Constructor for the device performance
  ThreadTrace() {}
  //! Update ThreadTrace status to true/false if new buffer was binded/unbinded respectively
  virtual void setNewBufferBinded(bool) = 0;
  //! Get the performance counter info
  virtual bool info(uint infoType, uint* info, uint infoSize) const = 0;
  //! Destructor for PerfCounter class
  virtual ~ThreadTrace() {}

 private:
  //! Disable default copy constructor
  ThreadTrace(const ThreadTrace&);

  //! Disable default operator=
  ThreadTrace& operator=(const ThreadTrace&);
};

//! A device execution environment.
class VirtualDevice : public amd::HeapObject {
 public:
  //! Construct a new virtual device for the given physical device.
  VirtualDevice(amd::Device& device)
      : device_(device),
        blitMgr_(NULL),
        execution_(true) /* Virtual device execution lock */
        ,
        index_(0) {}

  //! Destroy this virtual device.
  virtual ~VirtualDevice() {}

  //! Return the physical device for this virtual device.
  const amd::Device& device() const { return device_(); }
  virtual uint64_t getQueueID() = 0;
  virtual void submitReadMemory(amd::ReadMemoryCommand& cmd) = 0;
  virtual void submitWriteMemory(amd::WriteMemoryCommand& cmd) = 0;
  virtual void submitCopyMemory(amd::CopyMemoryCommand& cmd) = 0;
  virtual void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd) = 0;
  virtual void submitMapMemory(amd::MapMemoryCommand& cmd) = 0;
  virtual void submitUnmapMemory(amd::UnmapMemoryCommand& cmd) = 0;
  virtual void submitKernel(amd::NDRangeKernelCommand& command) = 0;
  virtual void submitNativeFn(amd::NativeFnCommand& cmd) = 0;
  virtual void submitMarker(amd::Marker& cmd) = 0;
  virtual void submitAccumulate(amd::AccumulateCommand& cmd) = 0;
  virtual void submitExternalSemaphoreCmd(amd::ExternalSemaphoreCmd& cmd) = 0;
  virtual void submitFillMemory(amd::FillMemoryCommand& cmd) = 0;
  virtual void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd) = 0;
  virtual void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd) = 0;
  virtual void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd) = 0;
  virtual void submitPerfCounter(amd::PerfCounterCommand& cmd) = 0;
  virtual void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) = 0;
  virtual void submitThreadTrace(amd::ThreadTraceCommand& cmd) = 0;
  virtual void flush(amd::Command* list = NULL, bool wait = false) = 0;
  virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd) = 0;
  virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd) = 0;
  virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd) = 0;
  virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd) = 0;
  virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd) = 0;
  /// Optional extensions
  virtual void submitSignal(amd::SignalCommand& cmd) = 0;
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd) = 0;
  virtual void submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd) { ShouldNotReachHere(); }
  virtual void submitStreamOperation(amd::StreamOperationCommand& cmd) { ShouldNotReachHere(); }
  virtual void submitBatchMemoryOperation(amd::BatchMemoryOperationCommand& cmd) {
    ShouldNotReachHere();
  }
  virtual void submitVirtualMap(amd::VirtualMapCommand& cmd) { ShouldNotReachHere(); }
  virtual void submitUserEvent(amd::UserEvent& vcmd) { ShouldNotReachHere(); }

  virtual address allocKernelArguments(size_t size, size_t alignment) { return nullptr; }
  virtual void ReleaseAllHwQueues() {}
  virtual void ReleaseHwQueue() {}

  //! Get the blit manager object
  device::BlitManager& blitMgr() const { return *blitMgr_; }

  //! Returns the monitor object for execution access by VirtualGPU
  amd::Monitor& execution() { return execution_; }

  //! Returns the virtual device unique index
  uint index() const { return index_; }

  //! Returns true if device has active wait setting
  bool ActiveWait() const;

  //! Returns fence state of the VirtualGPU
  virtual bool isFenceDirty() const = 0;
  //! Init hidden heap for device memory allocations
  virtual void HiddenHeapInit() = 0;

  //! Dispatches multiple AQL packets in a single batch operation
  virtual bool dispatchAqlPacketBatch(const std::vector<uint8_t*>& packets,
                                      const std::vector<std::string>& kernelNames,
                                      amd::AccumulateCommand* vcmd = nullptr) = 0 ;
  //! Returns the number of outstanding HSA async handlers
  std::atomic<uint64_t>& QueuedAsyncHandlers() const { return queued_async_handlers_; }

 private:
  //! Disable default copy constructor
  VirtualDevice& operator=(const VirtualDevice&);

  //! Disable operator=
  VirtualDevice(const VirtualDevice&);

  //! The physical device that this virtual device utilizes
  amd::SharedReference<amd::Device> device_;

 protected:
  device::BlitManager* blitMgr_;  //!< Blit manager

  amd::Monitor execution_;  //!< Lock to serialise access to all device objects
  uint index_;              //!< The virtual device unique index
  mutable std::atomic<uint64_t> queued_async_handlers_ = 0;  //!< Outstanding HSA async handlers
};

#if defined(USE_COMGR_LIBRARY)
extern bool getValueFromIsaMeta(const std::string& isa, const char* key, std::string& retValue);
#endif

}  // namespace amd::device

namespace amd {
/*! IHIP IPC MEMORY Structure */
#define AMD_IPC_MEM_HANDLE_SIZE 32

typedef enum SyncPolicy { Auto = 1, Spin = 2, Yield = 3, Blocking = 4 } SyncPolicy;

//! MemoryObject map lookup  class
class MemObjMap : public AllStatic {
 public:
  struct IpcMemHandle {
    char ipc_handle[AMD_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
    size_t psize;                              ///< Total size of the device memory allocation
    size_t poffset;                            ///< Offset within the allocation
    int owners_process_id;                     ///< ID of the process that owns the allocation
    int owners_device_id;                      ///< ID of the device that owns the allocation
    char reserved[LP64_SWITCH(16, 8)];        ///< Reserved for future extensions

    bool operator<(const IpcMemHandle& h) const {
      int cmp = std::memcmp(ipc_handle, h.ipc_handle, AMD_IPC_MEM_HANDLE_SIZE);
      if (cmp != 0) return cmp < 0;

      return poffset < h.poffset;
    }
  };
  //!< add the host mem pointer and buffer in the container
  static void AddMemObj(const void* k, amd::Memory* v);

  //!< Remove an entry of mem object from the container
  static void RemoveMemObj(const void* k);

  //!< Find the mem object based on the input pointer, outputs the offset
  static amd::Memory* FindMemObj(const void* k, size_t* offset = nullptr);
  static void UpdateAccess(amd::Device* peerDev);
  //!< Purge all user allocated memories on the given device
  static void Purge(amd::Device* dev);
  //!< Same as AddMemObj but for virtual addressing
  static void AddVirtualMemObj(const void* k, amd::Memory* v);

  //!< Same as RemoveMemObj but for virtual addressing
  static void RemoveVirtualMemObj(const void* k);
  //!< Same as FindMemObj but for virtual addressing
  static amd::Memory* FindVirtualMemObj(const void* k);

  //!< Same as AddMemObj but for virtual ipc handle to MemObj mapping
  static void AddIpcHandleMemObj(const IpcMemHandle& k, amd::Memory* v);
  //!< Remove entry from the map by searching values
  static void RemoveIpcHandleMemObj(amd::Memory* v);
  //!< Same as FindMemObj but for ipc handle to MemObj mapping
  static amd::Memory* FindIpcHandleMemObj(const IpcMemHandle& k);

 private:
  //!< the mem object<->hostptr information container
  static std::map<uintptr_t, amd::Memory*> MemObjMap_;
  //!< the virtual mem object<->hostptr information container
  static std::map<uintptr_t, amd::Memory*> VirtualMemObjMap_;
  //!< Shared read/write lock
  static std::shared_mutex AllocatedLock_;
  //!< the ipc handle<->mem object information container
  static std::map<IpcMemHandle, amd::Memory*> IpcHandleMemObjMap_;
};

/// @brief Instruction Set Architecture properties.
class Isa {
 public:
  /// @brief Isa's target feature setting type.
  enum class Feature : uint8_t {
    Unsupported,
    Any,
    Disabled,
    Enabled,
  };

  //! Return a non-zero uint64_t value that uniquely identifies the device.
  //! This can be used when a scalar value handle to the device is require.
  static uint64_t toHandle(const Isa* isa) {
    static_assert(sizeof(isa) <= sizeof(uint64_t), "Handle size does not match pointer size");
    assert((reinterpret_cast<uint64_t>(static_cast<const Isa*>(nullptr)) == 0) &&
           "nullptr value is not 0");
    return isa ? reinterpret_cast<uint64_t>(isa) : 0;
  }

  //! Return the device corresponding to a handle returned by Isa::handle,
  //! or nullptr if the handle is 0. This can be used when a scalar value
  //! handle for a device is provided.
  static const Isa* fromHandle(uint64_t handle) {
    static_assert(sizeof(handle) <= sizeof(uint64_t), "Handle size does not match pointer size");
    assert((reinterpret_cast<uint64_t>(static_cast<const Isa*>(nullptr)) == 0) &&
           "nullptr value is not 0");
    return handle ? reinterpret_cast<const Isa*>(handle) : nullptr;
  }

  /// @returns This Isa's target triple and target ID name.
  std::string isaName() const;

  /// @returns This Isa's processor name.
  std::string processorName() const;

  /// @returns This Isa's target ID name.
  const char* targetId() const { return targetId_; }

  /// @returns This Isa's name to use with the HSAIL compiler.
  const char* hsailName() const { return hsailId_; }

  /// @returns If the ROCm runtime supports the ISA.
  bool runtimeRocSupported() const {
    if (!IS_HIP && (versionMajor_ == 8)) {
      return false;
    }
    return runtimeRocSupported_;
  }

  /// @returns If the PAL runtime supports the ISA.
  bool runtimePalSupported() const {
    if (IS_LINUX && (GPU_ENABLE_PAL == 2) && (versionMajor_ >= 9)) {
      return false;
    }
    return runtimePalSupported_;
  }

  /// @returns SRAM ECC feature status.
  const Feature& sramecc() const { return sramecc_; }

  /// @returns XNACK feature status.
  const Feature& xnack() const { return xnack_; }

  /// @returns True if SRAMECC feature is supported, false otherwise.
  bool isSrameccSupported() const { return sramecc_ != Feature::Unsupported; }

  /// @returns True if XNACK feature is supported, false otherwise.
  bool isXnackSupported() const { return xnack_ != Feature::Unsupported; }

  /// @returns This Isa's major version.
  uint32_t versionMajor() const { return versionMajor_; }

  /// @returns This Isa's minor version.
  uint32_t versionMinor() const { return versionMinor_; }

  /// @returns This Isa's stepping version.
  uint32_t versionStepping() const { return versionStepping_; }

  /// @returns This Isa's number of SIMDs per CU.
  uint32_t simdPerCU() const { return simdPerCU_; }

  /// @returns This Isa's
  uint32_t simdWidth() const { return simdWidth_; }

  /// @returns This Isa's number of instructions processed per SIMD.
  uint32_t simdInstructionWidth() const { return simdInstructionWidth_; }

  /// @returns This Isa's memory channel bank width.
  uint32_t memChannelBankWidth() const { return memChannelBankWidth_; }

  /// @returns This Isa's local memory size per CU.
  uint32_t localMemSizePerCU() const { return localMemSizePerCU_; }

  /// @returns This Isa's number of banks of local memory.
  uint32_t localMemBanks() const { return localMemBanks_; }

  /// @returns True if @p codeObjectIsa and @p agentIsa are compatible,
  /// false otherwise.
  static bool isCompatible(const Isa& codeObjectIsa, const Isa& agentIsa);

  /// @returns Isa for requested @p isaName, null pointer if not supported.
  static const Isa* findIsa(const char* isaName);

  /// @returns Isa for requested @p version, null pointer if not supported.
  static const Isa* findIsa(uint32_t versionMajor, uint32_t versionMinor, uint32_t versionStepping,
                            Feature sramecc = Feature::Any, Feature xnack = Feature::Any);

  /// @returns Iterator for first isa.
  static const Isa* begin();

  /// @returns Iterator for one past the end isa.
  static const Isa* end();

 private:
  constexpr Isa(const char* targetId, const char* hsailId, bool runtimeRocSupported,
                bool runtimePalSupported, uint32_t versionMajor, uint32_t versionMinor,
                uint32_t versionStepping, Feature sramecc, Feature xnack, uint32_t simdPerCU,
                uint32_t simdWidth, uint32_t simdInstructionWidth, uint32_t memChannelBankWidth,
                uint32_t localMemSizePerCU, uint32_t localMemBanks)
      : targetId_(targetId),
        hsailId_(hsailId),
        runtimeRocSupported_(runtimeRocSupported),
        runtimePalSupported_(runtimePalSupported),
        versionMajor_(versionMajor),
        versionMinor_(versionMinor),
        versionStepping_(versionStepping),
        sramecc_(sramecc),
        xnack_(xnack),
        simdPerCU_(simdPerCU),
        simdWidth_(simdWidth),
        simdInstructionWidth_(simdInstructionWidth),
        memChannelBankWidth_(memChannelBankWidth),
        localMemSizePerCU_(localMemSizePerCU),
        localMemBanks_(localMemBanks) {}

  // @brief Returns the begin and end iterators for the suppported ISAs.
  static std::pair<const Isa*, const Isa*> supportedIsas();

  // @brief Isa's target ID name. Used for LLVM COde Object Manager
  // compilations.
  const char* targetId_;

  // @brief Isa's HSAIL name. Used for the Compiler Library for HSAIL
  // compilation using the Shader Compiler Finalizer. Empty string if
  // unsupported.
  const char* hsailId_;

  bool runtimeRocSupported_;       //!< ROCm runtime is supported.
  bool runtimePalSupported_;       //!< PAL runtime is supported.
  uint32_t versionMajor_;          //!< Isa's major version.
  uint32_t versionMinor_;          //!< Isa's minor version.
  uint32_t versionStepping_;       //!< Isa's stepping version.
  Feature sramecc_;                //!< SRAMECC feature.
  Feature xnack_;                  //!< XNACK feature.
  uint32_t simdPerCU_;             //!< Number of SIMDs per CU.
  uint32_t simdWidth_;             //!< Number of workitems processed per SIMD.
  uint32_t simdInstructionWidth_;  //!< Number of instructions processed per SIMD.
  uint32_t memChannelBankWidth_;   //!< Memory channel bank width.
  uint32_t localMemSizePerCU_;     //!< Local memory size per CU.
  uint32_t localMemBanks_;         //!< Number of banks of local memory.
};  // class Isa

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Device Device Abstraction
 *  @{
 */
class Device : public RuntimeObject {
 protected:
#if defined(WITH_COMPILER_LIB)
  typedef aclCompiler Compiler;
#endif

 public:
  // The structures below for MGPU launch match the device library format
  struct MGSyncData {
    uint32_t w0;
    uint32_t w1;
  };

  struct MGSyncInfo {
    struct MGSyncData* mgs;
    uint32_t grid_id;
    uint32_t num_grids;
    uint64_t prev_sum;
    uint64_t all_sum;
    struct MGSyncData sgs;
    uint num_wg;
  };

  // Attributes that could be retrived from hsa_amd_memory_pool_link_info_t.
  typedef enum LinkAttribute {
    kLinkLinkType = 0,
    kLinkHopCount,
    kLinkDistance,
    kLinkAtomicSupport
  } LinkAttribute;

  typedef enum MemorySegment {
    kNoAtomics = 0,
    kAtomics = 1,
    kKernArg = 2,
    kUncachedAtomics = 4
  } MemorySegment;

  typedef enum CacheState {
    kCacheStateInvalid = -1,
    kCacheStateIgnore = 0,
    kCacheStateAgent = 1,
    kCacheStateSystem = 2
  } CacheState;

  //<! Enum describing the access permissions of Virtual memory
  enum class VmmAccess { kNone = 0x0, kReadOnly = 0x1, kReadWrite = 0x3 };
  //<! Enum describing the location of Virtual memory
  enum class VmmLocationType { kNone = 0x0, kDevice = 0x1, kHost = 0x2 };

  typedef std::pair<LinkAttribute, int32_t /* value */> LinkAttrType;

  static constexpr size_t kP2PStagingSize = 4 * Mi;
  static constexpr size_t kMGSyncDataSize = sizeof(MGSyncData);
  static constexpr size_t kMGInfoSizePerDevice = kMGSyncDataSize + sizeof(MGSyncInfo);
  static constexpr size_t kSGInfoSize = kMGSyncDataSize;

  // Max Scratch size is based on ISA and thus per device.
  // Def value is as per GFX9 being the least among supported devices.
  size_t maxStackSize_ = kMaxStackSize9X;
  static cl_int gpu_error_;  //!< Store the GPU error cause during kernel launch

  typedef std::list<CommandQueue*> CommandQueues;

  struct BlitProgram : public amd::HeapObject {
    Program* program_;  //!< GPU program object
    Context* context_;  //!< A dummy context

    BlitProgram(Context* context) : program_(NULL), context_(context) {}
    ~BlitProgram();

    //! Creates blit program for this device
    bool create(Device* device,                  //!< Device object
                const std::string& extraKernel,  //!< Extra kernels from the device layer
                const std::string& extraOptions  //!< Extra compilation options
    );
  };

#if defined(WITH_COMPILER_LIB)
  virtual Compiler* compiler() const = 0;
  virtual Compiler* binCompiler() const { return compiler(); }
#endif

  Device();
  virtual ~Device();

  //! Initializes abstraction layer device object
  bool create(const Isa& isa);

  uint retain() {
    // Overwrite the RuntimeObject::retain().
    // There is an issue in the old SHOC11_DeviceMemory test on TC
    return 0u;
  }

  uint release() {
    // Overwrite the RuntimeObject::release().
    // There is an issue in the old SHOC11_DeviceMemory test on TC
    return 0u;
  }

  //! Register a device as available
  void registerDevice();

  //! Initialize the device layer (enumerate known devices)
  static bool init();

  //! Shutdown the device layer
  static void tearDown();

  static std::vector<Device*> getDevices(cl_device_type type,  //!< Device type
                                         bool offlineDevices   //!< Enable offline devices
  );

  static size_t numDevices(cl_device_type type,  //!< Device type
                           bool offlineDevices   //!< Enable offline devices
  );

  static bool getDeviceIDs(cl_device_type deviceType,  //!< Device type
                           uint32_t numEntries,        //!< Number of entries in the array
                           cl_device_id* devices,      //!< Array of the device ID(s)
                           uint32_t* numDevices,       //!< Number of available devices
                           bool offlineDevices         //!< Report offline devices
  );

  const device::Info& info() const { return info_; }

  //! Return svm support capability.
  bool svmSupport() const {
    return (info().svmCapabilities_ &
            (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
             CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) != 0
               ? true
               : false;
  }

  //! check svm FGS support capability.
  inline bool isFineGrainedSystem(bool FGSOPT = false) const {
    return FGSOPT && (info().svmCapabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0 ? true
                                                                                      : false;
  }

  //! Return this device's type.
  cl_device_type type() const { return info().type_ & ~(CL_DEVICE_TYPE_DEFAULT); }

  //! Create a new virtual device environment.
  virtual device::VirtualDevice* createVirtualDevice(CommandQueue* queue = NULL) = 0;

  //! Create a program for device.
  virtual device::Program* createProgram(amd::Program& owner, option::Options* options = NULL) = 0;

  //! Allocate a chunk of device memory as a cache for a CL memory object
  virtual device::Memory* createMemory(Memory& owner) const = 0;

  //! Allocate a chunk of device memory with address alignment
  virtual device::Memory* createMemory(size_t size, size_t alignment = 0) const = 0;

  //! Allocate a device sampler object
  virtual bool createSampler(const Sampler&, device::Sampler**) const = 0;

  //! Allocates a view object from the device memory
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const = 0;

  ///! Allocates a device signal object
  virtual device::Signal* createSignal() const = 0;

  //! Return true if initialized external API interop, otherwise false
  virtual bool bindExternalDevice(
      uint flags,             //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const pDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* pContext,         //!< HGLRC/GLXContext handle
      bool validateOnly  //! Only validate if the device can inter-operate with pDevice/pContext, do
                         //! not bind.
      ) = 0;

  virtual bool unbindExternalDevice(
      uint flags,             //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const pDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* pContext,         //!< HGLRC/GLXContext handle
      bool validateOnly  //! Only validate if the device can inter-operate with pDevice/pContext, do
                         //! not bind.
      ) = 0;

  //! resolves GL depth/msaa buffer
  virtual bool resolveGLMemory(device::Memory*) const { return true; }

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory  //!< Free memory information on a GPU device
  ) const = 0;

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
                       uint64_t* size_copied, int32_t* status) = 0;

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
                       uint64_t* size_copied, int32_t* status) = 0;

  virtual bool importExtSemaphore(void** extSemaphore, const amd::Os::FileDesc& handle,
                                  amd::ExternalSemaphoreHandleType sem_handle_type) = 0;
  virtual void DestroyExtSemaphore(void* extSemaphore) = 0;
  /**
   * @return True if the device has its own custom host allocator to be used
   * instead of the generic OS allocation routines
   */
  bool customHostAllocator() const { return settings().customHostAllocator_ == 1; }

  /**
   * @copydoc amd::Context::hostAlloc
   */
  virtual void* hostAlloc(size_t size, size_t alignment,
                          MemorySegment mem_seg = kNoAtomics,
                          const void* agentInfo = nullptr) const {
    ShouldNotCallThis();
    return NULL;
  }

  //! Flags for deviceLocalAlloc method
  typedef union {
    struct {
      uint32_t atomics_ : 1;           //!< True if atomics support is required
      uint32_t pseudo_fine_grain_ : 1; //!< True if pseudo fine grain memory is required
      uint32_t contiguous_ : 1;        //!< True if contiguous memory allocation is required
      uint32_t executable_ : 1;        //!< True if executable memory is required
      uint32_t reserved_ : 28;         //!< Reserved for future use
    };
    uint32_t data_;
  } AllocationFlags;

  virtual void* deviceLocalAlloc(
      size_t size, const AllocationFlags& flags = AllocationFlags{}) const {
    ShouldNotCallThis();
    return NULL;
  }

  virtual bool isXgmi() const {
    ShouldNotCallThis();
    return false;
  }

  virtual bool deviceAllowAccess(void* dst) const {
    ShouldNotCallThis();
    return true;
  }

  virtual bool allowPeerAccess(device::Memory* memory) const {
    ShouldNotCallThis();
    return true;
  }

  bool enableP2P(amd::Device* ptrDev);

  bool disableP2P(amd::Device* ptrDev);

  /**
   * @copydoc amd::Context::hostFree
   */
  virtual void hostFree(void* ptr, size_t size = 0) const { ShouldNotCallThis(); }

  /**
   * @copydoc amd::Context::svmAlloc
   */
  virtual void* svmAlloc(Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                         void* svmPtr) const = 0;
  /**
   * @copydoc amd::Context::svmFree
   */
  virtual void svmFree(void* ptr) const = 0;

  /**
   * Validatates Virtual Address range between parent and sub-buffer.
   *
   * @param vaddr_base_obj Parent/base object of the virtual address.
   * @param vaddr_sub_obj Sub Buffer object of the virtual address.
   */
  static bool ValidateVirtualAddressRange(amd::Memory* vaddr_base_obj, amd::Memory* vaddr_sub_obj);

  /**
   * Abstracts the Virtual Buffer creation and memobj/virtual memobj add/delete logic.
   *
   * @param device_context Context the virtual buffer should be created.
   * @param vptr virtual ptr to store in the buffer object.
   * @param size Size of the buffer
   * @param deviceId deviceId
   * @param parent base_obj or sub_obj
   * @param ForceAlloc force_alloc
   */
  amd::Memory* CreateVirtualBuffer(Context& device_context, void* vptr, size_t size, int deviceId,
                                   int locationType, bool parent, bool kForceAlloc = false);

  /**
   * Deletes Virtual Buffer and creates memob
   *
   * @param vaddr_mem_obj amd::Memory object of parent/sub buffer.
   */
  bool DestroyVirtualBuffer(amd::Memory* vaddr_mem_obj);

  /**
   * Reserve a VA range with no backing store
   *
   * @param addr Start address requested
   * @param size Size of the range in bytes
   * @param alignment Alignment in bytes
   */
  virtual void* virtualAlloc(void* addr, size_t size, size_t alignment) = 0;

  /**
   * Set Access permisions for a virtual memory object.
   *
   * @param va_addr Virtual Address ptr
   * @param va_size Virtual Address Size
   * @param access_flags Access permissions
   * @param count Number of access permissions
   */
  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice) = 0;

  /**
   * Get Access permisions for a virtual memory object.
   *
   * @param va_addr Virtual Address ptr
   * @param access_flags_ptr Access permissions to be filled
   */
  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const = 0;

  /**
   * Validate Access permisions for a virtual memory object.
   *
   * @param va_addr Virtual Address ptr
   * @param access_flags_ptr Access permissions to be filled
   */
  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const = 0;

  /**
   * Free a VA range
   *
   * @param addr Start address of the range
   */
  virtual bool virtualFree(void* addr) = 0;

  /**
   * Export Shareable VMM Handle to FD
   *
   * @param amd_mem_obj amd::Memory obj which holds the hsa_handle.
   * @param flags any flags to be passed
   * @param shareableHandle exported handle, points to fdesc.
   */
  virtual bool ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags,
                                        void* shareableHandle) {
    ShouldNotCallThis();
    return false;
  }

  /**
   * Import FD from Shareable VMM Handle
   *
   * @param osHandle os handle/fdesc/void*
   * @param amd_mem_obj amd_mem_obj with hsa_handle/memory_obj.
   */
  virtual amd::Memory* ImportShareableVMMHandle(void* osHandle) {
    ShouldNotCallThis();
    return nullptr;
  }

  /**
   * @return True if the device successfully applied the SVM attributes in HMM for device memory
   */
  virtual bool SetSvmAttributes(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                                bool use_cpu = false, int numa_id = kDefaultNumaNode) const {
    ShouldNotCallThis();
    return false;
  }

  /**
   * @return True if the device successfully retrieved the SVM attributes from HMM for device memory
   */
  virtual bool GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                                size_t num_attributes, const void* dev_ptr, size_t count) const {
    ShouldNotCallThis();
    return false;
  }

  //! Returns current scratch limit of the device. Valid only on rocm device.
  virtual size_t ScratchLimitCurrent() const { return 0; }

  //! Sets the current scratch limit of the device. Valid only on rocm device.
  virtual bool UpdateScratchLimitCurrent(size_t limit) const { return true; }

  //! Validate kernel
  virtual bool validateKernel(const amd::Kernel& kernel, const device::VirtualDevice* vdev,
                              bool coop_groups = false) {
    return true;
  };

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
    return true;
  };

  // Returns the status of HW event, associated with amd::Event
  virtual bool IsHwEventReady(const amd::Event& event,  //!< AMD event for HW status validation
                              bool wait = false,  //!< If true then forces the event completion
                              amd::SyncPolicy policy = amd::SyncPolicy::Auto) const {
    return false;
  };

  virtual void getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const {};

  virtual const uint32_t getPreferredNumaNode() const { return 0; }
  virtual void ReleaseGlobalSignal(void* signal) const {}
  virtual const bool isFineGrainSupported() const {
    return (info().svmCapabilities_ & CL_DEVICE_SVM_ATOMICS) != 0 ? true : false;
  }

  /// @brief  Creates HW user event for OpenCL implementation
  /// @return The pointer to a HW event structure, known to the HW backend
  virtual bool CreateUserEvent(amd::UserEvent* event) const { return false; }

  /// @brief  Sets HW user event to the complete status
  virtual void SetUserEvent(amd::UserEvent* event) const {}

  //! Returns TRUE if the device is available for computations
  bool isOnline() const { return online_; }

  //! Returns device isa.
  const Isa& isa() const {
    assert(isa_);
    return *isa_;
  }

  //! Return a non-zero uint64_t value that uniquely identifies the device.
  //! This can be used when a scalar value handle to the device is require.
  static uint64_t toHandle(const Device* device) {
    static_assert(sizeof(device) <= sizeof(uint64_t), "Handle size does not match pointer size");
    assert((reinterpret_cast<uint64_t>(static_cast<const Device*>(nullptr)) == 0) &&
           "nullptr value is not 0");
    return device ? reinterpret_cast<uint64_t>(device) : 0;
  }

  //! Return the device corresponding to a handle returned by Device::handle,
  //! or nullptr if the handle is 0. This can be used when a scalar value
  //! handle for a device is provided.
  static const Device* fromHhandle(uint64_t handle) {
    static_assert(sizeof(handle) <= sizeof(uint64_t), "Handle size does not match pointer size");
    assert((reinterpret_cast<uint64_t>(static_cast<const Device*>(nullptr)) == 0) &&
           "nullptr value is not 0");
    return handle ? reinterpret_cast<const Device*>(handle) : nullptr;
  }

  //! Returns device settings
  const device::Settings& settings() const { return *settings_; }

  //! Returns blit program info structure
  BlitProgram* blitProgram() const { return blitProgram_; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeDevice; }

  //! Returns app profile
  static const AppProfile* appProfile() { return &appProfile_; }

  //! Adds GPU memory to the VA cache list
  void addVACache(device::Memory* memory) const;

  //! Removes GPU memory from the VA cache list
  void removeVACache(const device::Memory* memory) const;

  //! Finds GPU memory from virtual address
  device::Memory* findMemoryFromVA(const void* ptr, size_t* offset) const;

  static std::vector<Device*>& devices() { return *devices_; }

  // P2P devices that are accessible from the current device
  std::vector<cl_device_id> p2pDevices_;

  // P2P devices for memory allocation. This list contains devices that can have access to the
  // current device
  std::vector<Device*> p2p_access_devices_;

  //! Checks if OCL runtime can use code object manager for compilation
  bool ValidateComgr();

  //! Checks if OCL runtime can use hsail for compilation
  bool ValidateHsail();

  bool IpcCreate(void* dev_ptr, size_t* mem_size, char* handle, size_t* mem_offset) const;

  bool IpcAttach(const char* handle, size_t mem_size, size_t mem_offset, unsigned int flags,
                 void** dev_ptr) const;

  void IpcDetach(amd::Memory* amd_mem_obj) const;

  //! Return context
  amd::Context& context() const { return *context_; }

  //! Return private global device context for P2P allocations
  amd::Context& GlbCtx() const { return *glb_ctx_; }

  //! Lock protect P2P staging operations
  Monitor& P2PStageOps() const { return p2p_stage_ops_; }

  //! Staging buffer for P2P transfer
  Memory* P2PStage() const { return p2p_stage_; }

  //! Returns heap buffer object for device allocator
  device::Memory* HeapBuffer() const { return heap_buffer_; }

  //! Returns stack size set for the device
  uint64_t StackSize() const { return stack_size_; }

  //! Sets the stack size of the device
  bool UpdateStackSize(uint64_t stackSize);

  //! Returns initial heap size
  uint64_t InitialHeapSize() const { return initial_heap_size_; }

  //! Sets the heap size of the device
  bool UpdateInitialHeapSize(uint64_t initialHeapSize);

  //! Does this device allow P2P access?
  bool P2PAccessAllowed() const { return (p2p_access_devices_.size() > 0) ? true : false; }

  //! Returns the list of devices that can have access to the current
  const std::vector<Device*>& P2PAccessDevices() const { return p2p_access_devices_; }

  //! Returns index of current device
  uint32_t index() const { return index_; }

  //! Returns value for LinkAttribute for lost of vectors
  virtual bool findLinkInfo(const amd::Device& other_device, std::vector<LinkAttrType>* link_attr) {
    return false;
  }

  //! Returns the queues that have at least one submitted command
  std::vector<amd::CommandQueue*> getActiveQueues();

  //! Adds the queue to the set of active command queues
  void addToActiveQueues(amd::CommandQueue* commandQueue) {
    amd::ScopedLock lock(activeQueuesLock_);
    activeQueues.insert(commandQueue);
  }

  //! Removes the queue from the set of active command queues
  void removeFromActiveQueues(amd::CommandQueue* commandQueue) {
    amd::ScopedLock lock(activeQueuesLock_);
    activeQueues.erase(commandQueue);
  }

  // Notifies device about context destroy
  virtual void ContextDestroy() {}

  //! Returns active wait state for this device
  bool ActiveWait() const { return activeWait_; }

  void SetActiveWait(bool state) { activeWait_ = state; }

  virtual amd::Memory* GetArenaMemObj(const void* ptr, size_t& offset, size_t size = 0) {
    return nullptr;
  }

  //! Returns stack size set for the device
  size_t MaxStackSize() const { return maxStackSize_; }

#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUriLocator() const = 0;
#endif
#endif

  static bool IsGPUInError() { return (gpu_error_ != CL_SUCCESS); }
  static cl_int GetGPUError() { return gpu_error_; }

  bool GetHandleForAddressRange(void* dev_ptr, size_t size, void* handle);

  // Registers a memory object allocated via hostcall for later cleanup.
  void TrackHostcallMemory(amd::Memory* memory);

  // Removes a memory object from hostcall tracking.
  void RemoveHostcallMemory(amd::Memory* memory);

  //! Enable the specified extension
  char* getExtensionString();

  device::Info info_;           //!< Device info structure
  device::Settings* settings_;  //!< Device settings
  union {
    struct {
      uint32_t online_ : 1;      //!< The device in online
      uint32_t activeWait_ : 1;  //!< If true device requires active wait
    };
    uint32_t state_;  //!< State bit mask
  };

  BlitProgram* blitProgram_;      //!< Blit program info
  static AppProfile appProfile_;  //!< application profile
  amd::Context* context_;         //!< Context

  static amd::Context* glb_ctx_;              //!< Global context with all devices
  static amd::Monitor p2p_stage_ops_;         //!< Lock to serialise cache for the P2P resources
  static Memory* p2p_stage_;                  //!< Staging resources
  std::vector<Device*> enabled_p2p_devices_;  //!< List of user enabled P2P devices for this device

  std::once_flag heap_initialized_;  //!< Heap buffer initialization flag
  std::once_flag heap_allocated_;    //!< Heap buffer allocation flag

  device::Memory* heap_buffer_;  //!< Preallocated heap buffer for memory allocations on device

  amd::Memory* arena_mem_obj_;                          //!< Arena memory object
  uint64_t stack_size_{1024};                           //!< Device stack size
  device::Memory* initial_heap_buffer_;                 //!< Initial heap buffer
  uint64_t initial_heap_size_{HIP_INITIAL_DM_SIZE};     //!< Initial device heap size
  amd::Monitor activeQueuesLock_{};                     //!< Guards access to the activeQueues set
  std::unordered_set<amd::CommandQueue*> activeQueues;  //!< The set of active queues

 private:
  const Isa* isa_;  //!< Device isa
  bool IsTypeMatching(cl_device_type type, bool offlineDevices);

#if defined(WITH_HSA_DEVICE)
  static AppProfile* rocAppProfile_;
#endif

  static std::vector<Device*>* devices_;  //!< All known devices
  static amd::Monitor lockP2P_;
  Monitor* vaCacheAccess_;                            //!< Lock to serialize VA caching access
  std::map<uintptr_t, device::Memory*>* vaCacheMap_;  //!< VA cache map
  uint32_t index_;                                    //!< Unique device index
  static constexpr int kDefaultNumaNode = -1;         //! Default NUMA node value for SVM operations
  // Tracks all amd::Memory objects allocated via hostcall for this device.
  std::vector<amd::Memory*> hostcall_allocated_memories_;
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*DEVICE_HPP_*/
