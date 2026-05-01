// rocclr/include/amd_types.hpp
// rocclr-owned type definitions — no CL or HIP headers.
// Numeric values match CL equivalents exactly; verified by cl_type_map.hpp.
#pragma once
#include <cstdint>
#include <cstddef>

namespace amd {

// Replaces cl_device_type (cl_bitfield = uint64_t)
enum class DeviceType : uint64_t {
  Default     = (1u << 0),   // CL_DEVICE_TYPE_DEFAULT
  CPU         = (1u << 1),   // CL_DEVICE_TYPE_CPU
  GPU         = (1u << 2),   // CL_DEVICE_TYPE_GPU
  Accelerator = (1u << 3),   // CL_DEVICE_TYPE_ACCELERATOR
  Custom      = (1u << 4),   // CL_DEVICE_TYPE_CUSTOM
  All         = 0xFFFFFFFFu, // CL_DEVICE_TYPE_ALL
};
inline DeviceType operator|(DeviceType a, DeviceType b) {
  return static_cast<DeviceType>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline DeviceType& operator|=(DeviceType& a, DeviceType b) {
  a = a | b;
  return a;
}
inline DeviceType operator&(DeviceType a, DeviceType b) {
  return static_cast<DeviceType>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline DeviceType operator~(DeviceType a) {
  return static_cast<DeviceType>(~static_cast<uint64_t>(a));
}

// Replaces cl_device_fp_config (cl_bitfield = uint64_t)
enum class FpConfig : uint64_t {
  Denorm                 = (1u << 0), // CL_FP_DENORM
  InfNan                 = (1u << 1), // CL_FP_INF_NAN
  RoundToNearest         = (1u << 2), // CL_FP_ROUND_TO_NEAREST
  RoundToZero            = (1u << 3), // CL_FP_ROUND_TO_ZERO
  RoundToInf             = (1u << 4), // CL_FP_ROUND_TO_INF
  Fma                    = (1u << 5), // CL_FP_FMA
  SoftFloat              = (1u << 6), // CL_FP_SOFT_FLOAT
  CorrectlyRounded       = (1u << 7), // CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT
};
inline FpConfig operator|(FpConfig a, FpConfig b) {
  return static_cast<FpConfig>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline FpConfig operator&(FpConfig a, FpConfig b) {
  return static_cast<FpConfig>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline FpConfig operator~(FpConfig a) {
  return static_cast<FpConfig>(~static_cast<uint64_t>(a));
}
inline FpConfig& operator|=(FpConfig& a, FpConfig b) {
  a = a | b;
  return a;
}
inline bool operator!(FpConfig a) {
  return static_cast<uint64_t>(a) == 0;
}

// Replaces cl_device_mem_cache_type (cl_uint)
enum class MemCacheType : uint32_t {
  None           = 0, // CL_NONE
  ReadOnlyCache  = 1, // CL_READ_ONLY_CACHE
  ReadWriteCache = 2, // CL_READ_WRITE_CACHE
};

// Replaces cl_device_local_mem_type (cl_uint)
enum class LocalMemType : uint32_t {
  None   = 0, // CL_NONE
  Local  = 1, // CL_LOCAL
  Global = 2, // CL_GLOBAL
};

// Replaces cl_device_exec_capabilities (cl_bitfield = uint64_t)
enum class ExecCapabilities : uint64_t {
  Kernel       = (1u << 0), // CL_EXEC_KERNEL
  NativeKernel = (1u << 1), // CL_EXEC_NATIVE_KERNEL
};
inline ExecCapabilities operator|(ExecCapabilities a, ExecCapabilities b) {
  return static_cast<ExecCapabilities>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline ExecCapabilities operator&(ExecCapabilities a, ExecCapabilities b) {
  return static_cast<ExecCapabilities>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline ExecCapabilities operator~(ExecCapabilities a) {
  return static_cast<ExecCapabilities>(~static_cast<uint64_t>(a));
}

// Replaces cl_device_svm_capabilities (cl_bitfield = uint64_t)
enum class SvmCapabilities : uint64_t {
  CoarseGrainBuffer = (1u << 0), // CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
  FineGrainBuffer   = (1u << 1), // CL_DEVICE_SVM_FINE_GRAIN_BUFFER
  FineGrainSystem   = (1u << 2), // CL_DEVICE_SVM_FINE_GRAIN_SYSTEM
  Atomics           = (1u << 3), // CL_DEVICE_SVM_ATOMICS
};
inline SvmCapabilities operator|(SvmCapabilities a, SvmCapabilities b) {
  return static_cast<SvmCapabilities>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline SvmCapabilities operator&(SvmCapabilities a, SvmCapabilities b) {
  return static_cast<SvmCapabilities>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline SvmCapabilities operator~(SvmCapabilities a) {
  return static_cast<SvmCapabilities>(~static_cast<uint64_t>(a));
}
inline SvmCapabilities& operator|=(SvmCapabilities& a, SvmCapabilities b) {
  a = a | b;
  return a;
}
inline bool operator!(SvmCapabilities a) {
  return static_cast<uint64_t>(a) == 0;
}

// Replaces cl_command_queue_properties (cl_bitfield = uint64_t)
enum class QueueProperties : uint64_t {
  None                  = 0,        // no flags
  OutOfOrderExec        = (1u << 0), // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  Profiling             = (1u << 1), // CL_QUEUE_PROFILING_ENABLE
  OnDevice              = (1u << 2), // CL_QUEUE_ON_DEVICE
  OnDeviceDefault       = (1u << 3), // CL_QUEUE_ON_DEVICE_DEFAULT
};
inline QueueProperties operator|(QueueProperties a, QueueProperties b) {
  return static_cast<QueueProperties>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline QueueProperties operator&(QueueProperties a, QueueProperties b) {
  return static_cast<QueueProperties>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline QueueProperties operator~(QueueProperties a) {
  return static_cast<QueueProperties>(~static_cast<uint64_t>(a));
}
inline QueueProperties& operator|=(QueueProperties& a, QueueProperties b) {
  a = a | b;
  return a;
}
inline QueueProperties& operator&=(QueueProperties& a, QueueProperties b) {
  a = a & b;
  return a;
}

// Replaces cl_mem_flags / cl_svm_mem_flags (cl_bitfield = uint64_t)
// Upper bits are rocclr-internal extensions (no CL equivalent).
enum class MemFlags : uint64_t {
  Empty              = 0,          // no flags set (avoid X11 "None" macro collision)
  // Standard CL_MEM_* flags
  ReadWrite          = (1u << 0),  // CL_MEM_READ_WRITE
  WriteOnly          = (1u << 1),  // CL_MEM_WRITE_ONLY
  ReadOnly           = (1u << 2),  // CL_MEM_READ_ONLY
  UseHostPtr         = (1u << 3),  // CL_MEM_USE_HOST_PTR
  AllocHostPtr       = (1u << 4),  // CL_MEM_ALLOC_HOST_PTR
  CopyHostPtr        = (1u << 5),  // CL_MEM_COPY_HOST_PTR
  UsePersistentMemAmd = (1u << 6), // CL_MEM_USE_PERSISTENT_MEM_AMD (AMD extension)
  HostWriteOnly      = (1u << 7),  // CL_MEM_HOST_WRITE_ONLY
  HostReadOnly       = (1u << 8),  // CL_MEM_HOST_READ_ONLY
  HostNoAccess       = (1u << 9),  // CL_MEM_HOST_NO_ACCESS
  SvmFineGrain       = (1u << 10), // CL_MEM_SVM_FINE_GRAIN_BUFFER
  SvmAtomics         = (1u << 11), // CL_MEM_SVM_ATOMICS
  KernelReadWrite    = (1u << 12), // CL_MEM_KERNEL_READ_AND_WRITE
  // rocclr-internal flags (upper bits, no CL equivalent)
  IoMemory           = (uint64_t(1) << 23), // ROCCLR_MEM_IO_MEMORY
  HsaContiguous      = (uint64_t(1) << 24), // ROCCLR_MEM_HSA_CONTIGUOUS
  PhyMem             = (uint64_t(1) << 25), // ROCCLR_MEM_PHYMEM
  Interprocess       = (uint64_t(1) << 26), // ROCCLR_MEM_INTERPROCESS
  HsaUncached        = (uint64_t(1) << 27), // ROCCLR_MEM_HSA_UNCACHED
  VaRangeAmd         = (uint64_t(1) << 28), // CL_MEM_VA_RANGE_AMD (AMD extension)
  InternalMemory     = (uint64_t(1) << 29), // ROCCLR_MEM_INTERNAL_MEMORY
  HsaSignalMemory      = (uint64_t(1) << 30), // ROCCLR_MEM_HSA_SIGNAL_MEMORY
  BusAddressable       = (uint64_t(1) << 30), // CL_MEM_BUS_ADDRESSABLE_AMD — intentional alias of HsaSignalMemory
  FollowUserNumaPolicy = (uint64_t(1) << 31), // CL_MEM_FOLLOW_USER_NUMA_POLICY (AMD extension)
  ExternalPhysical     = (uint64_t(1) << 31), // CL_MEM_EXTERNAL_PHYSICAL_AMD — intentional alias of FollowUserNumaPolicy
};
// Verify CL extension flag values match our enum aliases.
static_assert(static_cast<uint64_t>(MemFlags::BusAddressable) == (uint64_t(1) << 30),
              "BusAddressable must match CL_MEM_BUS_ADDRESSABLE_AMD (1<<30)");
static_assert(static_cast<uint64_t>(MemFlags::ExternalPhysical) == (uint64_t(1) << 31),
              "ExternalPhysical must match CL_MEM_EXTERNAL_PHYSICAL_AMD (1<<31)");
inline MemFlags operator|(MemFlags a, MemFlags b) {
  return static_cast<MemFlags>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline MemFlags operator&(MemFlags a, MemFlags b) {
  return static_cast<MemFlags>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline MemFlags operator~(MemFlags a) {
  return static_cast<MemFlags>(~static_cast<uint64_t>(a));
}
inline MemFlags& operator|=(MemFlags& a, MemFlags b) {
  a = a | b;
  return a;
}
inline MemFlags& operator&=(MemFlags& a, MemFlags b) {
  a = a & b;
  return a;
}

// Replaces cl_gl_object_type (cl_uint) from CL/cl_gl.h
enum class GlObjectType : uint32_t {
  Buffer        = 0x2000, // CL_GL_OBJECT_BUFFER
  Texture2D     = 0x2001, // CL_GL_OBJECT_TEXTURE2D
  Texture3D     = 0x2002, // CL_GL_OBJECT_TEXTURE3D
  Renderbuffer  = 0x2003, // CL_GL_OBJECT_RENDERBUFFER
  Texture2DArray= 0x200E, // CL_GL_OBJECT_TEXTURE2D_ARRAY
  Texture1D     = 0x200F, // CL_GL_OBJECT_TEXTURE1D
  Texture1DArray= 0x2010, // CL_GL_OBJECT_TEXTURE1D_ARRAY
  TextureBuffer = 0x2011, // CL_GL_OBJECT_TEXTURE_BUFFER
};

// Replaces cl_mem_object_type (cl_uint)
enum class MemObjectType : uint32_t {
  Buffer        = 0x10F0, // CL_MEM_OBJECT_BUFFER
  Image2D       = 0x10F1, // CL_MEM_OBJECT_IMAGE2D
  Image3D       = 0x10F2, // CL_MEM_OBJECT_IMAGE3D
  Image2DArray  = 0x10F3, // CL_MEM_OBJECT_IMAGE2D_ARRAY
  Image1D       = 0x10F4, // CL_MEM_OBJECT_IMAGE1D
  Image1DArray  = 0x10F5, // CL_MEM_OBJECT_IMAGE1D_ARRAY
  Image1DBuffer = 0x10F6, // CL_MEM_OBJECT_IMAGE1D_BUFFER
  Pipe          = 0x10F7, // CL_MEM_OBJECT_PIPE
};

// Replaces cl_channel_order (cl_uint)
enum class ChannelOrder : uint32_t {
  R            = 0x10B0, // CL_R
  A            = 0x10B1, // CL_A
  RG           = 0x10B2, // CL_RG
  RA           = 0x10B3, // CL_RA
  RGB          = 0x10B4, // CL_RGB
  RGBA         = 0x10B5, // CL_RGBA
  BGRA         = 0x10B6, // CL_BGRA
  ARGB         = 0x10B7, // CL_ARGB
  Intensity    = 0x10B8, // CL_INTENSITY
  Luminance    = 0x10B9, // CL_LUMINANCE
  Rx           = 0x10BA, // CL_Rx
  RGx          = 0x10BB, // CL_RGx
  RGBx         = 0x10BC, // CL_RGBx
  Depth        = 0x10BD, // CL_DEPTH
  DepthStencil = 0x10BE, // CL_DEPTH_STENCIL
  sRGB         = 0x10BF, // CL_sRGB
  sRGBx        = 0x10C0, // CL_sRGBx
  sRGBA        = 0x10C1, // CL_sRGBA
  sBGRA        = 0x10C2, // CL_sBGRA
  ABGR         = 0x10C3, // CL_ABGR
};

// Replaces cl_channel_type (cl_uint)
enum class ChannelDataType : uint32_t {
  SNormInt8       = 0x10D0, // CL_SNORM_INT8
  SNormInt16      = 0x10D1, // CL_SNORM_INT16
  UNormInt8       = 0x10D2, // CL_UNORM_INT8
  UNormInt16      = 0x10D3, // CL_UNORM_INT16
  UNormShort565   = 0x10D4, // CL_UNORM_SHORT_565
  UNormShort555   = 0x10D5, // CL_UNORM_SHORT_555
  UNormInt101010  = 0x10D6, // CL_UNORM_INT_101010
  SignedInt8      = 0x10D7, // CL_SIGNED_INT8
  SignedInt16     = 0x10D8, // CL_SIGNED_INT16
  SignedInt32     = 0x10D9, // CL_SIGNED_INT32
  UnsignedInt8    = 0x10DA, // CL_UNSIGNED_INT8
  UnsignedInt16   = 0x10DB, // CL_UNSIGNED_INT16
  UnsignedInt32   = 0x10DC, // CL_UNSIGNED_INT32
  HalfFloat       = 0x10DD, // CL_HALF_FLOAT
  Float           = 0x10DE, // CL_FLOAT
  UNormInt24      = 0x10DF, // CL_UNORM_INT24
  UNormInt101010_2 = 0x10E0, // CL_UNORM_INT_101010_2
};

// Replaces cl_image_format (struct)
struct ImageFormat {
  ChannelOrder    channelOrder;    // replaces image_channel_order
  ChannelDataType channelDataType; // replaces image_channel_data_type

  bool operator==(const ImageFormat& o) const {
    return channelOrder == o.channelOrder && channelDataType == o.channelDataType;
  }
};

// Replaces cl_int error codes. Values match CL error codes exactly.
enum class Status : int32_t {
  Success                         =   0, // CL_SUCCESS
  DeviceNotFound                  =  -1, // CL_DEVICE_NOT_FOUND
  DeviceNotAvailable              =  -2, // CL_DEVICE_NOT_AVAILABLE
  CompilerNotAvailable            =  -3, // CL_COMPILER_NOT_AVAILABLE
  MemObjectAllocationFailure      =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
  OutOfResources                  =  -5, // CL_OUT_OF_RESOURCES
  OutOfHostMemory                 =  -6, // CL_OUT_OF_HOST_MEMORY
  ProfilingInfoNotAvailable       =  -7, // CL_PROFILING_INFO_NOT_AVAILABLE
  MemCopyOverlap                  =  -8, // CL_MEM_COPY_OVERLAP
  ImageFormatMismatch             =  -9, // CL_IMAGE_FORMAT_MISMATCH
  ImageFormatNotSupported         = -10, // CL_IMAGE_FORMAT_NOT_SUPPORTED
  BuildProgramFailure             = -11, // CL_BUILD_PROGRAM_FAILURE
  MapFailure                      = -12, // CL_MAP_FAILURE
  MisalignedSubBufferOffset       = -13, // CL_MISALIGNED_SUB_BUFFER_OFFSET
  ExecStatusErrorForEventsInWaitList = -14, // CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
  CompileProgramFailure           = -15, // CL_COMPILE_PROGRAM_FAILURE
  LinkerNotAvailable              = -16, // CL_LINKER_NOT_AVAILABLE
  LinkProgramFailure              = -17, // CL_LINK_PROGRAM_FAILURE
  DevicePartitionFailed           = -18, // CL_DEVICE_PARTITION_FAILED
  KernelArgInfoNotAvailable       = -19, // CL_KERNEL_ARG_INFO_NOT_AVAILABLE
  InvalidValue                    = -30, // CL_INVALID_VALUE
  InvalidDeviceType               = -31, // CL_INVALID_DEVICE_TYPE
  InvalidPlatform                 = -32, // CL_INVALID_PLATFORM
  InvalidDevice                   = -33, // CL_INVALID_DEVICE
  InvalidContext                  = -34, // CL_INVALID_CONTEXT
  InvalidQueueProperties          = -35, // CL_INVALID_QUEUE_PROPERTIES
  InvalidCommandQueue             = -36, // CL_INVALID_COMMAND_QUEUE
  InvalidHostPtr                  = -37, // CL_INVALID_HOST_PTR
  InvalidMemObject                = -38, // CL_INVALID_MEM_OBJECT
  InvalidImageFormatDescriptor    = -39, // CL_INVALID_IMAGE_FORMAT_DESCRIPTOR
  InvalidImageSize                = -40, // CL_INVALID_IMAGE_SIZE
  InvalidSampler                  = -41, // CL_INVALID_SAMPLER
  InvalidBinary                   = -42, // CL_INVALID_BINARY
  InvalidBuildOptions             = -43, // CL_INVALID_BUILD_OPTIONS
  InvalidProgram                  = -44, // CL_INVALID_PROGRAM
  InvalidProgramExecutable        = -45, // CL_INVALID_PROGRAM_EXECUTABLE
  InvalidKernelName               = -46, // CL_INVALID_KERNEL_NAME
  InvalidKernelDefinition         = -47, // CL_INVALID_KERNEL_DEFINITION
  InvalidKernel                   = -48, // CL_INVALID_KERNEL
  InvalidArgIndex                 = -49, // CL_INVALID_ARG_INDEX
  InvalidArgValue                 = -50, // CL_INVALID_ARG_VALUE
  InvalidArgSize                  = -51, // CL_INVALID_ARG_SIZE
  InvalidKernelArgs               = -52, // CL_INVALID_KERNEL_ARGS
  InvalidWorkDimension            = -53, // CL_INVALID_WORK_DIMENSION
  InvalidWorkGroupSize            = -54, // CL_INVALID_WORK_GROUP_SIZE
  InvalidWorkItemSize             = -55, // CL_INVALID_WORK_ITEM_SIZE
  InvalidGlobalOffset             = -56, // CL_INVALID_GLOBAL_OFFSET
  InvalidEventWaitList            = -57, // CL_INVALID_EVENT_WAIT_LIST
  InvalidEvent                    = -58, // CL_INVALID_EVENT
  InvalidOperation                = -59, // CL_INVALID_OPERATION
  InvalidGlObject                 = -60, // CL_INVALID_GL_OBJECT
  InvalidBufferSize               = -61, // CL_INVALID_BUFFER_SIZE
  InvalidMipLevel                 = -62, // CL_INVALID_MIP_LEVEL
  InvalidGlobalWorkSize           = -63, // CL_INVALID_GLOBAL_WORK_SIZE
  InvalidProperty                 = -64, // CL_INVALID_PROPERTY
  InvalidImageDescriptor          = -65, // CL_INVALID_IMAGE_DESCRIPTOR
  InvalidCompilerOptions          = -66, // CL_INVALID_COMPILER_OPTIONS
  InvalidLinkerOptions            = -67, // CL_INVALID_LINKER_OPTIONS
  InvalidDevicePartitionCount     = -68, // CL_INVALID_DEVICE_PARTITION_COUNT
  InvalidPipeSize                 = -69, // CL_INVALID_PIPE_SIZE
  InvalidDeviceQueue              = -70, // CL_INVALID_DEVICE_QUEUE
  InvalidSpecId                   = -71, // CL_INVALID_SPEC_ID
  MaxSizeRestrictionExceeded      = -72, // CL_MAX_SIZE_RESTRICTION_EXCEEDED
};

// Replaces cl_build_status (cl_int)
// Values match CL_BUILD_* constants from cl.h.
enum class BuildStatus : int32_t {
  Success    =  0, // CL_BUILD_SUCCESS
  BuildNone  = -1, // CL_BUILD_NONE
  BuildError = -2, // CL_BUILD_ERROR
  InProgress = -3, // CL_BUILD_IN_PROGRESS
};

// Replaces cl_command_type (cl_uint)
enum class CommandType : uint32_t {
  NdRangeKernel      = 0x11F0, // CL_COMMAND_NDRANGE_KERNEL
  Task               = 0x11F1, // CL_COMMAND_TASK
  NativeKernel       = 0x11F2, // CL_COMMAND_NATIVE_KERNEL
  ReadBuffer         = 0x11F3, // CL_COMMAND_READ_BUFFER
  WriteBuffer        = 0x11F4, // CL_COMMAND_WRITE_BUFFER
  CopyBuffer         = 0x11F5, // CL_COMMAND_COPY_BUFFER
  ReadImage          = 0x11F6, // CL_COMMAND_READ_IMAGE
  WriteImage         = 0x11F7, // CL_COMMAND_WRITE_IMAGE
  CopyImage          = 0x11F8, // CL_COMMAND_COPY_IMAGE
  CopyImageToBuffer  = 0x11F9, // CL_COMMAND_COPY_IMAGE_TO_BUFFER
  CopyBufferToImage  = 0x11FA, // CL_COMMAND_COPY_BUFFER_TO_IMAGE
  MapBuffer          = 0x11FB, // CL_COMMAND_MAP_BUFFER
  MapImage           = 0x11FC, // CL_COMMAND_MAP_IMAGE
  UnmapMemObject     = 0x11FD, // CL_COMMAND_UNMAP_MEM_OBJECT
  Marker             = 0x11FE, // CL_COMMAND_MARKER
  AcquireGlObjects   = 0x11FF, // CL_COMMAND_ACQUIRE_GL_OBJECTS
  ReleaseGlObjects   = 0x1200, // CL_COMMAND_RELEASE_GL_OBJECTS
  ReadBufferRect     = 0x1201, // CL_COMMAND_READ_BUFFER_RECT
  WriteBufferRect    = 0x1202, // CL_COMMAND_WRITE_BUFFER_RECT
  CopyBufferRect     = 0x1203, // CL_COMMAND_COPY_BUFFER_RECT
  User               = 0x1204, // CL_COMMAND_USER
  Barrier            = 0x1205, // CL_COMMAND_BARRIER
  MigrateMemObjects  = 0x1206, // CL_COMMAND_MIGRATE_MEM_OBJECTS
  FillBuffer         = 0x1207, // CL_COMMAND_FILL_BUFFER
  FillImage          = 0x1208, // CL_COMMAND_FILL_IMAGE
  SvmFree            = 0x1209, // CL_COMMAND_SVM_FREE
  SvmMemcpy          = 0x120A, // CL_COMMAND_SVM_MEMCPY
  SvmMemfill         = 0x120B, // CL_COMMAND_SVM_MEMFILL
  SvmMap             = 0x120C, // CL_COMMAND_SVM_MAP
  SvmUnmap           = 0x120D, // CL_COMMAND_SVM_UNMAP
  AcquireGlFenceSyncObjectKHR = 0x200D, // CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR
};

// Event/command execution status codes.
// CL_COMPLETE=0, CL_RUNNING=1, CL_SUBMITTED=2, CL_QUEUED=3
// Negative values represent error codes (same as cl_int error codes from amd::Status).
// The status progresses from Queued(3) down to Complete(0); error codes are < 0.
enum class ExecutionStatus : int32_t {
  Complete  = 0, // CL_COMPLETE
  Running   = 1, // CL_RUNNING
  Submitted = 2, // CL_SUBMITTED
  Queued    = 3, // CL_QUEUED
};

// Map flags (replaces cl_map_flags)
enum class MapFlags : uint64_t {
  None            = 0,
  Read            = (1u << 0), // CL_MAP_READ
  Write           = (1u << 1), // CL_MAP_WRITE
  WriteInvalidate = (1u << 2), // CL_MAP_WRITE_INVALIDATE_REGION
};
inline MapFlags operator|(MapFlags a, MapFlags b) {
  return static_cast<MapFlags>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline MapFlags operator&(MapFlags a, MapFlags b) {
  return static_cast<MapFlags>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline bool operator!(MapFlags f) { return static_cast<uint64_t>(f) == 0; }

// Migration flags (replaces cl_mem_migration_flags)
enum class MemMigrationFlags : uint64_t {
  None             = 0,         // default: migrate with content
  Host             = (1u << 0), // CL_MIGRATE_MEM_OBJECT_HOST
  ContentUndefined = (1u << 1), // CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
};
inline MemMigrationFlags operator|(MemMigrationFlags a, MemMigrationFlags b) {
  return static_cast<MemMigrationFlags>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline MemMigrationFlags operator&(MemMigrationFlags a, MemMigrationFlags b) {
  return static_cast<MemMigrationFlags>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline bool operator!(MemMigrationFlags f) { return static_cast<uint64_t>(f) == 0; }

// Replaces cl_device_topology_amd (AMD extension struct)
struct DeviceTopology {
  uint32_t type;   // topology type (bus/pcie)
  uint8_t  bus;
  uint8_t  device;
  uint8_t  function;
};

// Replaces cl_kernel_arg_address_qualifier (cl_uint = uint32_t)
// Values match CL_KERNEL_ARG_ADDRESS_* constants from cl.h.
enum class KernelArgAddressQualifier : uint32_t {
  Global   = 0x119B, // CL_KERNEL_ARG_ADDRESS_GLOBAL
  Local    = 0x119C, // CL_KERNEL_ARG_ADDRESS_LOCAL
  Constant = 0x119D, // CL_KERNEL_ARG_ADDRESS_CONSTANT
  Private  = 0x119E, // CL_KERNEL_ARG_ADDRESS_PRIVATE
};

// Replaces cl_kernel_arg_access_qualifier (cl_uint = uint32_t)
// Values match CL_KERNEL_ARG_ACCESS_* constants from cl.h.
enum class KernelArgAccessQualifier : uint32_t {
  ReadOnly  = 0x11A0, // CL_KERNEL_ARG_ACCESS_READ_ONLY
  WriteOnly = 0x11A1, // CL_KERNEL_ARG_ACCESS_WRITE_ONLY
  ReadWrite = 0x11A2, // CL_KERNEL_ARG_ACCESS_READ_WRITE
  NoAccess  = 0x11A3, // CL_KERNEL_ARG_ACCESS_NONE (avoid X11 "None" macro collision)
  MaxSize   = 0x11A4, // sentinel for FindValue<V> not-found return
};

// Replaces cl_kernel_arg_type_qualifier (cl_bitfield = uint64_t)
// Values match CL_KERNEL_ARG_TYPE_* constants from cl.h.
enum class KernelArgTypeQualifier : uint64_t {
  None     = 0,        // CL_KERNEL_ARG_TYPE_NONE
  Const    = (1u << 0), // CL_KERNEL_ARG_TYPE_CONST
  Restrict = (1u << 1), // CL_KERNEL_ARG_TYPE_RESTRICT
  Volatile = (1u << 2), // CL_KERNEL_ARG_TYPE_VOLATILE
  Pipe     = (1u << 3), // CL_KERNEL_ARG_TYPE_PIPE
};
inline KernelArgTypeQualifier operator|(KernelArgTypeQualifier a, KernelArgTypeQualifier b) {
  return static_cast<KernelArgTypeQualifier>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline KernelArgTypeQualifier operator&(KernelArgTypeQualifier a, KernelArgTypeQualifier b) {
  return static_cast<KernelArgTypeQualifier>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}

// Replaces clk_value_type_t from amdocl/cl_kernel.h.
// Numeric values must stay in sync with clk_value_type_t.
enum class KernelArgValueType : int32_t {
  Void    = 0,  // T_VOID
  Char    = 1,  // T_CHAR
  Short   = 2,  // T_SHORT
  Int     = 3,  // T_INT
  Long    = 4,  // T_LONG
  Float   = 5,  // T_FLOAT
  Double  = 6,  // T_DOUBLE
  Pointer = 7,  // T_POINTER
  Char2   = 8,  // T_CHAR2
  Char3   = 9,  // T_CHAR3
  Char4   = 10, // T_CHAR4
  Char8   = 11, // T_CHAR8
  Char16  = 12, // T_CHAR16
  Short2  = 13, // T_SHORT2
  Short3  = 14, // T_SHORT3
  Short4  = 15, // T_SHORT4
  Short8  = 16, // T_SHORT8
  Short16 = 17, // T_SHORT16
  Int2    = 18, // T_INT2
  Int3    = 19, // T_INT3
  Int4    = 20, // T_INT4
  Int8    = 21, // T_INT8
  Int16   = 22, // T_INT16
  Long2   = 23, // T_LONG2
  Long3   = 24, // T_LONG3
  Long4   = 25, // T_LONG4
  Long8   = 26, // T_LONG8
  Long16  = 27, // T_LONG16
  Float2  = 28, // T_FLOAT2
  Float3  = 29, // T_FLOAT3
  Float4  = 30, // T_FLOAT4
  Float8  = 31, // T_FLOAT8
  Float16 = 32, // T_FLOAT16
  Double2 = 33, // T_DOUBLE2
  Double3 = 34, // T_DOUBLE3
  Double4 = 35, // T_DOUBLE4
  Double8 = 36, // T_DOUBLE8
  Double16 = 37, // T_DOUBLE16
  Sampler = 38, // T_SAMPLER
  Sema    = 39, // T_SEMA
  Struct  = 40, // T_STRUCT
  Queue   = 41, // T_QUEUE
  Pad     = 42, // T_PAD
};

// Replaces cl_DeviceClockMode_AMD from amdocl/cl_profile_amd.h.
enum class DeviceClockMode : uint32_t {
  Default       = 0x0, // CL_DEVICE_CLOCK_MODE_DEFAULT_AMD
  Query         = 0x1, // CL_DEVICE_CLOCK_MODE_QUERY_AMD
  Profiling     = 0x2, // CL_DEVICE_CLOCK_MODE_PROFILING_AMD
  MinimumMemory = 0x3, // CL_DEVICE_CLOCK_MODE_MINIMUMMEMORY_AMD
  MinimumEngine = 0x4, // CL_DEVICE_CLOCK_MODE_MINIMUMENGINE_AMD
  Peak          = 0x5, // CL_DEVICE_CLOCK_MODE_PEAK_AMD
  QueryProfiling = 0x6, // CL_DEVICE_CLOCK_MODE_QUERYPROFILING_AMD
  QueryPeak     = 0x7, // CL_DEVICE_CLOCK_MODE_QUERYPEAK_AMD
  Count         = 0x8, // CL_DEVICE_CLOCK_MODE_COUNT_AMD
};

// Replaces cl_set_device_clock_mode_input_amd from amdocl/cl_profile_amd.h.
struct SetDeviceClockModeInput {
  DeviceClockMode clockMode; // cl_DeviceClockMode_AMD clock_mode
};

// Replaces cl_set_device_clock_mode_output_amd from amdocl/cl_profile_amd.h.
struct SetDeviceClockModeOutput {
  float memoryClockRatioToPeak; // cl_float memory_clock_ratio_to_peak
  float engineClockRatioToPeak; // cl_float engine_clock_ratio_to_peak
};

// Replaces CL_FILTER_* constants (cl_filter_mode = cl_uint = uint32_t)
enum class FilterMode : uint32_t {
  Nearest = 0x1140, // CL_FILTER_NEAREST
  Linear  = 0x1141, // CL_FILTER_LINEAR
  None    = 0x1142, // No filtering (HIP extension, used for mipmap base case)
};

// Replaces CL_ADDRESS_* constants (cl_addressing_mode = cl_uint = uint32_t)
enum class AddressingMode : uint32_t {
  NoAddressing   = 0x1130, // CL_ADDRESS_NONE
  ClampToEdge    = 0x1131, // CL_ADDRESS_CLAMP_TO_EDGE
  Clamp          = 0x1132, // CL_ADDRESS_CLAMP
  Repeat         = 0x1133, // CL_ADDRESS_REPEAT
  MirroredRepeat = 0x1134, // CL_ADDRESS_MIRRORED_REPEAT
};

// Pipe object layout for CL 2.0 pipe built-in.
// Mirrors clk_pipe_t from opencl/amdocl/cl_kernel.h.
struct PipeObject {
  size_t read_idx;
  size_t write_idx;
  size_t end_idx;
  char padding[128 - 3 * sizeof(size_t)];
  // packets[] follow in device memory
};

// Bus address pair for the cl_amd_bus_addressable_memory extension.
// Equivalent to cl_bus_address_amd { cl_ulong surface_bus_address; cl_ulong marker_bus_address; }.
struct BusAddress {
  uint64_t surface_bus_address;
  uint64_t marker_bus_address;
};

// Replaces CL_PERFCOUNTER_* constants from cl_profile_amd.h (cl_perfcounter_info = cl_uint).
enum class PerfCounterInfo : uint32_t {
  None           = 0x0, // CL_PERFCOUNTER_NONE
  ReferenceCount = 0x1, // CL_PERFCOUNTER_REFERENCE_COUNT
  Data           = 0x2, // CL_PERFCOUNTER_DATA
  GpuBlockIndex  = 0x3, // CL_PERFCOUNTER_GPU_BLOCK_INDEX
  GpuCounterIndex = 0x4, // CL_PERFCOUNTER_GPU_COUNTER_INDEX
  GpuEventIndex  = 0x5, // CL_PERFCOUNTER_GPU_EVENT_INDEX
};

}  // namespace amd
