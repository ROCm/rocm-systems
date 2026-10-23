// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust representations of the HSA core and AMD-extension C ABI.
//!
//! Numeric constants, handle encodings, callback signatures, and `repr(C)`
//! records in this module are compatibility data, not internal conveniences.
//! They mirror the vendored HSA headers used by this workspace. Implementation
//! modules consume these definitions but must not reinterpret or renumber them.
//! Layout-sensitive records are checked by compile-time assertions and ABI
//! tests where the public contract requires an exact size or field offset.

use std::ffi::{c_char, c_void};

/// Numeric `hsa_status_t` result returned by HSA C entry points.
pub(crate) type Status = u32;
/// Signed value stored in a native HSA signal.
pub(crate) type SignalValue = i64;

/// The function has been executed successfully.
pub(crate) const SUCCESS: Status = 0;
/// A traversal over a list of elements has been interrupted by the application
/// before completing.
pub(crate) const INFO_BREAK: Status = 1;
/// A generic error has occurred.
pub(crate) const ERROR: Status = 0x1000;
/// One of the actual arguments does not meet a precondition stated in the
/// documentation of the corresponding formal argument.
pub(crate) const INVALID_ARGUMENT: Status = 0x1001;
/// The requested queue creation is not valid.
pub(crate) const INVALID_QUEUE_CREATION: Status = 0x1002;
/// The requested allocation is not valid.
pub(crate) const INVALID_ALLOCATION: Status = 0x1003;
/// The agent is invalid.
pub(crate) const INVALID_AGENT: Status = 0x1004;
/// The memory region is invalid.
pub(crate) const INVALID_REGION: Status = 0x1005;
/// The signal is invalid.
pub(crate) const INVALID_SIGNAL: Status = 0x1006;
/// The queue is invalid.
pub(crate) const INVALID_QUEUE: Status = 0x1007;
/// The runtime failed to allocate the necessary resources.
pub(crate) const OUT_OF_RESOURCES: Status = 0x1008;
/// The AQL packet is malformed.
pub(crate) const INVALID_PACKET_FORMAT: Status = 0x1009;
/// An error has been detected while releasing a resource.
pub(crate) const RESOURCE_FREE: Status = 0x100a;
/// An API other than `hsa_init` has been invoked while the reference count of the
/// HSA runtime is zero.
pub(crate) const NOT_INITIALIZED: Status = 0x100b;
/// The maximum reference count for the object has been reached.
pub(crate) const REFCOUNT_OVERFLOW: Status = 0x100c;
/// The arguments passed to a functions are not compatible.
pub(crate) const INCOMPATIBLE_ARGUMENTS: Status = 0x100d;
/// The index is invalid.
pub(crate) const INVALID_INDEX: Status = 0x100e;
/// The instruction set architecture is invalid.
pub(crate) const INVALID_ISA: Status = 0x100f;
/// The code object is invalid.
pub(crate) const INVALID_CODE_OBJECT: Status = 0x1010;
/// The executable is invalid.
pub(crate) const INVALID_EXECUTABLE: Status = 0x1011;
/// The executable is frozen.
pub(crate) const FROZEN_EXECUTABLE: Status = 0x1012;
/// There is no symbol with the given name.
pub(crate) const INVALID_SYMBOL_NAME: Status = 0x1013;
/// The variable is already defined.
pub(crate) const VARIABLE_ALREADY_DEFINED: Status = 0x1014;
/// The variable is undefined.
pub(crate) const VARIABLE_UNDEFINED: Status = 0x1015;
/// An HSAIL operation resulted in a hardware exception.
pub(crate) const EXCEPTION: Status = 0x1016;
/// The instruction set architecture name is invalid.
pub(crate) const INVALID_ISA_NAME: Status = 0x1017;
/// The code object symbol is invalid.
pub(crate) const INVALID_CODE_SYMBOL: Status = 0x1018;
/// The executable symbol is invalid.
pub(crate) const INVALID_EXECUTABLE_SYMBOL: Status = 0x1019;
/// The file descriptor is invalid.
pub(crate) const INVALID_FILE: Status = 0x1020;
/// The code object reader is invalid.
pub(crate) const INVALID_CODE_OBJECT_READER: Status = 0x1021;
/// The cache is invalid.
pub(crate) const INVALID_CACHE: Status = 0x1022;
/// The wavefront is invalid.
pub(crate) const INVALID_WAVEFRONT: Status = 0x1023;
/// The signal group is invalid.
pub(crate) const INVALID_SIGNAL_GROUP: Status = 0x1024;
/// The HSA runtime is not in the configuration state.
pub(crate) const INVALID_RUNTIME_STATE: Status = 0x1025;
/// The queue received an error that may require process termination.
pub(crate) const FATAL: Status = 0x1026;
/// The memory pool is invalid.
pub(crate) const INVALID_MEMORY_POOL: Status = 40;
/// The agent attempted to access memory beyond the largest legal address.
pub(crate) const MEMORY_APERTURE_VIOLATION: Status = 41;
/// The agent attempted to execute an illegal shader instruction.
pub(crate) const ILLEGAL_INSTRUCTION: Status = 42;
/// Agent attempted to access an inaccessible address.
pub(crate) const MEMORY_FAULT: Status = 43;
/// The CU mask was successfully set but the mask attempted to enable a CU which
/// was disabled for the process.
pub(crate) const CU_MASK_REDUCED: Status = 44;
/// Kernel has requested more VGPRs than are available on this agent.
pub(crate) const OUT_OF_REGISTERS: Status = 45;
/// Resource is busy or temporarily unavailable.
pub(crate) const RESOURCE_BUSY: Status = 46;
/// Request is not supported by this system.
pub(crate) const NOT_SUPPORTED: Status = 47;
/// Xnack is disabled on this system, but required by the requested operation.
pub(crate) const XNACK_DISABLED: Status = 48;
/// Kernel dispatch packet parameters exceed hardware limits for this agent (e.g.
pub(crate) const INVALID_DISPATCH_PARAMETERS: Status = 49;
/// Underlying resource is a valid resource, but it is not ready to be used.
pub(crate) const RESOURCE_NOT_READY: Status = 50;
/// Image format is not supported.
pub(crate) const IMAGE_FORMAT_UNSUPPORTED: Status = 0x3000;
/// Image size is not supported.
pub(crate) const IMAGE_SIZE_UNSUPPORTED: Status = 0x3001;
/// Image pitch is not supported or invalid.
pub(crate) const IMAGE_PITCH_UNSUPPORTED: Status = 0x3002;
/// Sampler descriptor is not supported or invalid.
pub(crate) const SAMPLER_DESCRIPTOR_UNSUPPORTED: Status = 0x3003;

/// Synthetic handle for the passive host CPU agent.
pub(crate) const CPU_AGENT: u64 = 0x4853_4100_0000_0001;
/// Base value of the synthetic GPU agent handle range.
pub(crate) const GPU_AGENT_BASE: u64 = 0x4853_4100_0010_0000;
/// Base value of the synthetic ISA handle range.
pub(crate) const ISA_BASE: u64 = 0x4853_4100_0020_0000;
/// Number of public ISA variants reserved for each GPU.
pub(crate) const ISA_COUNT_PER_GPU: u64 = 2;
/// Base value of the synthetic wavefront handle range.
pub(crate) const WAVEFRONT_BASE: u64 = 0x4853_4100_0028_0000;
/// Synthetic handle for the fine-grained host memory pool.
pub(crate) const CPU_POOL_FINE: u64 = 0x4853_4100_0030_0001;
/// Synthetic handle for the extended fine-grained host pool.
pub(crate) const CPU_POOL_EXTENDED: u64 = 0x4853_4100_0030_0002;
/// Synthetic handle for the host kernel-argument pool.
pub(crate) const CPU_POOL_KERNARG: u64 = 0x4853_4100_0030_0003;
/// Synthetic handle for the coarse-grained host pool.
pub(crate) const CPU_POOL_COARSE: u64 = 0x4853_4100_0030_0004;
/// Base value of the synthetic GPU pool handle range.
pub(crate) const GPU_POOL_BASE: u64 = 0x4853_4100_0040_0000;
/// Base value of the synthetic cache handle range.
pub(crate) const CACHE_BASE: u64 = 0x4853_4100_0050_0000;

/// HSA extension identifier for finalizer.
pub(crate) const EXTENSION_FINALIZER: u16 = 0;
/// HSA extension identifier for images.
pub(crate) const EXTENSION_IMAGES: u16 = 1;
/// HSA extension identifier for performance counters.
pub(crate) const EXTENSION_PERFORMANCE_COUNTERS: u16 = 2;
/// HSA extension identifier for profiling events.
pub(crate) const EXTENSION_PROFILING_EVENTS: u16 = 3;
/// HSA extension identifier for AMD profiler.
pub(crate) const EXTENSION_AMD_PROFILER: u16 = 0x200;
/// HSA extension identifier for AMD loader.
pub(crate) const EXTENSION_AMD_LOADER: u16 = 0x201;
/// HSA extension identifier for AMD aqlprofile.
pub(crate) const EXTENSION_AMD_AQLPROFILE: u16 = 0x202;
/// HSA extension identifier for AMD pc sampling.
pub(crate) const EXTENSION_AMD_PC_SAMPLING: u16 = 0x203;

/// Agent device-kind discriminant for CPU.
pub(crate) const DEVICE_CPU: u32 = 0;
/// Agent device-kind discriminant for GPU.
pub(crate) const DEVICE_GPU: u32 = 1;
/// Agent HSA profile discriminant for base.
pub(crate) const PROFILE_BASE: u32 = 0;
/// Agent HSA profile discriminant for full.
pub(crate) const PROFILE_FULL: u32 = 1;
/// HSA machine model with 64-bit address and size fields.
pub(crate) const MACHINE_MODEL_LARGE: u32 = 1;
/// Default floating-point rounding mode for default.
pub(crate) const DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 0;
/// Default floating-point rounding mode for zero.
pub(crate) const DEFAULT_FLOAT_ROUNDING_MODE_ZERO: u32 = 1;
/// Default floating-point rounding mode for near.
pub(crate) const DEFAULT_FLOAT_ROUNDING_MODE_NEAR: u32 = 2;
/// Agent feature bit for kernel dispatch.
pub(crate) const AGENT_FEATURE_KERNEL_DISPATCH: u32 = 1;
/// Queue producer-mode discriminant for multi.
pub(crate) const QUEUE_TYPE_MULTI: u32 = 0;
/// Queue producer-mode discriminant for single.
pub(crate) const QUEUE_TYPE_SINGLE: u32 = 1;
/// Queue producer-mode discriminant for cooperative.
pub(crate) const QUEUE_TYPE_COOPERATIVE: u32 = 2;
/// Queue feature bit for kernel dispatch.
pub(crate) const QUEUE_FEATURE_KERNEL_DISPATCH: u32 = 1;
/// Queue feature bit for agent dispatch.
pub(crate) const QUEUE_FEATURE_AGENT_DISPATCH: u32 = 1 << 1;

/// AMD queue priority discriminant for low.
pub(crate) const AMD_QUEUE_PRIORITY_LOW: u32 = 0;
/// AMD queue priority discriminant for normal.
pub(crate) const AMD_QUEUE_PRIORITY_NORMAL: u32 = 1;
/// AMD queue priority discriminant for high.
pub(crate) const AMD_QUEUE_PRIORITY_HIGH: u32 = 2;
/// ABI version value for AMD queue create desc.
pub(crate) const AMD_QUEUE_CREATE_DESC_VERSION: u16 = 1;
/// Flag requesting a device-memory queue ring.
pub(crate) const AMD_QUEUE_CREATE_DEVICE_MEM_RING: u16 = 1;
/// Flag requesting a device-memory queue descriptor.
pub(crate) const AMD_QUEUE_CREATE_DEVICE_MEM_DESCRIPTOR: u16 = 1 << 1;
/// AMD queue engine-kind discriminant for compute.
pub(crate) const AMD_QUEUE_ENGINE_COMPUTE: u8 = 0;
/// AMD queue engine-kind discriminant for SDMA.
pub(crate) const AMD_QUEUE_ENGINE_SDMA: u8 = 1;
/// AMD queue engine-kind discriminant for AIE.
pub(crate) const AMD_QUEUE_ENGINE_AIE: u8 = 2;
/// AMD coherency-mode discriminant for coherent.
pub(crate) const AMD_COHERENCY_TYPE_COHERENT: u32 = 0;
/// AMD coherency-mode discriminant for noncoherent.
pub(crate) const AMD_COHERENCY_TYPE_NONCOHERENT: u32 = 1;
/// AMD logging flag enabling informational messages.
pub(crate) const AMD_LOG_FLAG_INFO: u32 = 2;

/// System information selector for version major.
pub(crate) const SYSTEM_INFO_VERSION_MAJOR: u32 = 0;
/// System information selector for version minor.
pub(crate) const SYSTEM_INFO_VERSION_MINOR: u32 = 1;
/// System information selector for timestamp.
pub(crate) const SYSTEM_INFO_TIMESTAMP: u32 = 2;
/// System information selector for timestamp frequency.
pub(crate) const SYSTEM_INFO_TIMESTAMP_FREQUENCY: u32 = 3;
/// System information selector for signal max wait.
pub(crate) const SYSTEM_INFO_SIGNAL_MAX_WAIT: u32 = 4;
/// System information selector for endianness.
pub(crate) const SYSTEM_INFO_ENDIANNESS: u32 = 5;
/// System information selector for machine model.
pub(crate) const SYSTEM_INFO_MACHINE_MODEL: u32 = 6;
/// System information selector for extensions.
pub(crate) const SYSTEM_INFO_EXTENSIONS: u32 = 7;
/// AMD system information selector for SVM supported.
pub(crate) const AMD_SYSTEM_INFO_SVM_SUPPORTED: u32 = 0x201;
/// AMD system information selector for SVM accessible by default.
pub(crate) const AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT: u32 = 0x202;
/// AMD system information selector for MWAITX enabled.
pub(crate) const AMD_SYSTEM_INFO_MWAITX_ENABLED: u32 = 0x203;
/// AMD system information selector for DMA-BUF supported.
pub(crate) const AMD_SYSTEM_INFO_DMABUF_SUPPORTED: u32 = 0x204;
/// AMD system information selector for virtual mem api supported.
pub(crate) const AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED: u32 = 0x205;
/// AMD system information selector for XNACK enabled.
pub(crate) const AMD_SYSTEM_INFO_XNACK_ENABLED: u32 = 0x206;
/// AMD system information selector for ext version major.
pub(crate) const AMD_SYSTEM_INFO_EXT_VERSION_MAJOR: u32 = 0x207;
/// AMD system information selector for ext version minor.
pub(crate) const AMD_SYSTEM_INFO_EXT_VERSION_MINOR: u32 = 0x208;
/// AMD system information selector for fabric handles supported.
pub(crate) const AMD_SYSTEM_INFO_FABRIC_HANDLES_SUPPORTED: u32 = 0x209;
/// AMD system information selector for host alloc DMA-BUF supported.
pub(crate) const AMD_SYSTEM_INFO_HOST_ALLOC_DMABUF_SUPPORTED: u32 = 0x20a;

/// Agent information selector for name.
pub(crate) const AGENT_INFO_NAME: u32 = 0;
/// Agent information selector for vendor name.
pub(crate) const AGENT_INFO_VENDOR_NAME: u32 = 1;
/// Agent information selector for feature.
pub(crate) const AGENT_INFO_FEATURE: u32 = 2;
/// Agent information selector for machine model.
pub(crate) const AGENT_INFO_MACHINE_MODEL: u32 = 3;
/// Agent information selector for profile.
pub(crate) const AGENT_INFO_PROFILE: u32 = 4;
/// Agent information selector for default float rounding mode.
pub(crate) const AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 5;
/// Agent information selector for wavefront size.
pub(crate) const AGENT_INFO_WAVEFRONT_SIZE: u32 = 6;
/// Agent information selector for workgroup max dim.
pub(crate) const AGENT_INFO_WORKGROUP_MAX_DIM: u32 = 7;
/// Agent information selector for workgroup max size.
pub(crate) const AGENT_INFO_WORKGROUP_MAX_SIZE: u32 = 8;
/// Agent information selector for grid max dim.
pub(crate) const AGENT_INFO_GRID_MAX_DIM: u32 = 9;
/// Agent information selector for grid max size.
pub(crate) const AGENT_INFO_GRID_MAX_SIZE: u32 = 10;
/// Agent information selector for fbarrier max size.
pub(crate) const AGENT_INFO_FBARRIER_MAX_SIZE: u32 = 11;
/// Agent information selector for queues max.
pub(crate) const AGENT_INFO_QUEUES_MAX: u32 = 12;
/// Agent information selector for queue min size.
pub(crate) const AGENT_INFO_QUEUE_MIN_SIZE: u32 = 13;
/// Agent information selector for queue max size.
pub(crate) const AGENT_INFO_QUEUE_MAX_SIZE: u32 = 14;
/// Agent information selector for queue type.
pub(crate) const AGENT_INFO_QUEUE_TYPE: u32 = 15;
/// Agent information selector for node.
pub(crate) const AGENT_INFO_NODE: u32 = 16;
/// Agent information selector for device.
pub(crate) const AGENT_INFO_DEVICE: u32 = 17;
/// Agent information selector for cache size.
pub(crate) const AGENT_INFO_CACHE_SIZE: u32 = 18;
/// Agent information selector for ISA.
pub(crate) const AGENT_INFO_ISA: u32 = 19;
/// Agent information selector for extensions.
pub(crate) const AGENT_INFO_EXTENSIONS: u32 = 20;
/// Agent information selector for version major.
pub(crate) const AGENT_INFO_VERSION_MAJOR: u32 = 21;
/// Agent information selector for version minor.
pub(crate) const AGENT_INFO_VERSION_MINOR: u32 = 22;
/// Agent information selector for base profile default float rounding modes.
pub(crate) const AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES: u32 = 23;
/// Agent information selector for fast f16 operation.
pub(crate) const AGENT_INFO_FAST_F16_OPERATION: u32 = 24;

/// AMD agent information selector for chip ID.
pub(crate) const AMD_AGENT_INFO_CHIP_ID: u32 = 0xa000;
/// AMD agent information selector for cacheline size.
pub(crate) const AMD_AGENT_INFO_CACHELINE_SIZE: u32 = 0xa001;
/// AMD agent information selector for compute unit count.
pub(crate) const AMD_AGENT_INFO_COMPUTE_UNIT_COUNT: u32 = 0xa002;
/// AMD agent information selector for max clock frequency.
pub(crate) const AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY: u32 = 0xa003;
/// AMD agent information selector for driver node ID.
pub(crate) const AMD_AGENT_INFO_DRIVER_NODE_ID: u32 = 0xa004;
/// AMD agent information selector for max address watch points.
pub(crate) const AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS: u32 = 0xa005;
/// AMD agent information selector for PCI bus/device/function ID.
pub(crate) const AMD_AGENT_INFO_BDFID: u32 = 0xa006;
/// AMD agent information selector for memory width.
pub(crate) const AMD_AGENT_INFO_MEMORY_WIDTH: u32 = 0xa007;
/// AMD agent information selector for memory max frequency.
pub(crate) const AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY: u32 = 0xa008;
/// AMD agent information selector for product name.
pub(crate) const AMD_AGENT_INFO_PRODUCT_NAME: u32 = 0xa009;
/// AMD agent information selector for max waves per CU.
pub(crate) const AMD_AGENT_INFO_MAX_WAVES_PER_CU: u32 = 0xa00a;
/// AMD agent information selector for num simds per CU.
pub(crate) const AMD_AGENT_INFO_NUM_SIMDS_PER_CU: u32 = 0xa00b;
/// AMD agent information selector for num shader engines.
pub(crate) const AMD_AGENT_INFO_NUM_SHADER_ENGINES: u32 = 0xa00c;
/// AMD agent information selector for num shader arrays per se.
pub(crate) const AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE: u32 = 0xa00d;
/// AMD agent information selector for HDP flush.
pub(crate) const AMD_AGENT_INFO_HDP_FLUSH: u32 = 0xa00e;
/// AMD agent information selector for domain.
pub(crate) const AMD_AGENT_INFO_DOMAIN: u32 = 0xa00f;
/// AMD agent information selector for cooperative queues.
pub(crate) const AMD_AGENT_INFO_COOPERATIVE_QUEUES: u32 = 0xa010;
/// AMD agent information selector for uuid.
pub(crate) const AMD_AGENT_INFO_UUID: u32 = 0xa011;
/// AMD agent information selector for asic revision.
pub(crate) const AMD_AGENT_INFO_ASIC_REVISION: u32 = 0xa012;
/// AMD agent information selector for SVM direct host access.
pub(crate) const AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS: u32 = 0xa013;
/// AMD agent information selector for cooperative compute unit count.
pub(crate) const AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT: u32 = 0xa014;
/// AMD agent information selector for memory avail.
pub(crate) const AMD_AGENT_INFO_MEMORY_AVAIL: u32 = 0xa015;
/// AMD agent information selector for timestamp frequency.
pub(crate) const AMD_AGENT_INFO_TIMESTAMP_FREQUENCY: u32 = 0xa016;
/// AMD agent information selector for asic family ID.
pub(crate) const AMD_AGENT_INFO_ASIC_FAMILY_ID: u32 = 0xa107;
/// AMD agent information selector for ucode version.
pub(crate) const AMD_AGENT_INFO_UCODE_VERSION: u32 = 0xa108;
/// AMD agent information selector for SDMA ucode version.
pub(crate) const AMD_AGENT_INFO_SDMA_UCODE_VERSION: u32 = 0xa109;
/// AMD agent information selector for num SDMA eng.
pub(crate) const AMD_AGENT_INFO_NUM_SDMA_ENG: u32 = 0xa10a;
/// AMD agent information selector for num SDMA xgmi eng.
pub(crate) const AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG: u32 = 0xa10b;
/// AMD agent information selector for iommu support.
pub(crate) const AMD_AGENT_INFO_IOMMU_SUPPORT: u32 = 0xa110;
/// AMD agent information selector for num XCC.
pub(crate) const AMD_AGENT_INFO_NUM_XCC: u32 = 0xa111;
/// AMD agent information selector for driver uid.
pub(crate) const AMD_AGENT_INFO_DRIVER_UID: u32 = 0xa112;
/// AMD agent information selector for nearest CPU.
pub(crate) const AMD_AGENT_INFO_NEAREST_CPU: u32 = 0xa113;
/// AMD agent information selector for memory properties.
pub(crate) const AMD_AGENT_INFO_MEMORY_PROPERTIES: u32 = 0xa114;
/// AMD agent information selector for AQL extensions.
pub(crate) const AMD_AGENT_INFO_AQL_EXTENSIONS: u32 = 0xa115;
/// AMD agent information selector for scratch limit max.
pub(crate) const AMD_AGENT_INFO_SCRATCH_LIMIT_MAX: u32 = 0xa116;
/// AMD agent information selector for scratch limit current.
pub(crate) const AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT: u32 = 0xa117;
/// AMD agent information selector for clock counters.
pub(crate) const AMD_AGENT_INFO_CLOCK_COUNTERS: u32 = 0xa118;
/// AMD agent information selector for pm4 emulation.
pub(crate) const AMD_AGENT_INFO_PM4_EMULATION: u32 = 0xa119;
/// AMD agent information selector for LUID.
pub(crate) const AMD_AGENT_INFO_LUID: u32 = 0xa11a;
/// AMD agent information selector for has expert sched mode.
pub(crate) const AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE: u32 = 0xa11b;
/// AMD agent information selector for cuid.
pub(crate) const AMD_AGENT_INFO_CUID: u32 = 0xa11c;
/// AMD agent information selector for kernel wg max size.
pub(crate) const AMD_AGENT_INFO_KERNEL_WG_MAX_SIZE: u32 = 0xa11d;
/// AMD agent information selector for kernel cluster max dim.
pub(crate) const AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM: u32 = 0xa11e;
/// AMD agent information selector for kernel cluster max size.
pub(crate) const AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE: u32 = 0xa11f;
/// AMD agent information selector for cluster max dim.
pub(crate) const AMD_AGENT_INFO_CLUSTER_MAX_DIM: u32 = 0xa120;
/// AMD agent information selector for cluster max size.
pub(crate) const AMD_AGENT_INFO_CLUSTER_MAX_SIZE: u32 = 0xa121;
/// AMD agent information selector for kernel wg max dim.
pub(crate) const AMD_AGENT_INFO_KERNEL_WG_MAX_DIM: u32 = 0xa122;
/// AMD agent information selector for max data prefetch regions.
pub(crate) const AMD_AGENT_INFO_MAX_DATA_PREFETCH_REGIONS: u32 = 0xa123;
/// AMD agent information selector for host alloc DMA-BUF supported.
pub(crate) const AMD_AGENT_INFO_HOST_ALLOC_DMABUF_SUPPORTED: u32 = 0xa124;
/// AMD agent information selector for request persisting l2 cache size.
pub(crate) const AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE: u32 = 0xa125;
/// AMD agent information selector for max persisting l2 cache size.
pub(crate) const AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE: u32 = 0xa126;
/// AMD agent information selector for `PCIe` stores needing a flush.
pub(crate) const AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH: u32 = 0xa127;
/// AMD agent information selector for host atomics supported.
pub(crate) const AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED: u32 = 0xa128;
/// AMD agent information selector for signal hints supported.
pub(crate) const AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED: u32 = 0xa129;
/// AMD agent attribute requesting persistent L2 cache size.
pub(crate) const AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE: u32 = 0;

/// Image extension agent-attribute selector for image 1d max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS: u32 = 0x3000;
/// Image extension agent-attribute selector for image 1da max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS: u32 = 0x3001;
/// Image extension agent-attribute selector for image 1db max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS: u32 = 0x3002;
/// Image extension agent-attribute selector for image 2d max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS: u32 = 0x3003;
/// Image extension agent-attribute selector for image 2da max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS: u32 = 0x3004;
/// Image extension agent-attribute selector for image 2ddepth max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS: u32 = 0x3005;
/// Image extension agent-attribute selector for image 2dadepth max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS: u32 = 0x3006;
/// Image extension agent-attribute selector for image 3d max elements.
pub(crate) const EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS: u32 = 0x3007;
/// Image extension agent-attribute selector for image array max layers.
pub(crate) const EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS: u32 = 0x3008;
/// Image extension agent-attribute selector for max image rd handles.
pub(crate) const EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES: u32 = 0x3009;
/// Image extension agent-attribute selector for max image rorw handles.
pub(crate) const EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES: u32 = 0x300a;
/// Image extension agent-attribute selector for max sampler handlers.
pub(crate) const EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS: u32 = 0x300b;
/// Image extension agent-attribute selector for image linear row pitch alignment.
pub(crate) const EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT: u32 = 0x300c;
/// Image extension agent-attribute selector for image support.
pub(crate) const EXT_AGENT_INFO_IMAGE_SUPPORT: u32 = 0x300d;

/// ISA information selector for name length.
pub(crate) const ISA_INFO_NAME_LENGTH: u32 = 0;
/// ISA information selector for name.
pub(crate) const ISA_INFO_NAME: u32 = 1;
/// ISA information selector for call convention count.
pub(crate) const ISA_INFO_CALL_CONVENTION_COUNT: u32 = 2;
/// ISA information selector for machine models.
pub(crate) const ISA_INFO_MACHINE_MODELS: u32 = 5;
/// ISA information selector for profiles.
pub(crate) const ISA_INFO_PROFILES: u32 = 6;
/// ISA information selector for default float rounding modes.
pub(crate) const ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES: u32 = 7;
/// ISA information selector for base profile default float rounding modes.
pub(crate) const ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES: u32 = 8;
/// ISA information selector for fast f16 operation.
pub(crate) const ISA_INFO_FAST_F16_OPERATION: u32 = 9;
/// ISA information selector for workgroup max dim.
pub(crate) const ISA_INFO_WORKGROUP_MAX_DIM: u32 = 12;
/// ISA information selector for workgroup max size.
pub(crate) const ISA_INFO_WORKGROUP_MAX_SIZE: u32 = 13;
/// ISA information selector for grid max dim.
pub(crate) const ISA_INFO_GRID_MAX_DIM: u32 = 14;
/// ISA information selector for grid max size.
pub(crate) const ISA_INFO_GRID_MAX_SIZE: u32 = 16;
/// ISA information selector for fbarrier max size.
pub(crate) const ISA_INFO_FBARRIER_MAX_SIZE: u32 = 17;

/// HSA memory-segment discriminant for global.
pub(crate) const SEGMENT_GLOBAL: u32 = 0;
/// HSA memory-segment discriminant for group.
pub(crate) const SEGMENT_GROUP: u32 = 3;
/// Global-pool flag identifying kernel-argument storage.
pub(crate) const POOL_FLAG_KERNARG: u32 = 1;
/// Global-pool flag identifying fine-grained storage.
pub(crate) const POOL_FLAG_FINE: u32 = 2;
/// Global-pool flag identifying coarse-grained storage.
pub(crate) const POOL_FLAG_COARSE: u32 = 4;
/// Global-pool flag identifying extended fine-grained storage.
pub(crate) const POOL_FLAG_EXTENDED_FINE: u32 = 8;
/// Memory-pool information selector for segment.
pub(crate) const POOL_INFO_SEGMENT: u32 = 0;
/// Memory-pool information selector for global flags.
pub(crate) const POOL_INFO_GLOBAL_FLAGS: u32 = 1;
/// Memory-pool information selector for size.
pub(crate) const POOL_INFO_SIZE: u32 = 2;
/// Memory-pool information selector for runtime alloc allowed.
pub(crate) const POOL_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;
/// Memory-pool information selector for runtime alloc granule.
pub(crate) const POOL_INFO_RUNTIME_ALLOC_GRANULE: u32 = 6;
/// Memory-pool information selector for runtime alloc alignment.
pub(crate) const POOL_INFO_RUNTIME_ALLOC_ALIGNMENT: u32 = 7;
/// Memory-pool information selector for accessible by all.
pub(crate) const POOL_INFO_ACCESSIBLE_BY_ALL: u32 = 15;
/// Memory-pool information selector for alloc max size.
pub(crate) const POOL_INFO_ALLOC_MAX_SIZE: u32 = 16;
/// Memory-pool information selector for location.
pub(crate) const POOL_INFO_LOCATION: u32 = 17;
/// Memory-pool information selector for runtime alloc rec granule.
pub(crate) const POOL_INFO_RUNTIME_ALLOC_REC_GRANULE: u32 = 18;
/// Memory-pool information selector for signal capable.
pub(crate) const POOL_INFO_SIGNAL_CAPABLE: u32 = 19;
/// Agent-to-pool access discriminant for never.
pub(crate) const POOL_ACCESS_NEVER: u32 = 0;
/// Agent-to-pool access discriminant for default.
pub(crate) const POOL_ACCESS_DEFAULT: u32 = 1;
/// Agent-to-pool access discriminant for disallowed.
pub(crate) const POOL_ACCESS_DISALLOWED: u32 = 2;
/// Agent-to-pool relationship information selector for access.
pub(crate) const AGENT_POOL_INFO_ACCESS: u32 = 0;
/// Agent-to-pool relationship information selector for num link hops.
pub(crate) const AGENT_POOL_INFO_NUM_LINK_HOPS: u32 = 1;
/// Agent-to-pool relationship information selector for link info.
pub(crate) const AGENT_POOL_INFO_LINK_INFO: u32 = 2;

/// Memory-region information selector for alloc max size.
pub(crate) const REGION_INFO_ALLOC_MAX_SIZE: u32 = 4;
/// Memory-region information selector for alloc max private workgroup size.
pub(crate) const REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE: u32 = 8;
/// AMD region information selector for host accessible.
pub(crate) const AMD_REGION_INFO_HOST_ACCESSIBLE: u32 = 0xa000;
/// AMD region information selector for base.
pub(crate) const AMD_REGION_INFO_BASE: u32 = 0xa001;
/// AMD region information selector for bus width.
pub(crate) const AMD_REGION_INFO_BUS_WIDTH: u32 = 0xa002;
/// AMD region information selector for max clock frequency.
pub(crate) const AMD_REGION_INFO_MAX_CLOCK_FREQUENCY: u32 = 0xa003;

/// Cache information selector for name length.
pub(crate) const CACHE_INFO_NAME_LENGTH: u32 = 0;
/// Cache information selector for name.
pub(crate) const CACHE_INFO_NAME: u32 = 1;
/// Cache information selector for level.
pub(crate) const CACHE_INFO_LEVEL: u32 = 2;
/// Cache information selector for size.
pub(crate) const CACHE_INFO_SIZE: u32 = 3;

/// AMD pointer-information category for unknown.
pub(crate) const POINTER_TYPE_UNKNOWN: u32 = 0;
/// AMD pointer-information category for HSA.
pub(crate) const POINTER_TYPE_HSA: u32 = 1;
/// AMD pointer-information category for locked.
pub(crate) const POINTER_TYPE_LOCKED: u32 = 2;
/// AMD pointer-information category for graphics.
pub(crate) const POINTER_TYPE_GRAPHICS: u32 = 3;
/// AMD pointer-information category for IPC.
pub(crate) const POINTER_TYPE_IPC: u32 = 4;
/// AMD pointer-information category for reserved addr.
pub(crate) const POINTER_TYPE_RESERVED_ADDR: u32 = 5;
/// AMD pointer-information category for HSA vmem.
pub(crate) const POINTER_TYPE_HSA_VMEM: u32 = 6;
/// Allocation property bit permitting executable memory.
pub(crate) const POINTER_ALLOC_EXECUTABLE: u32 = 1 << 0;
/// Allocation property bit requesting contiguous memory.
pub(crate) const POINTER_ALLOC_CONTIGUOUS: u32 = 1 << 1;
/// Allocation property bit requesting nonpaged memory.
pub(crate) const POINTER_ALLOC_NONPAGED: u32 = 1 << 2;
/// Allocation property bit allowing host access.
pub(crate) const POINTER_ALLOC_HOST_ACCESS: u32 = 1 << 4;
/// Allocation property bit allowing full atomic access.
pub(crate) const POINTER_ALLOC_ATOMIC_FULL: u32 = 1 << 5;
/// Allocation property bit allowing partial atomic access.
pub(crate) const POINTER_ALLOC_ATOMIC_PARTIAL: u32 = 1 << 6;

/// AMD SVM attribute selector for global flag.
pub(crate) const AMD_SVM_ATTRIB_GLOBAL_FLAG: u64 = 0;
/// AMD SVM attribute selector for read only.
pub(crate) const AMD_SVM_ATTRIB_READ_ONLY: u64 = 1;
/// AMD SVM attribute selector for hive local.
pub(crate) const AMD_SVM_ATTRIB_HIVE_LOCAL: u64 = 2;
/// AMD SVM attribute selector for migration granularity.
pub(crate) const AMD_SVM_ATTRIB_MIGRATION_GRANULARITY: u64 = 3;
/// AMD SVM attribute selector for preferred location.
pub(crate) const AMD_SVM_ATTRIB_PREFERRED_LOCATION: u64 = 4;
/// AMD SVM attribute selector for prefetch location.
pub(crate) const AMD_SVM_ATTRIB_PREFETCH_LOCATION: u64 = 5;
/// AMD SVM attribute selector for read mostly.
pub(crate) const AMD_SVM_ATTRIB_READ_MOSTLY: u64 = 6;
/// AMD SVM attribute selector for gpu exec.
pub(crate) const AMD_SVM_ATTRIB_GPU_EXEC: u64 = 7;
/// AMD SVM attribute selector for agent accessible.
pub(crate) const AMD_SVM_ATTRIB_AGENT_ACCESSIBLE: u64 = 0x200;
/// AMD SVM attribute selector for agent accessible in place.
pub(crate) const AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE: u64 = 0x201;
/// AMD SVM attribute selector for agent no access.
pub(crate) const AMD_SVM_ATTRIB_AGENT_NO_ACCESS: u64 = 0x202;
/// AMD SVM attribute selector for access query.
pub(crate) const AMD_SVM_ATTRIB_ACCESS_QUERY: u64 = 0x203;

/// AMD SVM global-coherency mode for fine grained.
pub(crate) const AMD_SVM_GLOBAL_FLAG_FINE_GRAINED: u64 = 0;
/// AMD SVM global-coherency mode for coarse grained.
pub(crate) const AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED: u64 = 1;
/// AMD SVM global-coherency mode for indeterminate.
pub(crate) const AMD_SVM_GLOBAL_FLAG_INDETERMINATE: u64 = 2;

/// DMA-BUF mapping-mode discriminant for none.
pub(crate) const DMABUF_MAPPING_NONE: u64 = 0;
/// DMA-BUF mapping-mode discriminant for pcie.
pub(crate) const DMABUF_MAPPING_PCIE: u64 = 1;
/// Virtual-memory reservation flag that skips registration.
pub(crate) const VMEM_ADDRESS_NO_REGISTER: u64 = 1;
/// Virtual memory backing type for none.
pub(crate) const MEMORY_TYPE_NONE: u32 = 0;
/// Virtual memory backing type for pinned.
pub(crate) const MEMORY_TYPE_PINNED: u32 = 1;
/// Virtual memory access permission for none.
pub(crate) const ACCESS_PERMISSION_NONE: u32 = 0;
/// Virtual memory access permission for ro.
pub(crate) const ACCESS_PERMISSION_RO: u32 = 1;
/// Virtual memory access permission for wo.
pub(crate) const ACCESS_PERMISSION_WO: u32 = 2;
/// Virtual memory access permission for rw.
pub(crate) const ACCESS_PERMISSION_RW: u32 = 3;
/// Virtual memory access permission for ex.
pub(crate) const ACCESS_PERMISSION_EX: u32 = 4;

/// C ABI integer handle type used by legacy finalizer calls.
pub(crate) type HsaHandle = i32;

/// Symbol information selector for type.
pub(crate) const SYMBOL_INFO_TYPE: u32 = 0;
/// Symbol information selector for name length.
pub(crate) const SYMBOL_INFO_NAME_LENGTH: u32 = 1;
/// Symbol information selector for name.
pub(crate) const SYMBOL_INFO_NAME: u32 = 2;
/// Symbol information selector for module name length.
pub(crate) const SYMBOL_INFO_MODULE_NAME_LENGTH: u32 = 3;
/// Symbol information selector for module name.
pub(crate) const SYMBOL_INFO_MODULE_NAME: u32 = 4;
/// Symbol information selector for linkage.
pub(crate) const SYMBOL_INFO_LINKAGE: u32 = 5;
/// Symbol information selector for variable allocation.
pub(crate) const SYMBOL_INFO_VARIABLE_ALLOCATION: u32 = 6;
/// Symbol information selector for variable segment.
pub(crate) const SYMBOL_INFO_VARIABLE_SEGMENT: u32 = 7;
/// Symbol information selector for variable alignment.
pub(crate) const SYMBOL_INFO_VARIABLE_ALIGNMENT: u32 = 8;
/// Symbol information selector for variable size.
pub(crate) const SYMBOL_INFO_VARIABLE_SIZE: u32 = 9;
/// Symbol information selector for variable is const.
pub(crate) const SYMBOL_INFO_VARIABLE_IS_CONST: u32 = 10;
/// Symbol information selector for kernel kernel argument segment size.
pub(crate) const SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 11;
/// Symbol information selector for kernel kernel argument segment alignment.
pub(crate) const SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT: u32 = 12;
/// Symbol information selector for kernel group segment size.
pub(crate) const SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 13;
/// Symbol information selector for kernel private segment size.
pub(crate) const SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 14;
/// Symbol information selector for kernel dynamic callstack.
pub(crate) const SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK: u32 = 15;
/// Symbol information selector for indirect function call convention.
pub(crate) const SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION: u32 = 16;
/// Symbol information selector for is definition.
pub(crate) const SYMBOL_INFO_IS_DEFINITION: u32 = 17;
/// Symbol information selector for kernel call convention.
pub(crate) const SYMBOL_INFO_KERNEL_CALL_CONVENTION: u32 = 18;
/// Symbol information selector for kernel wavefront size.
pub(crate) const SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE: u32 = 19;
/// Symbol information selector for agent.
pub(crate) const SYMBOL_INFO_AGENT: u32 = 20;
/// Symbol information selector for variable address.
pub(crate) const SYMBOL_INFO_VARIABLE_ADDRESS: u32 = 21;
/// Symbol information selector for kernel object.
pub(crate) const SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
/// Symbol information selector for indirect function object.
pub(crate) const SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT: u32 = 23;
/// HSA code-symbol kind for variable.
pub(crate) const SYMBOL_KIND_VARIABLE: u32 = 0;
/// HSA code-symbol kind for kernel.
pub(crate) const SYMBOL_KIND_KERNEL: u32 = 1;
/// HSA code-symbol kind for indirect function.
pub(crate) const SYMBOL_KIND_INDIRECT_FUNCTION: u32 = 2;
/// HSA symbol linkage scope for module.
pub(crate) const SYMBOL_LINKAGE_MODULE: u32 = 0;
/// HSA symbol linkage scope for program.
pub(crate) const SYMBOL_LINKAGE_PROGRAM: u32 = 1;
/// HSA variable allocation scope for agent.
pub(crate) const VARIABLE_ALLOCATION_AGENT: u32 = 0;
/// HSA variable allocation scope for program.
pub(crate) const VARIABLE_ALLOCATION_PROGRAM: u32 = 1;
/// HSA variable memory segment for global.
pub(crate) const VARIABLE_SEGMENT_GLOBAL: u32 = 0;
/// HSA variable memory segment for readonly.
pub(crate) const VARIABLE_SEGMENT_READONLY: u32 = 1;

/// Executable information selector for profile.
pub(crate) const EXECUTABLE_INFO_PROFILE: u32 = 1;
/// Executable information selector for state.
pub(crate) const EXECUTABLE_INFO_STATE: u32 = 2;
/// Executable information selector for default float rounding mode.
pub(crate) const EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 3;
/// HSA executable state for unfrozen.
pub(crate) const EXECUTABLE_STATE_UNFROZEN: u32 = 0;
/// HSA executable state for frozen.
pub(crate) const EXECUTABLE_STATE_FROZEN: u32 = 1;

/// Code-object information selector for version.
pub(crate) const CODE_OBJECT_INFO_VERSION: u32 = 0;
/// Code-object information selector for type.
pub(crate) const CODE_OBJECT_INFO_TYPE: u32 = 1;
/// Code-object information selector for ISA.
pub(crate) const CODE_OBJECT_INFO_ISA: u32 = 2;
/// Code-object information selector for machine model.
pub(crate) const CODE_OBJECT_INFO_MACHINE_MODEL: u32 = 3;
/// Code-object information selector for profile.
pub(crate) const CODE_OBJECT_INFO_PROFILE: u32 = 4;
/// Code-object information selector for default float rounding mode.
pub(crate) const CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 5;
/// HSA code-object kind for program.
pub(crate) const CODE_OBJECT_TYPE_PROGRAM: u32 = 0;

/// HSA signal wait predicate for eq.
pub(crate) const SIGNAL_CONDITION_EQ: u32 = 0;
/// HSA signal wait predicate for ne.
pub(crate) const SIGNAL_CONDITION_NE: u32 = 1;
/// HSA signal wait predicate for lt.
pub(crate) const SIGNAL_CONDITION_LT: u32 = 2;
/// HSA signal wait predicate for gte.
pub(crate) const SIGNAL_CONDITION_GTE: u32 = 3;
/// HSA wait-state hint for blocked.
pub(crate) const WAIT_STATE_BLOCKED: u32 = 0;
/// HSA wait-state hint for active.
pub(crate) const WAIT_STATE_ACTIVE: u32 = 1;

/// Native AMD signal kind for invalid.
pub(crate) const AMD_SIGNAL_KIND_INVALID: i64 = 0;
/// Native AMD signal kind for user.
pub(crate) const AMD_SIGNAL_KIND_USER: i64 = 1;
/// Native AMD signal kind for doorbell.
pub(crate) const AMD_SIGNAL_KIND_DOORBELL: i64 = -1;
/// Native AMD signal kind for legacy doorbell.
pub(crate) const AMD_SIGNAL_KIND_LEGACY_DOORBELL: i64 = -2;
/// Signal-creation flag restricting consumers to AMD GPUs.
pub(crate) const AMD_SIGNAL_AMD_GPU_ONLY: u64 = 1;
/// Signal-creation flag enabling IPC export and attachment.
pub(crate) const AMD_SIGNAL_IPC: u64 = 1 << 1;
/// Queue property bit enabling dispatch timestamps.
pub(crate) const AMD_QUEUE_PROPERTIES_ENABLE_PROFILING: u32 = 1 << 3;

/// AMD queue information selector for agent.
pub(crate) const AMD_QUEUE_INFO_AGENT: u32 = 0;
/// AMD queue information selector for doorbell ID.
pub(crate) const AMD_QUEUE_INFO_DOORBELL_ID: u32 = 1;
/// Queue information selector for use count.
pub(crate) const QUEUE_INFO_USE_COUNT: u32 = 2;
/// Queue information selector for hw ID.
pub(crate) const QUEUE_INFO_HW_ID: u32 = 3;
/// AMD queue information selector for prefetch dispatch major.
pub(crate) const AMD_QUEUE_INFO_PREFETCH_DISPATCH_MAJOR: u32 = 4;
/// AMD queue information selector for prefetch dispatch minor.
pub(crate) const AMD_QUEUE_INFO_PREFETCH_DISPATCH_MINOR: u32 = 5;
/// AMD queue information selector for prefetch barrier major.
pub(crate) const AMD_QUEUE_INFO_PREFETCH_BARRIER_MAJOR: u32 = 6;
/// AMD queue information selector for prefetch barrier minor.
pub(crate) const AMD_QUEUE_INFO_PREFETCH_BARRIER_MINOR: u32 = 7;
/// AMD queue information selector for prefetch ring buffer.
pub(crate) const AMD_QUEUE_INFO_PREFETCH_RING_BUFFER: u32 = 8;
/// AMD queue information selector for properties.
pub(crate) const AMD_QUEUE_INFO_PROPERTIES: u32 = 9;
/// AMD queue information selector for vm fault status.
pub(crate) const AMD_QUEUE_INFO_VM_FAULT_STATUS: u32 = 10;
/// AMD queue information selector for vm fault address.
pub(crate) const AMD_QUEUE_INFO_VM_FAULT_ADDRESS: u32 = 11;
/// AMD queue information selector for vm fault reason.
pub(crate) const AMD_QUEUE_INFO_VM_FAULT_REASON: u32 = 12;
/// AMD queue information selector for engine type.
pub(crate) const AMD_QUEUE_INFO_ENGINE_TYPE: u32 = 13;
/// AMD queue information selector for SDMA engine ID.
pub(crate) const AMD_QUEUE_INFO_SDMA_ENGINE_ID: u32 = 14;
/// AMD queue information selector for read pointer.
pub(crate) const AMD_QUEUE_INFO_READ_POINTER: u32 = 15;
/// AMD queue information selector for write pointer.
pub(crate) const AMD_QUEUE_INFO_WRITE_POINTER: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Opaque agent handle passed by value at the HSA C boundary.
pub struct HsaAgent {
    /// Opaque identifier for the HSA agent.
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Opaque ISA handle identifying a GPU and wavefront variant.
pub struct HsaIsa {
    /// Opaque identifier for the HSA ISA.
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Opaque signal handle naming a stable native signal record.
pub struct HsaSignal {
    /// Opaque identifier for the HSA signal.
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Opaque handle for one PC sampling session.
pub struct HsaPcSampling {
    /// Opaque identifier for the HSA pc sampling.
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// PC sampling method and supported interval range.
pub(crate) struct HsaPcSamplingConfiguration {
    /// PC sampling method selected by the caller.
    pub(crate) method: u32,
    /// Units in which the sampling interval is expressed.
    pub(crate) units: u32,
    /// Smallest supported sampling interval in the selected units.
    pub(crate) minimum_interval: usize,
    /// Largest supported sampling interval in the selected units.
    pub(crate) maximum_interval: usize,
    /// Configuration flags, including the power-of-two interval requirement.
    pub(crate) flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Eight-word token used to export and attach shared memory.
pub struct HsaAmdIpcMemory {
    /// Eight-word IPC token interpreted by native shared-memory attachment.
    pub(crate) handle: [u32; 8],
}

/// IPC signal tokens use the same eight-word representation as memory.
pub(crate) type HsaAmdIpcSignal = HsaAmdIpcMemory;

#[repr(C)]
#[derive(Clone, Copy)]
/// Platform-specific file handle used by AIS transfers.
pub union HsaAmdAisFileHandle {
    /// Opaque platform handle occupying this union arm.
    pub(crate) handle: *mut c_void,
    /// POSIX file descriptor occupying this union arm.
    pub(crate) fd: i32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) pad: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Opaque handle for an imported external semaphore.
pub struct HsaAmdExternalSemaphore {
    /// Opaque identifier for the HSA AMD external semaphore.
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Platform-specific external semaphore handle.
pub union HsaAmdExternalSemaphoreHandle {
    /// Win32 semaphore handle occupying this union arm.
    pub(crate) win32_handle: *mut c_void,
    /// POSIX file descriptor occupying this union arm.
    pub(crate) fd: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Handle format and value supplied for semaphore import.
pub struct HsaAmdExternalSemaphoreHandleDescriptor {
    /// Discriminant selecting the external semaphore handle format.
    pub(crate) handle_type: u32,
    /// Platform handle selected by `handle_type`.
    pub(crate) handle: HsaAmdExternalSemaphoreHandle,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// One AMD SVM attribute selector and its value.
pub struct HsaAmdSvmAttributePair {
    /// AMD SVM attribute selector for this pair.
    pub(crate) attribute: u64,
    /// Value paired with the attribute selector.
    pub(crate) value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Opaque handle naming an HSA memory pool.
pub struct HsaMemoryPool {
    /// Opaque identifier for the HSA memory pool.
    pub(crate) handle: u64,
}

/// Generates the opaque handle ABI entry points.
macro_rules! opaque_handle {
    ($name:ident) => {
        #[doc = concat!("Opaque C ABI handle for `", stringify!($name), "`.")]
        #[repr(C)]
        #[derive(Clone, Copy, Default, Eq, PartialEq)]
        pub struct $name {
            /// Numeric identifier validated against the frontend registry.
            pub(crate) handle: u64,
        }
    };
}

opaque_handle!(HsaExecutable);
opaque_handle!(HsaCodeObject);
opaque_handle!(HsaExtProgram);
opaque_handle!(HsaCodeObjectReader);
opaque_handle!(HsaLoadedCodeObject);
opaque_handle!(HsaExecutableSymbol);
opaque_handle!(HsaCodeSymbol);
opaque_handle!(HsaCallbackData);
opaque_handle!(HsaCache);
opaque_handle!(HsaRegion);
opaque_handle!(HsaSignalGroup);
opaque_handle!(HsaWavefront);
opaque_handle!(HsaAmdVmemAllocHandle);
opaque_handle!(HsaAmdSignalBatch);

#[repr(C)]
#[derive(Clone, Copy)]
/// Size and owner information for a virtual-memory handle.
pub(crate) struct HsaAmdVmemHandleInfo {
    /// Byte size of the caller-provided record for ABI versioning.
    pub(crate) size: usize,
    /// Size in bytes of the underlying virtual-memory allocation.
    pub(crate) alloc_size: usize,
    /// Agent to which this ABI record or union arm refers.
    pub(crate) agent: HsaAgent,
}

const _: () = {
    assert!(std::mem::offset_of!(HsaAmdVmemHandleInfo, alloc_size) == std::mem::size_of::<usize>());
    assert!(std::mem::offset_of!(HsaAmdVmemHandleInfo, agent) == 2 * std::mem::size_of::<usize>());
    assert!(
        std::mem::size_of::<HsaAmdVmemHandleInfo>()
            == 2 * std::mem::size_of::<usize>() + std::mem::size_of::<HsaAgent>()
    );
};
opaque_handle!(HsaExtImage);
opaque_handle!(HsaExtSampler);

// hsa_ext_control_directives_t is passed by value to the legacy finalizer
// entry point, so its complete layout is part of that entry point's ABI.
#[repr(C)]
#[derive(Clone, Copy)]
/// By-value legacy finalizer directives with a fixed ABI layout.
pub(crate) struct HsaExtControlDirectives {
    /// Bitmask selecting active finalizer directives.
    pub(crate) control_directives_mask: u64,
    /// Exceptions that should interrupt finalization.
    pub(crate) break_exceptions_mask: u16,
    /// Exceptions the finalizer should detect.
    pub(crate) detect_exceptions_mask: u16,
    /// Maximum dynamic group memory per dispatch in bytes.
    pub(crate) max_dynamic_group_size: u32,
    /// Maximum number of work-items in a grid.
    pub(crate) max_flat_grid_size: u64,
    /// Maximum number of work-items in one workgroup.
    pub(crate) max_flat_workgroup_size: u32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved1: u32,
    /// Required work-item count in each grid dimension.
    pub(crate) required_grid_size: [u64; 3],
    /// Required work-item count in each workgroup dimension.
    pub(crate) required_workgroup_size: HsaDim3,
    /// Number of dimensions required by the finalizer directive.
    pub(crate) required_dim: u8,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved2: [u8; 75],
}

const _: () = {
    assert!(std::mem::size_of::<HsaDim3>() == 12);
    assert!(std::mem::size_of::<HsaExtControlDirectives>() == 144);
    assert!(std::mem::align_of::<HsaExtControlDirectives>() == 8);
};

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Channel data type and order of an HSA image.
pub(crate) struct HsaExtImageFormat {
    /// Numeric data representation of each image channel.
    pub(crate) channel_type: u32,
    /// Order and presence of channels in each image element.
    pub(crate) channel_order: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Geometry, dimensions, and format of an HSA image.
pub(crate) struct HsaExtImageDescriptor {
    /// Image geometry discriminant, including dimensionality and arrays.
    pub(crate) geometry: u32,
    /// Image width in elements.
    pub(crate) width: usize,
    /// Image height in elements.
    pub(crate) height: usize,
    /// Image depth in elements.
    pub(crate) depth: usize,
    /// Number of layers when the image geometry is an array.
    pub(crate) array_size: usize,
    /// Channel type and channel order of the image.
    pub(crate) format: HsaExtImageFormat,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Image descriptor extended with a mip-level count.
pub(crate) struct HsaExtImageDescriptorV2 {
    /// Image geometry discriminant, including dimensionality and arrays.
    pub(crate) geometry: u32,
    /// Image width in elements.
    pub(crate) width: usize,
    /// Image height in elements.
    pub(crate) height: usize,
    /// Image depth in elements.
    pub(crate) depth: usize,
    /// Number of layers when the image geometry is an array.
    pub(crate) array_size: usize,
    /// Channel type and channel order of the image.
    pub(crate) format: HsaExtImageFormat,
    /// Number of levels in the requested mip chain.
    pub(crate) mipmap_levels: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Required allocation size and alignment for image data.
pub(crate) struct HsaExtImageDataInfo {
    /// Bytes required to store the image data.
    pub(crate) size: usize,
    /// Required alignment of image data in bytes.
    pub(crate) alignment: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Origin and extent of a rectangular image region.
pub(crate) struct HsaExtImageRegion {
    /// Origin of the image region in element coordinates.
    pub(crate) offset: HsaDim3,
    /// Extent of the image region in elements.
    pub(crate) range: HsaDim3,
}

#[repr(C)]
/// Versioned AMD image descriptor with inline hardware words.
pub(crate) struct HsaAmdImageDescriptor {
    /// Version of this C ABI record.
    pub(crate) version: u32,
    /// Device identifier encoded in the AMD image descriptor.
    pub(crate) device_id: u32,
    /// First word of the variable-length native image descriptor.
    pub(crate) data: [u32; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
/// Coordinate, filter, and address modes of a sampler.
pub(crate) struct HsaExtSamplerDescriptor {
    /// Whether sampler coordinates are normalized.
    pub(crate) coordinate_mode: u32,
    /// Filtering rule used to sample image elements.
    pub(crate) filter_mode: u32,
    /// Addressing rule for out-of-range sampler coordinates.
    pub(crate) address_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Sampler descriptor with per-axis address and mip filter modes.
pub(crate) struct HsaExtSamplerDescriptorV2 {
    /// Whether sampler coordinates are normalized.
    pub(crate) coordinate_mode: u32,
    /// Filtering rule used to sample image elements.
    pub(crate) filter_mode: u32,
    /// Filtering rule used between mip levels.
    pub(crate) mipmap_filter_mode: u32,
    /// Addressing rules for X, Y, and Z coordinates.
    pub(crate) address_modes: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Agent and permissions for virtual-memory access.
pub struct HsaAmdMemoryAccessDesc {
    /// Access bits granted to the named agent.
    pub(crate) permissions: u32,
    /// Agent receiving the requested memory access.
    pub(crate) agent_handle: HsaAgent,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
/// Portable 16-byte fabric identity for virtual-memory sharing.
pub struct HsaFabricHandle {
    /// Sixteen-byte fabric identity used for sharing.
    pub(crate) handle: [u8; 16],
}

/// C callback signature for loaded code object callback.
pub(crate) type LoadedCodeObjectCallback =
    Option<unsafe extern "C" fn(HsaExecutable, HsaLoadedCodeObject, *mut c_void) -> Status>;
/// C callback signature for executable callback.
pub(crate) type ExecutableCallback =
    Option<unsafe extern "C" fn(HsaExecutable, *mut c_void) -> Status>;
/// C callback signature for code object alloc callback.
pub(crate) type CodeObjectAllocCallback =
    Option<unsafe extern "C" fn(usize, HsaCallbackData, *mut *mut c_void) -> Status>;
/// C callback signature for code object symbol callback.
pub(crate) type CodeObjectSymbolCallback =
    Option<unsafe extern "C" fn(HsaCodeObject, HsaCodeSymbol, *mut c_void) -> Status>;
/// C callback signature for executable symbol callback.
pub(crate) type ExecutableSymbolCallback =
    Option<unsafe extern "C" fn(HsaExecutable, HsaExecutableSymbol, *mut c_void) -> Status>;
/// C callback signature for agent executable symbol callback.
pub(crate) type AgentExecutableSymbolCallback = Option<
    unsafe extern "C" fn(HsaExecutable, HsaAgent, HsaExecutableSymbol, *mut c_void) -> Status,
>;
/// C callback signature for pc sampling configuration callback.
pub(crate) type PcSamplingConfigurationCallback =
    Option<unsafe extern "C" fn(*const HsaPcSamplingConfiguration, *mut c_void) -> Status>;
/// C callback signature for pc sampling data copy callback.
pub(crate) type PcSamplingDataCopyCallback =
    Option<unsafe extern "C" fn(*mut c_void, usize, *mut c_void) -> Status>;
/// C callback signature for pc sampling data ready callback.
pub(crate) type PcSamplingDataReadyCallback = Option<
    unsafe extern "C" fn(*mut c_void, usize, usize, PcSamplingDataCopyCallback, *mut c_void),
>;

/// AMD loader storage kind for in-memory code objects.
pub(crate) const LOADER_STORAGE_MEMORY: u32 = 2;
/// AMD loader object kind for program.
pub(crate) const LOADER_OBJECT_KIND_PROGRAM: u32 = 1;
/// AMD loader object kind for agent.
pub(crate) const LOADER_OBJECT_KIND_AGENT: u32 = 2;
/// Loader information selector for executable.
pub(crate) const LOADER_INFO_EXECUTABLE: u32 = 1;
/// Loader information selector for kind.
pub(crate) const LOADER_INFO_KIND: u32 = 2;
/// Loader information selector for agent.
pub(crate) const LOADER_INFO_AGENT: u32 = 3;
/// Loader information selector for storage type.
pub(crate) const LOADER_INFO_STORAGE_TYPE: u32 = 4;
/// Loader information selector for storage memory base.
pub(crate) const LOADER_INFO_STORAGE_MEMORY_BASE: u32 = 5;
/// Loader information selector for storage memory size.
pub(crate) const LOADER_INFO_STORAGE_MEMORY_SIZE: u32 = 6;
/// Loader information selector for storage file.
pub(crate) const LOADER_INFO_STORAGE_FILE: u32 = 7;
/// Loader information selector for load delta.
pub(crate) const LOADER_INFO_LOAD_DELTA: u32 = 8;
/// Loader information selector for load base.
pub(crate) const LOADER_INFO_LOAD_BASE: u32 = 9;
/// Loader information selector for load size.
pub(crate) const LOADER_INFO_LOAD_SIZE: u32 = 10;
/// Loader information selector for URI length.
pub(crate) const LOADER_INFO_URI_LENGTH: u32 = 11;
/// Loader information selector for URI.
pub(crate) const LOADER_INFO_URI: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy)]
/// Storage and loaded-address ranges of one code-object segment.
pub(crate) struct HsaLoaderSegmentDescriptor {
    /// Agent to which this ABI record or union arm refers.
    pub(crate) agent: HsaAgent,
    /// Executable that owns this loaded segment.
    pub(crate) executable: HsaExecutable,
    /// Kind of storage containing the code object.
    pub(crate) code_object_storage_type: u32,
    /// Base address of the code-object storage.
    pub(crate) code_object_storage_base: *const c_void,
    /// Size of the code-object storage in bytes.
    pub(crate) code_object_storage_size: usize,
    /// Byte offset of this segment within its code-object storage.
    pub(crate) code_object_storage_offset: usize,
    /// Base device address of the loaded segment.
    pub(crate) segment_base: *const c_void,
    /// Size of the loaded segment in bytes.
    pub(crate) segment_size: usize,
}

#[repr(C)]
/// Public queue header shared with HSA clients and device code.
pub struct HsaQueue {
    /// Single- or multi-producer queue mode.
    pub(crate) queue_type: u32,
    /// Bitmask of packet features supported by the queue.
    pub(crate) features: u32,
    /// Base address of the packet ring buffer.
    pub(crate) base_address: *mut c_void,
    /// Signal used to notify the device of new packets.
    pub(crate) doorbell_signal: HsaSignal,
    /// Ring capacity in packets for AQL, or bytes for SDMA.
    pub(crate) size: u32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: u32,
    /// Unique public queue identifier for this runtime generation.
    pub(crate) id: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Compute-engine options for an AMD queue creation request.
pub(crate) struct HsaAmdComputeQueueParams {
    /// Array of CU mask words, or null to enable every CU.
    pub(crate) cu_mask: *const u32,
    /// Single- or multi-producer queue mode.
    pub(crate) queue_type: u32,
    /// Initial private-segment bytes allocated per work-item.
    pub(crate) private_segment_size: u32,
    /// Number of CU mask bits; the array contains one word per 32 bits.
    pub(crate) cu_mask_count: u32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// SDMA engine options for an AMD queue creation request.
pub(crate) struct HsaAmdSdmaQueueParams {
    /// Native SDMA engine selected for the queue.
    pub(crate) engine_id: u32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: [u32; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Engine-specific AMD queue creation parameters.
pub(crate) union HsaAmdQueueEngineParams {
    /// Compute-queue parameters in this union arm.
    pub(crate) compute: HsaAmdComputeQueueParams,
    /// SDMA-queue parameters in this union arm.
    pub(crate) sdma: HsaAmdSdmaQueueParams,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: [u8; 32],
}

#[repr(C)]
/// Versioned AMD queue creation request and resulting queue pointer.
pub(crate) struct HsaAmdQueueCreateDesc {
    /// Version of this C ABI record.
    pub(crate) version: u16,
    /// Memory placement flags for the queue ring and descriptor.
    pub(crate) flags: u16,
    /// Engine-kind discriminant selecting the union arm.
    pub(crate) engine_type: u8,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved_header: [u8; 3],
    /// Requested queue ring capacity in bytes.
    pub(crate) queue_size_bytes: u32,
    /// Requested queue scheduling priority.
    pub(crate) priority: u32,
    /// Optional callback for asynchronous queue errors.
    pub(crate) callback: QueueErrorCallback,
    /// Opaque application argument passed to the error callback.
    pub(crate) callback_data: *mut c_void,
    /// Output queue pointer written on success and cleared on failure.
    pub(crate) queue: *mut HsaQueue,
    /// Engine-specific parameters for queue creation.
    pub(crate) engine: HsaAmdQueueEngineParams,
    /// Traffic-class hint; version one requires zero.
    pub(crate) traffic_class: u32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: [u8; 20],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Public AMD pointer classification, bounds, and ownership record.
pub(crate) struct HsaAmdPointerInfo {
    /// Caller record size in bytes, used for ABI version negotiation.
    pub(crate) size: u32,
    /// Allocation category reported for the queried pointer.
    pub(crate) pointer_type: u32,
    /// Allocation base address visible to non-host agents.
    pub(crate) agent_base_address: *mut c_void,
    /// Allocation base address visible to the host agent.
    pub(crate) host_base_address: *mut c_void,
    /// Total size of the tracked allocation in bytes.
    pub(crate) size_in_bytes: usize,
    /// Opaque caller data attached to the allocation.
    pub(crate) user_data: *mut c_void,
    /// Agent that owns the allocation.
    pub(crate) agent_owner: HsaAgent,
    /// Effective memory-pool global flags for this allocation.
    pub(crate) global_flags: u32,
    /// Whether host memory registration backs this pointer.
    pub(crate) registered: bool,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) registered_padding: [u8; 3],
    /// Native allocation property flags reported for this pointer.
    pub(crate) alloc_flags: u32,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) tail_padding: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Three 32-bit coordinates used by image regions.
pub struct HsaDim3 {
    /// X coordinate or extent component.
    pub(crate) x: u32,
    /// Y coordinate or extent component.
    pub(crate) y: u32,
    /// Z coordinate or extent component.
    pub(crate) z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Three 64-bit dimensions used by AMD agent queries.
pub(crate) struct HsaAmdDim3 {
    /// X coordinate or extent component.
    pub(crate) x: u64,
    /// Y coordinate or extent component.
    pub(crate) y: u64,
    /// Z coordinate or extent component.
    pub(crate) z: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Two-word locally unique identifier.
pub(crate) struct HsaLuid {
    /// Low half of the identifier.
    pub(crate) low: u32,
    /// High half of the identifier.
    pub(crate) high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Base address and byte strides of a pitched host buffer.
pub struct HsaPitchedPtr {
    /// Base address of the pitched host buffer.
    pub(crate) base: *mut c_void,
    /// Byte stride between adjacent rows.
    pub(crate) pitch: usize,
    /// Byte stride between adjacent slices.
    pub(crate) slice: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Predicate and address evaluated before a batch copy.
pub struct HsaAmdMemoryCopyWait {
    /// Wait predicate selected for the memory-copy control.
    pub(crate) function: u16,
    /// Memory scope for the wait or signal operation.
    pub(crate) scope: u16,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: u32,
    /// Address polled before the copy; null disables the wait.
    pub(crate) address: *mut c_void,
    /// Reference value used by the wait predicate.
    pub(crate) value: u64,
    /// Bits compared by the memory-copy wait predicate.
    pub(crate) mask: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Memory operation performed after a batch copy.
pub struct HsaAmdMemoryCopySignal {
    /// Operation kind encoded by this copy control or descriptor.
    pub(crate) operation: u16,
    /// Memory scope for the wait or signal operation.
    pub(crate) scope: u16,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: u32,
    /// Target address of the post-copy atomic operation.
    pub(crate) address: *mut c_void,
    /// Operand used by the post-copy atomic operation.
    pub(crate) data: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Single-agent or agent-list destination selector.
pub union HsaAmdMemoryCopyDestinationAgent {
    /// Agent to which this ABI record or union arm refers.
    pub(crate) agent: HsaAgent,
    /// Caller-owned destination-agent list for a multi-copy operation.
    pub(crate) list: *const HsaAgent,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Single byte count or list of per-entry byte counts.
pub union HsaAmdMemoryCopySize {
    /// Number of bytes copied by this operation.
    pub(crate) bytes: usize,
    /// Caller-owned byte-count list for a multi-copy operation.
    pub(crate) list: *const usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Versioned descriptor for one or more AMD copy operations.
pub struct HsaAmdMemoryCopyOp {
    /// Version of this C ABI record.
    pub(crate) version: u16,
    /// Operation kind encoded by this copy control or descriptor.
    pub(crate) operation: u16,
    /// Number of multi-copy entries; zero selects the scalar form.
    pub(crate) entry_count: u16,
    /// Requested queue or copy traffic class.
    pub(crate) traffic_class: u16,
    /// Signal decremented when the copy operation completes.
    pub(crate) completion_signal: HsaSignal,
    /// Source pointer or caller-owned source-pointer list.
    pub(crate) source: *mut c_void,
    /// Agent that can access the copy source.
    pub(crate) source_agent: HsaAgent,
    /// Agent or agent list that can access the destination.
    pub(crate) destination_agent: HsaAmdMemoryCopyDestinationAgent,
    /// Destination pointer or caller-owned destination list.
    pub(crate) destination: *mut c_void,
    /// Scalar copy byte count or caller-owned list of byte counts.
    pub(crate) size: HsaAmdMemoryCopySize,
    /// Second swap size; zero for operations that do not use it.
    pub(crate) secondary_size: usize,
    /// Optional wait control executed before the copy.
    pub(crate) wait: HsaAmdMemoryCopyWait,
    /// Optional memory signal operation after the copy.
    pub(crate) signal: HsaAmdMemoryCopySignal,
    /// Reserved ABI storage; initialized to zero.
    pub(crate) reserved: [u64; 1],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Start and end timestamps returned for a completed operation.
pub struct ProfilingTime {
    /// Start timestamp for this profiling time.
    pub(crate) start: u64,
    /// End timestamp for this profiling time.
    pub(crate) end: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Correlated GPU, CPU, and system clock readings.
pub(crate) struct HsaAmdClockCounters {
    /// GPU clock counter sampled at the correlation point.
    pub(crate) gpu_clock_counter: u64,
    /// CPU clock counter sampled at the correlation point.
    pub(crate) cpu_clock_counter: u64,
    /// System clock counter sampled at the correlation point.
    pub(crate) system_clock_counter: u64,
    /// System clock frequency in hertz.
    pub(crate) system_clock_frequency: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Latency, bandwidth, atomic, and coherence properties of a link.
pub(crate) struct PoolLinkInfo {
    /// Minimum reported link latency.
    pub(crate) min_latency: u32,
    /// Maximum reported link latency.
    pub(crate) max_latency: u32,
    /// Minimum reported link bandwidth.
    pub(crate) min_bandwidth: u32,
    /// Maximum reported link bandwidth.
    pub(crate) max_bandwidth: u32,
    /// Whether 32-bit atomics are supported over this link.
    pub(crate) atomic_support_32bit: bool,
    /// Whether 64-bit atomics are supported over this link.
    pub(crate) atomic_support_64bit: bool,
    /// Whether the link supports coherent access.
    pub(crate) coherent_support: bool,
    /// HSA link-kind discriminant.
    pub(crate) link_type: u32,
    /// Relative NUMA distance reported for this link.
    pub(crate) numa_distance: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// AMD system event kind and three-word payload.
pub struct HsaAmdEvent {
    /// AMD system event discriminant selecting the payload format.
    pub(crate) event_type: u32,
    /// Three ABI words holding event-specific data.
    pub(crate) payload: [u64; 3],
}

/// AMD system event kind for a GPU memory fault.
pub(crate) const AMD_GPU_MEMORY_FAULT_EVENT: u32 = 0;
/// AMD system event kind for a GPU hardware exception.
pub(crate) const AMD_GPU_HW_EXCEPTION_EVENT: u32 = 1;
/// AMD system event kind for runtime shutdown.
pub(crate) const AMD_SYSTEM_SHUTDOWN_EVENT: u32 = 3;
/// AMD reset-type value for an unspecified GPU reset.
pub(crate) const AMD_HW_EXCEPTION_RESET_TYPE_OTHER: u32 = 1;
/// AMD hardware-exception cause bit for GPU hang.
pub(crate) const AMD_HW_EXCEPTION_CAUSE_GPU_HANG: u32 = 1;
/// AMD hardware-exception cause bit for ecc.
pub(crate) const AMD_HW_EXCEPTION_CAUSE_ECC: u32 = 1 << 1;
/// Bit flag for AMD memory fault page not present.
pub(crate) const AMD_MEMORY_FAULT_PAGE_NOT_PRESENT: u32 = 1;
/// Bit flag for AMD memory fault read only.
pub(crate) const AMD_MEMORY_FAULT_READ_ONLY: u32 = 1 << 1;
/// Bit flag for AMD memory fault no execute.
pub(crate) const AMD_MEMORY_FAULT_NO_EXECUTE: u32 = 1 << 2;
/// Bit flag for AMD memory fault dram ecc.
pub(crate) const AMD_MEMORY_FAULT_DRAM_ECC: u32 = 1 << 4;
/// Bit flag for AMD memory fault imprecise.
pub(crate) const AMD_MEMORY_FAULT_IMPRECISE: u32 = 1 << 5;
/// Bit flag for AMD memory fault sram ecc.
pub(crate) const AMD_MEMORY_FAULT_SRAM_ECC: u32 = 1 << 6;
/// Bit flag for AMD memory fault hang.
pub(crate) const AMD_MEMORY_FAULT_HANG: u32 = 1 << 31;

/// C callback signature for agent callback.
pub(crate) type AgentCallback = Option<unsafe extern "C" fn(HsaAgent, *mut c_void) -> Status>;
/// C callback signature for ISA callback.
pub(crate) type IsaCallback = Option<unsafe extern "C" fn(HsaIsa, *mut c_void) -> Status>;
/// C callback signature for wavefront callback.
pub(crate) type WavefrontCallback =
    Option<unsafe extern "C" fn(HsaWavefront, *mut c_void) -> Status>;
/// C callback signature for pool callback.
pub(crate) type PoolCallback = Option<unsafe extern "C" fn(HsaMemoryPool, *mut c_void) -> Status>;
/// C callback signature for region callback.
pub(crate) type RegionCallback = Option<unsafe extern "C" fn(HsaRegion, *mut c_void) -> Status>;
/// C callback signature for cache callback.
pub(crate) type CacheCallback = Option<unsafe extern "C" fn(HsaCache, *mut c_void) -> Status>;
/// C callback signature for queue error callback.
pub(crate) type QueueErrorCallback =
    Option<unsafe extern "C" fn(Status, *mut HsaQueue, *mut c_void)>;
/// C callback signature for signal handler.
pub(crate) type SignalHandler = Option<unsafe extern "C" fn(SignalValue, *mut c_void) -> bool>;
/// C callback run by the asynchronous dispatcher.
pub(crate) type AsyncFunction = Option<unsafe extern "C" fn(*mut c_void)>;
/// C callback invoked when a tracked allocation is freed.
pub(crate) type DeallocationCallbackFn = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>;
/// C callback signature for system event handler.
pub(crate) type SystemEventHandler =
    unsafe extern "C" fn(*const HsaAmdEvent, *mut c_void) -> Status;
/// C callback signature for system event callback.
pub(crate) type SystemEventCallback = Option<SystemEventHandler>;
/// C callback used to allocate caller-owned pointer-info storage.
pub(crate) type PointerAllocator = Option<unsafe extern "C" fn(usize) -> *mut c_void>;

/// NUL-terminated diagnostic returned for `SUCCESS` by `hsa_status_string`.
pub(crate) const STATUS_SUCCESS: &[u8] =
    b"HSA_STATUS_SUCCESS: The function has been executed successfully.\0";
/// NUL-terminated diagnostic returned for `INFO_BREAK` by `hsa_status_string`.
pub(crate) const STATUS_INFO_BREAK: &[u8] = b"HSA_STATUS_INFO_BREAK: A traversal over a list of elements has been interrupted by the application before completing.\0";
/// NUL-terminated diagnostic returned for `ERROR` by `hsa_status_string`.
pub(crate) const STATUS_ERROR: &[u8] = b"HSA_STATUS_ERROR: A generic error has occurred.\0";
/// NUL-terminated diagnostic returned for `INVALID_ARGUMENT` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_ARGUMENT: &[u8] = b"HSA_STATUS_ERROR_INVALID_ARGUMENT: One of the actual arguments does not meet a precondition stated in the documentation of the corresponding formal argument.\0";
/// NUL-terminated diagnostic returned for `INVALID_QUEUE_CREATION` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_QUEUE_CREATION: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_QUEUE_CREATION: The requested queue creation is not valid.\0";
/// NUL-terminated diagnostic returned for `INVALID_ALLOCATION` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_ALLOCATION: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_ALLOCATION: The requested allocation is not valid.\0";
/// NUL-terminated diagnostic returned for `INVALID_AGENT` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_AGENT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_AGENT: The agent is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_REGION` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_REGION: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_REGION: The memory region is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_SIGNAL` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_SIGNAL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_SIGNAL: The signal is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_QUEUE` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_QUEUE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_QUEUE: The queue is invalid.\0";
/// NUL-terminated diagnostic returned for `OUT_OF_RESOURCES` by
/// `hsa_status_string`.
pub(crate) const STATUS_OUT_OF_RESOURCES: &[u8] = b"HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events.\0";
/// NUL-terminated diagnostic returned for `INVALID_PACKET_FORMAT` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_PACKET_FORMAT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is malformed.\0";
/// NUL-terminated diagnostic returned for `RESOURCE_FREE` by
/// `hsa_status_string`.
pub(crate) const STATUS_RESOURCE_FREE: &[u8] =
    b"HSA_STATUS_ERROR_RESOURCE_FREE: An error has been detected while releasing a resource.\0";
/// NUL-terminated diagnostic returned for `NOT_INITIALIZED` by
/// `hsa_status_string`.
pub(crate) const STATUS_NOT_INITIALIZED: &[u8] = b"HSA_STATUS_ERROR_NOT_INITIALIZED: An API other than hsa_init has been invoked while the reference count of the HSA runtime is zero.\0";
/// NUL-terminated diagnostic returned for `REFCOUNT_OVERFLOW` by
/// `hsa_status_string`.
pub(crate) const STATUS_REFCOUNT_OVERFLOW: &[u8] = b"HSA_STATUS_ERROR_REFCOUNT_OVERFLOW: The maximum reference count for the object has been reached.\0";
/// NUL-terminated diagnostic returned for `INCOMPATIBLE_ARGUMENTS` by
/// `hsa_status_string`.
pub(crate) const STATUS_INCOMPATIBLE_ARGUMENTS: &[u8] = b"HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS: The arguments passed to a functions are not compatible.\0";
/// NUL-terminated diagnostic returned for `INVALID_INDEX` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_INDEX: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_INDEX: The index is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_ISA` by `hsa_status_string`.
pub(crate) const STATUS_INVALID_ISA: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_ISA: The instruction set architecture is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_CODE_OBJECT` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_CODE_OBJECT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CODE_OBJECT: The code object is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_EXECUTABLE` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_EXECUTABLE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_EXECUTABLE: The executable is invalid.\0";
/// NUL-terminated diagnostic returned for `FROZEN_EXECUTABLE` by
/// `hsa_status_string`.
pub(crate) const STATUS_FROZEN_EXECUTABLE: &[u8] =
    b"HSA_STATUS_ERROR_FROZEN_EXECUTABLE: The executable is frozen.\0";
/// NUL-terminated diagnostic returned for `INVALID_SYMBOL_NAME` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_SYMBOL_NAME: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_SYMBOL_NAME: There is no symbol with the given name.\0";
/// NUL-terminated diagnostic returned for `VARIABLE_ALREADY_DEFINED` by
/// `hsa_status_string`.
pub(crate) const STATUS_VARIABLE_ALREADY_DEFINED: &[u8] =
    b"HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED: The variable is already defined.\0";
/// NUL-terminated diagnostic returned for `VARIABLE_UNDEFINED` by
/// `hsa_status_string`.
pub(crate) const STATUS_VARIABLE_UNDEFINED: &[u8] =
    b"HSA_STATUS_ERROR_VARIABLE_UNDEFINED: The variable is undefined.\0";
/// NUL-terminated diagnostic returned for `EXCEPTION` by `hsa_status_string`.
pub(crate) const STATUS_EXCEPTION: &[u8] =
    b"HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception.\0";
/// NUL-terminated diagnostic returned for `INVALID_ISA_NAME` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_ISA_NAME: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_ISA_NAME: The instruction set architecture name is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_CODE_SYMBOL` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_CODE_SYMBOL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CODE_SYMBOL: The code object symbol is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_EXECUTABLE_SYMBOL` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_EXECUTABLE_SYMBOL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL: The executable symbol is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_FILE` by `hsa_status_string`.
pub(crate) const STATUS_INVALID_FILE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_FILE: The file descriptor is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_CODE_OBJECT_READER` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_CODE_OBJECT_READER: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER: The code object reader is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_CACHE` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_CACHE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CACHE: The cache is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_WAVEFRONT` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_WAVEFRONT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_WAVEFRONT: The wavefront is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_SIGNAL_GROUP` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_SIGNAL_GROUP: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP: The signal group is invalid.\0";
/// NUL-terminated diagnostic returned for `INVALID_RUNTIME_STATE` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_RUNTIME_STATE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_RUNTIME_STATE: The HSA runtime is not in the configuration state.\0";
/// NUL-terminated diagnostic returned for `FATAL` by `hsa_status_string`.
pub(crate) const STATUS_FATAL: &[u8] =
    b"HSA_STATUS_ERROR_FATAL: The queue received an error that may require process termination.\0";
/// NUL-terminated diagnostic returned for `INVALID_MEMORY_POOL` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_MEMORY_POOL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_MEMORY_POOL: The memory pool is invalid.\0";
/// NUL-terminated diagnostic returned for `MEMORY_APERTURE_VIOLATION` by
/// `hsa_status_string`.
pub(crate) const STATUS_MEMORY_APERTURE_VIOLATION: &[u8] = b"HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address.\0";
/// NUL-terminated diagnostic returned for `ILLEGAL_INSTRUCTION` by
/// `hsa_status_string`.
pub(crate) const STATUS_ILLEGAL_INSTRUCTION: &[u8] = b"HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION: The agent attempted to execute an illegal shader instruction.\0";
/// NUL-terminated diagnostic returned for `MEMORY_FAULT` by `hsa_status_string`.
pub(crate) const STATUS_MEMORY_FAULT: &[u8] =
    b"HSA_STATUS_ERROR_MEMORY_FAULT: Agent attempted to access an inaccessible address.\0";
/// NUL-terminated diagnostic returned for `CU_MASK_REDUCED` by
/// `hsa_status_string`.
pub(crate) const STATUS_CU_MASK_REDUCED: &[u8] = b"HSA_STATUS_CU_MASK_REDUCED: The CU mask was successfully set but the mask attempted to enable a CU which was disabled for the process.  CUs disabled for the process remain disabled.\0";
/// NUL-terminated diagnostic returned for `OUT_OF_REGISTERS` by
/// `hsa_status_string`.
pub(crate) const STATUS_OUT_OF_REGISTERS: &[u8] = b"HSA_STATUS_ERROR_OUT_OF_REGISTERS: Kernel has requested more VGPRs than are available on this agent\0";
/// NUL-terminated diagnostic returned for `RESOURCE_BUSY` by
/// `hsa_status_string`.
pub(crate) const STATUS_RESOURCE_BUSY: &[u8] =
    b"HSA_STATUS_ERROR_RESOURCE_BUSY: Resource is busy or temporarily unavailable.\0";
/// NUL-terminated diagnostic returned for `NOT_SUPPORTED` by
/// `hsa_status_string`.
pub(crate) const STATUS_NOT_SUPPORTED: &[u8] =
    b"HSA_STATUS_ERROR_NOT_SUPPORTED: Request is not supported by this system.\0";
/// NUL-terminated diagnostic returned for `XNACK_DISABLED` by
/// `hsa_status_string`.
pub(crate) const STATUS_XNACK_DISABLED: &[u8] = b"HSA_STATUS_ERROR_XNACK_DISABLED: Xnack is disabled on this system, but required by the requested operation.\0";
/// NUL-terminated diagnostic returned for `INVALID_DISPATCH_PARAMETERS` by
/// `hsa_status_string`.
pub(crate) const STATUS_INVALID_DISPATCH_PARAMETERS: &[u8] = b"HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS: Kernel dispatch packet parameters exceed hardware limits for this agent (e.g. register usage, work-group dimensions, or other dispatch constraints)\0";
/// NUL-terminated diagnostic returned for `RESOURCE_NOT_READY` by
/// `hsa_status_string`.
pub(crate) const STATUS_RESOURCE_NOT_READY: &[u8] = b"HSA_STATUS_ERROR_RESOURCE_NOT_READY: Underlying resource is a valid resource, but it is not ready to be used.\0";
/// NUL-terminated diagnostic returned for `IMAGE_FORMAT_UNSUPPORTED` by
/// `hsa_status_string`.
pub(crate) const STATUS_IMAGE_FORMAT_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED: Image format is not supported.\0";
/// NUL-terminated diagnostic returned for `IMAGE_SIZE_UNSUPPORTED` by
/// `hsa_status_string`.
pub(crate) const STATUS_IMAGE_SIZE_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED: Image size is not supported.\0";
/// NUL-terminated diagnostic returned for `IMAGE_PITCH_UNSUPPORTED` by
/// `hsa_status_string`.
pub(crate) const STATUS_IMAGE_PITCH_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED: Image pitch is not supported or invalid.\0";
/// NUL-terminated diagnostic returned for `SAMPLER_DESCRIPTOR_UNSUPPORTED` by
/// `hsa_status_string`.
pub(crate) const STATUS_SAMPLER_DESCRIPTOR_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED: Sampler descriptor is not supported or invalid.\0";

/// Returns a pointer to a statically NUL-terminated status string.
pub(crate) const fn c_string(bytes: &'static [u8]) -> *const c_char {
    bytes.as_ptr().cast()
}
