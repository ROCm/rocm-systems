//! Rust representations of the HSA core and AMD-extension C ABI.
//!
//! Numeric constants, handle encodings, callback signatures, and `repr(C)`
//! records in this module are compatibility data, not internal conveniences.
//! They mirror the vendored HSA headers used by this workspace. Implementation
//! modules consume these definitions but must not reinterpret or renumber them.
//! Layout-sensitive records are checked by compile-time assertions and ABI
//! tests where the public contract requires an exact size or field offset.

use std::ffi::{c_char, c_void};

pub(crate) type Status = u32;
pub(crate) type SignalValue = i64;

pub(crate) const SUCCESS: Status = 0;
pub(crate) const INFO_BREAK: Status = 1;
pub(crate) const ERROR: Status = 0x1000;
pub(crate) const INVALID_ARGUMENT: Status = 0x1001;
pub(crate) const INVALID_QUEUE_CREATION: Status = 0x1002;
pub(crate) const INVALID_ALLOCATION: Status = 0x1003;
pub(crate) const INVALID_AGENT: Status = 0x1004;
pub(crate) const INVALID_REGION: Status = 0x1005;
pub(crate) const INVALID_SIGNAL: Status = 0x1006;
pub(crate) const INVALID_QUEUE: Status = 0x1007;
pub(crate) const OUT_OF_RESOURCES: Status = 0x1008;
pub(crate) const INVALID_PACKET_FORMAT: Status = 0x1009;
pub(crate) const RESOURCE_FREE: Status = 0x100a;
pub(crate) const NOT_INITIALIZED: Status = 0x100b;
pub(crate) const REFCOUNT_OVERFLOW: Status = 0x100c;
pub(crate) const INCOMPATIBLE_ARGUMENTS: Status = 0x100d;
pub(crate) const INVALID_INDEX: Status = 0x100e;
pub(crate) const INVALID_ISA: Status = 0x100f;
pub(crate) const INVALID_CODE_OBJECT: Status = 0x1010;
pub(crate) const INVALID_EXECUTABLE: Status = 0x1011;
pub(crate) const FROZEN_EXECUTABLE: Status = 0x1012;
pub(crate) const INVALID_SYMBOL_NAME: Status = 0x1013;
pub(crate) const VARIABLE_ALREADY_DEFINED: Status = 0x1014;
pub(crate) const VARIABLE_UNDEFINED: Status = 0x1015;
pub(crate) const EXCEPTION: Status = 0x1016;
pub(crate) const INVALID_ISA_NAME: Status = 0x1017;
pub(crate) const INVALID_CODE_SYMBOL: Status = 0x1018;
pub(crate) const INVALID_EXECUTABLE_SYMBOL: Status = 0x1019;
pub(crate) const INVALID_FILE: Status = 0x1020;
pub(crate) const INVALID_CODE_OBJECT_READER: Status = 0x1021;
pub(crate) const INVALID_CACHE: Status = 0x1022;
pub(crate) const INVALID_WAVEFRONT: Status = 0x1023;
pub(crate) const INVALID_SIGNAL_GROUP: Status = 0x1024;
pub(crate) const INVALID_RUNTIME_STATE: Status = 0x1025;
pub(crate) const FATAL: Status = 0x1026;
pub(crate) const INVALID_MEMORY_POOL: Status = 40;
pub(crate) const MEMORY_APERTURE_VIOLATION: Status = 41;
pub(crate) const ILLEGAL_INSTRUCTION: Status = 42;
pub(crate) const MEMORY_FAULT: Status = 43;
pub(crate) const CU_MASK_REDUCED: Status = 44;
pub(crate) const OUT_OF_REGISTERS: Status = 45;
pub(crate) const RESOURCE_BUSY: Status = 46;
pub(crate) const NOT_SUPPORTED: Status = 47;
pub(crate) const XNACK_DISABLED: Status = 48;
pub(crate) const INVALID_DISPATCH_PARAMETERS: Status = 49;
pub(crate) const RESOURCE_NOT_READY: Status = 50;
pub(crate) const IMAGE_FORMAT_UNSUPPORTED: Status = 0x3000;
pub(crate) const IMAGE_SIZE_UNSUPPORTED: Status = 0x3001;
pub(crate) const IMAGE_PITCH_UNSUPPORTED: Status = 0x3002;
pub(crate) const SAMPLER_DESCRIPTOR_UNSUPPORTED: Status = 0x3003;

pub(crate) const CPU_AGENT: u64 = 0x4853_4100_0000_0001;
pub(crate) const GPU_AGENT_BASE: u64 = 0x4853_4100_0010_0000;
pub(crate) const ISA_BASE: u64 = 0x4853_4100_0020_0000;
pub(crate) const ISA_COUNT_PER_GPU: u64 = 2;
pub(crate) const WAVEFRONT_BASE: u64 = 0x4853_4100_0028_0000;
pub(crate) const CPU_POOL_FINE: u64 = 0x4853_4100_0030_0001;
pub(crate) const CPU_POOL_EXTENDED: u64 = 0x4853_4100_0030_0002;
pub(crate) const CPU_POOL_KERNARG: u64 = 0x4853_4100_0030_0003;
pub(crate) const CPU_POOL_COARSE: u64 = 0x4853_4100_0030_0004;
pub(crate) const GPU_POOL_BASE: u64 = 0x4853_4100_0040_0000;
pub(crate) const CACHE_BASE: u64 = 0x4853_4100_0050_0000;

pub(crate) const EXTENSION_FINALIZER: u16 = 0;
pub(crate) const EXTENSION_IMAGES: u16 = 1;
pub(crate) const EXTENSION_PERFORMANCE_COUNTERS: u16 = 2;
pub(crate) const EXTENSION_PROFILING_EVENTS: u16 = 3;
pub(crate) const EXTENSION_AMD_PROFILER: u16 = 0x200;
pub(crate) const EXTENSION_AMD_LOADER: u16 = 0x201;
pub(crate) const EXTENSION_AMD_AQLPROFILE: u16 = 0x202;
pub(crate) const EXTENSION_AMD_PC_SAMPLING: u16 = 0x203;

pub(crate) const DEVICE_CPU: u32 = 0;
pub(crate) const DEVICE_GPU: u32 = 1;
pub(crate) const PROFILE_BASE: u32 = 0;
pub(crate) const PROFILE_FULL: u32 = 1;
pub(crate) const MACHINE_MODEL_LARGE: u32 = 1;
pub(crate) const DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 0;
pub(crate) const DEFAULT_FLOAT_ROUNDING_MODE_ZERO: u32 = 1;
pub(crate) const DEFAULT_FLOAT_ROUNDING_MODE_NEAR: u32 = 2;
pub(crate) const AGENT_FEATURE_KERNEL_DISPATCH: u32 = 1;
pub(crate) const QUEUE_TYPE_MULTI: u32 = 0;
pub(crate) const QUEUE_TYPE_SINGLE: u32 = 1;
pub(crate) const QUEUE_TYPE_COOPERATIVE: u32 = 2;
pub(crate) const QUEUE_FEATURE_KERNEL_DISPATCH: u32 = 1;
pub(crate) const QUEUE_FEATURE_AGENT_DISPATCH: u32 = 1 << 1;

pub(crate) const AMD_QUEUE_PRIORITY_LOW: u32 = 0;
pub(crate) const AMD_QUEUE_PRIORITY_NORMAL: u32 = 1;
pub(crate) const AMD_QUEUE_PRIORITY_HIGH: u32 = 2;
pub(crate) const AMD_QUEUE_CREATE_DESC_VERSION: u16 = 1;
pub(crate) const AMD_QUEUE_CREATE_DEVICE_MEM_RING: u16 = 1;
pub(crate) const AMD_QUEUE_CREATE_DEVICE_MEM_DESCRIPTOR: u16 = 1 << 1;
pub(crate) const AMD_QUEUE_ENGINE_COMPUTE: u8 = 0;
pub(crate) const AMD_QUEUE_ENGINE_SDMA: u8 = 1;
pub(crate) const AMD_QUEUE_ENGINE_AIE: u8 = 2;
pub(crate) const AMD_COHERENCY_TYPE_COHERENT: u32 = 0;
pub(crate) const AMD_COHERENCY_TYPE_NONCOHERENT: u32 = 1;
pub(crate) const AMD_LOG_FLAG_INFO: u32 = 2;

pub(crate) const SYSTEM_INFO_VERSION_MAJOR: u32 = 0;
pub(crate) const SYSTEM_INFO_VERSION_MINOR: u32 = 1;
pub(crate) const SYSTEM_INFO_TIMESTAMP: u32 = 2;
pub(crate) const SYSTEM_INFO_TIMESTAMP_FREQUENCY: u32 = 3;
pub(crate) const SYSTEM_INFO_SIGNAL_MAX_WAIT: u32 = 4;
pub(crate) const SYSTEM_INFO_ENDIANNESS: u32 = 5;
pub(crate) const SYSTEM_INFO_MACHINE_MODEL: u32 = 6;
pub(crate) const SYSTEM_INFO_EXTENSIONS: u32 = 7;
pub(crate) const AMD_SYSTEM_INFO_SVM_SUPPORTED: u32 = 0x201;
pub(crate) const AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT: u32 = 0x202;
pub(crate) const AMD_SYSTEM_INFO_MWAITX_ENABLED: u32 = 0x203;
pub(crate) const AMD_SYSTEM_INFO_DMABUF_SUPPORTED: u32 = 0x204;
pub(crate) const AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED: u32 = 0x205;
pub(crate) const AMD_SYSTEM_INFO_XNACK_ENABLED: u32 = 0x206;
pub(crate) const AMD_SYSTEM_INFO_EXT_VERSION_MAJOR: u32 = 0x207;
pub(crate) const AMD_SYSTEM_INFO_EXT_VERSION_MINOR: u32 = 0x208;
pub(crate) const AMD_SYSTEM_INFO_FABRIC_HANDLES_SUPPORTED: u32 = 0x209;
pub(crate) const AMD_SYSTEM_INFO_HOST_ALLOC_DMABUF_SUPPORTED: u32 = 0x20a;

pub(crate) const AGENT_INFO_NAME: u32 = 0;
pub(crate) const AGENT_INFO_VENDOR_NAME: u32 = 1;
pub(crate) const AGENT_INFO_FEATURE: u32 = 2;
pub(crate) const AGENT_INFO_MACHINE_MODEL: u32 = 3;
pub(crate) const AGENT_INFO_PROFILE: u32 = 4;
pub(crate) const AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 5;
pub(crate) const AGENT_INFO_WAVEFRONT_SIZE: u32 = 6;
pub(crate) const AGENT_INFO_WORKGROUP_MAX_DIM: u32 = 7;
pub(crate) const AGENT_INFO_WORKGROUP_MAX_SIZE: u32 = 8;
pub(crate) const AGENT_INFO_GRID_MAX_DIM: u32 = 9;
pub(crate) const AGENT_INFO_GRID_MAX_SIZE: u32 = 10;
pub(crate) const AGENT_INFO_FBARRIER_MAX_SIZE: u32 = 11;
pub(crate) const AGENT_INFO_QUEUES_MAX: u32 = 12;
pub(crate) const AGENT_INFO_QUEUE_MIN_SIZE: u32 = 13;
pub(crate) const AGENT_INFO_QUEUE_MAX_SIZE: u32 = 14;
pub(crate) const AGENT_INFO_QUEUE_TYPE: u32 = 15;
pub(crate) const AGENT_INFO_NODE: u32 = 16;
pub(crate) const AGENT_INFO_DEVICE: u32 = 17;
pub(crate) const AGENT_INFO_CACHE_SIZE: u32 = 18;
pub(crate) const AGENT_INFO_ISA: u32 = 19;
pub(crate) const AGENT_INFO_EXTENSIONS: u32 = 20;
pub(crate) const AGENT_INFO_VERSION_MAJOR: u32 = 21;
pub(crate) const AGENT_INFO_VERSION_MINOR: u32 = 22;
pub(crate) const AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES: u32 = 23;
pub(crate) const AGENT_INFO_FAST_F16_OPERATION: u32 = 24;

pub(crate) const AMD_AGENT_INFO_CHIP_ID: u32 = 0xa000;
pub(crate) const AMD_AGENT_INFO_CACHELINE_SIZE: u32 = 0xa001;
pub(crate) const AMD_AGENT_INFO_COMPUTE_UNIT_COUNT: u32 = 0xa002;
pub(crate) const AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY: u32 = 0xa003;
pub(crate) const AMD_AGENT_INFO_DRIVER_NODE_ID: u32 = 0xa004;
pub(crate) const AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS: u32 = 0xa005;
pub(crate) const AMD_AGENT_INFO_BDFID: u32 = 0xa006;
pub(crate) const AMD_AGENT_INFO_MEMORY_WIDTH: u32 = 0xa007;
pub(crate) const AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY: u32 = 0xa008;
pub(crate) const AMD_AGENT_INFO_PRODUCT_NAME: u32 = 0xa009;
pub(crate) const AMD_AGENT_INFO_MAX_WAVES_PER_CU: u32 = 0xa00a;
pub(crate) const AMD_AGENT_INFO_NUM_SIMDS_PER_CU: u32 = 0xa00b;
pub(crate) const AMD_AGENT_INFO_NUM_SHADER_ENGINES: u32 = 0xa00c;
pub(crate) const AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE: u32 = 0xa00d;
pub(crate) const AMD_AGENT_INFO_HDP_FLUSH: u32 = 0xa00e;
pub(crate) const AMD_AGENT_INFO_DOMAIN: u32 = 0xa00f;
pub(crate) const AMD_AGENT_INFO_COOPERATIVE_QUEUES: u32 = 0xa010;
pub(crate) const AMD_AGENT_INFO_UUID: u32 = 0xa011;
pub(crate) const AMD_AGENT_INFO_ASIC_REVISION: u32 = 0xa012;
pub(crate) const AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS: u32 = 0xa013;
pub(crate) const AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT: u32 = 0xa014;
pub(crate) const AMD_AGENT_INFO_MEMORY_AVAIL: u32 = 0xa015;
pub(crate) const AMD_AGENT_INFO_TIMESTAMP_FREQUENCY: u32 = 0xa016;
pub(crate) const AMD_AGENT_INFO_ASIC_FAMILY_ID: u32 = 0xa107;
pub(crate) const AMD_AGENT_INFO_UCODE_VERSION: u32 = 0xa108;
pub(crate) const AMD_AGENT_INFO_SDMA_UCODE_VERSION: u32 = 0xa109;
pub(crate) const AMD_AGENT_INFO_NUM_SDMA_ENG: u32 = 0xa10a;
pub(crate) const AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG: u32 = 0xa10b;
pub(crate) const AMD_AGENT_INFO_IOMMU_SUPPORT: u32 = 0xa110;
pub(crate) const AMD_AGENT_INFO_NUM_XCC: u32 = 0xa111;
pub(crate) const AMD_AGENT_INFO_DRIVER_UID: u32 = 0xa112;
pub(crate) const AMD_AGENT_INFO_NEAREST_CPU: u32 = 0xa113;
pub(crate) const AMD_AGENT_INFO_MEMORY_PROPERTIES: u32 = 0xa114;
pub(crate) const AMD_AGENT_INFO_AQL_EXTENSIONS: u32 = 0xa115;
pub(crate) const AMD_AGENT_INFO_SCRATCH_LIMIT_MAX: u32 = 0xa116;
pub(crate) const AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT: u32 = 0xa117;
pub(crate) const AMD_AGENT_INFO_CLOCK_COUNTERS: u32 = 0xa118;
pub(crate) const AMD_AGENT_INFO_PM4_EMULATION: u32 = 0xa119;
pub(crate) const AMD_AGENT_INFO_LUID: u32 = 0xa11a;
pub(crate) const AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE: u32 = 0xa11b;
pub(crate) const AMD_AGENT_INFO_CUID: u32 = 0xa11c;
pub(crate) const AMD_AGENT_INFO_KERNEL_WG_MAX_SIZE: u32 = 0xa11d;
pub(crate) const AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM: u32 = 0xa11e;
pub(crate) const AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE: u32 = 0xa11f;
pub(crate) const AMD_AGENT_INFO_CLUSTER_MAX_DIM: u32 = 0xa120;
pub(crate) const AMD_AGENT_INFO_CLUSTER_MAX_SIZE: u32 = 0xa121;
pub(crate) const AMD_AGENT_INFO_KERNEL_WG_MAX_DIM: u32 = 0xa122;
pub(crate) const AMD_AGENT_INFO_MAX_DATA_PREFETCH_REGIONS: u32 = 0xa123;
pub(crate) const AMD_AGENT_INFO_HOST_ALLOC_DMABUF_SUPPORTED: u32 = 0xa124;

pub(crate) const EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS: u32 = 0x3000;
pub(crate) const EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS: u32 = 0x3001;
pub(crate) const EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS: u32 = 0x3002;
pub(crate) const EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS: u32 = 0x3003;
pub(crate) const EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS: u32 = 0x3004;
pub(crate) const EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS: u32 = 0x3005;
pub(crate) const EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS: u32 = 0x3006;
pub(crate) const EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS: u32 = 0x3007;
pub(crate) const EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS: u32 = 0x3008;
pub(crate) const EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES: u32 = 0x3009;
pub(crate) const EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES: u32 = 0x300a;
pub(crate) const EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS: u32 = 0x300b;
pub(crate) const EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT: u32 = 0x300c;
pub(crate) const EXT_AGENT_INFO_IMAGE_SUPPORT: u32 = 0x300d;

pub(crate) const ISA_INFO_NAME_LENGTH: u32 = 0;
pub(crate) const ISA_INFO_NAME: u32 = 1;
pub(crate) const ISA_INFO_CALL_CONVENTION_COUNT: u32 = 2;
pub(crate) const ISA_INFO_MACHINE_MODELS: u32 = 5;
pub(crate) const ISA_INFO_PROFILES: u32 = 6;
pub(crate) const ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES: u32 = 7;
pub(crate) const ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES: u32 = 8;
pub(crate) const ISA_INFO_FAST_F16_OPERATION: u32 = 9;
pub(crate) const ISA_INFO_WORKGROUP_MAX_DIM: u32 = 12;
pub(crate) const ISA_INFO_WORKGROUP_MAX_SIZE: u32 = 13;
pub(crate) const ISA_INFO_GRID_MAX_DIM: u32 = 14;
pub(crate) const ISA_INFO_GRID_MAX_SIZE: u32 = 16;
pub(crate) const ISA_INFO_FBARRIER_MAX_SIZE: u32 = 17;

pub(crate) const SEGMENT_GLOBAL: u32 = 0;
pub(crate) const SEGMENT_GROUP: u32 = 3;
pub(crate) const POOL_FLAG_KERNARG: u32 = 1;
pub(crate) const POOL_FLAG_FINE: u32 = 2;
pub(crate) const POOL_FLAG_COARSE: u32 = 4;
pub(crate) const POOL_FLAG_EXTENDED_FINE: u32 = 8;
pub(crate) const POOL_INFO_SEGMENT: u32 = 0;
pub(crate) const POOL_INFO_GLOBAL_FLAGS: u32 = 1;
pub(crate) const POOL_INFO_SIZE: u32 = 2;
pub(crate) const POOL_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;
pub(crate) const POOL_INFO_RUNTIME_ALLOC_GRANULE: u32 = 6;
pub(crate) const POOL_INFO_RUNTIME_ALLOC_ALIGNMENT: u32 = 7;
pub(crate) const POOL_INFO_ACCESSIBLE_BY_ALL: u32 = 15;
pub(crate) const POOL_INFO_ALLOC_MAX_SIZE: u32 = 16;
pub(crate) const POOL_INFO_LOCATION: u32 = 17;
pub(crate) const POOL_INFO_RUNTIME_ALLOC_REC_GRANULE: u32 = 18;
pub(crate) const POOL_ACCESS_NEVER: u32 = 0;
pub(crate) const POOL_ACCESS_DEFAULT: u32 = 1;
pub(crate) const POOL_ACCESS_DISALLOWED: u32 = 2;
pub(crate) const AGENT_POOL_INFO_ACCESS: u32 = 0;
pub(crate) const AGENT_POOL_INFO_NUM_LINK_HOPS: u32 = 1;
pub(crate) const AGENT_POOL_INFO_LINK_INFO: u32 = 2;

pub(crate) const REGION_INFO_ALLOC_MAX_SIZE: u32 = 4;
pub(crate) const REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE: u32 = 8;
pub(crate) const AMD_REGION_INFO_HOST_ACCESSIBLE: u32 = 0xa000;
pub(crate) const AMD_REGION_INFO_BASE: u32 = 0xa001;
pub(crate) const AMD_REGION_INFO_BUS_WIDTH: u32 = 0xa002;
pub(crate) const AMD_REGION_INFO_MAX_CLOCK_FREQUENCY: u32 = 0xa003;

pub(crate) const CACHE_INFO_NAME_LENGTH: u32 = 0;
pub(crate) const CACHE_INFO_NAME: u32 = 1;
pub(crate) const CACHE_INFO_LEVEL: u32 = 2;
pub(crate) const CACHE_INFO_SIZE: u32 = 3;

pub(crate) const POINTER_TYPE_UNKNOWN: u32 = 0;
pub(crate) const POINTER_TYPE_HSA: u32 = 1;
pub(crate) const POINTER_TYPE_LOCKED: u32 = 2;
pub(crate) const POINTER_TYPE_GRAPHICS: u32 = 3;
pub(crate) const POINTER_TYPE_IPC: u32 = 4;
pub(crate) const POINTER_TYPE_RESERVED_ADDR: u32 = 5;
pub(crate) const POINTER_TYPE_HSA_VMEM: u32 = 6;
pub(crate) const POINTER_ALLOC_EXECUTABLE: u32 = 1 << 0;
pub(crate) const POINTER_ALLOC_CONTIGUOUS: u32 = 1 << 1;
pub(crate) const POINTER_ALLOC_NONPAGED: u32 = 1 << 2;
pub(crate) const POINTER_ALLOC_HOST_ACCESS: u32 = 1 << 4;
pub(crate) const POINTER_ALLOC_ATOMIC_FULL: u32 = 1 << 5;
pub(crate) const POINTER_ALLOC_ATOMIC_PARTIAL: u32 = 1 << 6;

pub(crate) const AMD_SVM_ATTRIB_GLOBAL_FLAG: u64 = 0;
pub(crate) const AMD_SVM_ATTRIB_READ_ONLY: u64 = 1;
pub(crate) const AMD_SVM_ATTRIB_HIVE_LOCAL: u64 = 2;
pub(crate) const AMD_SVM_ATTRIB_MIGRATION_GRANULARITY: u64 = 3;
pub(crate) const AMD_SVM_ATTRIB_PREFERRED_LOCATION: u64 = 4;
pub(crate) const AMD_SVM_ATTRIB_PREFETCH_LOCATION: u64 = 5;
pub(crate) const AMD_SVM_ATTRIB_READ_MOSTLY: u64 = 6;
pub(crate) const AMD_SVM_ATTRIB_GPU_EXEC: u64 = 7;
pub(crate) const AMD_SVM_ATTRIB_AGENT_ACCESSIBLE: u64 = 0x200;
pub(crate) const AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE: u64 = 0x201;
pub(crate) const AMD_SVM_ATTRIB_AGENT_NO_ACCESS: u64 = 0x202;
pub(crate) const AMD_SVM_ATTRIB_ACCESS_QUERY: u64 = 0x203;

pub(crate) const AMD_SVM_GLOBAL_FLAG_FINE_GRAINED: u64 = 0;
pub(crate) const AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED: u64 = 1;
pub(crate) const AMD_SVM_GLOBAL_FLAG_INDETERMINATE: u64 = 2;

pub(crate) const DMABUF_MAPPING_NONE: u64 = 0;
pub(crate) const DMABUF_MAPPING_PCIE: u64 = 1;
pub(crate) const VMEM_ADDRESS_NO_REGISTER: u64 = 1;
pub(crate) const MEMORY_TYPE_NONE: u32 = 0;
pub(crate) const MEMORY_TYPE_PINNED: u32 = 1;
pub(crate) const ACCESS_PERMISSION_NONE: u32 = 0;
pub(crate) const ACCESS_PERMISSION_RO: u32 = 1;
pub(crate) const ACCESS_PERMISSION_WO: u32 = 2;
pub(crate) const ACCESS_PERMISSION_RW: u32 = 3;

pub(crate) const IMAGE_GEOMETRY_1D: u32 = 0;
pub(crate) const IMAGE_GEOMETRY_2D: u32 = 1;
pub(crate) const IMAGE_GEOMETRY_3D: u32 = 2;
pub(crate) const IMAGE_GEOMETRY_1DA: u32 = 3;
pub(crate) const IMAGE_GEOMETRY_2DA: u32 = 4;
pub(crate) const IMAGE_GEOMETRY_1DB: u32 = 5;
pub(crate) const IMAGE_GEOMETRY_2DDEPTH: u32 = 6;
pub(crate) const IMAGE_GEOMETRY_2DADEPTH: u32 = 7;
pub(crate) const IMAGE_DATA_LAYOUT_LINEAR: u32 = 1;
pub(crate) const IMAGE_CAPABILITY_READ_ONLY: u32 = 1;
pub(crate) const IMAGE_CAPABILITY_WRITE_ONLY: u32 = 1 << 1;
pub(crate) const IMAGE_CAPABILITY_READ_WRITE: u32 = 1 << 2;

pub(crate) type HsaHandle = i32;

pub(crate) const SYMBOL_INFO_TYPE: u32 = 0;
pub(crate) const SYMBOL_INFO_NAME_LENGTH: u32 = 1;
pub(crate) const SYMBOL_INFO_NAME: u32 = 2;
pub(crate) const SYMBOL_INFO_MODULE_NAME_LENGTH: u32 = 3;
pub(crate) const SYMBOL_INFO_MODULE_NAME: u32 = 4;
pub(crate) const SYMBOL_INFO_LINKAGE: u32 = 5;
pub(crate) const SYMBOL_INFO_VARIABLE_ALLOCATION: u32 = 6;
pub(crate) const SYMBOL_INFO_VARIABLE_SEGMENT: u32 = 7;
pub(crate) const SYMBOL_INFO_VARIABLE_ALIGNMENT: u32 = 8;
pub(crate) const SYMBOL_INFO_VARIABLE_SIZE: u32 = 9;
pub(crate) const SYMBOL_INFO_VARIABLE_IS_CONST: u32 = 10;
pub(crate) const SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 11;
pub(crate) const SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT: u32 = 12;
pub(crate) const SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 13;
pub(crate) const SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 14;
pub(crate) const SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK: u32 = 15;
pub(crate) const SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION: u32 = 16;
pub(crate) const SYMBOL_INFO_IS_DEFINITION: u32 = 17;
pub(crate) const SYMBOL_INFO_KERNEL_CALL_CONVENTION: u32 = 18;
pub(crate) const SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE: u32 = 19;
pub(crate) const SYMBOL_INFO_AGENT: u32 = 20;
pub(crate) const SYMBOL_INFO_VARIABLE_ADDRESS: u32 = 21;
pub(crate) const SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
pub(crate) const SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT: u32 = 23;
pub(crate) const SYMBOL_KIND_VARIABLE: u32 = 0;
pub(crate) const SYMBOL_KIND_KERNEL: u32 = 1;
pub(crate) const SYMBOL_KIND_INDIRECT_FUNCTION: u32 = 2;
pub(crate) const SYMBOL_LINKAGE_MODULE: u32 = 0;
pub(crate) const SYMBOL_LINKAGE_PROGRAM: u32 = 1;
pub(crate) const VARIABLE_ALLOCATION_AGENT: u32 = 0;
pub(crate) const VARIABLE_ALLOCATION_PROGRAM: u32 = 1;
pub(crate) const VARIABLE_SEGMENT_GLOBAL: u32 = 0;
pub(crate) const VARIABLE_SEGMENT_READONLY: u32 = 1;

pub(crate) const EXECUTABLE_INFO_PROFILE: u32 = 1;
pub(crate) const EXECUTABLE_INFO_STATE: u32 = 2;
pub(crate) const EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 3;
pub(crate) const EXECUTABLE_STATE_UNFROZEN: u32 = 0;
pub(crate) const EXECUTABLE_STATE_FROZEN: u32 = 1;

pub(crate) const CODE_OBJECT_INFO_VERSION: u32 = 0;
pub(crate) const CODE_OBJECT_INFO_TYPE: u32 = 1;
pub(crate) const CODE_OBJECT_INFO_ISA: u32 = 2;
pub(crate) const CODE_OBJECT_INFO_MACHINE_MODEL: u32 = 3;
pub(crate) const CODE_OBJECT_INFO_PROFILE: u32 = 4;
pub(crate) const CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 5;
pub(crate) const CODE_OBJECT_TYPE_PROGRAM: u32 = 0;

pub(crate) const SIGNAL_CONDITION_EQ: u32 = 0;
pub(crate) const SIGNAL_CONDITION_NE: u32 = 1;
pub(crate) const SIGNAL_CONDITION_LT: u32 = 2;
pub(crate) const SIGNAL_CONDITION_GTE: u32 = 3;
pub(crate) const WAIT_STATE_BLOCKED: u32 = 0;
pub(crate) const WAIT_STATE_ACTIVE: u32 = 1;

pub(crate) const AMD_SIGNAL_KIND_INVALID: i64 = 0;
pub(crate) const AMD_SIGNAL_KIND_USER: i64 = 1;
pub(crate) const AMD_SIGNAL_KIND_DOORBELL: i64 = -1;
pub(crate) const AMD_SIGNAL_AMD_GPU_ONLY: u64 = 1;
pub(crate) const AMD_SIGNAL_IPC: u64 = 1 << 1;
pub(crate) const AMD_MEMORY_COPY_OP_VERSION: u16 = 1;
pub(crate) const AMD_MEMORY_COPY_OP_LINEAR: u16 = 0;
pub(crate) const AMD_MEMORY_COPY_OP_LINEAR_BROADCAST: u16 = 1;
pub(crate) const AMD_MEMORY_COPY_OP_LINEAR_SWAP: u16 = 2;
pub(crate) const AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC: u16 = 3;
pub(crate) const AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST: u16 = 4;
pub(crate) const AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST: u16 = 5;
pub(crate) const AMD_MEMORY_COPY_WAIT_ALWAYS: u16 = 0;
pub(crate) const AMD_MEMORY_COPY_WAIT_LT: u16 = 1;
pub(crate) const AMD_MEMORY_COPY_WAIT_LE: u16 = 2;
pub(crate) const AMD_MEMORY_COPY_WAIT_EQ: u16 = 3;
pub(crate) const AMD_MEMORY_COPY_WAIT_NE: u16 = 4;
pub(crate) const AMD_MEMORY_COPY_WAIT_GE: u16 = 5;
pub(crate) const AMD_MEMORY_COPY_WAIT_GT: u16 = 6;
pub(crate) const AMD_MEMORY_COPY_SIGNAL_NONE: u16 = 0;
pub(crate) const AMD_MEMORY_COPY_SIGNAL_WRITE: u16 = 1;
pub(crate) const AMD_MEMORY_COPY_SIGNAL_ADD: u16 = 2;
pub(crate) const AMD_MEMORY_COPY_SIGNAL_SUB: u16 = 3;
pub(crate) const FENCE_SCOPE_SYSTEM: u16 = 2;
pub(crate) const AMD_QUEUE_PROPERTIES_ENABLE_PROFILING: u32 = 1 << 3;

pub(crate) const AMD_QUEUE_INFO_AGENT: u32 = 0;
pub(crate) const AMD_QUEUE_INFO_DOORBELL_ID: u32 = 1;
pub(crate) const QUEUE_INFO_USE_COUNT: u32 = 2;
pub(crate) const QUEUE_INFO_HW_ID: u32 = 3;
pub(crate) const AMD_QUEUE_INFO_PREFETCH_DISPATCH_MAJOR: u32 = 4;
pub(crate) const AMD_QUEUE_INFO_PREFETCH_DISPATCH_MINOR: u32 = 5;
pub(crate) const AMD_QUEUE_INFO_PREFETCH_BARRIER_MAJOR: u32 = 6;
pub(crate) const AMD_QUEUE_INFO_PREFETCH_BARRIER_MINOR: u32 = 7;
pub(crate) const AMD_QUEUE_INFO_PREFETCH_RING_BUFFER: u32 = 8;
pub(crate) const AMD_QUEUE_INFO_PROPERTIES: u32 = 9;

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaAgent {
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaIsa {
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaSignal {
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaPcSampling {
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct HsaPcSamplingConfiguration {
    pub(crate) method: u32,
    pub(crate) units: u32,
    pub(crate) minimum_interval: usize,
    pub(crate) maximum_interval: usize,
    pub(crate) flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaAmdIpcMemory {
    pub(crate) handle: [u32; 8],
}

pub(crate) type HsaAmdIpcSignal = HsaAmdIpcMemory;

#[repr(C)]
#[derive(Clone, Copy)]
pub union HsaAmdAisFileHandle {
    pub(crate) handle: *mut c_void,
    pub(crate) fd: i32,
    pub(crate) pad: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaAmdExternalSemaphore {
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union HsaAmdExternalSemaphoreHandle {
    pub(crate) win32_handle: *mut c_void,
    pub(crate) fd: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaAmdExternalSemaphoreHandleDescriptor {
    pub(crate) handle_type: u32,
    pub(crate) handle: HsaAmdExternalSemaphoreHandle,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaAmdSvmAttributePair {
    pub(crate) attribute: u64,
    pub(crate) value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaMemoryPool {
    pub(crate) handle: u64,
}

macro_rules! opaque_handle {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy, Default, Eq, PartialEq)]
        pub struct $name {
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
opaque_handle!(HsaExtImage);
opaque_handle!(HsaExtSampler);

// hsa_ext_control_directives_t is passed by value to the legacy finalizer
// entry point, so its complete layout is part of that entry point's ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaExtControlDirectives {
    pub(crate) control_directives_mask: u64,
    pub(crate) break_exceptions_mask: u16,
    pub(crate) detect_exceptions_mask: u16,
    pub(crate) max_dynamic_group_size: u32,
    pub(crate) max_flat_grid_size: u64,
    pub(crate) max_flat_workgroup_size: u32,
    pub(crate) reserved1: u32,
    pub(crate) required_grid_size: [u64; 3],
    pub(crate) required_workgroup_size: HsaDim3,
    pub(crate) required_dim: u8,
    pub(crate) reserved2: [u8; 75],
}

const _: () = {
    assert!(std::mem::size_of::<HsaDim3>() == 12);
    assert!(std::mem::size_of::<HsaExtControlDirectives>() == 144);
    assert!(std::mem::align_of::<HsaExtControlDirectives>() == 8);
};

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct HsaExtImageFormat {
    pub(crate) channel_type: u32,
    pub(crate) channel_order: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct HsaExtImageDescriptor {
    pub(crate) geometry: u32,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
    pub(crate) array_size: usize,
    pub(crate) format: HsaExtImageFormat,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct HsaExtImageDescriptorV2 {
    pub(crate) geometry: u32,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
    pub(crate) array_size: usize,
    pub(crate) format: HsaExtImageFormat,
    pub(crate) mipmap_levels: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct HsaExtImageDataInfo {
    pub(crate) size: usize,
    pub(crate) alignment: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaExtImageRegion {
    pub(crate) offset: HsaDim3,
    pub(crate) range: HsaDim3,
}

#[repr(C)]
pub(crate) struct HsaAmdImageDescriptor {
    pub(crate) version: u32,
    pub(crate) device_id: u32,
    pub(crate) data: [u32; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub(crate) struct HsaExtSamplerDescriptor {
    pub(crate) coordinate_mode: u32,
    pub(crate) filter_mode: u32,
    pub(crate) address_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct HsaExtSamplerDescriptorV2 {
    pub(crate) coordinate_mode: u32,
    pub(crate) filter_mode: u32,
    pub(crate) mipmap_filter_mode: u32,
    pub(crate) address_modes: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaAmdMemoryAccessDesc {
    pub(crate) permissions: u32,
    pub(crate) agent_handle: HsaAgent,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct HsaFabricHandle {
    pub(crate) handle: [u8; 16],
}

pub(crate) type LoadedCodeObjectCallback =
    Option<unsafe extern "C" fn(HsaExecutable, HsaLoadedCodeObject, *mut c_void) -> Status>;
pub(crate) type ExecutableCallback =
    Option<unsafe extern "C" fn(HsaExecutable, *mut c_void) -> Status>;
pub(crate) type CodeObjectAllocCallback =
    Option<unsafe extern "C" fn(usize, HsaCallbackData, *mut *mut c_void) -> Status>;
pub(crate) type CodeObjectSymbolCallback =
    Option<unsafe extern "C" fn(HsaCodeObject, HsaCodeSymbol, *mut c_void) -> Status>;
pub(crate) type ExecutableSymbolCallback =
    Option<unsafe extern "C" fn(HsaExecutable, HsaExecutableSymbol, *mut c_void) -> Status>;
pub(crate) type AgentExecutableSymbolCallback = Option<
    unsafe extern "C" fn(HsaExecutable, HsaAgent, HsaExecutableSymbol, *mut c_void) -> Status,
>;
pub(crate) type PcSamplingConfigurationCallback =
    Option<unsafe extern "C" fn(*const HsaPcSamplingConfiguration, *mut c_void) -> Status>;
pub(crate) type PcSamplingDataCopyCallback =
    Option<unsafe extern "C" fn(*mut c_void, usize, *mut c_void) -> Status>;
pub(crate) type PcSamplingDataReadyCallback = Option<
    unsafe extern "C" fn(*mut c_void, usize, usize, PcSamplingDataCopyCallback, *mut c_void),
>;

pub(crate) const LOADER_STORAGE_MEMORY: u32 = 2;
pub(crate) const LOADER_OBJECT_KIND_PROGRAM: u32 = 1;
pub(crate) const LOADER_OBJECT_KIND_AGENT: u32 = 2;
pub(crate) const LOADER_INFO_EXECUTABLE: u32 = 1;
pub(crate) const LOADER_INFO_KIND: u32 = 2;
pub(crate) const LOADER_INFO_AGENT: u32 = 3;
pub(crate) const LOADER_INFO_STORAGE_TYPE: u32 = 4;
pub(crate) const LOADER_INFO_STORAGE_MEMORY_BASE: u32 = 5;
pub(crate) const LOADER_INFO_STORAGE_MEMORY_SIZE: u32 = 6;
pub(crate) const LOADER_INFO_STORAGE_FILE: u32 = 7;
pub(crate) const LOADER_INFO_LOAD_DELTA: u32 = 8;
pub(crate) const LOADER_INFO_LOAD_BASE: u32 = 9;
pub(crate) const LOADER_INFO_LOAD_SIZE: u32 = 10;
pub(crate) const LOADER_INFO_URI_LENGTH: u32 = 11;
pub(crate) const LOADER_INFO_URI: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaLoaderSegmentDescriptor {
    pub(crate) agent: HsaAgent,
    pub(crate) executable: HsaExecutable,
    pub(crate) code_object_storage_type: u32,
    pub(crate) code_object_storage_base: *const c_void,
    pub(crate) code_object_storage_size: usize,
    pub(crate) code_object_storage_offset: usize,
    pub(crate) segment_base: *const c_void,
    pub(crate) segment_size: usize,
}

#[repr(C)]
pub struct HsaQueue {
    pub(crate) queue_type: u32,
    pub(crate) features: u32,
    pub(crate) base_address: *mut c_void,
    pub(crate) doorbell_signal: HsaSignal,
    pub(crate) size: u32,
    pub(crate) reserved: u32,
    pub(crate) id: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaAmdComputeQueueParams {
    pub(crate) cu_mask: *const u32,
    pub(crate) queue_type: u32,
    pub(crate) private_segment_size: u32,
    pub(crate) cu_mask_count: u32,
    pub(crate) reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaAmdSdmaQueueParams {
    pub(crate) engine_id: u32,
    pub(crate) reserved: [u32; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union HsaAmdQueueEngineParams {
    pub(crate) compute: HsaAmdComputeQueueParams,
    pub(crate) sdma: HsaAmdSdmaQueueParams,
    pub(crate) reserved: [u8; 32],
}

#[repr(C)]
pub(crate) struct HsaAmdQueueCreateDesc {
    pub(crate) version: u16,
    pub(crate) flags: u16,
    pub(crate) engine_type: u8,
    pub(crate) reserved_header: [u8; 3],
    pub(crate) queue_size_bytes: u32,
    pub(crate) priority: u32,
    pub(crate) callback: QueueErrorCallback,
    pub(crate) callback_data: *mut c_void,
    pub(crate) queue: *mut HsaQueue,
    pub(crate) engine: HsaAmdQueueEngineParams,
    pub(crate) traffic_class: u32,
    pub(crate) reserved: [u8; 20],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaAmdPointerInfo {
    pub(crate) size: u32,
    pub(crate) pointer_type: u32,
    pub(crate) agent_base_address: *mut c_void,
    pub(crate) host_base_address: *mut c_void,
    pub(crate) size_in_bytes: usize,
    pub(crate) user_data: *mut c_void,
    pub(crate) agent_owner: HsaAgent,
    pub(crate) global_flags: u32,
    pub(crate) registered: bool,
    pub(crate) registered_padding: [u8; 3],
    pub(crate) alloc_flags: u32,
    pub(crate) tail_padding: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaDim3 {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct HsaAmdDim3 {
    pub(crate) x: u64,
    pub(crate) y: u64,
    pub(crate) z: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct HsaLuid {
    pub(crate) low: u32,
    pub(crate) high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaPitchedPtr {
    pub(crate) base: *mut c_void,
    pub(crate) pitch: usize,
    pub(crate) slice: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaAmdMemoryCopyWait {
    pub(crate) function: u16,
    pub(crate) scope: u16,
    pub(crate) reserved: u32,
    pub(crate) address: *mut c_void,
    pub(crate) value: u64,
    pub(crate) mask: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaAmdMemoryCopySignal {
    pub(crate) operation: u16,
    pub(crate) scope: u16,
    pub(crate) reserved: u32,
    pub(crate) address: *mut c_void,
    pub(crate) data: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaAmdMemoryCopyOp {
    pub(crate) version: u16,
    pub(crate) operation: u16,
    pub(crate) entry_count: u16,
    pub(crate) traffic_class: u16,
    pub(crate) completion_signal: HsaSignal,
    pub(crate) source: *mut c_void,
    pub(crate) source_agent: HsaAgent,
    pub(crate) destination_agent: HsaAgent,
    pub(crate) destination: *mut c_void,
    pub(crate) size: usize,
    pub(crate) secondary_size: usize,
    pub(crate) wait: HsaAmdMemoryCopyWait,
    pub(crate) signal: HsaAmdMemoryCopySignal,
    pub(crate) reserved: [u64; 1],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProfilingTime {
    pub(crate) start: u64,
    pub(crate) end: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HsaAmdClockCounters {
    pub(crate) gpu_clock_counter: u64,
    pub(crate) cpu_clock_counter: u64,
    pub(crate) system_clock_counter: u64,
    pub(crate) system_clock_frequency: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PoolLinkInfo {
    pub(crate) min_latency: u32,
    pub(crate) max_latency: u32,
    pub(crate) min_bandwidth: u32,
    pub(crate) max_bandwidth: u32,
    pub(crate) atomic_support_32bit: bool,
    pub(crate) atomic_support_64bit: bool,
    pub(crate) coherent_support: bool,
    pub(crate) link_type: u32,
    pub(crate) numa_distance: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaAmdEvent {
    pub(crate) event_type: u32,
    pub(crate) payload: [u64; 3],
}

pub(crate) const AMD_GPU_MEMORY_FAULT_EVENT: u32 = 0;
pub(crate) const AMD_SYSTEM_SHUTDOWN_EVENT: u32 = 3;
pub(crate) const AMD_MEMORY_FAULT_PAGE_NOT_PRESENT: u32 = 1;
pub(crate) const AMD_MEMORY_FAULT_READ_ONLY: u32 = 1 << 1;
pub(crate) const AMD_MEMORY_FAULT_NO_EXECUTE: u32 = 1 << 2;
pub(crate) const AMD_MEMORY_FAULT_DRAM_ECC: u32 = 1 << 4;
pub(crate) const AMD_MEMORY_FAULT_IMPRECISE: u32 = 1 << 5;
pub(crate) const AMD_MEMORY_FAULT_SRAM_ECC: u32 = 1 << 6;
pub(crate) const AMD_MEMORY_FAULT_HANG: u32 = 1 << 31;

pub(crate) type AgentCallback = Option<unsafe extern "C" fn(HsaAgent, *mut c_void) -> Status>;
pub(crate) type IsaCallback = Option<unsafe extern "C" fn(HsaIsa, *mut c_void) -> Status>;
pub(crate) type WavefrontCallback =
    Option<unsafe extern "C" fn(HsaWavefront, *mut c_void) -> Status>;
pub(crate) type PoolCallback = Option<unsafe extern "C" fn(HsaMemoryPool, *mut c_void) -> Status>;
pub(crate) type RegionCallback = Option<unsafe extern "C" fn(HsaRegion, *mut c_void) -> Status>;
pub(crate) type CacheCallback = Option<unsafe extern "C" fn(HsaCache, *mut c_void) -> Status>;
pub(crate) type QueueErrorCallback =
    Option<unsafe extern "C" fn(Status, *mut HsaQueue, *mut c_void)>;
pub(crate) type SignalHandler = Option<unsafe extern "C" fn(SignalValue, *mut c_void) -> bool>;
pub(crate) type AsyncFunction = Option<unsafe extern "C" fn(*mut c_void)>;
pub(crate) type DeallocationCallbackFn = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>;
pub(crate) type SystemEventHandler =
    unsafe extern "C" fn(*const HsaAmdEvent, *mut c_void) -> Status;
pub(crate) type SystemEventCallback = Option<SystemEventHandler>;
pub(crate) type PointerAllocator = Option<unsafe extern "C" fn(usize) -> *mut c_void>;

pub(crate) const STATUS_SUCCESS: &[u8] =
    b"HSA_STATUS_SUCCESS: The function has been executed successfully.\0";
pub(crate) const STATUS_INFO_BREAK: &[u8] = b"HSA_STATUS_INFO_BREAK: A traversal over a list of elements has been interrupted by the application before completing.\0";
pub(crate) const STATUS_ERROR: &[u8] = b"HSA_STATUS_ERROR: A generic error has occurred.\0";
pub(crate) const STATUS_INVALID_ARGUMENT: &[u8] = b"HSA_STATUS_ERROR_INVALID_ARGUMENT: One of the actual arguments does not meet a precondition stated in the documentation of the corresponding formal argument.\0";
pub(crate) const STATUS_INVALID_QUEUE_CREATION: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_QUEUE_CREATION: The requested queue creation is not valid.\0";
pub(crate) const STATUS_INVALID_ALLOCATION: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_ALLOCATION: The requested allocation is not valid.\0";
pub(crate) const STATUS_INVALID_AGENT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_AGENT: The agent is invalid.\0";
pub(crate) const STATUS_INVALID_REGION: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_REGION: The memory region is invalid.\0";
pub(crate) const STATUS_INVALID_SIGNAL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_SIGNAL: The signal is invalid.\0";
pub(crate) const STATUS_INVALID_QUEUE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_QUEUE: The queue is invalid.\0";
pub(crate) const STATUS_OUT_OF_RESOURCES: &[u8] = b"HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events.\0";
pub(crate) const STATUS_INVALID_PACKET_FORMAT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is malformed.\0";
pub(crate) const STATUS_RESOURCE_FREE: &[u8] =
    b"HSA_STATUS_ERROR_RESOURCE_FREE: An error has been detected while releasing a resource.\0";
pub(crate) const STATUS_NOT_INITIALIZED: &[u8] = b"HSA_STATUS_ERROR_NOT_INITIALIZED: An API other than hsa_init has been invoked while the reference count of the HSA runtime is zero.\0";
pub(crate) const STATUS_REFCOUNT_OVERFLOW: &[u8] = b"HSA_STATUS_ERROR_REFCOUNT_OVERFLOW: The maximum reference count for the object has been reached.\0";
pub(crate) const STATUS_INCOMPATIBLE_ARGUMENTS: &[u8] = b"HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS: The arguments passed to a functions are not compatible.\0";
pub(crate) const STATUS_INVALID_INDEX: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_INDEX: The index is invalid.\0";
pub(crate) const STATUS_INVALID_ISA: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_ISA: The instruction set architecture is invalid.\0";
pub(crate) const STATUS_INVALID_CODE_OBJECT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CODE_OBJECT: The code object is invalid.\0";
pub(crate) const STATUS_INVALID_EXECUTABLE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_EXECUTABLE: The executable is invalid.\0";
pub(crate) const STATUS_FROZEN_EXECUTABLE: &[u8] =
    b"HSA_STATUS_ERROR_FROZEN_EXECUTABLE: The executable is frozen.\0";
pub(crate) const STATUS_INVALID_SYMBOL_NAME: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_SYMBOL_NAME: There is no symbol with the given name.\0";
pub(crate) const STATUS_VARIABLE_ALREADY_DEFINED: &[u8] =
    b"HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED: The variable is already defined.\0";
pub(crate) const STATUS_VARIABLE_UNDEFINED: &[u8] =
    b"HSA_STATUS_ERROR_VARIABLE_UNDEFINED: The variable is undefined.\0";
pub(crate) const STATUS_EXCEPTION: &[u8] =
    b"HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception.\0";
pub(crate) const STATUS_INVALID_ISA_NAME: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_ISA_NAME: The instruction set architecture name is invalid.\0";
pub(crate) const STATUS_INVALID_CODE_SYMBOL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CODE_SYMBOL: The code object symbol is invalid.\0";
pub(crate) const STATUS_INVALID_EXECUTABLE_SYMBOL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL: The executable symbol is invalid.\0";
pub(crate) const STATUS_INVALID_FILE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_FILE: The file descriptor is invalid.\0";
pub(crate) const STATUS_INVALID_CODE_OBJECT_READER: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER: The code object reader is invalid.\0";
pub(crate) const STATUS_INVALID_CACHE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_CACHE: The cache is invalid.\0";
pub(crate) const STATUS_INVALID_WAVEFRONT: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_WAVEFRONT: The wavefront is invalid.\0";
pub(crate) const STATUS_INVALID_SIGNAL_GROUP: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP: The signal group is invalid.\0";
pub(crate) const STATUS_INVALID_RUNTIME_STATE: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_RUNTIME_STATE: The HSA runtime is not in the configuration state.\0";
pub(crate) const STATUS_FATAL: &[u8] =
    b"HSA_STATUS_ERROR_FATAL: The queue received an error that may require process termination.\0";
pub(crate) const STATUS_INVALID_MEMORY_POOL: &[u8] =
    b"HSA_STATUS_ERROR_INVALID_MEMORY_POOL: The memory pool is invalid.\0";
pub(crate) const STATUS_MEMORY_APERTURE_VIOLATION: &[u8] = b"HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address.\0";
pub(crate) const STATUS_ILLEGAL_INSTRUCTION: &[u8] = b"HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION: The agent attempted to execute an illegal shader instruction.\0";
pub(crate) const STATUS_MEMORY_FAULT: &[u8] =
    b"HSA_STATUS_ERROR_MEMORY_FAULT: Agent attempted to access an inaccessible address.\0";
pub(crate) const STATUS_CU_MASK_REDUCED: &[u8] = b"HSA_STATUS_CU_MASK_REDUCED: The CU mask was successfully set but the mask attempted to enable a CU which was disabled for the process.  CUs disabled for the process remain disabled.\0";
pub(crate) const STATUS_OUT_OF_REGISTERS: &[u8] = b"HSA_STATUS_ERROR_OUT_OF_REGISTERS: Kernel has requested more VGPRs than are available on this agent\0";
pub(crate) const STATUS_RESOURCE_BUSY: &[u8] =
    b"HSA_STATUS_ERROR_RESOURCE_BUSY: Resource is busy or temporarily unavailable.\0";
pub(crate) const STATUS_NOT_SUPPORTED: &[u8] =
    b"HSA_STATUS_ERROR_NOT_SUPPORTED: Request is not supported by this system.\0";
pub(crate) const STATUS_XNACK_DISABLED: &[u8] = b"HSA_STATUS_ERROR_XNACK_DISABLED: Xnack is disabled on this system, but required by the requested operation.\0";
pub(crate) const STATUS_INVALID_DISPATCH_PARAMETERS: &[u8] = b"HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS: Kernel dispatch packet parameters exceed hardware limits for this agent (e.g. register usage, work-group dimensions, or other dispatch constraints)\0";
pub(crate) const STATUS_RESOURCE_NOT_READY: &[u8] = b"HSA_STATUS_ERROR_RESOURCE_NOT_READY: Underlying resource is a valid resource, but it is not ready to be used.\0";
pub(crate) const STATUS_IMAGE_FORMAT_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED: Image format is not supported.\0";
pub(crate) const STATUS_IMAGE_SIZE_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED: Image size is not supported.\0";
pub(crate) const STATUS_IMAGE_PITCH_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED: Image pitch is not supported or invalid.\0";
pub(crate) const STATUS_SAMPLER_DESCRIPTOR_UNSUPPORTED: &[u8] =
    b"HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED: Sampler descriptor is not supported or invalid.\0";

pub(crate) const fn c_string(bytes: &'static [u8]) -> *const c_char {
    bytes.as_ptr().cast()
}
