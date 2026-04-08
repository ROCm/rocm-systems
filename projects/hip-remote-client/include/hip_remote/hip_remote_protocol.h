/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hip_remote_protocol.h
 * @brief Binary protocol for remote HIP API execution
 *
 * This protocol enables HIP API calls to be forwarded from a client
 * (e.g., macOS development machine) to a worker service running on
 * a Linux system with AMD GPUs.
 *
 * Protocol design principles:
 * - Fixed-size headers for efficient parsing
 * - Request-response model with correlation IDs
 * - Support for bulk data transfer (memory copies)
 * - Extensible via reserved fields and flags
 */

#ifndef HIP_REMOTE_PROTOCOL_H
#define HIP_REMOTE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef _MSC_VER
#define HIP_PACKED_ATTR
#pragma pack(push, 1)
#else
#define HIP_PACKED_ATTR __attribute__((packed))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Protocol Constants
 * ============================================================================ */

/** Protocol magic number: 'HIPR' in ASCII */
#define HIP_REMOTE_MAGIC 0x48495052

/** Protocol version (major.minor as 0xMMmm) */
#define HIP_REMOTE_VERSION 0x0100

/** Default port for worker service */
#define HIP_REMOTE_DEFAULT_PORT 18515

/** Maximum payload size (64MB) */
#define HIP_REMOTE_MAX_PAYLOAD_SIZE (2048u * 1024u * 1024u)

/** Maximum number of kernel arguments */
#define HIP_REMOTE_MAX_KERNEL_ARGS 64

/** Maximum size of a single kernel argument */
#define HIP_REMOTE_MAX_ARG_SIZE 256

/* ============================================================================
 * Operation Codes
 * ============================================================================ */

typedef enum {
    /* Connection management (0x00xx) */
    HIP_OP_INIT                     = 0x0001,
    HIP_OP_SHUTDOWN                 = 0x0002,
    HIP_OP_PING                     = 0x0003,

    /* Device management (0x01xx) */
    HIP_OP_GET_DEVICE_COUNT         = 0x0100,
    HIP_OP_SET_DEVICE               = 0x0101,
    HIP_OP_GET_DEVICE               = 0x0102,
    HIP_OP_GET_DEVICE_PROPERTIES    = 0x0103,
    HIP_OP_DEVICE_SYNCHRONIZE       = 0x0104,
    HIP_OP_DEVICE_RESET             = 0x0105,
    HIP_OP_DEVICE_GET_ATTRIBUTE     = 0x0106,
    HIP_OP_DEVICE_GET_LIMIT         = 0x0107,
    HIP_OP_DEVICE_SET_LIMIT         = 0x0108,
    HIP_OP_DEVICE_CAN_ACCESS_PEER   = 0x0109,
    HIP_OP_DEVICE_ENABLE_PEER_ACCESS = 0x010A,
    HIP_OP_DEVICE_DISABLE_PEER_ACCESS = 0x010B,
    HIP_OP_DEVICE_GET               = 0x010C,  /* hipDeviceGet */
    HIP_OP_DEVICE_GET_NAME          = 0x010D,  /* hipDeviceGetName */
    HIP_OP_DEVICE_TOTAL_MEM         = 0x010E,  /* hipDeviceTotalMem */
    HIP_OP_DEVICE_GET_PCI_BUS_ID    = 0x010F,  /* hipDeviceGetPCIBusId */
    HIP_OP_DEVICE_GET_BY_PCI_BUS_ID = 0x0110,  /* hipDeviceGetByPCIBusId */
    HIP_OP_DEVICE_COMPUTE_CAPABILITY = 0x0111, /* hipDeviceComputeCapability */
    HIP_OP_DEVICE_GET_UUID          = 0x0112,  /* hipDeviceGetUuid */
    HIP_OP_DEVICE_GET_CACHE_CONFIG  = 0x0113,  /* hipDeviceGetCacheConfig */
    HIP_OP_DEVICE_SET_CACHE_CONFIG  = 0x0114,  /* hipDeviceSetCacheConfig */
    HIP_OP_DEVICE_GET_SHARED_MEM_CONFIG = 0x0115, /* hipDeviceGetSharedMemConfig */
    HIP_OP_DEVICE_SET_SHARED_MEM_CONFIG = 0x0116, /* hipDeviceSetSharedMemConfig */
    HIP_OP_SET_DEVICE_FLAGS         = 0x0117,  /* hipSetDeviceFlags */
    HIP_OP_GET_DEVICE_FLAGS         = 0x0118,  /* hipGetDeviceFlags */
    HIP_OP_DEVICE_GET_P2P_ATTRIBUTE = 0x0119,  /* hipDeviceGetP2PAttribute */
    HIP_OP_SET_VALID_DEVICES        = 0x011A,  /* hipSetValidDevices */
    HIP_OP_CHOOSE_DEVICE            = 0x011B,  /* hipChooseDevice */
    HIP_OP_DEVICE_GET_STREAM_PRIORITY_RANGE = 0x011C,  /* hipDeviceGetStreamPriorityRange */

    /* Memory allocation (0x02xx) */
    HIP_OP_MALLOC                   = 0x0200,
    HIP_OP_FREE                     = 0x0201,
    HIP_OP_MALLOC_HOST              = 0x0202,
    HIP_OP_FREE_HOST                = 0x0203,
    HIP_OP_MALLOC_MANAGED           = 0x0204,
    HIP_OP_MALLOC_ASYNC             = 0x0205,
    HIP_OP_FREE_ASYNC               = 0x0206,

    /* Memory transfer (0x021x) */
    HIP_OP_MEMCPY                   = 0x0210,
    HIP_OP_MEMCPY_ASYNC             = 0x0211,
    HIP_OP_MEMCPY_2D                = 0x0212,
    HIP_OP_MEMCPY_2D_ASYNC          = 0x0213,
    HIP_OP_MEMCPY_3D                = 0x0214,
    HIP_OP_MEMCPY_3D_ASYNC          = 0x0215,
    HIP_OP_MEMCPY_DTOD              = 0x0216,
    HIP_OP_MEMCPY_DTOD_ASYNC        = 0x0217,
    HIP_OP_MEMCPY_HTOD              = 0x0218,
    HIP_OP_MEMCPY_HTOD_ASYNC        = 0x0219,
    HIP_OP_MEMCPY_DTOH              = 0x021A,
    HIP_OP_MEMCPY_DTOH_ASYNC        = 0x021B,
    HIP_OP_MEMCPY_PEER              = 0x021C,
    HIP_OP_MEMCPY_PEER_ASYNC        = 0x021D,
    HIP_OP_MEMCPY_HTOD_CACHED       = 0x021E,

    /* Memory set (0x022x) */
    HIP_OP_MEMSET                   = 0x0220,
    HIP_OP_MEMSET_ASYNC             = 0x0221,
    HIP_OP_MEMSET_D8                = 0x0222,
    HIP_OP_MEMSET_D16               = 0x0223,
    HIP_OP_MEMSET_D32               = 0x0224,

    /* Memory info (0x023x) */
    HIP_OP_MEM_GET_INFO             = 0x0230,
    HIP_OP_POINTER_GET_ATTRIBUTES   = 0x0231,
    HIP_OP_POINTER_GET_ATTRIBUTE    = 0x0232,

    /* IPC operations (0x024x) */
    HIP_OP_IPC_GET_MEM_HANDLE       = 0x0240,
    HIP_OP_IPC_OPEN_MEM_HANDLE      = 0x0241,
    HIP_OP_IPC_CLOSE_MEM_HANDLE     = 0x0242,
    HIP_OP_IPC_GET_EVENT_HANDLE     = 0x0243,
    HIP_OP_IPC_OPEN_EVENT_HANDLE    = 0x0244,

    /* Memory Pool operations (0x025x) */
    HIP_OP_MEM_POOL_CREATE          = 0x0250,
    HIP_OP_MEM_POOL_DESTROY         = 0x0251,
    HIP_OP_MEM_POOL_SET_ATTRIBUTE   = 0x0252,
    HIP_OP_MEM_POOL_GET_ATTRIBUTE   = 0x0253,
    HIP_OP_MALLOC_FROM_POOL_ASYNC   = 0x0254,
    HIP_OP_MEM_POOL_TRIM_TO         = 0x0255,
    HIP_OP_MEMPOOL_GET_ATTRIBUTE    = HIP_OP_MEM_POOL_GET_ATTRIBUTE,
    HIP_OP_MEMPOOL_SET_ATTRIBUTE    = HIP_OP_MEM_POOL_SET_ATTRIBUTE,
    HIP_OP_MEMPOOL_SET_ACCESS       = 0x025A,
    HIP_OP_MEMPOOL_TRIM_TO          = HIP_OP_MEM_POOL_TRIM_TO,
    HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL = 0x0256,
    HIP_OP_DEVICE_SET_MEM_POOL      = 0x0257,
    HIP_OP_DEVICE_GET_MEM_POOL      = 0x0258,

    /* Host memory registration (0x026x) */
    HIP_OP_HOST_REGISTER            = 0x0260,
    HIP_OP_HOST_UNREGISTER          = 0x0261,
    HIP_OP_HOST_GET_DEVICE_POINTER  = 0x0262,
    HIP_OP_HOST_GET_FLAGS           = 0x0263,
    HIP_OP_HOST_ALLOC               = 0x0264,
    HIP_OP_HOST_FREE                = 0x0265,
    HIP_OP_MEM_ALLOC_PITCH          = 0x0266,

    /* Unified memory management (0x027x) */
    HIP_OP_MEM_ADVISE               = 0x0270,
    HIP_OP_MEM_PREFETCH_ASYNC       = 0x0271,
    HIP_OP_MEM_RANGE_GET_ATTRIBUTE  = 0x0272,
    HIP_OP_MEM_RANGE_GET_ATTRIBUTES = 0x0273,

    /* Graph Node operations (0x064x) */
    HIP_OP_GRAPH_ADD_MEMCPY_NODE    = 0x0640,
    HIP_OP_GRAPH_ADD_MEMSET_NODE    = 0x0641,
    HIP_OP_GRAPH_ADD_KERNEL_NODE    = 0x0642,
    HIP_OP_GRAPH_ADD_DEPENDENCIES   = 0x0643,
    HIP_OP_GRAPH_ADD_EMPTY_NODE     = 0x0644,
    HIP_OP_GRAPH_GET_NODES          = 0x0645,
    HIP_OP_GRAPH_GET_ROOT_NODES     = 0x0646,
    HIP_OP_GRAPH_GET_EDGES          = 0x0647,
    HIP_OP_GRAPH_NODE_GET_TYPE      = 0x0648,
    HIP_OP_GRAPH_DESTROY_NODE       = 0x0649,
    HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D = 0x064A,
    HIP_OP_GRAPH_ADD_HOST_NODE      = 0x064B,
    HIP_OP_GRAPH_ADD_CHILD_GRAPH_NODE = 0x064C,
    HIP_OP_GRAPH_ADD_EVENT_RECORD_NODE = 0x064D,
    HIP_OP_GRAPH_ADD_EVENT_WAIT_NODE = 0x064E,

    /* Stream operations (0x03xx) */
    HIP_OP_STREAM_CREATE            = 0x0300,
    HIP_OP_STREAM_CREATE_WITH_FLAGS = 0x0301,
    HIP_OP_STREAM_CREATE_WITH_PRIORITY = 0x0302,
    HIP_OP_STREAM_DESTROY           = 0x0303,
    HIP_OP_STREAM_SYNCHRONIZE       = 0x0304,
    HIP_OP_STREAM_QUERY             = 0x0305,
    HIP_OP_STREAM_WAIT_EVENT        = 0x0306,
    HIP_OP_STREAM_GET_FLAGS         = 0x0307,
    HIP_OP_STREAM_GET_PRIORITY      = 0x0308,
    HIP_OP_STREAM_GET_CAPTURE_INFO  = 0x0309,
    HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES = 0x030A,

    /* Event operations (0x04xx) */
    HIP_OP_EVENT_CREATE             = 0x0400,
    HIP_OP_EVENT_CREATE_WITH_FLAGS  = 0x0401,
    HIP_OP_EVENT_DESTROY            = 0x0402,
    HIP_OP_EVENT_RECORD             = 0x0403,
    HIP_OP_EVENT_SYNCHRONIZE        = 0x0404,
    HIP_OP_EVENT_QUERY              = 0x0405,
    HIP_OP_EVENT_ELAPSED_TIME       = 0x0406,

    /* Module operations (0x05xx) */
    HIP_OP_MODULE_LOAD_DATA         = 0x0500,
    HIP_OP_MODULE_LOAD_DATA_EX      = 0x0501,
    HIP_OP_MODULE_UNLOAD            = 0x0502,
    HIP_OP_MODULE_GET_FUNCTION      = 0x0503,
    HIP_OP_MODULE_GET_GLOBAL        = 0x0504,

    /* Kernel launch (0x051x) */
    HIP_OP_LAUNCH_KERNEL            = 0x0510,
    HIP_OP_LAUNCH_COOPERATIVE_KERNEL = 0x0511,
    HIP_OP_MODULE_LAUNCH_KERNEL     = 0x0512,
    HIP_OP_LAUNCH_COOPERATIVE_KERNEL_MULTI_DEVICE = 0x0513,
    HIP_OP_FUNC_GET_ATTRIBUTES      = 0x0514,  /* hipFuncGetAttributes */
    HIP_OP_FUNC_SET_ATTRIBUTE       = 0x0515,  /* hipFuncSetAttribute */
    HIP_OP_FUNC_SET_CACHE_CONFIG    = 0x0516,  /* hipFuncSetCacheConfig */

    /* Virtual-handle allocation (client assigns handle, worker maps to real) */
    HIP_OP_MALLOC_VADDR                 = 0x0280,
    HIP_OP_MALLOC_ASYNC_VADDR           = 0x0281,

    /* Batched/combined operations (0x074x-0x075x) */
    HIP_OP_MODULE_LOAD_AND_GET_FUNCTION = 0x0744,
    HIP_OP_MALLOC_BATCH                 = 0x0745,
    HIP_OP_STREAM_CREATE_BATCH          = 0x0746,
    HIP_OP_EVENT_CREATE_BATCH           = 0x0747,

    /* Function/pointer introspection (0x075x) */
    HIP_OP_MEM_PTR_GET_INFO             = 0x0751,

    /* Error handling (0x06xx) */
    HIP_OP_GET_LAST_ERROR           = 0x0600,
    HIP_OP_PEEK_AT_LAST_ERROR       = 0x0601,
    HIP_OP_GET_ERROR_STRING         = 0x0602,
    HIP_OP_GET_ERROR_NAME           = 0x0603,

    /* Runtime info (0x07xx) */
    HIP_OP_RUNTIME_GET_VERSION      = 0x0700,
    HIP_OP_DRIVER_GET_VERSION       = 0x0701,

    /* Occupancy (0x071x) */
    HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE = 0x0710,
    HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM = 0x0711,

    /* Graph operations (0x072x) */
    HIP_OP_GRAPH_CREATE                 = 0x0720,
    HIP_OP_GRAPH_DESTROY                = 0x0721,
    HIP_OP_GRAPH_INSTANTIATE            = 0x0722,
    HIP_OP_GRAPH_LAUNCH                 = 0x0723,
    HIP_OP_GRAPH_EXEC_DESTROY           = 0x0724,
    HIP_OP_STREAM_BEGIN_CAPTURE         = 0x0725,
    HIP_OP_STREAM_END_CAPTURE           = 0x0726,
    HIP_OP_STREAM_IS_CAPTURING          = 0x0727,
    HIP_OP_GRAPH_CLONE                  = 0x0728,
    HIP_OP_GRAPH_NODE_GET_DEPENDENCIES  = 0x0729,
    HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES = 0x072A,
    HIP_OP_GRAPH_EXEC_UPDATE            = 0x072B,
    HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS = 0x072C,

    /* Stream callbacks (0x073x) */
    HIP_OP_STREAM_ADD_CALLBACK          = 0x0730,

    /* Context operations [Deprecated] (0x09xx) */
    HIP_OP_CTX_CREATE                   = 0x0900,
    HIP_OP_CTX_DESTROY                  = 0x0901,
    HIP_OP_CTX_SET_CURRENT              = 0x0902,
    HIP_OP_CTX_GET_CURRENT              = 0x0903,
    HIP_OP_CTX_PUSH_CURRENT             = 0x0904,
    HIP_OP_CTX_POP_CURRENT              = 0x0905,
    HIP_OP_CTX_GET_DEVICE               = 0x0906,
    HIP_OP_CTX_GET_API_VERSION          = 0x0907,
    HIP_OP_CTX_GET_CACHE_CONFIG         = 0x0908,
    HIP_OP_CTX_SET_CACHE_CONFIG         = 0x0909,
    HIP_OP_CTX_GET_SHARED_MEM_CONFIG    = 0x090A,
    HIP_OP_CTX_SET_SHARED_MEM_CONFIG    = 0x090B,
    HIP_OP_CTX_SYNCHRONIZE              = 0x090C,
    HIP_OP_CTX_GET_FLAGS                = 0x090D,
    HIP_OP_CTX_ENABLE_PEER_ACCESS       = 0x090E,
    HIP_OP_CTX_DISABLE_PEER_ACCESS      = 0x090F,
    HIP_OP_DEVICE_PRIMARY_CTX_GET_STATE = 0x0910,
    HIP_OP_DEVICE_PRIMARY_CTX_RETAIN    = 0x0911,
    HIP_OP_DEVICE_PRIMARY_CTX_RELEASE   = 0x0912,
    HIP_OP_DEVICE_PRIMARY_CTX_RESET     = 0x0913,
    HIP_OP_DEVICE_PRIMARY_CTX_SET_FLAGS = 0x0914,

    /* AMD SMI operations (0x08xx) */
    SMI_OP_INIT                     = 0x0800,
    SMI_OP_SHUTDOWN                 = 0x0801,
    SMI_OP_GET_PROCESSOR_COUNT      = 0x0802,
    SMI_OP_GET_GPU_METRICS          = 0x0820,
    SMI_OP_GET_POWER_INFO           = 0x0821,
    SMI_OP_GET_CLOCK_INFO           = 0x0822,
    SMI_OP_GET_TEMP_METRIC          = 0x0823,
    SMI_OP_GET_GPU_ACTIVITY         = 0x0824,
    SMI_OP_GET_VRAM_USAGE           = 0x0825,
    SMI_OP_GET_ASIC_INFO            = 0x0830,

} HipRemoteOpCode;

/* ============================================================================
 * Message Flags
 * ============================================================================ */

/** Response flag - set in responses */
#define HIP_REMOTE_FLAG_RESPONSE        (1u << 0)

/** Error flag - set when operation failed */
#define HIP_REMOTE_FLAG_ERROR           (1u << 1)

/** Has inline data - payload contains bulk data after structured payload */
#define HIP_REMOTE_FLAG_HAS_INLINE_DATA (1u << 2)

/** Fire-and-forget flag - client does not expect a reply.
 *  Used to eliminate round-trip latency on async GPU operations. */
#define HIP_REMOTE_FLAG_NO_REPLY        (1u << 3)

/* ============================================================================
 * Protocol Header
 * ============================================================================ */

/**
 * Common header for all protocol messages.
 * Total size: 20 bytes
 */
typedef struct HIP_PACKED_ATTR {
    uint32_t magic;           /**< Must be HIP_REMOTE_MAGIC */
    uint16_t version;         /**< Protocol version */
    uint16_t op_code;         /**< Operation code (HipRemoteOpCode) */
    uint32_t request_id;      /**< Correlation ID for async matching */
    uint32_t payload_length;  /**< Bytes following this header */
    uint32_t flags;           /**< Message flags */
} HipRemoteHeader;

/* ============================================================================
 * Common Response Header
 * ============================================================================ */

/**
 * Common response header included in all responses.
 */
typedef struct HIP_PACKED_ATTR {
    int32_t error_code;       /**< hipError_t value */
} HipRemoteResponseHeader;

/* ============================================================================
 * Init Response (extended with content cache manifest)
 * ============================================================================ */

/**
 * Extended init response sent by the worker when content caching is active.
 * Contains a cache epoch and a list of content hashes currently cached in
 * GPU memory.  The client uses this to avoid re-sending data that the
 * worker already has.
 */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t cache_epoch;     /**< Worker identifier (PID); changes on restart */
    uint32_t cache_count;     /**< Number of cached hashes that follow */
    /* Followed by cache_count uint64_t hashes */
} HipRemoteInitResponse;

/* ============================================================================
 * Device Operations
 * ============================================================================ */

/* HIP_OP_SET_DEVICE / HIP_OP_GET_DEVICE / HIP_OP_DEVICE_GET_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    int32_t device_id;
} HipRemoteDeviceRequest;

typedef struct HIP_PACKED_ATTR {
    int32_t device_id;
    int32_t attribute;        /**< For HIP_OP_DEVICE_GET_ATTRIBUTE */
} HipRemoteDeviceAttributeRequest;

/* HIP_OP_GET_DEVICE_COUNT response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t count;
} HipRemoteDeviceCountResponse;

/* HIP_OP_GET_DEVICE response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t device_id;
} HipRemoteGetDeviceResponse;

/* HIP_OP_DEVICE_GET_ATTRIBUTE response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t value;
} HipRemoteDeviceAttributeResponse;

/* HIP_OP_GET_DEVICE_PROPERTIES response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    /* Device properties - matches hipDeviceProp_t layout for key fields */
    char name[256];
    uint64_t total_global_mem;
    uint64_t shared_mem_per_block;
    int32_t regs_per_block;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t max_threads_dim[3];
    int32_t max_grid_size[3];
    int32_t clock_rate;
    int32_t memory_clock_rate;
    int32_t memory_bus_width;
    int32_t major;
    int32_t minor;
    int32_t multi_processor_count;
    int32_t l2_cache_size;
    int32_t max_threads_per_multi_processor;
    int32_t compute_mode;
    int32_t pci_bus_id;
    int32_t pci_device_id;
    int32_t pci_domain_id;
    int32_t integrated;
    int32_t can_map_host_memory;
    int32_t concurrent_kernels;
    char gcn_arch_name[256];
} HipRemoteDevicePropertiesResponse;

/* HIP_OP_DEVICE_GET_LIMIT / HIP_OP_DEVICE_SET_LIMIT */
typedef struct HIP_PACKED_ATTR {
    int32_t limit;                /**< hipLimit_t enum value */
    uint64_t value;               /**< Value to set (for SET_LIMIT) */
} HipRemoteDeviceLimitRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t value;               /**< Current limit value */
} HipRemoteDeviceLimitResponse;

/* HIP_OP_DEVICE_CAN_ACCESS_PEER */
typedef struct HIP_PACKED_ATTR {
    int32_t device_id;            /**< Current device */
    int32_t peer_device_id;       /**< Peer device to check */
} HipRemoteDeviceCanAccessPeerRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t can_access_peer;      /**< 1 if peer access possible, 0 otherwise */
} HipRemoteDeviceCanAccessPeerResponse;

/* HIP_OP_DEVICE_ENABLE_PEER_ACCESS / HIP_OP_DEVICE_DISABLE_PEER_ACCESS */
typedef struct HIP_PACKED_ATTR {
    int32_t peer_device_id;       /**< Peer device to enable/disable access to */
    uint32_t flags;               /**< Flags (reserved, must be 0) */
} HipRemoteDevicePeerAccessRequest;

/* HIP_OP_DEVICE_GET */
typedef struct HIP_PACKED_ATTR {
    int32_t ordinal;              /**< Device ordinal */
} HipRemoteDeviceGetRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t device;              /**< Device handle (opaque) */
} HipRemoteDeviceGetResponse;

/* HIP_OP_DEVICE_GET_NAME */
typedef struct HIP_PACKED_ATTR {
    uint64_t device;              /**< Device handle */
    int32_t len;                  /**< Maximum length of name buffer */
} HipRemoteDeviceGetNameRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    char name[256];               /**< Device name string */
} HipRemoteDeviceGetNameResponse;

/* HIP_OP_DEVICE_TOTAL_MEM */
typedef struct HIP_PACKED_ATTR {
    uint64_t device;              /**< Device handle */
} HipRemoteDeviceTotalMemRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t bytes;               /**< Total memory in bytes */
} HipRemoteDeviceTotalMemResponse;

/* HIP_OP_DEVICE_GET_PCI_BUS_ID */
typedef struct HIP_PACKED_ATTR {
    int32_t device;               /**< Device ordinal */
    int32_t len;                  /**< Maximum length of PCI bus ID string */
} HipRemoteDeviceGetPCIBusIdRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    char pci_bus_id[32];          /**< PCI bus ID string (e.g., "0000:03:00.0") */
} HipRemoteDeviceGetPCIBusIdResponse;

/* HIP_OP_DEVICE_GET_BY_PCI_BUS_ID */
typedef struct HIP_PACKED_ATTR {
    char pci_bus_id[32];          /**< PCI bus ID string */
} HipRemoteDeviceGetByPCIBusIdRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t device;               /**< Device ordinal */
} HipRemoteDeviceGetByPCIBusIdResponse;

/* HIP_OP_DEVICE_COMPUTE_CAPABILITY */
typedef struct HIP_PACKED_ATTR {
    uint64_t device;              /**< Device handle */
} HipRemoteDeviceComputeCapabilityRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t major;                /**< Major compute capability version */
    int32_t minor;                /**< Minor compute capability version */
} HipRemoteDeviceComputeCapabilityResponse;

/* HIP_OP_DEVICE_GET_UUID */
#define HIP_UUID_SIZE 16

typedef struct HIP_PACKED_ATTR {
    uint64_t device;              /**< Device handle */
} HipRemoteDeviceGetUuidRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint8_t uuid[HIP_UUID_SIZE];  /**< Device UUID bytes */
} HipRemoteDeviceGetUuidResponse;

/* HIP_OP_DEVICE_GET_CACHE_CONFIG / HIP_OP_DEVICE_SET_CACHE_CONFIG */
typedef struct HIP_PACKED_ATTR {
    int32_t cache_config;         /**< hipFuncCache_t enum value */
} HipRemoteDeviceCacheConfigRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t cache_config;         /**< hipFuncCache_t enum value */
} HipRemoteDeviceCacheConfigResponse;

/* HIP_OP_DEVICE_GET_SHARED_MEM_CONFIG / HIP_OP_DEVICE_SET_SHARED_MEM_CONFIG */
typedef struct HIP_PACKED_ATTR {
    int32_t shared_mem_config;    /**< hipSharedMemConfig enum value */
} HipRemoteDeviceSharedMemConfigRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t shared_mem_config;    /**< hipSharedMemConfig enum value */
} HipRemoteDeviceSharedMemConfigResponse;

/* HIP_OP_SET_DEVICE_FLAGS / HIP_OP_GET_DEVICE_FLAGS */
typedef struct HIP_PACKED_ATTR {
    uint32_t flags;               /**< Device flags */
} HipRemoteDeviceFlagsRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t flags;               /**< Device flags */
} HipRemoteDeviceFlagsResponse;

/* HIP_OP_DEVICE_GET_P2P_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    int32_t attr;                 /**< hipDeviceP2PAttr enum value */
    int32_t src_device;           /**< Source device ordinal */
    int32_t dst_device;           /**< Destination device ordinal */
} HipRemoteDeviceGetP2PAttributeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t value;                /**< Attribute value */
} HipRemoteDeviceGetP2PAttributeResponse;

/* HIP_OP_SET_VALID_DEVICES */
typedef struct HIP_PACKED_ATTR {
    int32_t len;                  /**< Number of valid devices */
    /* Followed by len int32_t device IDs */
} HipRemoteSetValidDevicesRequest;

/* HIP_OP_CHOOSE_DEVICE */
typedef struct HIP_PACKED_ATTR {
    /* Contains key fields from hipDeviceProp_t for matching */
    uint64_t total_global_mem;
    int32_t major;
    int32_t minor;
    int32_t multi_processor_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
} HipRemoteChooseDeviceRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t device;               /**< Best matching device ordinal */
} HipRemoteChooseDeviceResponse;

/* HIP_OP_DEVICE_GET_STREAM_PRIORITY_RANGE */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t least_priority;       /**< Least priority value */
    int32_t greatest_priority;    /**< Greatest priority value */
} HipRemoteDeviceGetStreamPriorityRangeResponse;

/* ============================================================================
 * Memory Operations
 * ============================================================================ */

/* HIP_OP_MALLOC / HIP_OP_MALLOC_HOST / HIP_OP_MALLOC_MANAGED */
typedef struct HIP_PACKED_ATTR {
    uint64_t size;
    uint32_t flags;           /**< For managed memory flags */
} HipRemoteMallocRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t device_ptr;      /**< Remote device pointer (opaque handle) */
} HipRemoteMallocResponse;

/* HIP_OP_FREE / HIP_OP_FREE_HOST */
typedef struct HIP_PACKED_ATTR {
    uint64_t device_ptr;
} HipRemoteFreeRequest;

/* HIP_OP_MALLOC_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t size;
    uint64_t stream;          /**< Stream for async allocation */
} HipRemoteMallocAsyncRequest;

/* HIP_OP_FREE_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t device_ptr;
    uint64_t stream;          /**< Stream for async deallocation */
} HipRemoteFreeAsyncRequest;

/* HIP_OP_MALLOC_VADDR / HIP_OP_MALLOC_ASYNC_VADDR
 * Client assigns a virtual handle and sends FnF; worker allocates real memory
 * and stores the mapping. Eliminates the round-trip for hipMalloc. */
typedef struct HIP_PACKED_ATTR {
    uint64_t vaddr;           /**< Client-assigned virtual address */
    uint64_t size;
    uint64_t stream;          /**< For MALLOC_ASYNC_VADDR; 0 for sync */
    uint32_t flags;
} HipRemoteMallocVaddrRequest;

/* HIP_OP_MEMCPY / HIP_OP_MEMCPY_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t dst;             /**< Destination pointer */
    uint64_t src;             /**< Source pointer */
    uint64_t size;            /**< Size in bytes */
    int32_t kind;             /**< hipMemcpyKind */
    uint64_t stream;          /**< Stream handle (0 for default) */
} HipRemoteMemcpyRequest;

/* For H2D copies, inline data follows this header */
/* For D2H copies, response includes inline data */

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    /* For D2H: inline data follows of size 'size' from request */
} HipRemoteMemcpyResponse;

/* HIP_OP_MEMCPY_HTOD_CACHED
 * Content-addressed H2D memcpy.  Client hashes the source buffer and sends
 * the hash.  If the worker already has the data cached in GPU memory it
 * performs a fast D2D copy; otherwise the client includes inline data and
 * the worker caches it for future reuse.
 *
 * When inline data is present: HIP_REMOTE_FLAG_HAS_INLINE_DATA is set and
 * the raw bytes follow immediately after this struct.
 * When hash-only (cache hit expected): no inline data flag, no trailing bytes.
 */
typedef struct HIP_PACKED_ATTR {
    uint64_t hash;            /**< XXH64 of source data */
    uint64_t dst;             /**< Destination device pointer (vaddr) */
    uint64_t size;            /**< Data size in bytes */
    uint64_t stream;          /**< Stream handle (0 for default) */
} HipRemoteMemcpyCachedRequest;

/* HIP_OP_MEMCPY_2D / HIP_OP_MEMCPY_2D_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t dst;             /**< Destination pointer */
    uint64_t dpitch;          /**< Destination pitch (bytes per row including padding) */
    uint64_t src;             /**< Source pointer */
    uint64_t spitch;          /**< Source pitch (bytes per row including padding) */
    uint64_t width;           /**< Width of data to copy (bytes) */
    uint64_t height;          /**< Number of rows to copy */
    int32_t kind;             /**< hipMemcpyKind */
    uint32_t reserved;        /**< Padding for alignment */
    uint64_t stream;          /**< Stream handle (0 for default, used by async) */
} HipRemoteMemcpy2DRequest;

/* HIP_OP_MEMCPY_3D / HIP_OP_MEMCPY_3D_ASYNC */
typedef struct HIP_PACKED_ATTR {
    /* Source parameters */
    uint64_t src_ptr;         /**< Source pointer (device or host) */
    uint64_t src_pitch;       /**< Source pitch (bytes per row) */
    uint64_t src_height;      /**< Source height (rows per slice) */
    uint64_t src_x_offset;    /**< Source X offset in bytes */
    uint64_t src_y_offset;    /**< Source Y offset in rows */
    uint64_t src_z_offset;    /**< Source Z offset in slices */
    /* Destination parameters */
    uint64_t dst_ptr;         /**< Destination pointer (device or host) */
    uint64_t dst_pitch;       /**< Destination pitch (bytes per row) */
    uint64_t dst_height;      /**< Destination height (rows per slice) */
    uint64_t dst_x_offset;    /**< Destination X offset in bytes */
    uint64_t dst_y_offset;    /**< Destination Y offset in rows */
    uint64_t dst_z_offset;    /**< Destination Z offset in slices */
    /* Extent */
    uint64_t width;           /**< Width to copy (bytes) */
    uint64_t height;          /**< Height to copy (rows) */
    uint64_t depth;           /**< Depth to copy (slices) */
    /* Kind and stream */
    int32_t kind;             /**< hipMemcpyKind */
    uint32_t reserved;        /**< Padding for alignment */
    uint64_t stream;          /**< Stream handle (0 for default) */
} HipRemoteMemcpy3DRequest;

/* HIP_OP_MEMCPY_PEER / HIP_OP_MEMCPY_PEER_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t dst;             /**< Destination pointer on dstDevice */
    int32_t dst_device;       /**< Destination device ID */
    uint64_t src;             /**< Source pointer on srcDevice */
    int32_t src_device;       /**< Source device ID */
    uint64_t size;            /**< Size in bytes */
    uint64_t stream;          /**< Stream handle (for async version) */
} HipRemoteMemcpyPeerRequest;

/* HIP_OP_MEMSET / HIP_OP_MEMSET_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t dst;
    int32_t value;
    uint64_t size;
    uint64_t stream;
} HipRemoteMemsetRequest;

/* HIP_OP_MEM_GET_INFO response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t free_bytes;
    uint64_t total_bytes;
} HipRemoteMemGetInfoResponse;

/* HIP_OP_POINTER_GET_ATTRIBUTES */
typedef struct HIP_PACKED_ATTR {
    uint64_t ptr;             /**< Pointer to query */
} HipRemotePointerGetAttributesRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t memory_type;      /**< hipMemoryType enum value */
    int32_t device;           /**< Device ID */
    uint64_t device_pointer;  /**< Device pointer */
    uint64_t host_pointer;    /**< Host pointer */
    int32_t is_managed;       /**< Is managed memory */
    uint32_t allocation_flags;/**< Allocation flags */
} HipRemotePointerGetAttributesResponse;

/* HIP_OP_POINTER_GET_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    uint64_t ptr;             /**< Pointer to query */
    int32_t attribute;        /**< hipPointer_attribute enum value */
    uint32_t reserved;        /**< Padding */
} HipRemotePointerGetAttributeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    /* Data follows - size depends on attribute type */
    uint64_t data;            /**< Attribute data (up to 8 bytes) */
} HipRemotePointerGetAttributeResponse;

/* ============================================================================
 * IPC Operations
 * ============================================================================ */

/** IPC handle size - matches HIP's hipIpcMemHandle_t (64 bytes) */
#define HIP_REMOTE_IPC_HANDLE_SIZE 64

/* HIP_OP_IPC_GET_MEM_HANDLE */
typedef struct HIP_PACKED_ATTR {
    uint64_t device_ptr;      /**< Device pointer to get handle for */
} HipRemoteIpcGetMemHandleRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint8_t handle[HIP_REMOTE_IPC_HANDLE_SIZE];  /**< IPC memory handle */
} HipRemoteIpcGetMemHandleResponse;

/* HIP_OP_IPC_OPEN_MEM_HANDLE */
typedef struct HIP_PACKED_ATTR {
    uint8_t handle[HIP_REMOTE_IPC_HANDLE_SIZE];  /**< IPC memory handle */
    uint32_t flags;           /**< Flags (hipIpcMemLazyEnablePeerAccess) */
} HipRemoteIpcOpenMemHandleRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t device_ptr;      /**< Opened device pointer */
} HipRemoteIpcOpenMemHandleResponse;

/* HIP_OP_IPC_CLOSE_MEM_HANDLE */
typedef struct HIP_PACKED_ATTR {
    uint64_t device_ptr;      /**< Device pointer to close */
} HipRemoteIpcCloseMemHandleRequest;

/* HIP_OP_IPC_GET_EVENT_HANDLE */
typedef struct HIP_PACKED_ATTR {
    uint64_t event;           /**< Event to get handle for */
} HipRemoteIpcGetEventHandleRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint8_t handle[HIP_REMOTE_IPC_HANDLE_SIZE];  /**< IPC event handle */
} HipRemoteIpcGetEventHandleResponse;

/* HIP_OP_IPC_OPEN_EVENT_HANDLE */
typedef struct HIP_PACKED_ATTR {
    uint8_t handle[HIP_REMOTE_IPC_HANDLE_SIZE];  /**< IPC event handle */
} HipRemoteIpcOpenEventHandleRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t event;           /**< Opened event handle */
} HipRemoteIpcOpenEventHandleResponse;

/* ============================================================================
 * Memory Pool Operations
 * ============================================================================ */

/* Memory pool allocation type */
typedef enum {
    HIP_MEM_ALLOCATION_TYPE_INVALID = 0,
    HIP_MEM_ALLOCATION_TYPE_PINNED = 1,
    HIP_MEM_ALLOCATION_TYPE_MAX = 2
} HipMemAllocationType;

/* Memory pool location type */
typedef enum {
    HIP_MEM_LOCATION_TYPE_INVALID = 0,
    HIP_MEM_LOCATION_TYPE_DEVICE = 1,
    HIP_MEM_LOCATION_TYPE_MAX = 2
} HipMemLocationType;

/* Memory pool handle type */
typedef enum {
    HIP_MEM_HANDLE_TYPE_NONE = 0,
    HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
    HIP_MEM_HANDLE_TYPE_WIN32 = 2,
    HIP_MEM_HANDLE_TYPE_WIN32_KMT = 4
} HipMemHandleType;

/* Memory pool attributes */
typedef enum {
    HIP_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    HIP_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
    HIP_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD = 4,
    HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT = 5,
    HIP_MEM_POOL_ATTR_RESERVED_MEM_HIGH = 6,
    HIP_MEM_POOL_ATTR_USED_MEM_CURRENT = 7,
    HIP_MEM_POOL_ATTR_USED_MEM_HIGH = 8
} HipMemPoolAttrValue;

/* HIP_OP_MEM_POOL_CREATE */
typedef struct HIP_PACKED_ATTR {
    int32_t alloc_type;       /**< HipMemAllocationType */
    int32_t handle_types;     /**< Bitmask of HipMemHandleType */
    int32_t location_type;    /**< HipMemLocationType */
    int32_t location_id;      /**< Device ID for DEVICE type */
    uint64_t max_size;        /**< Maximum pool size (0 = unlimited) */
    uint8_t reserved[32];     /**< Reserved for future use */
} HipRemoteMemPoolCreateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t mem_pool;        /**< Memory pool handle */
} HipRemoteMemPoolCreateResponse;

/* HIP_OP_MEM_POOL_DESTROY */
typedef struct HIP_PACKED_ATTR {
    uint64_t mem_pool;        /**< Memory pool to destroy */
} HipRemoteMemPoolDestroyRequest;

/* HIP_OP_MEM_POOL_SET_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    uint64_t mem_pool;        /**< Memory pool */
    int32_t attr;             /**< Attribute to set (HipMemPoolAttrValue) */
    uint32_t reserved;        /**< Padding */
    uint64_t value;           /**< Attribute value */
} HipRemoteMemPoolSetAttributeRequest;

/* HIP_OP_MEM_POOL_GET_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    uint64_t mem_pool;        /**< Memory pool */
    int32_t attr;             /**< Attribute to get (HipMemPoolAttrValue) */
} HipRemoteMemPoolGetAttributeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t value;           /**< Attribute value */
} HipRemoteMemPoolGetAttributeResponse;

/* HIP_OP_MALLOC_FROM_POOL_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t size;            /**< Allocation size */
    uint64_t mem_pool;        /**< Memory pool to allocate from */
    uint64_t stream;          /**< Stream for async allocation */
} HipRemoteMallocFromPoolAsyncRequest;

/* Response uses HipRemoteMallocResponse */

/* HIP_OP_MEM_POOL_TRIM_TO */
typedef struct HIP_PACKED_ATTR {
    uint64_t mem_pool;        /**< Memory pool */
    uint64_t min_bytes_to_hold; /**< Minimum bytes to retain */
} HipRemoteMemPoolTrimToRequest;

/* HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL / HIP_OP_DEVICE_GET_MEM_POOL */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ID */
} HipRemoteDeviceGetMemPoolRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t mem_pool;        /**< Memory pool handle */
} HipRemoteDeviceGetMemPoolResponse;

/* HIP_OP_DEVICE_SET_MEM_POOL */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ID */
    uint32_t reserved;        /**< Padding */
    uint64_t mem_pool;        /**< Memory pool to set */
} HipRemoteDeviceSetMemPoolRequest;

/* HIP_OP_MEMPOOL_GET_ATTRIBUTE (abbreviated alias) */
typedef struct HIP_PACKED_ATTR {
    uint64_t mem_pool;        /**< Memory pool */
    int32_t attr;             /**< Attribute to get (HipMemPoolAttrValue) */
    int32_t reserved;         /**< Padding */
} HipRemoteMemPoolAttrRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t value;           /**< Attribute value */
} HipRemoteMemPoolAttrResponse;

/* HIP_OP_MEMPOOL_SET_ATTRIBUTE (abbreviated alias) */
typedef HipRemoteMemPoolSetAttributeRequest HipRemoteMemPoolSetAttrRequest;

/* HIP_OP_MEMPOOL_TRIM_TO (abbreviated alias) */
typedef struct HIP_PACKED_ATTR {
    uint64_t mem_pool;        /**< Memory pool */
    uint64_t min_bytes_to_keep; /**< Minimum bytes to retain */
} HipRemoteMemPoolTrimRequest;

/* ============================================================================
 * Host Memory Registration
 * ============================================================================ */

/* HIP_OP_HOST_REGISTER */
typedef struct HIP_PACKED_ATTR {
    uint64_t host_ptr;        /**< Host pointer to register */
    uint64_t size_bytes;      /**< Size in bytes */
    uint32_t flags;           /**< Registration flags */
} HipRemoteHostRegisterRequest;

/* HIP_OP_HOST_UNREGISTER */
typedef struct HIP_PACKED_ATTR {
    uint64_t host_ptr;        /**< Host pointer to unregister */
} HipRemoteHostUnregisterRequest;

/* HIP_OP_HOST_GET_DEVICE_POINTER */
typedef struct HIP_PACKED_ATTR {
    uint64_t host_ptr;        /**< Host pointer */
    uint32_t flags;           /**< Flags (reserved) */
} HipRemoteHostGetDevicePointerRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t device_ptr;      /**< Device pointer */
} HipRemoteHostGetDevicePointerResponse;

/* HIP_OP_HOST_GET_FLAGS */
typedef struct HIP_PACKED_ATTR {
    uint64_t host_ptr;        /**< Host pointer */
} HipRemoteHostGetFlagsRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t flags;           /**< Allocation flags */
} HipRemoteHostGetFlagsResponse;

/* HIP_OP_HOST_ALLOC */
typedef struct HIP_PACKED_ATTR {
    uint64_t size;            /**< Allocation size */
    uint32_t flags;           /**< Allocation flags */
} HipRemoteHostAllocRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t ptr;             /**< Allocated pointer */
} HipRemoteHostAllocResponse;

/* HIP_OP_HOST_FREE */
typedef struct HIP_PACKED_ATTR {
    uint64_t ptr;             /**< Pointer to free */
} HipRemoteHostFreeRequest;

/* HIP_OP_MEM_ALLOC_PITCH */
typedef struct HIP_PACKED_ATTR {
    uint64_t width_in_bytes;  /**< Width in bytes */
    uint64_t height;          /**< Height */
    uint32_t element_size;    /**< Element size in bytes */
} HipRemoteMemAllocPitchRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t dptr;            /**< Allocated device pointer */
    uint64_t pitch;           /**< Pitch in bytes */
} HipRemoteMemAllocPitchResponse;

/* ============================================================================
 * Unified Memory Management
 * ============================================================================ */

/* Memory advise values */
typedef enum {
    HIP_MEM_ADVISE_SET_READ_MOSTLY = 1,
    HIP_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    HIP_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    HIP_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    HIP_MEM_ADVISE_SET_ACCESSED_BY = 5,
    HIP_MEM_ADVISE_UNSET_ACCESSED_BY = 6
} HipMemoryAdvise;

/* Memory range attributes */
typedef enum {
    HIP_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    HIP_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    HIP_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    HIP_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
    HIP_MEM_RANGE_ATTRIBUTE_COHERENCY_MODE = 100
} HipMemRangeAttribute;

/* HIP_OP_MEM_ADVISE */
typedef struct HIP_PACKED_ATTR {
    uint64_t dev_ptr;         /**< Device pointer */
    uint64_t count;           /**< Size in bytes */
    int32_t advice;           /**< HipMemoryAdvise enum value */
    int32_t device;           /**< Device ID */
} HipRemoteMemAdviseRequest;

/* HIP_OP_MEM_PREFETCH_ASYNC */
typedef struct HIP_PACKED_ATTR {
    uint64_t dev_ptr;         /**< Device pointer */
    uint64_t count;           /**< Size in bytes */
    int32_t device;           /**< Destination device */
    uint32_t reserved;        /**< Padding */
    uint64_t stream;          /**< Stream for async operation */
} HipRemoteMemPrefetchAsyncRequest;

/* HIP_OP_MEM_RANGE_GET_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    uint64_t data_size;       /**< Size of data buffer */
    int32_t attribute;        /**< HipMemRangeAttribute enum */
    uint32_t reserved;        /**< Padding */
    uint64_t dev_ptr;         /**< Device pointer */
    uint64_t count;           /**< Range size in bytes */
} HipRemoteMemRangeGetAttributeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    /* Followed by data of size data_size */
} HipRemoteMemRangeGetAttributeResponse;

/* HIP_OP_MEM_RANGE_GET_ATTRIBUTES */
#define HIP_REMOTE_MAX_MEM_RANGE_ATTRIBUTES 8

typedef struct HIP_PACKED_ATTR {
    uint32_t num_attributes;  /**< Number of attributes to query */
    uint32_t reserved;        /**< Padding */
    uint64_t dev_ptr;         /**< Device pointer */
    uint64_t count;           /**< Range size in bytes */
    /* Followed by num_attributes int32_t attribute values */
    /* Followed by num_attributes uint64_t data_sizes */
} HipRemoteMemRangeGetAttributesRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    /* Followed by data for each attribute */
} HipRemoteMemRangeGetAttributesResponse;

/* ============================================================================
 * Context Operations [Deprecated]
 * ============================================================================ */

/* HIP_OP_CTX_CREATE */
typedef struct HIP_PACKED_ATTR {
    uint32_t flags;           /**< Context creation flags */
    uint32_t device;          /**< Device ordinal */
} HipRemoteCtxCreateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t ctx;             /**< Created context handle */
} HipRemoteCtxCreateResponse;

/* HIP_OP_CTX_DESTROY */
typedef struct HIP_PACKED_ATTR {
    uint64_t ctx;             /**< Context to destroy */
} HipRemoteCtxDestroyRequest;

/* HIP_OP_CTX_SET_CURRENT */
typedef struct HIP_PACKED_ATTR {
    uint64_t ctx;             /**< Context to set as current */
} HipRemoteCtxSetCurrentRequest;

/* HIP_OP_CTX_GET_CURRENT */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t ctx;             /**< Current context handle */
} HipRemoteCtxGetCurrentResponse;

/* HIP_OP_CTX_PUSH_CURRENT */
typedef struct HIP_PACKED_ATTR {
    uint64_t ctx;             /**< Context to push */
} HipRemoteCtxPushCurrentRequest;

/* HIP_OP_CTX_POP_CURRENT */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t ctx;             /**< Popped context handle */
} HipRemoteCtxPopCurrentResponse;

/* HIP_OP_CTX_GET_DEVICE */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t device;           /**< Device ordinal */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxGetDeviceResponse;

/* HIP_OP_CTX_GET_API_VERSION */
typedef struct HIP_PACKED_ATTR {
    uint64_t ctx;             /**< Context handle */
} HipRemoteCtxGetApiVersionRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t version;         /**< API version */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxGetApiVersionResponse;

/* HIP_OP_CTX_GET_CACHE_CONFIG */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t config;           /**< Cache config value */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxGetCacheConfigResponse;

/* HIP_OP_CTX_SET_CACHE_CONFIG */
typedef struct HIP_PACKED_ATTR {
    int32_t config;           /**< Cache config to set */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxSetCacheConfigRequest;

/* HIP_OP_CTX_GET_SHARED_MEM_CONFIG */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t config;           /**< Shared memory config */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxGetSharedMemConfigResponse;

/* HIP_OP_CTX_SET_SHARED_MEM_CONFIG */
typedef struct HIP_PACKED_ATTR {
    int32_t config;           /**< Shared memory config to set */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxSetSharedMemConfigRequest;

/* HIP_OP_CTX_GET_FLAGS */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t flags;           /**< Context flags */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxGetFlagsResponse;

/* HIP_OP_CTX_ENABLE_PEER_ACCESS */
typedef struct HIP_PACKED_ATTR {
    uint64_t peer_ctx;        /**< Peer context */
    uint32_t flags;           /**< Flags (must be 0) */
    uint32_t reserved;        /**< Padding */
} HipRemoteCtxEnablePeerAccessRequest;

/* HIP_OP_CTX_DISABLE_PEER_ACCESS */
typedef struct HIP_PACKED_ATTR {
    uint64_t peer_ctx;        /**< Peer context */
} HipRemoteCtxDisablePeerAccessRequest;

/* HIP_OP_DEVICE_PRIMARY_CTX_GET_STATE */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ordinal */
    uint32_t reserved;        /**< Padding */
} HipRemoteDevicePrimaryCtxGetStateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t flags;           /**< Context flags */
    int32_t active;           /**< Context active state */
} HipRemoteDevicePrimaryCtxGetStateResponse;

/* HIP_OP_DEVICE_PRIMARY_CTX_RETAIN */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ordinal */
    uint32_t reserved;        /**< Padding */
} HipRemoteDevicePrimaryCtxRetainRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t ctx;             /**< Primary context handle */
} HipRemoteDevicePrimaryCtxRetainResponse;

/* HIP_OP_DEVICE_PRIMARY_CTX_RELEASE */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ordinal */
    uint32_t reserved;        /**< Padding */
} HipRemoteDevicePrimaryCtxReleaseRequest;

/* HIP_OP_DEVICE_PRIMARY_CTX_RESET */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ordinal */
    uint32_t reserved;        /**< Padding */
} HipRemoteDevicePrimaryCtxResetRequest;

/* HIP_OP_DEVICE_PRIMARY_CTX_SET_FLAGS */
typedef struct HIP_PACKED_ATTR {
    int32_t device;           /**< Device ordinal */
    uint32_t flags;           /**< Context flags */
} HipRemoteDevicePrimaryCtxSetFlagsRequest;

/* ============================================================================
 * Graph Node Operations
 * ============================================================================ */

/** Max dependencies per graph node operation */
#define HIP_REMOTE_MAX_GRAPH_DEPENDENCIES 64

/** Graph node types */
typedef enum {
    HIP_GRAPH_NODE_TYPE_KERNEL = 0,
    HIP_GRAPH_NODE_TYPE_MEMCPY = 1,
    HIP_GRAPH_NODE_TYPE_MEMSET = 2,
    HIP_GRAPH_NODE_TYPE_HOST = 3,
    HIP_GRAPH_NODE_TYPE_GRAPH = 4,
    HIP_GRAPH_NODE_TYPE_EMPTY = 5,
    HIP_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    HIP_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    HIP_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    HIP_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    HIP_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    HIP_GRAPH_NODE_TYPE_MEM_FREE = 11,
    HIP_GRAPH_NODE_TYPE_COUNT = 12
} HipGraphNodeType;

/* HIP_OP_GRAPH_ADD_MEMCPY_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    uint32_t reserved;        /**< Padding */
    uint64_t dst;             /**< Destination pointer */
    uint64_t src;             /**< Source pointer */
    uint64_t width;           /**< Copy width */
    uint64_t height;          /**< Copy height */
    uint64_t dst_pitch;       /**< Destination pitch */
    uint64_t src_pitch;       /**< Source pitch */
    int32_t kind;             /**< hipMemcpyKind */
    uint32_t reserved2;       /**< Padding */
    /* Followed by num_deps uint64_t dependency node handles */
} HipRemoteGraphAddMemcpyNodeRequest;

/* HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    uint32_t reserved;        /**< Padding */
    uint64_t dst;             /**< Destination pointer */
    uint64_t src;             /**< Source pointer */
    uint64_t count;           /**< Number of bytes */
    int32_t kind;             /**< hipMemcpyKind */
    /* Followed by num_deps uint64_t dependency node handles */
} HipRemoteGraphAddMemcpyNode1DRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t node;            /**< Created graph node handle */
} HipRemoteGraphAddNodeResponse;

/* HIP_OP_GRAPH_ADD_MEMSET_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    uint32_t reserved;        /**< Padding */
    uint64_t dst;             /**< Destination pointer */
    uint64_t pitch;           /**< Pitch */
    int32_t value;            /**< Value to set */
    uint32_t element_size;    /**< Element size (1, 2, or 4) */
    uint64_t width;           /**< Width in elements */
    uint64_t height;          /**< Height */
    /* Followed by num_deps uint64_t dependency node handles */
} HipRemoteGraphAddMemsetNodeRequest;

/* HIP_OP_GRAPH_ADD_KERNEL_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    uint32_t reserved;        /**< Padding */
    uint64_t func;            /**< Kernel function handle */
    uint32_t grid_dim_x;      /**< Grid dimensions */
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;     /**< Block dimensions */
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;/**< Shared memory size */
    uint32_t num_args;        /**< Number of kernel arguments */
    /* Followed by num_deps uint64_t dependency node handles */
    /* Followed by num_args uint64_t kernel arguments */
} HipRemoteGraphAddKernelNodeRequest;

/* HIP_OP_GRAPH_ADD_DEPENDENCIES */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies to add */
    /* Followed by num_deps pairs of (from_node, to_node) uint64_t handles */
} HipRemoteGraphAddDependenciesRequest;

/* HIP_OP_GRAPH_ADD_EMPTY_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    /* Followed by num_deps uint64_t dependency node handles */
} HipRemoteGraphAddEmptyNodeRequest;

/* HIP_OP_GRAPH_GET_NODES / HIP_OP_GRAPH_GET_ROOT_NODES */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t max_nodes;       /**< Maximum nodes to return */
} HipRemoteGraphGetNodesRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t num_nodes;       /**< Number of nodes returned */
    uint32_t reserved;        /**< Padding */
    /* Followed by num_nodes uint64_t node handles */
} HipRemoteGraphGetNodesResponse;

/* HIP_OP_GRAPH_GET_EDGES */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t max_edges;       /**< Maximum edges to return */
} HipRemoteGraphGetEdgesRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t num_edges;       /**< Number of edges returned */
    uint32_t reserved;        /**< Padding */
    /* Followed by num_edges pairs of (from_node, to_node) uint64_t handles */
} HipRemoteGraphGetEdgesResponse;

/* HIP_OP_GRAPH_NODE_GET_TYPE */
typedef struct HIP_PACKED_ATTR {
    uint64_t node;            /**< Node handle */
} HipRemoteGraphNodeGetTypeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t type;             /**< HipGraphNodeType */
} HipRemoteGraphNodeGetTypeResponse;

/* HIP_OP_GRAPH_DESTROY_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t node;            /**< Node handle to destroy */
} HipRemoteGraphDestroyNodeRequest;

/* HIP_OP_GRAPH_ADD_HOST_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    uint32_t reserved;        /**< Padding */
    /* Note: Host callbacks not supported in remote mode - stub only */
    /* Followed by num_deps uint64_t dependency node handles */
} HipRemoteGraphAddHostNodeRequest;

/* HIP_OP_GRAPH_ADD_EVENT_RECORD_NODE / HIP_OP_GRAPH_ADD_EVENT_WAIT_NODE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t num_deps;        /**< Number of dependencies */
    uint32_t reserved;        /**< Padding */
    uint64_t event;           /**< Event handle */
    /* Followed by num_deps uint64_t dependency node handles */
} HipRemoteGraphAddEventNodeRequest;

/* ============================================================================
 * Stream Operations
 * ============================================================================ */

/* HIP_OP_STREAM_CREATE / HIP_OP_STREAM_CREATE_WITH_FLAGS */
typedef struct HIP_PACKED_ATTR {
    uint32_t flags;
    int32_t priority;         /**< For HIP_OP_STREAM_CREATE_WITH_PRIORITY */
    uint64_t vhandle;         /**< Client-assigned virtual stream handle (0 = legacy) */
} HipRemoteStreamCreateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t stream;          /**< Remote stream handle */
} HipRemoteStreamCreateResponse;

/* HIP_OP_STREAM_DESTROY / HIP_OP_STREAM_SYNCHRONIZE / HIP_OP_STREAM_QUERY */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;
} HipRemoteStreamRequest;

/* HIP_OP_STREAM_WAIT_EVENT */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;
    uint64_t event;
    uint32_t flags;
} HipRemoteStreamWaitEventRequest;

/* HIP_OP_STREAM_GET_FLAGS response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t flags;           /**< Stream flags */
} HipRemoteStreamGetFlagsResponse;

/* HIP_OP_STREAM_GET_PRIORITY response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t priority;         /**< Stream priority */
} HipRemoteStreamGetPriorityResponse;

/* HIP_OP_STREAM_GET_CAPTURE_INFO */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;
} HipRemoteStreamGetCaptureInfoRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t capture_status;   /**< hipStreamCaptureStatus */
    uint64_t graph;           /**< Graph being captured */
    uint32_t reserved;        /**< Padding */
} HipRemoteStreamGetCaptureInfoResponse;

/* HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;
    uint32_t num_dependencies;
    uint32_t flags;
    /* Followed by num_dependencies uint64_t node handles */
} HipRemoteStreamUpdateCaptureDependenciesRequest;

/* ============================================================================
 * Event Operations
 * ============================================================================ */

/* HIP_OP_EVENT_CREATE / HIP_OP_EVENT_CREATE_WITH_FLAGS */
typedef struct HIP_PACKED_ATTR {
    uint32_t flags;
} HipRemoteEventCreateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t event;           /**< Remote event handle */
} HipRemoteEventCreateResponse;

/* HIP_OP_EVENT_DESTROY / HIP_OP_EVENT_SYNCHRONIZE / HIP_OP_EVENT_QUERY */
typedef struct HIP_PACKED_ATTR {
    uint64_t event;
} HipRemoteEventRequest;

/* HIP_OP_EVENT_RECORD */
typedef struct HIP_PACKED_ATTR {
    uint64_t event;
    uint64_t stream;
} HipRemoteEventRecordRequest;

/* HIP_OP_EVENT_ELAPSED_TIME */
typedef struct HIP_PACKED_ATTR {
    uint64_t start_event;
    uint64_t end_event;
} HipRemoteEventElapsedTimeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    float milliseconds;
} HipRemoteEventElapsedTimeResponse;

/* ============================================================================
 * Module Operations
 * ============================================================================ */

/* HIP_OP_MODULE_LOAD_DATA
 * Payload: module data (code object) follows immediately after this struct
 */
typedef struct HIP_PACKED_ATTR {
    uint64_t data_size;       /**< Size of code object data */
    uint64_t vhandle;         /**< Client-assigned virtual module handle (0 = legacy sync) */
    /* Code object data follows */
} HipRemoteModuleLoadRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t module;          /**< Remote module handle */
} HipRemoteModuleLoadResponse;

/* HIP_OP_MODULE_UNLOAD */
typedef struct HIP_PACKED_ATTR {
    uint64_t module;
} HipRemoteModuleUnloadRequest;

/* HIP_OP_MODULE_GET_FUNCTION */
typedef struct HIP_PACKED_ATTR {
    uint64_t module;
    uint64_t vhandle;         /**< Client-assigned virtual function handle (0 = sync) */
    char function_name[1024];
} HipRemoteModuleGetFunctionRequest;

typedef struct HIP_PACKED_ATTR {
    uint32_t offset;
    uint32_t size;
    uint8_t  is_pointer;  /* 1 if value_kind == global_buffer */
    uint8_t  _pad[3];
} HipRemoteParamDesc;

#define HIP_REMOTE_MAX_PARAM_DESCS 64

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t function;        /**< Remote function handle */
    uint32_t num_args;        /**< kernarg_segment_size (legacy, kept for compat) */
    uint32_t num_params;      /**< Actual number of kernel parameters */
    HipRemoteParamDesc params[HIP_REMOTE_MAX_PARAM_DESCS];
} HipRemoteModuleGetFunctionResponse;

/* HIP_OP_MODULE_LOAD_AND_GET_FUNCTION: combined module load + get function */
typedef struct HIP_PACKED_ATTR {
    uint32_t name_length;
    uint32_t _pad;
    uint64_t vmodule;         /**< Client-assigned virtual module handle (0 = legacy sync) */
    uint64_t vfunc;           /**< Client-assigned virtual function handle (0 = legacy sync) */
} HipRemoteModuleLoadAndGetFunctionRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t module;
    uint64_t function;
    uint32_t num_params;
    uint32_t kernarg_size;
    HipRemoteParamDesc params[HIP_REMOTE_MAX_PARAM_DESCS];
} HipRemoteModuleLoadAndGetFunctionResponse;

/* HIP_OP_MALLOC_BATCH */
#define HIP_REMOTE_MAX_BATCH_MALLOC 64

typedef struct HIP_PACKED_ATTR {
    uint32_t count;
    uint32_t _pad;
    uint64_t sizes[HIP_REMOTE_MAX_BATCH_MALLOC];
} HipRemoteMallocBatchRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t count;
    uint64_t ptrs[HIP_REMOTE_MAX_BATCH_MALLOC];
} HipRemoteMallocBatchResponse;

/* HIP_OP_STREAM_CREATE_BATCH / HIP_OP_EVENT_CREATE_BATCH */
#define HIP_REMOTE_MAX_BATCH_HANDLES 32

typedef struct HIP_PACKED_ATTR {
    uint32_t count;
    uint32_t flags;
} HipRemoteHandleBatchRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t count;
    uint32_t _pad;
    uint64_t handles[HIP_REMOTE_MAX_BATCH_HANDLES];
} HipRemoteHandleBatchResponse;

/* HIP_OP_MEM_PTR_GET_INFO */
typedef struct HIP_PACKED_ATTR {
    uint64_t ptr;
} HipRemoteMemPtrGetInfoRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t size;
} HipRemoteMemPtrGetInfoResponse;

/* ============================================================================
 * Kernel Launch
 * ============================================================================ */

/**
 * Kernel argument descriptor
 */
typedef struct HIP_PACKED_ATTR {
    uint32_t size;            /**< Argument size in bytes */
    uint32_t offset;          /**< Offset into arg_data array */
} HipRemoteKernelArg;

/**
 * HIP_OP_LAUNCH_KERNEL
 * Variable-size message: arg_data follows the fixed portion
 */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle from MODULE_GET_FUNCTION */
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint64_t stream;
    uint32_t num_args;
    uint32_t launch_flags;
    uint64_t start_event;     /**< Optional start event for timing (0 = none) */
    uint64_t stop_event;      /**< Optional stop event for timing (0 = none) */
    uint32_t ext_flags;       /**< Extended launch flags */
    /* HipRemoteKernelArg args[num_args] follows */
    /* uint8_t arg_data[] follows (concatenated argument values) */
} HipRemoteLaunchKernelRequest;

/* HIP_OP_LAUNCH_COOPERATIVE_KERNEL_MULTI_DEVICE
 * For multi-device cooperative kernel launch
 */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle */
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint64_t stream;
    int32_t num_devices;
    uint32_t flags;
    uint32_t num_args;
    /* Followed by num_devices int32_t device IDs */
    /* Followed by HipRemoteKernelArg args[num_args] */
    /* Followed by uint8_t arg_data[] */
} HipRemoteLaunchCooperativeKernelMultiDeviceRequest;

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/* HIP_OP_GET_ERROR_STRING / HIP_OP_GET_ERROR_NAME */
typedef struct HIP_PACKED_ATTR {
    int32_t error_code;
} HipRemoteErrorStringRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    char error_string[256];
} HipRemoteErrorStringResponse;

/* ============================================================================
 * Runtime Info
 * ============================================================================ */

/* HIP_OP_RUNTIME_GET_VERSION / HIP_OP_DRIVER_GET_VERSION response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t version;
} HipRemoteVersionResponse;

/* ============================================================================
 * Function Attributes Operations
 * ============================================================================ */

/* HIP_OP_FUNC_GET_ATTRIBUTES */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle */
} HipRemoteFuncGetAttributesRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t shared_size_bytes;           /**< Size of shared memory per block */
    int32_t const_size_bytes;            /**< Size of constant memory */
    int32_t local_size_bytes;            /**< Size of local memory per thread */
    int32_t num_regs;                    /**< Number of registers per thread */
    int32_t max_threads_per_block;       /**< Maximum threads per block */
    int32_t ptx_version;                 /**< PTX version */
    int32_t binary_version;              /**< Binary version */
    int32_t cache_mode_ca;               /**< Cache mode */
    int32_t max_dynamic_shared_size_bytes;  /**< Max dynamic shared memory */
    int32_t preferred_shared_memory_carveout; /**< Preferred shared memory carveout */
} HipRemoteFuncGetAttributesResponse;

/* HIP_OP_FUNC_SET_ATTRIBUTE */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle */
    int32_t attribute;        /**< Attribute to set */
    int32_t value;            /**< Attribute value */
} HipRemoteFuncSetAttributeRequest;

/* HIP_OP_FUNC_SET_CACHE_CONFIG */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle */
    int32_t cache_config;     /**< Cache configuration */
} HipRemoteFuncSetCacheConfigRequest;

/* ============================================================================
 * Occupancy Operations
 * ============================================================================ */

/* HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle */
    uint64_t dyn_shared_mem;  /**< Dynamic shared memory per block (bytes) */
    int32_t block_size_limit; /**< Max block size limit (0 = no limit) */
    uint32_t flags;           /**< Flags (reserved) */
} HipRemoteOccupancyMaxPotentialBlockSizeRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t min_grid_size;    /**< Minimum grid size for max occupancy */
    int32_t block_size;       /**< Optimal block size */
} HipRemoteOccupancyMaxPotentialBlockSizeResponse;

/* HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM */
typedef struct HIP_PACKED_ATTR {
    uint64_t function;        /**< Function handle */
    int32_t block_size;       /**< Block size to query */
    uint64_t dyn_shared_mem;  /**< Dynamic shared memory per block (bytes) */
} HipRemoteOccupancyMaxActiveBlocksPerSMRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t num_blocks;       /**< Max active blocks per SM */
} HipRemoteOccupancyMaxActiveBlocksPerSMResponse;

/* ============================================================================
 * Graph Operations
 * ============================================================================ */

/* HIP_OP_GRAPH_CREATE */
typedef struct HIP_PACKED_ATTR {
    uint32_t flags;           /**< Graph creation flags */
} HipRemoteGraphCreateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t graph;           /**< Graph handle */
} HipRemoteGraphCreateResponse;

/* HIP_OP_GRAPH_DESTROY */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle to destroy */
} HipRemoteGraphDestroyRequest;

/* HIP_OP_GRAPH_INSTANTIATE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph;           /**< Graph handle */
    uint32_t flags;           /**< Instantiation flags */
} HipRemoteGraphInstantiateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t graph_exec;      /**< Executable graph handle */
} HipRemoteGraphInstantiateResponse;

/* HIP_OP_GRAPH_LAUNCH */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph_exec;      /**< Executable graph handle */
    uint64_t stream;          /**< Stream to launch on */
} HipRemoteGraphLaunchRequest;

/* HIP_OP_GRAPH_EXEC_DESTROY */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph_exec;      /**< Executable graph handle to destroy */
} HipRemoteGraphExecDestroyRequest;

/* HIP_OP_STREAM_BEGIN_CAPTURE */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;          /**< Stream to begin capture on */
    int32_t mode;             /**< Capture mode (hipStreamCaptureMode) */
} HipRemoteStreamBeginCaptureRequest;

/* HIP_OP_STREAM_END_CAPTURE */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;          /**< Stream to end capture on */
} HipRemoteStreamEndCaptureRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t graph;           /**< Captured graph handle */
} HipRemoteStreamEndCaptureResponse;

/* HIP_OP_STREAM_IS_CAPTURING */
typedef struct HIP_PACKED_ATTR {
    uint64_t stream;          /**< Stream to query */
} HipRemoteStreamIsCapturingRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t capture_status;   /**< hipStreamCaptureStatus */
} HipRemoteStreamIsCapturingResponse;

/* HIP_OP_GRAPH_CLONE */
typedef struct HIP_PACKED_ATTR {
    uint64_t original_graph;  /**< Original graph to clone */
} HipRemoteGraphCloneRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t cloned_graph;    /**< Cloned graph handle */
} HipRemoteGraphCloneResponse;

/* HIP_OP_GRAPH_NODE_GET_DEPENDENCIES / HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES */
typedef struct HIP_PACKED_ATTR {
    uint64_t node;            /**< Node to query */
    uint32_t max_nodes;       /**< Maximum number of nodes to return */
} HipRemoteGraphNodeGetDependenciesRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t num_nodes;       /**< Number of dependency nodes */
    uint32_t reserved;        /**< Padding */
    /* Followed by num_nodes uint64_t node handles */
} HipRemoteGraphNodeGetDependenciesResponse;

/* HIP_OP_GRAPH_EXEC_UPDATE */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph_exec;      /**< Executable graph to update */
    uint64_t graph;           /**< Graph with new topology/parameters */
} HipRemoteGraphExecUpdateRequest;

typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t update_result;    /**< hipGraphExecUpdateResult */
} HipRemoteGraphExecUpdateResponse;

/* HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS */
typedef struct HIP_PACKED_ATTR {
    uint64_t graph_exec;      /**< Executable graph */
    uint64_t node;            /**< Kernel node to update */
    uint64_t func;            /**< Function handle */
    uint32_t grid_dim_x;      /**< Grid dimensions */
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;     /**< Block dimensions */
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem;      /**< Shared memory size */
    uint32_t num_params;      /**< Number of kernel parameters */
    /* Followed by num_params kernel parameter pointers */
} HipRemoteGraphExecKernelNodeSetParamsRequest;

/* ============================================================================
 * AMD SMI Operations
 * ============================================================================ */

/* SMI_OP_INIT request */
typedef struct HIP_PACKED_ATTR {
    uint64_t init_flags;          /**< amdsmi_init_flags_t */
} SmiRemoteInitRequest;

/* SMI_OP_GET_PROCESSOR_COUNT response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t processor_count;
} SmiRemoteProcessorCountResponse;

/* Request with processor index (used by most SMI queries) */
typedef struct HIP_PACKED_ATTR {
    uint32_t processor_index;     /**< Maps to remote amdsmi_processor_handle */
} SmiRemoteProcessorRequest;

/* SMI_OP_GET_GPU_METRICS response - summary of key metrics */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t temperature_edge;     /**< Edge temperature (C) */
    int32_t temperature_hotspot;  /**< Hotspot/junction temperature (C) */
    int32_t temperature_mem;      /**< Memory temperature (C) */
    uint32_t average_socket_power;/**< Average socket power (W) */
    uint32_t gfx_activity;        /**< GFX engine activity (%) */
    uint32_t umc_activity;        /**< Memory controller activity (%) */
    uint32_t mm_activity;         /**< Multimedia engine activity (%) */
    uint32_t current_gfxclk;      /**< Current GFX clock (MHz) */
    uint32_t current_uclk;        /**< Current memory clock (MHz) */
    uint32_t current_socclk;      /**< Current SOC clock (MHz) */
    uint64_t vram_total;          /**< Total VRAM (bytes) */
    uint64_t vram_used;           /**< Used VRAM (bytes) */
    uint32_t fan_speed_rpm;       /**< Fan speed (RPM) */
    uint32_t pcie_bandwidth;      /**< PCIe bandwidth (MB/s) */
    uint32_t throttle_status;     /**< Throttle status flags */
    uint32_t reserved;            /**< Padding for alignment */
} SmiRemoteGpuMetricsResponse;

/* SMI_OP_GET_POWER_INFO response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t current_socket_power;/**< Current socket power (W) */
    uint32_t average_socket_power;/**< Average socket power (W) */
    uint32_t gfx_voltage;         /**< GFX voltage (mV) */
    uint32_t soc_voltage;         /**< SOC voltage (mV) */
    uint32_t mem_voltage;         /**< Memory voltage (mV) */
    uint32_t power_limit;         /**< Power limit/cap (W) */
} SmiRemotePowerInfoResponse;

/* SMI_OP_GET_CLOCK_INFO request */
typedef struct HIP_PACKED_ATTR {
    uint32_t processor_index;
    uint32_t clock_type;          /**< amdsmi_clk_type_t */
} SmiRemoteClockInfoRequest;

/* SMI_OP_GET_CLOCK_INFO response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t current_clk;         /**< Current clock (MHz) */
    uint32_t min_clk;             /**< Minimum clock (MHz) */
    uint32_t max_clk;             /**< Maximum clock (MHz) */
    uint8_t clk_locked;           /**< Clock locked flag */
    uint8_t clk_deep_sleep;       /**< Deep sleep flag */
    uint16_t reserved;            /**< Padding */
} SmiRemoteClockInfoResponse;

/* SMI_OP_GET_TEMP_METRIC request */
typedef struct HIP_PACKED_ATTR {
    uint32_t processor_index;
    uint32_t sensor_type;         /**< amdsmi_temperature_type_t */
} SmiRemoteTempMetricRequest;

/* SMI_OP_GET_TEMP_METRIC response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    int32_t temperature;          /**< Temperature (milli-Celsius) */
} SmiRemoteTempMetricResponse;

/* SMI_OP_GET_GPU_ACTIVITY response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint32_t gfx_activity;        /**< GFX activity (%) */
    uint32_t umc_activity;        /**< Memory controller activity (%) */
    uint32_t mm_activity;         /**< Multimedia activity (%) */
    uint32_t reserved;            /**< Padding */
} SmiRemoteGpuActivityResponse;

/* SMI_OP_GET_VRAM_USAGE response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    uint64_t vram_total;          /**< Total VRAM (bytes) */
    uint64_t vram_used;           /**< Used VRAM (bytes) */
} SmiRemoteVramUsageResponse;

/* SMI_OP_GET_ASIC_INFO response */
typedef struct HIP_PACKED_ATTR {
    HipRemoteResponseHeader header;
    char market_name[256];        /**< Marketing name (e.g., "AMD Instinct MI300X") */
    uint32_t vendor_id;           /**< PCI vendor ID */
    uint32_t device_id;           /**< PCI device ID */
    uint32_t rev_id;              /**< Revision ID */
    uint32_t num_compute_units;   /**< Number of compute units */
    char asic_serial[64];         /**< ASIC serial number */
} SmiRemoteAsicInfoResponse;

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Initialize a protocol header with the given parameters.
 */
static inline void hip_remote_init_header(
    HipRemoteHeader* header,
    HipRemoteOpCode op_code,
    uint32_t request_id,
    uint32_t payload_length
) {
    header->magic = HIP_REMOTE_MAGIC;
    header->version = HIP_REMOTE_VERSION;
    header->op_code = (uint16_t)op_code;
    header->request_id = request_id;
    header->payload_length = payload_length;
    header->flags = 0;
}

/**
 * Validate a received protocol header.
 * @return 0 on success, negative error code on failure
 */
static inline int hip_remote_validate_header(const HipRemoteHeader* header) {
    if (header->magic != HIP_REMOTE_MAGIC) {
        return -1;  /* Invalid magic */
    }
    if ((header->version >> 8) != (HIP_REMOTE_VERSION >> 8)) {
        return -2;  /* Major version mismatch */
    }
    if (header->payload_length > HIP_REMOTE_MAX_PAYLOAD_SIZE) {
        return -3;  /* Payload too large */
    }
    return 0;
}

/**
 * Get human-readable name for an operation code.
 */
static inline const char* hip_remote_op_name(HipRemoteOpCode op_code) {
    switch (op_code) {
        case HIP_OP_INIT: return "hipInit(remote)";
        case HIP_OP_SHUTDOWN: return "hipShutdown(remote)";
        case HIP_OP_PING: return "ping";

        case HIP_OP_GET_DEVICE_COUNT: return "hipGetDeviceCount";
        case HIP_OP_SET_DEVICE: return "hipSetDevice";
        case HIP_OP_GET_DEVICE: return "hipGetDevice";
        case HIP_OP_GET_DEVICE_PROPERTIES: return "hipGetDeviceProperties";
        case HIP_OP_DEVICE_SYNCHRONIZE: return "hipDeviceSynchronize";
        case HIP_OP_DEVICE_RESET: return "hipDeviceReset";
        case HIP_OP_DEVICE_GET_ATTRIBUTE: return "hipDeviceGetAttribute";
        case HIP_OP_DEVICE_GET_LIMIT: return "hipDeviceGetLimit";
        case HIP_OP_DEVICE_SET_LIMIT: return "hipDeviceSetLimit";
        case HIP_OP_DEVICE_CAN_ACCESS_PEER: return "hipDeviceCanAccessPeer";
        case HIP_OP_DEVICE_ENABLE_PEER_ACCESS: return "hipDeviceEnablePeerAccess";
        case HIP_OP_DEVICE_DISABLE_PEER_ACCESS: return "hipDeviceDisablePeerAccess";
        case HIP_OP_DEVICE_GET: return "hipDeviceGet";
        case HIP_OP_DEVICE_GET_NAME: return "hipDeviceGetName";
        case HIP_OP_DEVICE_TOTAL_MEM: return "hipDeviceTotalMem";
        case HIP_OP_DEVICE_GET_PCI_BUS_ID: return "hipDeviceGetPCIBusId";
        case HIP_OP_DEVICE_GET_BY_PCI_BUS_ID: return "hipDeviceGetByPCIBusId";
        case HIP_OP_DEVICE_COMPUTE_CAPABILITY: return "hipDeviceComputeCapability";
        case HIP_OP_DEVICE_GET_UUID: return "hipDeviceGetUuid";
        case HIP_OP_DEVICE_GET_CACHE_CONFIG: return "hipDeviceGetCacheConfig";
        case HIP_OP_DEVICE_SET_CACHE_CONFIG: return "hipDeviceSetCacheConfig";
        case HIP_OP_DEVICE_GET_SHARED_MEM_CONFIG: return "hipDeviceGetSharedMemConfig";
        case HIP_OP_DEVICE_SET_SHARED_MEM_CONFIG: return "hipDeviceSetSharedMemConfig";
        case HIP_OP_SET_DEVICE_FLAGS: return "hipSetDeviceFlags";
        case HIP_OP_GET_DEVICE_FLAGS: return "hipGetDeviceFlags";
        case HIP_OP_DEVICE_GET_P2P_ATTRIBUTE: return "hipDeviceGetP2PAttribute";
        case HIP_OP_SET_VALID_DEVICES: return "hipSetValidDevices";
        case HIP_OP_CHOOSE_DEVICE: return "hipChooseDevice";
        case HIP_OP_DEVICE_GET_STREAM_PRIORITY_RANGE: return "hipDeviceGetStreamPriorityRange";

        case HIP_OP_MALLOC: return "hipMalloc";
        case HIP_OP_FREE: return "hipFree";
        case HIP_OP_MALLOC_HOST: return "hipMallocHost";
        case HIP_OP_FREE_HOST: return "hipFreeHost";
        case HIP_OP_MALLOC_MANAGED: return "hipMallocManaged";
        case HIP_OP_MALLOC_ASYNC: return "hipMallocAsync";
        case HIP_OP_MALLOC_VADDR: return "hipMalloc(vaddr)";
        case HIP_OP_MALLOC_ASYNC_VADDR: return "hipMallocAsync(vaddr)";
        case HIP_OP_FREE_ASYNC: return "hipFreeAsync";

        case HIP_OP_MEMCPY: return "hipMemcpy";
        case HIP_OP_MEMCPY_ASYNC: return "hipMemcpyAsync";
        case HIP_OP_MEMCPY_DTOD: return "hipMemcpyDtoD";
        case HIP_OP_MEMCPY_DTOD_ASYNC: return "hipMemcpyDtoDAsync";
        case HIP_OP_MEMCPY_HTOD: return "hipMemcpyHtoD";
        case HIP_OP_MEMCPY_HTOD_ASYNC: return "hipMemcpyHtoDAsync";
        case HIP_OP_MEMCPY_DTOH: return "hipMemcpyDtoH";
        case HIP_OP_MEMCPY_DTOH_ASYNC: return "hipMemcpyDtoHAsync";
        case HIP_OP_MEMCPY_PEER: return "hipMemcpyPeer";
        case HIP_OP_MEMCPY_PEER_ASYNC: return "hipMemcpyPeerAsync";
        case HIP_OP_MEMCPY_HTOD_CACHED: return "hipMemcpyHtoD(cached)";
        case HIP_OP_MEMCPY_3D: return "hipMemcpy3D";
        case HIP_OP_MEMCPY_3D_ASYNC: return "hipMemcpy3DAsync";

        case HIP_OP_MEMSET: return "hipMemset";
        case HIP_OP_MEMSET_ASYNC: return "hipMemsetAsync";
        case HIP_OP_MEMSET_D8: return "hipMemsetD8";
        case HIP_OP_MEMSET_D16: return "hipMemsetD16";
        case HIP_OP_MEMSET_D32: return "hipMemsetD32";

        case HIP_OP_MEM_GET_INFO: return "hipMemGetInfo";
        case HIP_OP_POINTER_GET_ATTRIBUTES: return "hipPointerGetAttributes";
        case HIP_OP_POINTER_GET_ATTRIBUTE: return "hipPointerGetAttribute";

        case HIP_OP_IPC_GET_MEM_HANDLE: return "hipIpcGetMemHandle";
        case HIP_OP_IPC_OPEN_MEM_HANDLE: return "hipIpcOpenMemHandle";
        case HIP_OP_IPC_CLOSE_MEM_HANDLE: return "hipIpcCloseMemHandle";
        case HIP_OP_IPC_GET_EVENT_HANDLE: return "hipIpcGetEventHandle";
        case HIP_OP_IPC_OPEN_EVENT_HANDLE: return "hipIpcOpenEventHandle";

        case HIP_OP_MEM_POOL_CREATE: return "hipMemPoolCreate";
        case HIP_OP_MEM_POOL_DESTROY: return "hipMemPoolDestroy";
        case HIP_OP_MEM_POOL_SET_ATTRIBUTE: return "hipMemPoolSetAttribute";
        case HIP_OP_MEM_POOL_GET_ATTRIBUTE: return "hipMemPoolGetAttribute";
        case HIP_OP_MALLOC_FROM_POOL_ASYNC: return "hipMallocFromPoolAsync";
        case HIP_OP_MEM_POOL_TRIM_TO: return "hipMemPoolTrimTo";
        case HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL: return "hipDeviceGetDefaultMemPool";
        case HIP_OP_DEVICE_SET_MEM_POOL: return "hipDeviceSetMemPool";
        case HIP_OP_DEVICE_GET_MEM_POOL: return "hipDeviceGetMemPool";

        case HIP_OP_HOST_REGISTER: return "hipHostRegister";
        case HIP_OP_HOST_UNREGISTER: return "hipHostUnregister";
        case HIP_OP_HOST_GET_DEVICE_POINTER: return "hipHostGetDevicePointer";
        case HIP_OP_HOST_GET_FLAGS: return "hipHostGetFlags";
        case HIP_OP_HOST_ALLOC: return "hipHostAlloc";
        case HIP_OP_HOST_FREE: return "hipHostFree";
        case HIP_OP_MEM_ALLOC_PITCH: return "hipMemAllocPitch";

        case HIP_OP_MEM_ADVISE: return "hipMemAdvise";
        case HIP_OP_MEM_PREFETCH_ASYNC: return "hipMemPrefetchAsync";
        case HIP_OP_MEM_RANGE_GET_ATTRIBUTE: return "hipMemRangeGetAttribute";
        case HIP_OP_MEM_RANGE_GET_ATTRIBUTES: return "hipMemRangeGetAttributes";

        case HIP_OP_GRAPH_ADD_MEMCPY_NODE: return "hipGraphAddMemcpyNode";
        case HIP_OP_GRAPH_ADD_MEMSET_NODE: return "hipGraphAddMemsetNode";
        case HIP_OP_GRAPH_ADD_KERNEL_NODE: return "hipGraphAddKernelNode";
        case HIP_OP_GRAPH_ADD_DEPENDENCIES: return "hipGraphAddDependencies";
        case HIP_OP_GRAPH_ADD_EMPTY_NODE: return "hipGraphAddEmptyNode";
        case HIP_OP_GRAPH_GET_NODES: return "hipGraphGetNodes";
        case HIP_OP_GRAPH_GET_ROOT_NODES: return "hipGraphGetRootNodes";
        case HIP_OP_GRAPH_GET_EDGES: return "hipGraphGetEdges";
        case HIP_OP_GRAPH_NODE_GET_TYPE: return "hipGraphNodeGetType";
        case HIP_OP_GRAPH_DESTROY_NODE: return "hipGraphDestroyNode";
        case HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D: return "hipGraphAddMemcpyNode1D";
        case HIP_OP_GRAPH_ADD_HOST_NODE: return "hipGraphAddHostNode";
        case HIP_OP_GRAPH_ADD_CHILD_GRAPH_NODE: return "hipGraphAddChildGraphNode";
        case HIP_OP_GRAPH_ADD_EVENT_RECORD_NODE: return "hipGraphAddEventRecordNode";
        case HIP_OP_GRAPH_ADD_EVENT_WAIT_NODE: return "hipGraphAddEventWaitNode";

        case HIP_OP_STREAM_CREATE: return "hipStreamCreate";
        case HIP_OP_STREAM_CREATE_WITH_FLAGS: return "hipStreamCreateWithFlags";
        case HIP_OP_STREAM_CREATE_WITH_PRIORITY: return "hipStreamCreateWithPriority";
        case HIP_OP_STREAM_DESTROY: return "hipStreamDestroy";
        case HIP_OP_STREAM_SYNCHRONIZE: return "hipStreamSynchronize";
        case HIP_OP_STREAM_QUERY: return "hipStreamQuery";
        case HIP_OP_STREAM_WAIT_EVENT: return "hipStreamWaitEvent";
        case HIP_OP_STREAM_GET_FLAGS: return "hipStreamGetFlags";
        case HIP_OP_STREAM_GET_PRIORITY: return "hipStreamGetPriority";
        case HIP_OP_STREAM_GET_CAPTURE_INFO: return "hipStreamGetCaptureInfo";
        case HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES: return "hipStreamUpdateCaptureDependencies";

        case HIP_OP_EVENT_CREATE: return "hipEventCreate";
        case HIP_OP_EVENT_CREATE_WITH_FLAGS: return "hipEventCreateWithFlags";
        case HIP_OP_EVENT_DESTROY: return "hipEventDestroy";
        case HIP_OP_EVENT_RECORD: return "hipEventRecord";
        case HIP_OP_EVENT_SYNCHRONIZE: return "hipEventSynchronize";
        case HIP_OP_EVENT_QUERY: return "hipEventQuery";
        case HIP_OP_EVENT_ELAPSED_TIME: return "hipEventElapsedTime";

        case HIP_OP_MODULE_LOAD_DATA: return "hipModuleLoadData";
        case HIP_OP_MODULE_LOAD_DATA_EX: return "hipModuleLoadDataEx";
        case HIP_OP_MODULE_UNLOAD: return "hipModuleUnload";
        case HIP_OP_MODULE_GET_FUNCTION: return "hipModuleGetFunction";
        case HIP_OP_MODULE_GET_GLOBAL: return "hipModuleGetGlobal";
        case HIP_OP_MODULE_LOAD_AND_GET_FUNCTION: return "hipModuleLoadAndGetFunction";
        case HIP_OP_MALLOC_BATCH: return "hipMallocBatch";
        case HIP_OP_STREAM_CREATE_BATCH: return "hipStreamCreateBatch";
        case HIP_OP_EVENT_CREATE_BATCH: return "hipEventCreateBatch";
        case HIP_OP_MEM_PTR_GET_INFO: return "hipMemPtrGetInfo";

        case HIP_OP_LAUNCH_KERNEL: return "hipLaunchKernel";
        case HIP_OP_LAUNCH_COOPERATIVE_KERNEL: return "hipLaunchCooperativeKernel";
        case HIP_OP_LAUNCH_COOPERATIVE_KERNEL_MULTI_DEVICE: return "hipLaunchCooperativeKernelMultiDevice";
        case HIP_OP_MODULE_LAUNCH_KERNEL: return "hipModuleLaunchKernel";
        case HIP_OP_FUNC_GET_ATTRIBUTES: return "hipFuncGetAttributes";
        case HIP_OP_FUNC_SET_ATTRIBUTE: return "hipFuncSetAttribute";
        case HIP_OP_FUNC_SET_CACHE_CONFIG: return "hipFuncSetCacheConfig";

        case HIP_OP_GET_LAST_ERROR: return "hipGetLastError";
        case HIP_OP_PEEK_AT_LAST_ERROR: return "hipPeekAtLastError";
        case HIP_OP_GET_ERROR_STRING: return "hipGetErrorString";
        case HIP_OP_GET_ERROR_NAME: return "hipGetErrorName";

        case HIP_OP_RUNTIME_GET_VERSION: return "hipRuntimeGetVersion";
        case HIP_OP_DRIVER_GET_VERSION: return "hipDriverGetVersion";

        case HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE: return "hipOccupancyMaxPotentialBlockSize";
        case HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM: return "hipOccupancyMaxActiveBlocksPerMultiprocessor";

        case HIP_OP_GRAPH_CREATE: return "hipGraphCreate";
        case HIP_OP_GRAPH_DESTROY: return "hipGraphDestroy";
        case HIP_OP_GRAPH_INSTANTIATE: return "hipGraphInstantiate";
        case HIP_OP_GRAPH_LAUNCH: return "hipGraphLaunch";
        case HIP_OP_GRAPH_EXEC_DESTROY: return "hipGraphExecDestroy";
        case HIP_OP_STREAM_BEGIN_CAPTURE: return "hipStreamBeginCapture";
        case HIP_OP_STREAM_END_CAPTURE: return "hipStreamEndCapture";
        case HIP_OP_STREAM_IS_CAPTURING: return "hipStreamIsCapturing";
        case HIP_OP_GRAPH_CLONE: return "hipGraphClone";
        case HIP_OP_GRAPH_NODE_GET_DEPENDENCIES: return "hipGraphNodeGetDependencies";
        case HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES: return "hipGraphNodeGetDependentNodes";
        case HIP_OP_GRAPH_EXEC_UPDATE: return "hipGraphExecUpdate";
        case HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS: return "hipGraphExecKernelNodeSetParams";
        case HIP_OP_STREAM_ADD_CALLBACK: return "hipStreamAddCallback";

        case HIP_OP_CTX_CREATE: return "hipCtxCreate";
        case HIP_OP_CTX_DESTROY: return "hipCtxDestroy";
        case HIP_OP_CTX_SET_CURRENT: return "hipCtxSetCurrent";
        case HIP_OP_CTX_GET_CURRENT: return "hipCtxGetCurrent";
        case HIP_OP_CTX_PUSH_CURRENT: return "hipCtxPushCurrent";
        case HIP_OP_CTX_POP_CURRENT: return "hipCtxPopCurrent";
        case HIP_OP_CTX_GET_DEVICE: return "hipCtxGetDevice";
        case HIP_OP_CTX_GET_API_VERSION: return "hipCtxGetApiVersion";
        case HIP_OP_CTX_GET_CACHE_CONFIG: return "hipCtxGetCacheConfig";
        case HIP_OP_CTX_SET_CACHE_CONFIG: return "hipCtxSetCacheConfig";
        case HIP_OP_CTX_GET_SHARED_MEM_CONFIG: return "hipCtxGetSharedMemConfig";
        case HIP_OP_CTX_SET_SHARED_MEM_CONFIG: return "hipCtxSetSharedMemConfig";
        case HIP_OP_CTX_SYNCHRONIZE: return "hipCtxSynchronize";
        case HIP_OP_CTX_GET_FLAGS: return "hipCtxGetFlags";
        case HIP_OP_CTX_ENABLE_PEER_ACCESS: return "hipCtxEnablePeerAccess";
        case HIP_OP_CTX_DISABLE_PEER_ACCESS: return "hipCtxDisablePeerAccess";
        case HIP_OP_DEVICE_PRIMARY_CTX_GET_STATE: return "hipDevicePrimaryCtxGetState";
        case HIP_OP_DEVICE_PRIMARY_CTX_RETAIN: return "hipDevicePrimaryCtxRetain";
        case HIP_OP_DEVICE_PRIMARY_CTX_RELEASE: return "hipDevicePrimaryCtxRelease";
        case HIP_OP_DEVICE_PRIMARY_CTX_RESET: return "hipDevicePrimaryCtxReset";
        case HIP_OP_DEVICE_PRIMARY_CTX_SET_FLAGS: return "hipDevicePrimaryCtxSetFlags";

        case SMI_OP_INIT: return "amdsmi_init";
        case SMI_OP_SHUTDOWN: return "amdsmi_shut_down";
        case SMI_OP_GET_PROCESSOR_COUNT: return "amdsmi_get_processor_count";
        case SMI_OP_GET_GPU_METRICS: return "amdsmi_get_gpu_metrics";
        case SMI_OP_GET_POWER_INFO: return "amdsmi_get_power_info";
        case SMI_OP_GET_CLOCK_INFO: return "amdsmi_get_clock_info";
        case SMI_OP_GET_TEMP_METRIC: return "amdsmi_get_temp_metric";
        case SMI_OP_GET_GPU_ACTIVITY: return "amdsmi_get_gpu_activity";
        case SMI_OP_GET_VRAM_USAGE: return "amdsmi_get_vram_usage";
        case SMI_OP_GET_ASIC_INFO: return "amdsmi_get_asic_info";

        default: return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#ifdef _MSC_VER
#pragma pack(pop)
#endif

#endif /* HIP_REMOTE_PROTOCOL_H */
