/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <vector>

namespace Wkmi {
/// @brief Memory allocation domain types
/// Specifies the memory domain for GPU allocations
enum AllocDomain {
  kSystem,        ///< System memory (GTT - Graphics Translation Table)
  kLocal,         ///< Local GPU memory (VRAM)
  kUserMemory,    ///< User-space memory mapping
  kUserQueue,     ///< User queue allocation
  kDomainCount,   ///< Total number of allocation domains
};

/// @brief Memory allocation flags
/// Bitfield flags to specify memory properties and coherency
enum MemFlag {
  kFineGrain    = (1ULL << 0),  ///< Fine-grained coherent memory access
  kKernarg      = (1ULL << 1),  ///< Kernel argument memory (uncached)
  kQueueObject  = (1ULL << 2),  ///< Queue object memory (AQL queue support)
};

/// @brief GPU engine type flags
/// Bitfield flags representing different GPU engine types
enum EngineFlag {
  KCOMPUTE0   = (1ULL << 0),  ///< Compute engine 0
  KDRMDMA     = (1ULL << 1),  ///< DMA engine 0 (SDMA)
  KDRMDMA1    = (1ULL << 2),  ///< DMA engine 1 (SDMA1)
};

/// @brief Queue scheduling priority levels
/// Defines priority levels for hardware queue scheduling
enum SchedLevel {
  kLow = 0,     ///< Low priority scheduling
  kNormal = 1,  ///< Normal priority scheduling (default)
  kHigh = 2,    ///< High priority scheduling
};

/// @brief Hardware scheduling information
/// Contains hardware scheduling (HWS) capabilities and engine support information
struct HwsInfo {
  union {
    struct {
      uint32_t gfxHwsEnabled     : 1;  ///< Graphics engine HWS enabled
      uint32_t computeHwsEnabled : 1;  ///< Compute engine HWS enabled
      uint32_t dmaHwsEnabled     : 1;  ///< DMA engine HWS enabled
      uint32_t dma1HwsEnabled    : 1;  ///< DMA1 engine HWS enabled
      uint32_t aql_queue         : 1;  ///< Kernel mode driver supports native AQL queue
      uint32_t reserved : 27;          ///< Reserved for future use
    } hwsMask;                         ///< Hardware scheduling mask bitfield
    uint32_t osHwsEnableFlags;         ///< OS HWS enable flags (raw value)
  };
  uint64_t engineOrdinalMask;          ///< Bitfield indicating which engines (by ordinal) support MES HWS
};

/// @brief Comprehensive GPU device information structure
/// Contains all device capabilities, hardware specifications, and driver information
/// for an AMD GPU. This structure is populated by ParseAdapterInfo().
struct DeviceInfo {
  // Architecture identification
  int major;                                  ///< GFX IP major version (e.g., 10, 11, 12)
  int minor;                                  ///< GFX IP minor version
  int stepping;                               ///< GFX IP stepping version
  bool is_dgpu;                               ///< True if discrete GPU, false if APU
  char product_name[MAX_PATH];                ///< Product name string (e.g., "Radeon RX 7900 XTX")
  uint64_t uuid;                              ///< Unique device identifier (product serial number)
  uint32_t family;                            ///< GPU family ID (e.g., FAMILY_NV, FAMILY_NV3)
  uint32_t device_id;                         ///< PCI device ID
  
  // Compute capabilities
  uint32_t wavefront_size;                    ///< Wavefront size (32 or 64)
  uint32_t compute_unit_count;                ///< Total number of compute units (CUs)
  uint32_t wave_per_cu;                       ///< Maximum waves per compute unit
  uint32_t simd_per_cu;                       ///< Number of SIMD units per CU (2 for RDNA, 4 for GCN)
  uint32_t max_scratch_slots_per_cu;          ///< Maximum scratch slots per CU
  uint32_t num_shader_engine;                 ///< Number of shader engines
  uint32_t shader_array_per_shader_engine;    ///< Number of shader arrays per shader engine
  
  // Clock frequencies
  uint32_t max_engine_clock_mhz;              ///< Maximum engine clock frequency in MHz
  uint32_t max_memory_clock_mhz;              ///< Maximum memory clock frequency in MHz
  uint64_t gpu_counter_frequency;             ///< GPU performance counter frequency in Hz
  
  // Hardware features
  uint32_t watch_points_num;                  ///< Number of hardware watchpoints for debugging
  uint32_t pci_bus_addr;                      ///< PCI bus address (bus << 8 | device << 3 | function)
  uint32_t memory_bus_width;                  ///< Memory bus width in bits
  uint32_t domain;                            ///< PCI domain
  uint32_t num_gws;                           ///< Number of global wave sync resources
  uint32_t asic_revision;                     ///< ASIC revision ID
  
  // Memory configuration
  uint64_t local_visible_heap_size;           ///< Local visible VRAM size in bytes (CPU accessible)
  uint64_t local_invisible_heap_size;         ///< Local invisible VRAM size in bytes (GPU only)
  uint64_t non_local_heap_size;               ///< Non-local (system) memory size in bytes
  uint64_t private_aperture_base;             ///< Private memory aperture base address
  uint64_t private_aperture_size;             ///< Private memory aperture size in bytes
  uint64_t shared_aperture_base;              ///< Shared memory aperture base address
  uint64_t shared_aperture_size;              ///< Shared memory aperture size in bytes
  
  // Queue and memory features
  uint32_t user_queue_size;                   ///< User queue size in bytes
  uint32_t lds_size;                          ///< Local data share (LDS) size in bytes
  uint32_t big_page_alignment_size;           ///< Large page alignment size (e.g., 64KB, 256KB)
  uint32_t hw_big_page_min_alignment_size;    ///< Hardware minimum large page alignment
  uint32_t hw_big_page_alignment_size;        ///< Hardware large page alignment size
  bool enable_big_page_alignment;             ///< True if large page alignment is enabled
  
  // Firmware versions
  uint32_t mec_fw_version;                    ///< Micro Engine Compute (MEC) firmware version
  uint32_t sdma_fw_version;                   ///< System DMA (SDMA) firmware version
  
  // Cache hierarchy
  uint32_t l1_cache_size;                     ///< L1 cache size in bytes
  uint32_t l2_cache_size;                     ///< L2 (GL2) cache size in bytes
  uint32_t l3_cache_size;                     ///< L3 (Mall) cache size in bytes
  uint32_t gl2_cacheline_size;                ///< GL2 cache line size in bytes (128 or 256)
  
  // Hardware scheduling
  uint32_t num_cp_queues;                     ///< Number of command processor queues
  HwsInfo hwsInfo;                            ///< Hardware scheduling information
  std::vector<int> sdma_schedid;              ///< SDMA scheduler IDs
  uint32_t compute_schedid;                   ///< Compute scheduler ID
  
  // Advanced features
  bool state_shadowing_by_cpfw;               ///< True if CP firmware supports state shadowing
  bool platform_atomic_support;               ///< True if platform supports atomic operations
  
  // Internal driver data
  void* adapter_info;                         ///< Pointer to internal adapter information (ADAPTERINFOEX)
  uint32_t kmd_version;                       ///< Kernel mode driver version (e.g., 2520 for 25.20)
  uint32_t gb_addr_config;                    ///< Graphics block address configuration
  uint32_t num_xcc;                           ///< Number of XCC (eXtended Compute Complex) units
};

struct KmdDbgVersion {
  uint32_t major;
  uint32_t minor;
};

// ============================================================================
// Device Query Functions
// ============================================================================

/// @brief Get the engine ordinal for a specific scheduler ID
/// @return Engine ordinal index, or -1 if not found
int EngineOrdinal(int engine,                 ///< Scheduler ID (e.g., DXUMD_SCHEDULERIDENTIFIER_COMPUTE0)
                  DeviceInfo* device_info);   ///< Pointer to device information structure

/// @brief Check if hardware scheduling (HWS) is enabled for an engine
/// @return True if HWS is enabled for the specified engine
bool GetHwsEnabled(int engine,                ///< Scheduler ID
                   DeviceInfo* device_info);  ///< Pointer to device information structure

/// @brief Query if GPU timeout should be disabled for an engine
/// @return True if GPU timeout should be disabled (engine supports status queries)
bool ShouldDisableGpuTimeout(int engine,                ///< Scheduler ID
                             DeviceInfo* device_info);  ///< Pointer to device information structure

/// @brief Check if the specified device ID is supported
/// @return True if the device is supported by this library
bool QueryAdapterSupported(unsigned int device_id);  ///< PCI device ID to check

// ============================================================================
// Memory Management Functions
// ============================================================================

/// @brief Convert queue engine ID to engine flag
/// @return Engine flag (KCOMPUTE0, KDRMDMA, or KDRMDMA1)
uint32_t QueueEngine2EngineFlag(uint32_t queue_engine);  ///< Queue engine scheduler ID

/// @brief Configure GPU memory allocation parameters
void SetAllocationInfo(void* data,                    ///< Pointer to allocation private data structure
                       uint64_t size,                 ///< Size of allocation in bytes
                       AllocDomain domain,            ///< Memory domain (kSystem, kLocal, kUserMemory, kUserQueue)
                       uint64_t addr,                 ///< Virtual address for user queue allocations
                       uint32_t mem_flags,            ///< Memory flags (kFineGrain, kKernarg, kQueueObject)
                       uint32_t engine_flag,          ///< Engine flag for queue allocations
                       const DeviceInfo& device_info);///< Device information structure (const reference)

/// @brief Get the required sizes for allocation private data
void GetAllocPrivDataSize(int* priv_drv_data_size,      ///< [out] Size of driver private data structure
                          int* priv_alloc_data_size);   ///< [out] Size of allocation private data structure

/// @brief Fill in driver private data for allocation
void FillinAllocPrivDrvData(void* drv_priv,              ///< Pointer to driver private data structure
                            int priv_alloc_data_size);   ///< Size of allocation private data

// ============================================================================
// Command Submission Functions
// ============================================================================

/// @brief Get the size of submission private data structure
/// @return Size in bytes of the submission private data structure
int GetSubmitPrivDataSize();

/// @brief Configure PM4 command buffer submission parameters
void FillinSubmitPrivData(void* priv_data,              ///< Pointer to submission private data structure
                          D3DKMT_HANDLE queue,          ///< Queue handle (user queue allocation handle for non-HW queues)
                          uint64_t command_addr,        ///< Virtual address of command buffer
                          uint64_t command_size,        ///< Size of command buffer in bytes
                          bool is_hw_queue);            ///< True if submitting to hardware queue, false for user queue

/// @brief Get the size of hardware queue private data structure
/// @return Size in bytes of the HW queue private data structure
int GetHwQueuePrivDataSize();

/// @brief Configure hardware queue creation parameters (Windows)
void FillinHwQueuePrivData(void* priv_data,              ///< Pointer to HW queue private data structure
                           bool FwManagedGfxState,       ///< True to enable firmware-managed graphics state
                           SchedLevel level = kNormal,   ///< Queue scheduling priority level (default: kNormal)
                           bool aql = false,             ///< True if creating an AQL queue (default: false)
                           uint64_t queue_va = 0,        ///< Virtual address of AQL queue memory (for AQL queues)
                           uint64_t queue_size = 0,      ///< Size of AQL queue memory in bytes (for AQL queues)
                           uint64_t wptr = 0,            ///< Initial write pointer value (for AQL queues)
                           uint64_t rptr = 0,            ///< Initial read pointer value (for AQL queues)
                           D3DKMT_HANDLE aql_queue_desc = 0, ///< Handle to AQL queue descriptor (for AQL queues)
                           uint32_t** doorbell_ptr = nullptr); ///< [out] Pointer to receive doorbell offset address (for AQL queues)

// ============================================================================
// Context Management Functions
// ============================================================================

/// @brief Get the size of context private data structure
/// @return Size in bytes of the context private data structure
int GetContextPrivDataSize();

/// @brief Query and parse GPU adapter information (Windows)
/// @return NTSTATUS code (STATUS_SUCCESS on success)
NTSTATUS ParseAdapterInfo(D3DKMT_HANDLE adapter,      ///< Handle to the D3DKMT adapter
                          DeviceInfo* device_info);   ///< Pointer to DeviceInfo structure to populate

/// @brief Fill in context creation private data (Windows)
void FillinContextPrivData(void* priv_data,                ///< Pointer to context private data structure
                           bool FwManagedGfxState,         ///< True to enable firmware-managed graphics state
                           uint32_t schedId = 0,           ///< Scheduler ID for the context (default: 0)
                           uint64_t debuggerData = 0);

// ============================================================================
// AQL Queue Submit Interfaces (Windows only)
// ============================================================================

/// @brief Get the size of AQL submission private data structure
/// @return Size in bytes of the AQL submission private data structure
int GetAqlSubmitPrivDataSize();

/// @brief Configure AQL packet submission with doorbell value
void FillinAqlSubmitPrivData(void* priv_data,              ///< Pointer to AQL submission private data structure
                             uint64_t doorbell_value);     ///< Write pointer (doorbell) value for AQL submission

// ============================================================================
// Queue CU Mask Interface (Windows only)
// ============================================================================

/// @brief Get the size of CU mask private data structure
/// @return Size in bytes of the CU mask private data structure
int GetCuMaskPrivDataSize();

/// @brief Configure compute unit mask for a specific queue
void FillinCuMaskPrivData(void* priv_data,                 ///< Pointer to CU mask private data structure
                          uint32_t doorbell,               ///< Doorbell offset identifying the queue
                          uint32_t cu_mask_count,          ///< Number of bits in the CU mask
                          const uint32_t* queue_cu_mask);  ///< Pointer to CU mask array

// ============================================================================
// User Mode Event Interfaces (Windows only)
// ============================================================================

/// @brief Get the size of event registration private data structure
/// @return Size in bytes of the event registration private data structure
int GetRegisterEventPrivDataSize();

/// @brief Register a user-mode event for HSA signal support
void FillinRegisterEventPrivData(void* priv_data,   ///< Pointer to event registration private data structure
                                 uint64_t handle,   ///< Event handle to register
                                 uint32_t event_id);///< Event identifier

/// @brief Get the mailbox array virtual address for event delivery
/// @return Mailbox array virtual address
uint64_t GetRegisterEventMailbox(void* priv_data);  ///< Pointer to event registration private data structure

/// @brief Get the size of event unregistration private data structure
/// @return Size in bytes of the event unregistration private data structure
int GetUnregisterEventPrivDataSize();

/// @brief Unregister a previously registered user-mode event
void FillinUnregisterEventPrivData(void* priv_data,  ///< Pointer to event unregistration private data structure
                                   uint64_t handle); ///< Event handle to unregister

// ============================================================================
// Interop Interfaces (Windows only)
// ============================================================================

/// @brief Query the size of a memory allocation from its private data
/// @return Allocation size in bytes
uint64_t GetMemoryAllocationSize(const void* priv_data);  ///< Pointer to allocation private data structure

/// @brief Get the size of the proxy resource info structure
/// @return Size in bytes of the PROXY_RESOURCE_INFO structure
size_t GetProxyResourceInfoSize();

// ============================================================================
// Power Management Functions
// ============================================================================

/// @brief Get the size of power optimization private data structure
/// @return Size in bytes of the power optimization private data structure
int GetPowerOptPrivDataSize();

/// @brief Configure power optimization settings
void FillinPowerOptPrivData(void* priv_data,  ///< Pointer to power optimization private data structure
                            bool restore);    ///< True to restore default settings, false to apply ML workload optimization

// ============================================================================
// Debugger Support Functions
// ============================================================================

/// @brief Query the size of a debugger command private data structure
/// @return Size in bytes
int GetDebuggerCmdPrivDataSize();

/// @brief Configure the KmdDbgVersion query
void FillinKmdDbgVersionPrivData(void* priv_data);

/// @brief Extract the KmdDbgVersion from the private data structure
void GetKmdDbgVersion(void* priv_data, KmdDbgVersion* version);

/// @brief Configure the RegisterRuntimeState debugger command
void FillinRegisterRuntimeStatePrivData(void* priv_data, uint32_t runtime_state,
                                        const void* r_debug, bool ttmp_setup_hint, HANDLE event);

/// @brief Configure the SetTrapHandler private data
void FillinTrapHandlerPrivData(void* priv_data, uint64_t tba, uint64_t tma);

}  // namespace Wkmi
