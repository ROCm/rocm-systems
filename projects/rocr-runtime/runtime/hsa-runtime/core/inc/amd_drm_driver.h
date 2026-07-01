////////////////////////////////////////////////////////////////////////////////
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: NCSA
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_DRM_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_DRM_DRIVER_H_

#include <unordered_map>

#include <amdgpu.h>
#include <amdgpu_drm.h>

#include "core/inc/amd_kfd_driver.h"
#include "core/inc/runtime.h"

namespace rocr {

namespace core {

class Queue;

}

namespace AMD {
  class GpuAgent;

  // Type definitions for doorbell

  // Max number of doorbell per process
  constexpr uint32_t DOORBELL_PER_PROCESS = 1024;

  // Doorbell memory size
  constexpr uint32_t DOORBELL_SIZE = 8 * DOORBELL_PER_PROCESS;

  // Number of doorbells per page
  constexpr uint32_t DOORBELL_PER_PAGE = 4096 / 8;

  // Size in bits of each element in bitmap
  constexpr uint32_t DOORBELL_BITMAP_BIT_SIZE = 8 * sizeof(uint64_t);

  // Number of elements in doorbell bitmap
  constexpr uint32_t DOORBELL_BITMAP_COUNT = DOORBELL_PER_PROCESS / DOORBELL_BITMAP_BIT_SIZE;

  // Maximum number of doorbell ranges per queue type
  constexpr uint32_t MAX_RANGES_PER_QUEUE_TYPE = 2;

/// @brief AMD Kernel Fusion Driver (KFD) for AMD GPU and CPU agents.
///
/// @details The user-mode driver into the Linux KFD for AMD GPU and CPU HSA
/// agents. Provides APIs for the ROCr core to discover the topology produced
/// by the KFD, allocate memory out of the KFD, manage DMA bufs, allocate queues,
/// and more.
class DrmDriver final : public KfdDriver {
public:
  typedef enum {
    AQL_QUEUE = 0,
    SDMA_QUEUE = 1,
    MAX_QUEUE_TYPE
  } queue_type;

  typedef struct doorbell_info_s {
    // Doorbell BO handle - to manage doorbell memory
    amdgpu_bo_handle doorbell_bo_handle;
    // Doorbell handle to pass to libdrm for queue creation
    __u32 doorbell_handle;
    // CPU virtual pointer for doorbell memory
    uint64_t *doorbellCpuPtr;
    // Bitmap for doorbell offset allocation
    uint64_t doorbell_bitmap[DOORBELL_BITMAP_COUNT];

    struct {
      uint32_t start;
      uint32_t end;
    } ranges[MAX_QUEUE_TYPE][MAX_RANGES_PER_QUEUE_TYPE];
  } doorbell_info_t;


  // Class containing queue creation input parameters for DRM interface
  class CreateQueueInParams {
  public:
    GpuAgent  &gpu_agent;
    queue_type type;
    void      *ring_buf;
    uint32_t   ring_buf_size;
    uint64_t  rptr_addr;
    uint64_t  wptr_addr;
    uint64_t  extra_va;      // Compute: EOP va, SDMA: CSA va
    uint64_t  cwsr_va;       // Compute: CWSR buffer va
    uint32_t  cwsr_size;     // Compute: CWSR buffer size

    CreateQueueInParams(GpuAgent &gpu_agent, queue_type type, void *ring_buf, uint32_t ring_buf_size, uint64_t rptr_addr, uint64_t wptr_addr, uint64_t extra_va, uint64_t cwsr_va = 0, uint32_t cwsr_size = 0) : gpu_agent(gpu_agent), type(type), ring_buf(ring_buf), ring_buf_size(ring_buf_size), rptr_addr(rptr_addr), wptr_addr(wptr_addr), extra_va(extra_va), cwsr_va(cwsr_va), cwsr_size(cwsr_size) { }
  };

  // Class containing queue creation output parameters for DRM interface
  class CreateQueueOutParams {
  public:
    uint32_t  queue_id;
    uint32_t  doorbell_offset;
    uint64_t *doorbell_ptr;
    CreateQueueOutParams() : queue_id(0), doorbell_offset(0), doorbell_ptr(nullptr) { }
  };

  // Class containing queue destruction input parameters for DRM interface
  class DestroyQueueInParams {
  public:
    GpuAgent &gpu_agent;
    uint32_t  queue_id;
    uint32_t  doorbell_offset;
    DestroyQueueInParams(GpuAgent &gpu_agent, uint32_t queue_id, uint32_t doorbell_offset) : gpu_agent(gpu_agent), queue_id(queue_id), doorbell_offset(doorbell_offset) { }
  };

  // Class containing queue modification input parameters for DRM interface
  class ModifyQueueInParams {
  public:
    GpuAgent  &gpu_agent;
    uint32_t   queue_id;
    queue_type type;
    void      *ring_buf;
    uint32_t   ring_buf_size;
    uint64_t   rptr_addr;
    uint64_t   wptr_addr;
    uint64_t   extra_va;  // Compute: EOP va

    // Optional modification parameters (use nullptr/0 to skip)
    uint32_t  *cu_mask;
    uint32_t   cu_mask_count;
    uint32_t   queue_percentage;
    uint32_t   hqd_queue_priority;
    uint32_t   pm4_target_xcc;

    ModifyQueueInParams(GpuAgent &gpu_agent, uint32_t queue_id,
                        queue_type type, void *ring_buf,
                        uint32_t ring_buf_size, uint64_t rptr_addr,
                        uint64_t wptr_addr, uint64_t extra_va)
      : gpu_agent(gpu_agent), queue_id(queue_id), type(type),
        ring_buf(ring_buf), ring_buf_size(ring_buf_size),
        rptr_addr(rptr_addr), wptr_addr(wptr_addr),
        extra_va(extra_va), cu_mask(nullptr), cu_mask_count(0),
        queue_percentage(0), hqd_queue_priority(0),
        pm4_target_xcc(0) { }
  };

  DrmDriver(std::string devnode_name);

  /// @brief Determine if should use DRM driver interface.
  ///
  /// @param[out] Driver object for the DRM interface.
  /// @return HSA_STATUS_SUCCESS if driver found and opened.
  /// @return HSA_STATUS_ERROR if unable to find or open DRM driver.
  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  hsa_status_t CreateQueue(const CreateQueueInParams *queueIn, CreateQueueOutParams *queueOut);
  hsa_status_t DestroyQueue(const DestroyQueueInParams *queueIn);
  hsa_status_t ModifyQueue(const ModifyQueueInParams *queueIn);
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority, void* queue_addr,
                           uint64_t queue_size_bytes, HsaEvent* event) const override;
  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;

  // Legacy DestroyQueue signature for backward compatibility
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;

  void ReleaseResources(core::Agent &agent) override;

  /// @brief Get hardware-specific parameters for user queue.
  ///
  /// @param [in] agent  The agent doing the request.
  /// @param [in] type   The type of queue for the request.
  /// @param[out] info   A non-NULL pointer where the information requested will be stored.
  hsa_status_t GetUserQueueMetadata(core::Agent &agent, queue_type type, struct drm_amdgpu_info_uq_metadata *info);

  /// @brief Map HSA queue priority to hardware queue priority
  ///
  /// @param hsa_priority HSA priority value (range: -3 to +3)
  /// @return Hardware queue priority value (range: 0-15)
  ///
  /// @details Maps HSA_QUEUE_PRIORITY to hqd_queue_priority.
  /// Uses shared mapping from libhsakmt (hsaKmtMapPriorityToHw).
  /// HSA priorities: MINIMUM=-3, LOW=-2, BELOW_NORMAL=-1,
  ///                 NORMAL=0, ABOVE_NORMAL=+1, HIGH=+2,
  ///                 MAXIMUM=+3
  /// HW priorities: 0-15, higher values = higher priority
  static uint32_t MapHsaPriorityToHqd(
      HSA_QUEUE_PRIORITY hsa_priority);

  private:
  /// @brief Allocate (if needed) doorbell memory
  ///
  /// @param [in] agent  The agent doing the request.
  /// @return HSA_STATUS_SUCCESS on success, error code on failure.
  ///
  hsa_status_t AllocateDoorbellMemory(core::Agent &agent);

  ///
  /// @brief Free doorbell resources
  ///
  /// @param [in] agent  The agent doing the request.
  ///
  void FreeDoorbellMemory(core::Agent &agent);

  ///
  /// @brief Allocate doorbell offset
  ///
  /// @param [ in] agent            The agent doing the request.
  /// @param [ in] type             The type of queue that will use the doorbell.
  /// @param [out] doorbell_handle  Doorbell handle to pass to libdrm userq API.
  /// @param [out] doorbell_offset  Offset of doorbell within doorbell memory.
  /// @param [out] cpuPtr           CPU pointer to doorbell allocated for queue.
  ///
  /// @return HSA_STATUS_SUCCESS on success, error code on failure.
  ///
  /// @details
  ///    Note: This function will allocate doorbell memory if not already allocated.
  ///       @see AllocateDoorbellMemory
  ///
  hsa_status_t AllocateDoorbellOffset(core::Agent &agent, queue_type type, __u32 *doorbell_handle, uint32_t *offset, uint64_t **cpuPtr);

  ///
  /// @brief Free doorbell offset.
  ///
  /// @param [in] agent  The agent doing the request.
  /// @param [in] offset The doorbell offset to be released.
  ///
  /// @details Offset can be re-used by a new queue once this function is called.
  ///
  void FreeDoorbellOffset(core::Agent &agent, uint32_t offset);

  /// @brief Store doorbell info state per agent.
  std::unordered_map<core::Agent *, doorbell_info_t *> db_info_by_agent;

};

} // namespace AMD
} // namespace rocr

#endif // header guard
