////////////////////////////////////////////////////////////////////////////////
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: NCSA
//
////////////////////////////////////////////////////////////////////////////////

#include <cerrno>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/inc/amd_drm_driver.h"
#include "core/inc/amd_aql_queue.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/exceptions.h"

namespace rocr {
namespace AMD {

DrmDriver::DrmDriver(std::string devnode_name)
    : KfdDriver(core::DriverType::DRM, std::move(devnode_name)) {}

hsa_status_t DrmDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  auto tmp_driver = std::unique_ptr<core::Driver>(new DrmDriver("/dev/kfd"));

  if (core::Runtime::runtime_singleton_->flag().enable_drm()) {
    if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
      driver = std::move(tmp_driver);
      return HSA_STATUS_SUCCESS;
    }
  }

  return HSA_STATUS_ERROR;
}

hsa_status_t DrmDriver::CreateQueue(const CreateQueueInParams *queueIn, CreateQueueOutParams *queueOut) {
  if (!queueIn || !queueOut) {
    debug_print("queueIn or queueOut is NULL!\n");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto * drmQueueIn = static_cast<const DrmDriver::CreateQueueInParams *>(queueIn);
  auto * drmQueueOut = static_cast<DrmDriver::CreateQueueOutParams *>(queueOut);

  GpuAgent & gpu_agent = const_cast<GpuAgent&>(drmQueueIn->gpu_agent);
  const queue_type type = drmQueueIn->type;

  void *mqd_in = NULL;
  uint32_t ip_type = 0;

  __u32 doorbell_handle = 0;
  __u32 doorbell_offset = 0;
  uint64_t *queueDoorbellPtr = NULL;
  __u32 queue_id = 0;
  __u32 flags = 0;

  assert(drmQueueIn);
  assert(drmQueueOut);

  // Struct used to pass some parameters for the queue
  struct drm_amdgpu_userq_mqd_compute_gfx11 compute_mqd = { 0 };
  struct drm_amdgpu_userq_mqd_sdma_gfx11 sdma_mqd = { 0 };

  if (type == AQL_QUEUE) {
    compute_mqd.eop_va = drmQueueIn->extra_va;
    compute_mqd.ctx_save_area_addr = drmQueueIn->cwsr_va;
    compute_mqd.ctx_save_area_size = drmQueueIn->cwsr_size;
    fprintf(stderr, "DEBUG DRM: Creating compute queue with eop_va=0x%llx, cwsr_va=0x%llx, cwsr_size=%u\n",
            (unsigned long long)compute_mqd.eop_va, (unsigned long long)compute_mqd.ctx_save_area_addr, compute_mqd.ctx_save_area_size);
    mqd_in = &compute_mqd;
    // This is an HSA AQL queue — tell the kernel/MES to configure it in AQL mode.
    // Previously set SECURE (1<<2) which made MES treat it as a non-AQL (PM4) TMZ
    // queue; AQL packets then never executed → dispatch hang. AQL_COMPUTE is 1<<3.
    flags = AMDGPU_USERQ_CREATE_FLAGS_QUEUE_AQL_COMPUTE;
    ip_type = AMDGPU_HW_IP_COMPUTE;
  } else if (type == SDMA_QUEUE) {
    sdma_mqd.csa_va = drmQueueIn->extra_va;
    mqd_in = &sdma_mqd;
    flags = 0;
    ip_type = AMDGPU_HW_IP_DMA;
  } else {
    debug_print("Invalid queue type! (type: %u)\n", type);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (AllocateDoorbellOffset(gpu_agent, type, &doorbell_handle, &doorbell_offset, &queueDoorbellPtr) != HSA_STATUS_SUCCESS) {
    debug_print("Failed to allocate doorbell offset!\n");
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  int result = amdgpu_create_userqueue(gpu_agent.libDrmDev(),
                                       ip_type,
                                       doorbell_handle,
                                       doorbell_offset,
                                       (uint64_t) drmQueueIn->ring_buf,
                                       drmQueueIn->ring_buf_size,
                                       drmQueueIn->wptr_addr,
                                       drmQueueIn->rptr_addr,
                                       mqd_in,
                                       flags,
                                       &queue_id);

  if (result != 0) {
    debug_print("Failed to create user queue! Error code: %d (%s)\n", result, strerror(-result));
    FreeDoorbellOffset(gpu_agent, doorbell_offset);
    return HSA_STATUS_ERROR;
  }

  // Success, fill out result paramaters
  drmQueueOut->queue_id = queue_id;
  drmQueueOut->doorbell_ptr = queueDoorbellPtr;
  drmQueueOut->doorbell_offset = doorbell_offset;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DrmDriver::DestroyQueue(const DestroyQueueInParams *queueIn) {
  if (!queueIn) {
    debug_print("queueIn is NULL!\n");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto * drmQueueIn = static_cast<const DrmDriver::DestroyQueueInParams *>(queueIn);

  FreeDoorbellOffset(drmQueueIn->gpu_agent, drmQueueIn->doorbell_offset);

  if (amdgpu_free_userqueue(drmQueueIn->gpu_agent.libDrmDev(), drmQueueIn->queue_id) != 0) {
    debug_print("Failed to free userqueue!\n");
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DrmDriver::ModifyQueue(const ModifyQueueInParams *queueIn) {
  if (!queueIn) {
    debug_print("DrmDriver::ModifyQueue(); queueIn is NULL!\n");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto * drmQueueIn = static_cast<const DrmDriver::ModifyQueueInParams *>(queueIn);

  GpuAgent & gpu_agent = const_cast<GpuAgent&>(drmQueueIn->gpu_agent);
  const queue_type type = drmQueueIn->type;

  void *mqd_in = NULL;
  uint32_t ip_type = 0;

  // Struct used to pass modification parameters for the queue
  struct drm_amdgpu_userq_mqd_compute_gfx11 compute_mqd = { 0 };

  if (type != AQL_QUEUE) {
    debug_print("Only AQL queues can be modified (type: %u)\n", type);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  compute_mqd.eop_va = drmQueueIn->extra_va;

  // Set optional modification parameters if provided
  if (drmQueueIn->cu_mask && drmQueueIn->cu_mask_count > 0) {
    compute_mqd.cu_mask_ptr = (uint64_t)drmQueueIn->cu_mask;
    compute_mqd.cu_mask_count = drmQueueIn->cu_mask_count;
  }

  compute_mqd.queue_percentage = drmQueueIn->queue_percentage;
  compute_mqd.hqd_queue_priority = drmQueueIn->hqd_queue_priority;
  compute_mqd.pm4_target_xcc = drmQueueIn->pm4_target_xcc;

  mqd_in = &compute_mqd;
  ip_type = AMDGPU_HW_IP_COMPUTE;

  int result = amdgpu_modify_userqueue(gpu_agent.libDrmDev(),
                                       ip_type,
                                       drmQueueIn->queue_id,
                                       (uint64_t)drmQueueIn->ring_buf,
                                       drmQueueIn->ring_buf_size,
                                       drmQueueIn->wptr_addr,
                                       drmQueueIn->rptr_addr,
                                       mqd_in);

  if (result != 0) {
    debug_print("Failed to modify user queue!\n");
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

void DrmDriver::ReleaseResources(core::Agent &agent) {
  FreeDoorbellMemory(agent);
}

hsa_status_t DrmDriver::AllocateDoorbellMemory(core::Agent &agent) {
  auto &gpu_agent = static_cast<GpuAgent &>(agent);

  /*
   * Doorbell memory is allocated per process.
   *
   * Allocation is done with a lazy approach, it will be done only when first queue is created.
   *
   */
  if (!db_info_by_agent[&agent]) {
    struct amdgpu_bo_alloc_request req = {0};
    amdgpu_bo_handle buf_bo_handle = { 0 };
    __u32 buf_handle = 0;
    uint64_t *cpuPtr = NULL;

    // Numbers are hardcoded for now, this will be replaced with INFO ioctl once it is available.
    const uint32_t first_non_cp = 0x100;
    const uint32_t last_non_cp = 0x190;
    const uint32_t first_sdma = 0x100;
    const uint32_t last_sdma = 0x100 + 10 * 2 - 1;

    req.alloc_size = DOORBELL_SIZE;
    req.preferred_heap =  AMDGPU_GEM_DOMAIN_DOORBELL;

    if (amdgpu_bo_alloc(gpu_agent.libDrmDev(), &req, &buf_bo_handle) != 0)
    {
      debug_print("Failed to allocate doorbell memory!\n");
      return HSA_STATUS_ERROR;
    }

    if (amdgpu_bo_cpu_map(buf_bo_handle, (void **) &cpuPtr) != 0)
    {
      debug_print("Failed to cpu map doorbell memory!\n");
      amdgpu_bo_free(buf_bo_handle);
      return HSA_STATUS_ERROR;
    }

    if (amdgpu_bo_export(buf_bo_handle, amdgpu_bo_handle_type_kms, &buf_handle) != 0)
    {
      debug_print("amdgpu_bo_export failed!\n");

      amdgpu_bo_cpu_unmap(buf_bo_handle);
      amdgpu_bo_free(buf_bo_handle);
      return HSA_STATUS_ERROR;
    }

    // success, create object and setup all fields.
    doorbell_info_t *db_info = new doorbell_info_t;
    memset(db_info, 0, sizeof(*db_info));
    db_info->doorbell_bo_handle = buf_bo_handle;
    db_info->doorbell_handle = buf_handle;
    db_info->doorbellCpuPtr = cpuPtr;
    db_info_by_agent[&agent] = db_info;

    db_info->ranges[AQL_QUEUE][0] = { 0, first_non_cp - 1 };
    db_info->ranges[AQL_QUEUE][1] = { last_non_cp + 1, DOORBELL_PER_PAGE - 1};
    db_info->ranges[SDMA_QUEUE][0] = { first_sdma, last_sdma };
    db_info->ranges[SDMA_QUEUE][1] = { 1, 0 };  // using (start > end) to indicate empty range

  }

  return HSA_STATUS_SUCCESS;
}

void DrmDriver::FreeDoorbellMemory(core::Agent &agent) {
  doorbell_info_t *db_info = db_info_by_agent[&agent];

  if (db_info) {
    if (db_info->doorbellCpuPtr) {
      amdgpu_bo_cpu_unmap(db_info->doorbell_bo_handle);
      db_info->doorbellCpuPtr = NULL;
    }

    if (db_info->doorbell_handle)
    {
      amdgpu_bo_free(db_info->doorbell_bo_handle);
      db_info->doorbell_handle = 0;
    }

    db_info_by_agent.erase(&agent);
    delete db_info;
  }
}

hsa_status_t DrmDriver::AllocateDoorbellOffset(core::Agent &agent, queue_type type, __u32 *doorbell_handle, uint32_t *offset, uint64_t **cpuPtr) {
  doorbell_info_t *db_info = db_info_by_agent[&agent];

  if (!db_info) {
    hsa_status_t ret = AllocateDoorbellMemory(agent);
    if (ret == HSA_STATUS_SUCCESS) {
      db_info = db_info_by_agent[&agent];
    } else {
      return ret;
    }
  }

  assert(db_info);
  assert(db_info->doorbell_handle);
  assert(db_info->doorbellCpuPtr);
  assert((type == AQL_QUEUE) || (type == SDMA_QUEUE));

  /*
   * Each page has a range reserved for SDMA and other non-CP engines.
   * The range is the same on each page.
   * We currently have only 2 pages of doorbell per process, code is however written
   * to be general and to work if number of pages is increased.
   *
   */
  for (uint32_t page = 0; page < (DOORBELL_PER_PROCESS / DOORBELL_PER_PAGE); page++) {
    for (uint32_t r = 0; r < MAX_RANGES_PER_QUEUE_TYPE; r++) {
      const uint32_t start = db_info->ranges[type][r].start + page * DOORBELL_PER_PAGE;
      const uint32_t end = db_info->ranges[type][r].end + page * DOORBELL_PER_PAGE;

      assert(start < DOORBELL_PER_PROCESS);
      assert(end < DOORBELL_PER_PROCESS);

      for (int i = start; i < end; i++) {
        if ((db_info->doorbell_bitmap[i / DOORBELL_BITMAP_BIT_SIZE] & (1 << (i % DOORBELL_BITMAP_BIT_SIZE))) == 0) {
          db_info->doorbell_bitmap[i / DOORBELL_BITMAP_BIT_SIZE] |= (1 << (i % DOORBELL_BITMAP_BIT_SIZE));

          *doorbell_handle = db_info->doorbell_handle;
          *offset = i;
          *cpuPtr = &db_info->doorbellCpuPtr[i];
          return HSA_STATUS_SUCCESS;
        }
      }
    }
  }

  debug_warning("Exceeded available doorbells for process!\n");
  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

void DrmDriver::FreeDoorbellOffset(core::Agent &agent, uint32_t offset) {
  doorbell_info_t *db_info = db_info_by_agent[&agent];

  if (offset < DOORBELL_PER_PROCESS)
    db_info->doorbell_bitmap[offset / DOORBELL_BITMAP_BIT_SIZE] &= ~(1 << (offset % DOORBELL_BITMAP_BIT_SIZE));
  else
    assert(false && "Invalid offset");
}

hsa_status_t DrmDriver::GetUserQueueMetadata(core::Agent &agent, queue_type type, struct drm_amdgpu_info_uq_metadata *info) {
  if (!info)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  GpuAgent &gpu_agent = static_cast<GpuAgent &>(agent);
  uint32_t ip_type = 0;

  if (type == AQL_QUEUE) {
    ip_type = AMDGPU_HW_IP_COMPUTE;
  } else if (type == SDMA_QUEUE) {
    ip_type = AMDGPU_HW_IP_DMA;
  } else {
    debug_print("Invalid queue type! (type: %u)\n", type);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  int result = amdgpu_query_uq_fw_area_info(gpu_agent.libDrmDev(), ip_type, 0, info);

  return (result == 0) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

uint32_t DrmDriver::MapHsaPriorityToHqd(HSA_QUEUE_PRIORITY hsa_priority) {
  // Use the shared priority mapping from libhsakmt
  // Maps HSA_QUEUE_PRIORITY (-3 to +3) to hardware priority (0-15)
  // Mapping: -3→0, -2→3, -1→5, 0→7, +1→9, +2→11, +3→15
  return hsaKmtMapPriorityToHw(hsa_priority);
}

// KFD fallback implementations - used when DRM queue creation falls back to KFD.
hsa_status_t DrmDriver::UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                                    HSA::hsa_amd_queue_priority_internal_t priority, void* queue_addr,
                                    uint64_t queue_size_bytes, HsaEvent* event) const {
  // Delegate to KFD for queues created via KFD fallback when DRM is unsupported on this GPU.
  if (HSAKMT_CALL(hsaKmtUpdateQueue(queue_id, queue_pct, (HSA_QUEUE_PRIORITY)priority,
                                    queue_addr, queue_size_bytes, event)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t DrmDriver::SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                                       uint32_t* queue_cu_mask) const {
  // Delegate to KFD for queues created via KFD fallback when DRM is unsupported on this GPU.
  if (HSAKMT_CALL(hsaKmtSetQueueCUMask(queue_id, cu_mask_count, queue_cu_mask)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t DrmDriver::AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                                      uint32_t* first_gws) const {
  // Stub: GWS (Global Wave Sync) allocation not yet supported in DRM mode
  // May require new kernel ioctl support
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

// Legacy DestroyQueue: used for KFD-created queues when DRM queue creation falls back to KFD.
hsa_status_t DrmDriver::DestroyQueue(HSA_QUEUEID queue_id) const {
  if (HSAKMT_CALL(hsaKmtDestroyQueue(queue_id)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

namespace {

// Boolean DRM SVM attributes occupy the contiguous enum range
// [AMDGPU_SVM_ATTR_HOST_ACCESS, AMDGPU_SVM_ATTR_GPU_READ_MOSTLY]. Their enum
// values double as bit positions in the set/clear masks used below.
constexpr uint32_t kFirstDrmBoolAttr = AMDGPU_SVM_ATTR_HOST_ACCESS;
constexpr uint32_t kLastDrmBoolAttr = AMDGPU_SVM_ATTR_GPU_READ_MOSTLY;

// Guard against UB if DRM enum grows past 31 (shift would overflow uint32_t).
static_assert(kLastDrmBoolAttr < 32,
              "DRM boolean attributes exceed 32; DrmFlagBit would overflow");

constexpr uint32_t DrmFlagBit(uint32_t drm_attr) { return 1u << drm_attr; }

// Boolean DRM SVM attributes that SvmGetAttr can read back. AMDGPU_SVM_ATTR_GPU_EXEC
// is settable but not reported, so it is intentionally omitted.
constexpr uint32_t kReportedDrmFlags[] = {
    AMDGPU_SVM_ATTR_HOST_ACCESS,  AMDGPU_SVM_ATTR_COHERENT,
    AMDGPU_SVM_ATTR_EXT_COHERENT, AMDGPU_SVM_ATTR_HIVE_LOCAL,
    AMDGPU_SVM_ATTR_GPU_RO,       AMDGPU_SVM_ATTR_GPU_READ_MOSTLY,
};

// SVM location value meaning "this GPU/VRAM". The kernel's SVM model is one
// context per device, so it only distinguishes SYSMEM(0), UNDEFINED(0xffffffff),
// and "any other value => the device that received the ioctl".
constexpr uint32_t kSvmLocationThisGpu = 1u;

}  // namespace

hsa_status_t DrmDriver::SvmSetAttr(void* base, size_t size,
                                   const hsa_amd_svm_attribute_pair_t* attribs, size_t count) {
  const auto& gpu_agents = core::Runtime::runtime_singleton_->gpu_agents();

  if (gpu_agents.empty())
    return HSA_STATUS_ERROR;

  // if count is 0, reset all attributes to default values
  if (count == 0) {
    for (core::Agent* agent : gpu_agents) {
      amdgpu_device_handle dev = static_cast<GpuAgent*>(agent)->libDrmDev();
      if (dev == nullptr)
        continue;
      if (amdgpu_svm_reset_attr(dev, reinterpret_cast<uint64_t>(base), size))
        return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
  }

  std::vector<drm_amdgpu_svm_attribute> common;
  std::unordered_map<core::Agent*, std::vector<drm_amdgpu_svm_attribute>> per_device;
  // Bitmasks of boolean DRM attributes to set (=1) or clear (=0), indexed by
  // the DRM attribute enum value (see DrmFlagBit).
  uint32_t set_mask = 0, clr_mask = 0;

  SvmAttrParser parser("DrmDriver::SvmSetAttr");

  // Route an access policy for a specific agent to that GPU's context.
  auto PushAccess = [&](core::Agent* agent, uint32_t policy) {
    per_device[agent].push_back({AMDGPU_SVM_ATTR_ACCESS, policy});
  };

  for (size_t i = 0; i < count; ++i) {
    const auto attrib = attribs[i].attribute;
    const auto value = attribs[i].value;

    switch (attrib) {
      case HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG: {
        parser.CheckOnce(attrib);
        switch (value) {
          case HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED:
            set_mask |= DrmFlagBit(AMDGPU_SVM_ATTR_COHERENT);
            break;
          case HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED:
            clr_mask |= DrmFlagBit(AMDGPU_SVM_ATTR_COHERENT);
            break;
          default:
            throw hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                                "Invalid HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG value.");
        }
        break;
      }
      case HSA_AMD_SVM_ATTRIB_READ_ONLY:
        parser.CheckOnce(attrib);
        (value ? set_mask : clr_mask) |= DrmFlagBit(AMDGPU_SVM_ATTR_GPU_RO);
        break;
      case HSA_AMD_SVM_ATTRIB_HIVE_LOCAL:
        parser.CheckOnce(attrib);
        (value ? set_mask : clr_mask) |= DrmFlagBit(AMDGPU_SVM_ATTR_HIVE_LOCAL);
        break;
      case HSA_AMD_SVM_ATTRIB_READ_MOSTLY:
        parser.CheckOnce(attrib);
        (value ? set_mask : clr_mask) |= DrmFlagBit(AMDGPU_SVM_ATTR_GPU_READ_MOSTLY);
        break;
      case HSA_AMD_SVM_ATTRIB_GPU_EXEC:
        parser.CheckOnce(attrib);
        (value ? set_mask : clr_mask) |= DrmFlagBit(AMDGPU_SVM_ATTR_GPU_EXEC);
        break;
      case HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY: {
        parser.CheckOnce(attrib);
        uint32_t granularity = value;
        // Max migration size is 1GB.
        if (granularity > 18) granularity = 18;
        common.push_back({AMDGPU_SVM_ATTR_GRANULARITY, granularity});
        break;
      }
      case HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION: {
        parser.CheckOnce(attrib);
        core::Agent* agent = parser.ConvertAllowNull(value);
        if (agent == nullptr)
          common.push_back({AMDGPU_SVM_ATTR_PREFERRED_LOC, AMDGPU_SVM_LOCATION_UNDEFINED});
        else if (agent->device_type() == core::Agent::kAmdGpuDevice)
          per_device[agent].push_back(
              {AMDGPU_SVM_ATTR_PREFERRED_LOC, kSvmLocationThisGpu});
        else
          common.push_back({AMDGPU_SVM_ATTR_PREFERRED_LOC, AMDGPU_SVM_LOCATION_SYSMEM});
        break;
      }
      case HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE: {
        core::Agent* agent = parser.Convert(value);
        parser.ConfirmNew(agent);
        if (agent->device_type() == core::Agent::kAmdCpuDevice)
          set_mask |= DrmFlagBit(AMDGPU_SVM_ATTR_HOST_ACCESS);
        else
          PushAccess(agent, AMDGPU_SVM_ACCESS_ALLOW_MIGRATE);
        break;
      }
      case HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE: {
        core::Agent* agent = parser.Convert(value);
        parser.ConfirmNew(agent);
        if (agent->device_type() == core::Agent::kAmdCpuDevice)
          set_mask |= DrmFlagBit(AMDGPU_SVM_ATTR_HOST_ACCESS);
        else
          PushAccess(agent, AMDGPU_SVM_ACCESS_IN_PLACE);
        break;
      }
      case HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS: {
        core::Agent* agent = parser.Convert(value);
        parser.ConfirmNew(agent);
        if (agent->device_type() == core::Agent::kAmdCpuDevice)
          clr_mask |= DrmFlagBit(AMDGPU_SVM_ATTR_HOST_ACCESS);
        else
          PushAccess(agent, AMDGPU_SVM_ACCESS_INACCESSIBLE);
        break;
      }
      default:
        throw hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                            "Illegal or invalid attribute in DrmDriver::SvmSetAttr");
    }
  }

  // Merge CPU access properties - grant access if any CPU needs access.
  if (set_mask & DrmFlagBit(AMDGPU_SVM_ATTR_HOST_ACCESS))
    clr_mask &= ~DrmFlagBit(AMDGPU_SVM_ATTR_HOST_ACCESS);

  for (uint32_t attr = kFirstDrmBoolAttr; attr <= kLastDrmBoolAttr; ++attr) {
    if (set_mask & DrmFlagBit(attr))
      common.push_back({attr, 1});
    else if (clr_mask & DrmFlagBit(attr))
      common.push_back({attr, 0});
  }

  // Issue one ioctl per GPU with the shared entries plus that GPU's own ones.
  for (core::Agent* agent : gpu_agents) {
    amdgpu_device_handle dev = static_cast<GpuAgent*>(agent)->libDrmDev();
    if (dev == nullptr)
      continue;

    std::vector<drm_amdgpu_svm_attribute> attrs = common;
    auto it = per_device.find(agent);
    if (it != per_device.end())
      attrs.insert(attrs.end(), it->second.begin(), it->second.end());

    if (attrs.empty())
      continue;

    /* TODO: Handle partial failure.
     * If amdgpu_svm_set_attr succeeds for some devices but fails for a later
     * one, SVM attributes are left inconsistent across GPUs.
     * Two possible recovery strategies:
     *   1. Retry: On transient errors (e.g. EBUSY), retry the ioctl for the
     *      failing device with exponential back-off before giving up.
     *   2. Rollback: On unrecoverable errors, re-issue amdgpu_svm_set_attr
     *      with the original values on already-updated devices to restore a
     *      consistent state before returning an error.
     */
    if (amdgpu_svm_set_attr(dev, reinterpret_cast<uint64_t>(base), size,
                            static_cast<uint32_t>(attrs.size()), attrs.data()) != 0)
      return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DrmDriver::SvmGetAttr(void* base, size_t size,
                                   hsa_amd_svm_attribute_pair_t* attribs, size_t count) {
  const auto& gpu_agents = core::Runtime::runtime_singleton_->gpu_agents();
  if (gpu_agents.empty())
    return HSA_STATUS_ERROR;

  SvmAttrParser parser("DrmDriver::SvmGetAttr");

  // Boolean flags and migration granularity are range properties shared
  // across devices, so they are read from a representative device (gpu_agents[0]);
  // preferred/prefetch locations must be scanned across all GPUs; per-agent
  // access is read from that agent's device. A first pass determines what is
  // needed, then a single batched amdgpu_svm_get_attr per device collects it.
  bool need_flags = false;
  bool need_gran = false;
  bool need_pref = false;
  bool need_prefetch = false;
  std::unordered_set<core::Agent*> access_agents;

  for (size_t i = 0; i < count; ++i) {
    switch (attribs[i].attribute) {
      case HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG:
      case HSA_AMD_SVM_ATTRIB_READ_ONLY:
      case HSA_AMD_SVM_ATTRIB_HIVE_LOCAL:
      case HSA_AMD_SVM_ATTRIB_READ_MOSTLY:
        need_flags = true;
        break;
      case HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY:
        need_gran = true;
        break;
      case HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION:
        need_pref = true;
        break;
      case HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION:
        need_prefetch = true;
        break;
      case HSA_AMD_SVM_ATTRIB_ACCESS_QUERY: {
        core::Agent* agent = parser.Convert(attribs[i].value);
        if (agent->device_type() == core::Agent::kAmdCpuDevice)
          need_flags = true;
        else
          access_agents.insert(agent);
        break;
      }
      default:
        throw hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                            "Illegal or invalid attribute in DrmDriver::SvmGetAttr.");
    }
  }

  // agent -> (DRM attribute type -> value), populated only for successful reads.
  std::unordered_map<core::Agent*, std::unordered_map<uint32_t, uint32_t>> node_vals;

  for (core::Agent* agent : gpu_agents) {
    std::vector<drm_amdgpu_svm_attribute> req;
    if (agent == gpu_agents[0]) {
      if (need_flags)
        for (uint32_t drm_attr : kReportedDrmFlags)
          req.push_back({drm_attr, 0});
      if (need_gran)
        req.push_back({AMDGPU_SVM_ATTR_GRANULARITY, 0});
    }
    if (need_pref)
      req.push_back({AMDGPU_SVM_ATTR_PREFERRED_LOC, 0});
    if (need_prefetch)
      req.push_back({AMDGPU_SVM_ATTR_PREFETCH_LOC, 0});
    if (access_agents.count(agent))
      req.push_back({AMDGPU_SVM_ATTR_ACCESS, 0});

    if (req.empty())
      continue;

    amdgpu_device_handle dev = static_cast<GpuAgent*>(agent)->libDrmDev();
    if (dev == nullptr)
      continue;

    if (amdgpu_svm_get_attr(dev, reinterpret_cast<uint64_t>(base), size,
                            static_cast<uint32_t>(req.size()), req.data()) != 0)
      continue;

    auto& vals = node_vals[agent];
    for (const auto& a : req)
      vals[a.type] = a.value;
  }

  // Look up a cached attribute value for an agent. Returns false if the agent
  // was not queried successfully or did not report the attribute.
  auto node_val = [&](core::Agent* agent, uint32_t drm_attr, uint32_t& out) -> bool {
    auto nit = node_vals.find(agent);
    if (nit == node_vals.end())
      return false;
    auto vit = nit->second.find(drm_attr);
    if (vit == nit->second.end())
      return false;
    out = vit->second;
    return true;
  };

  // Reconstruct the boolean-flag bitmask (indexed by DRM attribute enum value,
  // see DrmFlagBit) from the representative device's cached values.
  auto flags = [&]() -> uint32_t {
    uint32_t f = 0;
    uint32_t v = 0;
    for (uint32_t drm_attr : kReportedDrmFlags)
      if (node_val(gpu_agents[0], drm_attr, v) && v)
        f |= DrmFlagBit(drm_attr);
    return f;
  };

  // Resolve a preferred/prefetch location to a public agent handle (handle 0
  // when the range has no or a mixed destination), scanning cached per-agent
  // values.
  auto location_handle = [&](uint32_t drm_type) -> uint64_t {
    core::Agent* result = nullptr;
    bool sysmem_seen = false;
    for (core::Agent* agent : gpu_agents) {
      uint32_t v = 0;
      if (!node_val(agent, drm_type, v))
        continue;
      if (v == AMDGPU_SVM_LOCATION_UNDEFINED) continue;
      if (v == AMDGPU_SVM_LOCATION_SYSMEM) {
        sysmem_seen = true;
        continue;
      }
      // Not sysmem and not undefined => the range lives on this GPU.
      result = agent;
      break;
    }
    // No GPU claims the range but some device reported sysmem => resolve to the
    // system-memory (CPU) agent at node 0.
    if (result == nullptr && sysmem_seen)
      result = core::Runtime::runtime_singleton_->agent_by_nodeid(0);

    return core::Agent::Convert(result).handle;
  };

  for (size_t i = 0; i < count; ++i) {
    auto& attrib = attribs[i].attribute;
    auto& value = attribs[i].value;

    switch (attrib) {
      case HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG: {
        value = (flags() & DrmFlagBit(AMDGPU_SVM_ATTR_COHERENT))
                    ? HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED
                    : HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED;
        break;
      }
      case HSA_AMD_SVM_ATTRIB_READ_ONLY: {
        value = (flags() & DrmFlagBit(AMDGPU_SVM_ATTR_GPU_RO)) ? 1 : 0;
        break;
      }
      case HSA_AMD_SVM_ATTRIB_HIVE_LOCAL: {
        value = (flags() & DrmFlagBit(AMDGPU_SVM_ATTR_HIVE_LOCAL)) ? 1 : 0;
        break;
      }
      case HSA_AMD_SVM_ATTRIB_READ_MOSTLY: {
        value = (flags() & DrmFlagBit(AMDGPU_SVM_ATTR_GPU_READ_MOSTLY)) ? 1 : 0;
        break;
      }
      case HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY: {
        uint32_t v = 0;
        if (node_val(gpu_agents[0], AMDGPU_SVM_ATTR_GRANULARITY, v))
          value = v;
        break;
      }
      case HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION: {
        value = location_handle(AMDGPU_SVM_ATTR_PREFERRED_LOC);
        break;
      }
      case HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION: {
        value = location_handle(AMDGPU_SVM_ATTR_PREFETCH_LOC);
        break;
      }
      case HSA_AMD_SVM_ATTRIB_ACCESS_QUERY: {
        core::Agent* agent = parser.Convert(value);
        if (agent->device_type() == core::Agent::kAmdCpuDevice) {
          attrib = (flags() & DrmFlagBit(AMDGPU_SVM_ATTR_HOST_ACCESS))
                       ? HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                       : HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS;
          break;
        }
        uint32_t v = 0;
        if (node_val(agent, AMDGPU_SVM_ATTR_ACCESS, v)) {
          switch (v) {
            case AMDGPU_SVM_ACCESS_IN_PLACE:
              attrib = HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE;
              break;
            case AMDGPU_SVM_ACCESS_INACCESSIBLE:
              attrib = HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS;
              break;
            case AMDGPU_SVM_ACCESS_ALLOW_MIGRATE:
            default:
              attrib = HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE;
              break;
          }
        }
        break;
      }
      default:
        throw hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                            "Illegal or invalid attribute in DrmDriver::SvmGetAttr.");
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DrmDriver::SvmPrefetch(const core::Agent& dstAgent, void* base, size_t size) {
  if (dstAgent.device_type() == core::Agent::kAmdGpuDevice) {
    // Migrate to a specific GPU: prefetch on that GPU's fd only.
    amdgpu_device_handle dev = static_cast<const GpuAgent&>(dstAgent).libDrmDev();
    if (dev == nullptr)
      return HSA_STATUS_ERROR;

    drm_amdgpu_svm_attribute attr{AMDGPU_SVM_ATTR_PREFETCH_LOC, kSvmLocationThisGpu};
    if (amdgpu_svm_set_attr(dev, reinterpret_cast<uint64_t>(base), size, 1, &attr) != 0)
      return HSA_STATUS_ERROR;

    return HSA_STATUS_SUCCESS;
  }

  const auto& gpu_agents = core::Runtime::runtime_singleton_->gpu_agents();
  if (gpu_agents.empty())
    return HSA_STATUS_ERROR;

  // Prefetch to system memory: every GPU's independent SVM context must be
  // updated, so issue the ioctl on each device's fd.
  /* TODO: NUMA awareness - the AMDGPU_SVM_ATTR_PREFETCH_LOC value for system
   * memory should reflect the NUMA node, rather than always using a fixed
   * SYSMEM location. This requires selecting the appropriate CPU agent (i.e.
   * the one whose NUMA node is nearest to the current GPU agent) and encoding
   * its NUMA node ID into the prefetch location attribute. Without this,
   * prefetched pages may land on a remote NUMA node, incurring higher memory
   * latency for the GPU.
   */
  drm_amdgpu_svm_attribute attr{AMDGPU_SVM_ATTR_PREFETCH_LOC, AMDGPU_SVM_LOCATION_SYSMEM};

  for (core::Agent* agent : gpu_agents) {
    amdgpu_device_handle dev = static_cast<GpuAgent*>(agent)->libDrmDev();
    if (dev == nullptr)
      continue;

    amdgpu_svm_set_attr(dev, reinterpret_cast<uint64_t>(base), size, 1, &attr);
  }
  return HSA_STATUS_SUCCESS;
}


} // namespace AMD
} // namespace rocr
