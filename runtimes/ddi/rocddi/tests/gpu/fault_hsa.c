// SPDX-License-Identifier: MIT

#include "fault_ioctl.h"
#include <dlfcn.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static hsa_agent_t gpu;
static hsa_status_t visit(hsa_agent_t agent, void *data) {
  (void)data;
  hsa_device_type_t type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU &&
      !gpu.handle)
    gpu = agent;
  return status;
}
int main(int argc, char **argv) {
  if (argc != 2 || (strcmp(argv[1], "create") && strcmp(argv[1], "mask")))
    return 2;
  void (*arm)(int) = dlsym(RTLD_DEFAULT, "rocddi_fault_arm");
  void (*stats)(struct fault_stats *) =
      dlsym(RTLD_DEFAULT, "rocddi_fault_stats");
  if (!arm || !stats)
    return 3;
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return 4;
  if (hsa_iterate_agents(visit, NULL) != HSA_STATUS_SUCCESS || !gpu.handle)
    return 5;
  hsa_status_t status;
  if (!strcmp(argv[1], "create")) {
    arm(ROCDDI_FAULT_CREATE_EFAULT);
    hsa_queue_t *queue = NULL;
    status = hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 4096,
                              0, &queue);
    if (queue)
      return 6;
  } else {
    uint32_t mask = 1;
    hsa_amd_queue_create_desc_t desc = {0};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
    desc.queue_size_bytes = 256 * 64;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
    desc.engine.compute.private_segment_size = 4096;
    desc.engine.compute.cu_mask = &mask;
    desc.engine.compute.cu_mask_count = 32;
    arm(ROCDDI_FAULT_CU_MASK_ROLLBACK);
    status = hsa_amd_queue_create(gpu, &desc, 1);
    if (desc.queue)
      return 7;
  }
  struct fault_stats got;
  stats(&got);
  printf("HSA %s status=0x%x creates=%u create_faults=%u mask_faults=%u "
         "destroy_faults=%u alloc=%u free=%u\n",
         argv[1], status, got.creates, got.create_faults, got.mask_faults,
         got.destroy_faults, got.allocations, got.frees);
  printf("HSA borrowed owners: signal=%u freed=%u scratch=%u freed=%u\n",
         got.signal_seen, got.signal_freed, got.scratch_seen,
         got.scratch_freed);
  if (status == HSA_STATUS_SUCCESS || got.creates != 1)
    return 8;
  if (!got.signal_seen || got.signal_freed || !got.scratch_seen ||
      got.scratch_freed || !got.signal_handle || !got.scratch_handle ||
      got.signal_handle == got.scratch_handle)
    return 11;
  if (!strcmp(argv[1], "create")) {
    if (got.create_faults != 1 || got.destroy_faults != 0)
      return 9;
  } else if (got.mask_faults != 1 || got.destroy_faults != 1)
    return 10;
  return 0;
}
