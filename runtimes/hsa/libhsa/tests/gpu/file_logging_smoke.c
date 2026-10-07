// SPDX-License-Identifier: MIT

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static hsa_agent_t gpu;

static hsa_status_t find_gpu(hsa_agent_t agent, void *unused) {
  (void)unused;
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU)
    gpu = agent;
  return status;
}

static int exercise_queue(void) {
  hsa_queue_t *queue = NULL;
  if (hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0,
                       &queue) != HSA_STATUS_SUCCESS ||
      !queue)
    return 1;
  return hsa_queue_destroy(queue) != HSA_STATUS_SUCCESS;
}

int main(void) {
  if (hsa_init() != HSA_STATUS_SUCCESS) return 1;
  if (hsa_iterate_agents(find_gpu, NULL) != HSA_STATUS_SUCCESS || !gpu.handle)
    return 2;

  FILE *stream = tmpfile();
  if (!stream) return 3;
  uint8_t flags[8] = {1u << HSA_AMD_LOG_FLAG_INFO};
  if (hsa_amd_enable_logging(flags, stream) != HSA_STATUS_SUCCESS) return 4;
  if (exercise_queue() != 0) return 5;
  flags[0] = 0;
  if (hsa_amd_enable_logging(flags, NULL) != HSA_STATUS_SUCCESS) return 6;
  rewind(stream);
  char line[512];
  int found = 0;
  while (fgets(line, sizeof(line), stream)) {
    if (strstr(line, "created AQL queue")) found = 1;
  }
  if (!found || ferror(stream)) return 7;
  if (fclose(stream) != 0) return 8;

  flags[0] = 1u << HSA_AMD_LOG_FLAG_INFO;
  if (hsa_amd_enable_logging(flags, NULL) != HSA_STATUS_SUCCESS) return 9;
  if (exercise_queue() != 0) return 10;
  flags[0] = 0;
  if (hsa_amd_enable_logging(flags, NULL) != HSA_STATUS_SUCCESS) return 11;
  if (hsa_shut_down() != HSA_STATUS_SUCCESS) return 12;
  puts("HSA logging boundary smoke passed");
  return 0;
}
