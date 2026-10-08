// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Example consumer: uses only the public C API, with no mock control calls.
#include <amd_smi/amdsmi.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static void check(amdsmi_status_t status) {
  if (status != AMDSMI_STATUS_SUCCESS) {
    fprintf(stderr, "AMD SMI call failed: status=%d\n", status);
    exit(EXIT_FAILURE);
  }
}

int main(void) {
  check(amdsmi_init(AMDSMI_INIT_AMD_GPUS));
  uint32_t count = 0;
  check(amdsmi_get_socket_handles(&count, NULL));
  if (!count) {
    puts("No mock GPUs configured");
    check(amdsmi_shut_down());
    return EXIT_SUCCESS;
  }
  amdsmi_socket_handle socket;
  check(amdsmi_get_socket_handles(&count, &socket));
  check(amdsmi_get_processor_handles(socket, &count, NULL));
  amdsmi_processor_handle* processors = calloc(count, sizeof(*processors));
  if (!processors) return EXIT_FAILURE;
  check(amdsmi_get_processor_handles(socket, &count, processors));
  for (uint32_t index = 0; index < count; ++index) {
    int64_t temperature;
    amdsmi_engine_usage_t activity;
    amdsmi_power_info_t power;
    amdsmi_vram_usage_t memory;
    amdsmi_error_count_t ecc;
    check(amdsmi_get_temp_metric(processors[index], AMDSMI_TEMPERATURE_TYPE_EDGE,
                                 AMDSMI_TEMP_CURRENT, &temperature));
    check(amdsmi_get_gpu_activity(processors[index], &activity));
    check(amdsmi_get_power_info(processors[index], &power));
    check(amdsmi_get_gpu_vram_usage(processors[index], &memory));
    check(amdsmi_get_gpu_total_ecc_count(processors[index], &ecc));
    printf("gpu=%u temperature_c=%" PRId64 " utilization_percent=%u power_w=%" PRIu64
           " vram_used_mib=%u ecc_correctable=%" PRIu64 " ecc_uncorrectable=%" PRIu64 "\n",
           index, temperature, activity.gfx_activity, power.socket_power, memory.vram_used,
           ecc.correctable_count, ecc.uncorrectable_count);
  }
  free(processors);
  check(amdsmi_shut_down());
  return EXIT_SUCCESS;
}
