// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_pc_sampling.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static hsa_agent_t gpu;

static hsa_status_t visit_agent(hsa_agent_t agent, void *data) {
  (void)data;
  hsa_device_type_t type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU &&
      gpu.handle == 0) {
    gpu = agent;
  }
  return status;
}

static hsa_status_t visit_configuration(
    const hsa_ven_amd_pcs_configuration_t *configuration, void *data) {
  (void)configuration;
  (void)data;
  return HSA_STATUS_SUCCESS;
}

static void ready(void *client_data, size_t data_size, size_t lost,
                  hsa_ven_amd_pcs_data_copy_callback_t copy, void *copy_data) {
  (void)client_data;
  (void)data_size;
  (void)lost;
  (void)copy;
  (void)copy_data;
}

int main(void) {
  if (hsa_init() != HSA_STATUS_SUCCESS) return 1;
  if (hsa_iterate_agents(visit_agent, NULL) != HSA_STATUS_SUCCESS ||
      gpu.handle == 0) {
    fprintf(stderr, "no GPU agent\n");
    hsa_shut_down();
    return 2;
  }

  uint16_t system_minor = 0, agent_minor = 0;
  bool system_supported = true, agent_supported = true;
  hsa_status_t system_status = hsa_system_major_extension_supported(
      HSA_EXTENSION_AMD_PC_SAMPLING, 1, &system_minor, &system_supported);
  hsa_status_t agent_status = hsa_agent_major_extension_supported(
      HSA_EXTENSION_AMD_PC_SAMPLING, gpu, 1, &agent_minor, &agent_supported);
  hsa_status_t iterate_status = hsa_ven_amd_pcs_iterate_configuration(
      gpu, visit_configuration, NULL);
  hsa_ven_amd_pcs_t session = {.handle = UINT64_C(0x12345678)};
  hsa_status_t create_status = hsa_ven_amd_pcs_create(
      gpu, HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1,
      HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS, 512, 0, 4096,
      ready, NULL, &session);
  hsa_status_t shutdown_status = hsa_shut_down();

  if (system_status != HSA_STATUS_SUCCESS ||
      agent_status != HSA_STATUS_SUCCESS ||
      system_supported || agent_supported ||
      iterate_status != (hsa_status_t)HSA_STATUS_ERROR_NOT_SUPPORTED ||
      create_status != (hsa_status_t)HSA_STATUS_ERROR_NOT_SUPPORTED ||
      session.handle != UINT64_C(0x12345678) ||
      shutdown_status != HSA_STATUS_SUCCESS) {
    fprintf(stderr,
            "PC sampling capability mismatch: system=%d agent=%d "
            "iterate=0x%x create=0x%x session=0x%llx shutdown=0x%x\n",
            system_supported, agent_supported, iterate_status, create_status,
            (unsigned long long)session.handle, shutdown_status);
    return 3;
  }
  puts("PC sampling remains unadvertised and session creation is unsupported");
  return 0;
}
