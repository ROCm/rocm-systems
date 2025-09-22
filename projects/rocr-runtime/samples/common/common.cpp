#include "common.hpp"

void ErrorCheck(hsa_status_t hsa_error_code) {
  if (hsa_error_code != HSA_STATUS_SUCCESS) {
    std::cerr << "HSA reported error!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

hsa_status_t FindGpuDevice(hsa_agent_t agent, void *data) {
  if (data == NULL) {
     return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code = hsa_agent_get_info(
    agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type
  );
  if (hsa_error_code != HSA_STATUS_SUCCESS) {
    return hsa_error_code;
  }

  if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {
    *((hsa_agent_t*)data) = agent;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t FindHostRegion(hsa_region_t region, void *data) {
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  bool is_host_region = false;
  hsa_status_t hsa_error_code = hsa_region_get_info(
      region, (hsa_region_info_t)HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
      &is_host_region);
  if (hsa_error_code != HSA_STATUS_SUCCESS) {
    return hsa_error_code;
  }

  if (is_host_region) {
    *((hsa_region_t*)data) = region;
  }

  return HSA_STATUS_SUCCESS;
}
