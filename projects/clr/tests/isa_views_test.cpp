/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<hsa_agent_t> gpus;

hsa_status_t findGpu(hsa_agent_t agent, void*) {
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (type == HSA_DEVICE_TYPE_GPU) {
    gpus.push_back(agent);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_agent_t findGpuForHipDevice(const hipDeviceProp_t& properties) {
  for (hsa_agent_t gpu : gpus) {
    uint32_t domain = 0;
    uint32_t bdf = 0;
    if (hsa_agent_get_info(gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
                           &domain) != HSA_STATUS_SUCCESS ||
        hsa_agent_get_info(gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdf) !=
            HSA_STATUS_SUCCESS) {
      continue;
    }
    const uint32_t bus = (bdf >> 8) & 0xff;
    const uint32_t device = (bdf >> 3) & 0x1f;
    if (domain == static_cast<uint32_t>(properties.pciDomainID) &&
        bus == static_cast<uint32_t>(properties.pciBusID) &&
        device == static_cast<uint32_t>(properties.pciDeviceID)) {
      return gpu;
    }
  }
  return {};
}

bool isaName(hsa_isa_t isa, std::string& name) {
  uint32_t length = 0;
  if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &length) != HSA_STATUS_SUCCESS) {
    return false;
  }
  std::vector<char> buffer(length + 1, '\0');
  if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, buffer.data()) != HSA_STATUS_SUCCESS) {
    return false;
  }
  name = buffer.data();
  return true;
}

bool containsTarget(const std::string& isa, const char* target) {
  if (target == nullptr) {
    return true;
  }
  const std::string needle = std::string("--") + target;
  const size_t position = isa.find(needle);
  if (position == std::string::npos) {
    return false;
  }
  const size_t end = position + needle.size();
  return end == isa.size() || isa[end] == ':';
}

bool checkBlit() {
  constexpr size_t size = 4096;
  constexpr unsigned char pattern = 0x5a;
  void* deviceBuffer = nullptr;
  if (hipMalloc(&deviceBuffer, size) != hipSuccess) {
    return false;
  }

  std::vector<unsigned char> source(size, pattern);
  std::vector<unsigned char> destination(size, 0);
  hipError_t status = hipMemcpy(deviceBuffer, source.data(), size, hipMemcpyHostToDevice);
  if (status == hipSuccess) {
    status = hipMemcpy(destination.data(), deviceBuffer, size, hipMemcpyDeviceToHost);
  }
  hipError_t freeStatus = hipFree(deviceBuffer);
  if (status != hipSuccess || freeStatus != hipSuccess) {
    return false;
  }
  for (unsigned char value : destination) {
    if (value != pattern) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const bool expectQueryFailure = argc == 2 && std::strcmp(argv[1], "--expect-query-failure") == 0;
  const char* expectedPresented = argc >= 2 && !expectQueryFailure ? argv[1] : nullptr;
  const char* expectedExecution = argc >= 3 ? argv[2] : nullptr;

  // Check for physical hardware before HIP initialization so that a machine
  // without a GPU can skip without hiding a CLR execution-ISA query failure.
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    return 1;
  }
  hsa_status_t hsaStatus = hsa_iterate_agents(findGpu, nullptr);
  if (hsaStatus != HSA_STATUS_SUCCESS || gpus.empty()) {
    hsa_shut_down();
    return 77;
  }

  if (expectQueryFailure) {
    const hsa_agent_t gpu = gpus.front();
    hsa_isa_t executionIsa = {};
    hsaStatus = hsa_agent_get_info(
        gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_EXECUTION_ISA), &executionIsa);
    if (hsaStatus == HSA_STATUS_SUCCESS) {
      hsa_shut_down();
      return 1;
    }
  }

  int deviceCount = 0;
  hipError_t hipStatus = hipGetDeviceCount(&deviceCount);
  if (expectQueryFailure) {
    std::printf(
        "execution_isa_query_status=%d\nhip_status=%d\n"
        "device_count=%d\n",
        static_cast<int>(hsaStatus), static_cast<int>(hipStatus), deviceCount);
    hsa_shut_down();
    return hipStatus == hipErrorNoDevice && deviceCount == 0 ? 0 : 1;
  }
  if (hipStatus != hipSuccess || deviceCount == 0) {
    std::fprintf(stderr, "HIP device initialization failed: %d\n", static_cast<int>(hipStatus));
    hsa_shut_down();
    return 1;
  }

  hipDeviceProp_t properties = {};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
    hsa_shut_down();
    return 1;
  }

  const hsa_agent_t gpu = findGpuForHipDevice(properties);
  if (gpu.handle == 0) {
    hsa_shut_down();
    return 1;
  }

  hsa_isa_t presentedIsa = {};
  hsa_isa_t executionIsa = {};
  std::string presentedName;
  std::string executionName;
  if (hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &presentedIsa) != HSA_STATUS_SUCCESS ||
      hsa_agent_get_info(gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_EXECUTION_ISA),
                         &executionIsa) != HSA_STATUS_SUCCESS ||
      !isaName(presentedIsa, presentedName) || !isaName(executionIsa, executionName)) {
    hsa_shut_down();
    return 1;
  }

  const bool native = expectedPresented == nullptr;
  const char* presentedTarget = native ? properties.gcnArchName : expectedPresented;
  const std::string gcnArchIsa = std::string("amdgcn-amd-amdhsa--") + properties.gcnArchName;
  uint32_t presentedWavefrontSize = 0;
  if (hsa_agent_get_info(gpu, HSA_AGENT_INFO_WAVEFRONT_SIZE, &presentedWavefrontSize) !=
      HSA_STATUS_SUCCESS) {
    hsa_shut_down();
    return 1;
  }
  bool passed = containsTarget(presentedName, presentedTarget) &&
                containsTarget(gcnArchIsa, presentedTarget) &&
                containsTarget(executionName, expectedExecution) &&
                properties.warpSize == static_cast<int>(presentedWavefrontSize) &&
                (!native || presentedName == executionName) && checkBlit();

  std::printf(
      "presented_isa=%s\nexecution_isa=%s\ngcn_arch=%s\n"
      "warp_size=%d\nmax_threads_per_multiprocessor=%d\n",
      presentedName.c_str(), executionName.c_str(), properties.gcnArchName, properties.warpSize,
      properties.maxThreadsPerMultiProcessor);
  hsa_shut_down();
  return passed ? 0 : 1;
}
