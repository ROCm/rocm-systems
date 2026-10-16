// SPDX-License-Identifier: MIT
// Create and destroy one empty GPU queue using only public HSA APIs.
#include <hsa/hsa.h>
#include <cstdio>
#include <cstdlib>

static void check(hsa_status_t status, const char* call) {
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: HSA status 0x%x\n", call, unsigned(status));
    std::exit(1);
  }
}
#define CHECK(call) check((call), #call)

static hsa_status_t find_gpu(hsa_agent_t agent, void* data) {
  hsa_device_type_t device;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device);
  if (status == HSA_STATUS_SUCCESS && device == HSA_DEVICE_TYPE_GPU)
    *static_cast<hsa_agent_t*>(data) = agent;
  return status;
}

int main() {
  CHECK(hsa_init());
  hsa_agent_t gpu{};
  CHECK(hsa_iterate_agents(find_gpu, &gpu));
  if (gpu.handle == 0) {
    CHECK(hsa_shut_down());
    std::fprintf(stderr, "No GPU agent available\n");
    return 77;
  }

  // Use the agent's advertised queue size and type, with unspecified
  // private/group segment hints as documented by hsa_queue_create.
  uint32_t size;
  hsa_queue_type32_t type;
  CHECK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &size));
  CHECK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_TYPE, &type));
  hsa_queue_t* queue = nullptr;
  CHECK(hsa_queue_create(gpu, size, type, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue));
  // No packets were submitted, so there is no application work to wait for.
  CHECK(hsa_queue_destroy(queue));
  CHECK(hsa_shut_down());
  std::puts("All HSA calls returned SUCCESS");
}
