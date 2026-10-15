#ifndef RCCL_TEST_HOST_RAS_REGISTRY_TEST_SUPPORT_H_
#define RCCL_TEST_HOST_RAS_REGISTRY_TEST_SUPPORT_H_

#include <cstdlib>
#include <mutex>
#include <vector>

#include <gtest/gtest.h>
#include "ras/ras_internal.h"

namespace ras_test {

inline void InstallNcclComms(const std::vector<ncclComm*>& comms) {
  std::lock_guard<std::mutex> lock(ncclCommsMutex);
  auto** replacement = static_cast<ncclComm**>(std::calloc(comms.empty() ? 1 : comms.size(), sizeof(ncclComm*)));
  ASSERT_NE(nullptr, replacement);
  std::free(ncclComms);
  ncclComms = replacement;
  nNcclComms = static_cast<int>(comms.size());
  ncclCommsSorted = false;
  for (size_t index = 0; index < comms.size(); ++index) ncclComms[index] = comms[index];
}

inline void ResetNcclComms() {
  std::lock_guard<std::mutex> lock(ncclCommsMutex);
  std::free(ncclComms);
  ncclComms = nullptr;
  nNcclComms = 0;
  ncclCommsSorted = false;
}

}

#endif
