#ifndef RCCL_TEST_HOST_RAS_PARAM_FAKES_H_
#define RCCL_TEST_HOST_RAS_PARAM_FAKES_H_

#include <cstdint>
#include <functional>

extern std::function<int64_t(int64_t)> g_rasTimeoutFactorNs;

void ResetRasParamFakes();

#endif
