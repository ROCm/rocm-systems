#ifndef RCCL_TEST_HOST_RAS_CLIENT_SUPPORT_FAKES_H_
#define RCCL_TEST_HOST_RAS_CLIENT_SUPPORT_FAKES_H_

#include <functional>
#include "ras/ras_internal.h"

extern std::function<const char*(const ncclSocketAddress*, char*, size_t)> g_ncclSocketToHost;
const char* ncclSocketToHost(const ncclSocketAddress* addr, char* buf, size_t size);
extern std::function<const char*(uint64_t, uint64_t, char*, size_t)> g_rasGpuDevsToString;
const char* rasGpuDevsToString(uint64_t cudaDevs, uint64_t nvmlDevs, char* buf, size_t size);
extern std::function<void(rasEventGroup, const rasEventNotification*)> g_rasClientsNotifyEvent;
void rasClientsNotifyEvent(rasEventGroup group, const rasEventNotification* event);

void ResetRasClientSupportFakes();

#endif
