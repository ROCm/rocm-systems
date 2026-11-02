#include "fakes/ras_client_support_fakes.h"

#include <cstdio>
#include "fakes/signature-drift.h"

static const char* DefaultNcclSocketToHost(const ncclSocketAddress*, char* buf, size_t size) {
  snprintf(buf, size, "<host>");
  return buf;
}
std::function<const char*(const ncclSocketAddress*, char*, size_t)> g_ncclSocketToHost = DefaultNcclSocketToHost;
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketToHost, ncclSocketToHost);
const char* ncclSocketToHost(const ncclSocketAddress* addr, char* buf, size_t size) { return g_ncclSocketToHost(addr, buf, size); }

static const char* DefaultRasGpuDevsToString(uint64_t, uint64_t, char* buf, size_t size) {
  snprintf(buf, size, "<devs>");
  return buf;
}
std::function<const char*(uint64_t, uint64_t, char*, size_t)> g_rasGpuDevsToString = DefaultRasGpuDevsToString;
ASSERT_HOOK_MATCHES_PROD(g_rasGpuDevsToString, rasGpuDevsToString);
const char* rasGpuDevsToString(uint64_t cudaDevs, uint64_t nvmlDevs, char* buf, size_t size) {
  return g_rasGpuDevsToString(cudaDevs, nvmlDevs, buf, size);
}

static void DefaultRasClientsNotifyEvent(rasEventGroup, const rasEventNotification*) {}
std::function<void(rasEventGroup, const rasEventNotification*)> g_rasClientsNotifyEvent = DefaultRasClientsNotifyEvent;
ASSERT_HOOK_MATCHES_PROD(g_rasClientsNotifyEvent, rasClientsNotifyEvent);
void rasClientsNotifyEvent(rasEventGroup group, const rasEventNotification* event) {
  g_rasClientsNotifyEvent(group, event);
}

void ResetRasClientSupportFakes() {
  g_ncclSocketToHost = DefaultNcclSocketToHost;
  g_rasGpuDevsToString = DefaultRasGpuDevsToString;
  g_rasClientsNotifyEvent = DefaultRasClientsNotifyEvent;
}
