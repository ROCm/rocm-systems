#ifndef RCCL_TEST_HOST_RAS_NET_FAKES_H_
#define RCCL_TEST_HOST_RAS_NET_FAKES_H_

#include <functional>
#include "ras/ras_internal.h"

extern std::function<rasConnection*(const ncclSocketAddress*)> g_rasConnFind;
rasConnection* rasConnFind(const ncclSocketAddress* addr);
extern std::function<ncclResult_t(const ncclSocketAddress*, rasConnection**)> g_rasConnCreate;
ncclResult_t rasConnCreate(const ncclSocketAddress* addr, rasConnection** conn);
extern std::function<void(const ncclSocketAddress*)> g_rasConnDisconnect;
void rasConnDisconnect(const ncclSocketAddress* addr);
extern std::function<ncclResult_t(rasLink*, const rasConnection*)> g_rasLinkAddFallback;
ncclResult_t rasLinkAddFallback(rasLink* link, const rasConnection* conn);

void ResetRasNetFakes();

#endif
