#include "fakes/ras_net_fakes.h"

#include "fakes/signature-drift.h"

static rasConnection* DefaultRasConnFind(const ncclSocketAddress*) { return nullptr; }
std::function<rasConnection*(const ncclSocketAddress*)> g_rasConnFind = DefaultRasConnFind;
ASSERT_HOOK_MATCHES_PROD(g_rasConnFind, rasConnFind);
rasConnection* rasConnFind(const ncclSocketAddress* addr) { return g_rasConnFind(addr); }

static ncclResult_t DefaultRasConnCreate(const ncclSocketAddress*, rasConnection** conn) {
  *conn = nullptr;
  return ncclInternalError;
}
std::function<ncclResult_t(const ncclSocketAddress*, rasConnection**)> g_rasConnCreate = DefaultRasConnCreate;
ASSERT_HOOK_MATCHES_PROD(g_rasConnCreate, rasConnCreate);
ncclResult_t rasConnCreate(const ncclSocketAddress* addr, rasConnection** conn) { return g_rasConnCreate(addr, conn); }

static void DefaultRasConnDisconnect(const ncclSocketAddress*) {}
std::function<void(const ncclSocketAddress*)> g_rasConnDisconnect = DefaultRasConnDisconnect;
ASSERT_HOOK_MATCHES_PROD(g_rasConnDisconnect, rasConnDisconnect);
void rasConnDisconnect(const ncclSocketAddress* addr) { g_rasConnDisconnect(addr); }

static ncclResult_t DefaultRasLinkAddFallback(rasLink*, const rasConnection*) { return ncclSuccess; }
std::function<ncclResult_t(rasLink*, const rasConnection*)> g_rasLinkAddFallback = DefaultRasLinkAddFallback;
ASSERT_HOOK_MATCHES_PROD(g_rasLinkAddFallback, rasLinkAddFallback);
ncclResult_t rasLinkAddFallback(rasLink* link, const rasConnection* conn) { return g_rasLinkAddFallback(link, conn); }

void ResetRasNetFakes() {
  g_rasConnFind = DefaultRasConnFind;
  g_rasConnCreate = DefaultRasConnCreate;
  g_rasConnDisconnect = DefaultRasConnDisconnect;
  g_rasLinkAddFallback = DefaultRasLinkAddFallback;
}
