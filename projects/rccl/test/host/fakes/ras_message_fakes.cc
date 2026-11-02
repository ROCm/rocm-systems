#include "fakes/ras_message_fakes.h"

#include <algorithm>
#include <cstdlib>
#include "fakes/signature-drift.h"

static ncclResult_t DefaultRasMessageTestAlloc(rasMsg** msg, size_t length) {
  const size_t size = offsetof(rasMsgMeta, msg) + std::max(length, sizeof(rasMsg));
  auto* storage = static_cast<rasMsgMeta*>(calloc(1, size));
  if (storage == nullptr) return ncclSystemError;
  *msg = &storage->msg;
  return ncclSuccess;
}
std::function<ncclResult_t(rasMsg**, size_t)> g_rasMessageAlloc = DefaultRasMessageTestAlloc;
ASSERT_HOOK_MATCHES_PROD(g_rasMessageAlloc, rasMsgAlloc);
ncclResult_t RasMessageTestAlloc(rasMsg** msg, size_t length) { return g_rasMessageAlloc(msg, length); }

static void DefaultRasMessageTestFree(rasMsg* msg) {
  if (msg) free(reinterpret_cast<char*>(msg) - offsetof(rasMsgMeta, msg));
}
std::function<void(rasMsg*)> g_rasMessageFree = DefaultRasMessageTestFree;
ASSERT_HOOK_MATCHES_PROD(g_rasMessageFree, rasMsgFree);
void RasMessageTestFree(rasMsg* msg) { g_rasMessageFree(msg); }

static void DefaultRasMessageTestEnqueue(rasConnection*, rasMsg* msg, size_t, bool) { RasMessageTestFree(msg); }
std::function<void(rasConnection*, rasMsg*, size_t, bool)> g_rasMessageEnqueue = DefaultRasMessageTestEnqueue;
ASSERT_HOOK_MATCHES_PROD(g_rasMessageEnqueue, rasConnEnqueueMsg);
void RasMessageTestEnqueue(rasConnection* conn, rasMsg* msg, size_t length, bool front) {
  g_rasMessageEnqueue(conn, msg, length, front);
}

void ResetRasMessageFakes() {
  g_rasMessageAlloc = DefaultRasMessageTestAlloc;
  g_rasMessageFree = DefaultRasMessageTestFree;
  g_rasMessageEnqueue = DefaultRasMessageTestEnqueue;
}
