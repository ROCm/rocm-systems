#ifndef RCCL_TEST_HOST_RAS_MESSAGE_FAKES_H_
#define RCCL_TEST_HOST_RAS_MESSAGE_FAKES_H_

#include <functional>
#include "ras/ras_internal.h"

extern std::function<ncclResult_t(rasMsg**, size_t)> g_rasMessageAlloc;
ncclResult_t RasMessageTestAlloc(rasMsg** msg, size_t length);
extern std::function<void(rasMsg*)> g_rasMessageFree;
void RasMessageTestFree(rasMsg* msg);
extern std::function<void(rasConnection*, rasMsg*, size_t, bool)> g_rasMessageEnqueue;
void RasMessageTestEnqueue(rasConnection* conn, rasMsg* msg, size_t length, bool front);

void ResetRasMessageFakes();

#endif
