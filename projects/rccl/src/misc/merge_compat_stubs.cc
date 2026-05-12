// RCCL_COMPAT_STUB: aggregated stubs for NCCL upstream symbols not yet
// implemented in RCCL. Each function/object below is tagged with
// RCCL_COMPAT_STUB on its closing brace for traceability.

#include "scheduler.h"
#include "socket.h"
#include "param.h"
#include "env.h"
#include "plugin/plugin.h"
#include "dev_runtime.h"

#include <cstdint>
#include <cstdlib>

#define NOWARN(cmd, subsys) do { \
  (void)(subsys); \
  (void)(cmd); \
} while (0) // RCCL_COMPAT_STUB

ncclResult_t ncclMakeSymmetricTaskList(
    struct ncclComm*,
    struct ncclTaskColl* task,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>*,
    struct ncclTaskColl** remainTasksHead
  ) {
  if (remainTasksHead) *remainTasksHead = task;
  return ncclSuccess;
} // RCCL_COMPAT_STUB

ncclResult_t ncclSymmetricTaskScheduler(
    struct ncclComm*,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>*,
    struct ncclKernelPlan*
  ) {
  return ncclInternalError;
} // RCCL_COMPAT_STUB

ncclResult_t ncclScheduleBcastTasksToPlan(
    struct ncclComm*, struct ncclKernelPlan*, struct ncclKernelPlanBudget*
  ) {
  return ncclInternalError;
} // RCCL_COMPAT_STUB

ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* /*ops*/, int /*numOps*/) {
  return ncclInternalError;
} // RCCL_COMPAT_STUB

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized,
                      int64_t* cache, int8_t* noCache);
void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static int8_t noCache = -1;
  ncclLoadParam(env, deftVal, uninitialized, cache, &noCache);
} // RCCL_COMPAT_STUB

void* ncclOpenEnvPluginLib(const char* /*name*/) {
  return nullptr;
} // RCCL_COMPAT_STUB

ncclResult_t ncclInitEnv(void) {
  if (ncclEnvPluginInitialized()) return ncclSuccess;
  return ncclEnvPluginInit();
} // RCCL_COMPAT_STUB

void ncclDevCommCopyLsaData(void* /*dstRankPtr*/, void const* /*srcRankPtr*/) {
} // RCCL_COMPAT_STUB

NCCL_PARAM(PollTimeOut, "SOCKET_POLL_TIMEOUT", 100); // RCCL_COMPAT_STUB

const char* ncclVersionToString(int version, char* buf, size_t bufLen) {
  int major = version / 10000;
  int minor = (version / 100) % 100;
  int patch = version % 100;
  snprintf(buf, bufLen, "%d.%d.%d", major, minor, patch);
  return buf;
} // RCCL_COMPAT_STUB
