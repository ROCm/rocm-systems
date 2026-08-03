#include "param.h"
#include "nccl.h"
#include <stdlib.h>

// Getter defined in graph/paths.cc (not compiled here); macro routes to real ncclLoadParam.
NCCL_PARAM(PxnC2c, "PXN_C2C", 0);

// Leaf env/os plumbing: behaviorally-faithful stubs (env-var driven) to avoid
// pulling plugin/env.cc + os avalanche into the host UT.
const char* ncclEnvPluginGetEnv(const char* name) { return getenv(name); }
ncclResult_t ncclInitEnv(void) { return ncclSuccess; }
void ncclOsSetEnv(const char* name, const char* value) { setenv(name, value, 1); }
