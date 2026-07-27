// Compiles NullParentTests.cpp with real RCCL topo types via hipcc.
#include "topo.h"

ncclResult_t test_ncclTopoSetPaths(struct ncclTopoNode* baseNode,
                                   struct ncclTopoSystem* system);
int test_gpuPciBw(struct ncclTopoNode* gpu);

#include "NullParentTests.cpp"
