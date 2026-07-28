// Compiles NullParentTests.cpp with real RCCL topo types via hipcc.
#include "topo.h"
#include "xml.h"

ncclResult_t test_ncclTopoSetPaths(struct ncclTopoNode* baseNode,
                                   struct ncclTopoSystem* system);
int test_gpuPciBw(struct ncclTopoNode* gpu);
ncclResult_t test_ncclTopoGetChannelFromXml(struct ncclXmlNode* xmlChannel, int c,
                                            struct ncclTopoSystem* system,
                                            struct ncclTopoGraph* graph);

#include "NullParentTests.cpp"
