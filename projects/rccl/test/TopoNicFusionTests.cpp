/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for NIC fusion input validation in ncclTopoForceMerge()
// (src/graph/topo.cc), driven with a hand-built propsList (no real NICs, GPUs,
// or net plugin).
//
// Two ways a NCCL_NET_FORCE_MERGE group can ask for more devices than
// ncclNetVDeviceProps_t::devs[] holds:
//
//   1. More devices matched than patterns given. parseStringList() caps the number of
//      parsed patterns at NCCL_NET_MAX_DEVS_PER_NIC, but a pattern written without
//      ":port" matches every port of a multi-port NIC (matchPort() accepts any port
//      against -1), so the number of matched devices is not bounded by the number of
//      patterns and vProps.devs[vProps.ndevs++] runs off the end of the stack array.
//   2. More patterns given than the limit. parseStringList() stops at maxList and drops
//      the rest silently, so the request looks like a legal full-size one and gets
//      merged short of what was asked for instead of rejected.
//
// Case 2 fails on any build without the fix. Case 1 does not: the pre-existing
// "vProps.ndevs != nUserIfs" check fails right after the overflowing writes, so the
// result is ncclInvalidUsage either way, and only BUILD_ADDRESS_SANITIZER=ON tells the
// two apart (pre-fix aborts on the store into devs[], fixed returns cleanly). No CI job
// currently builds this binary with sanitizers, so case 1 pays off when the suite is run
// from an ASAN build (install.sh --address-sanitizer) -- do not drop it as vacuous on the
// basis of an ordinary build.
//
// Debug-only target (rccl-UnitTestsFixturesDebug): ncclTopoForceMerge has hidden
// visibility in Release, so it is only linkable from the Debug fixtures binary.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "graph/topo.h"  // struct ncclTopoNetInfo, ncclTopoForceMerge()
#include "graph/xml.h"   // struct ncclXml, xmlAlloc(), xmlAddNode()
#include "nccl_net.h"    // ncclNetProperties_t, NCCL_NET_MAX_DEVS_PER_NIC

namespace RcclUnitTesting
{

namespace
{

// One device past what a vNIC can hold, which is all these tests need.
constexpr int kFakeDevs = NCCL_NET_MAX_DEVS_PER_NIC + 1;

// The arrays ncclTopoForceMerge() works on, for a made-up set of NICs. Frees the XML tree
// in the destructor, so a failing ASSERT_* in the middle of a test does not leak it.
struct FakeNetTopo
{
    struct ncclXml*     xml = nullptr;
    char                names[kFakeDevs][16];
    ncclNetProperties_t propsList[kFakeDevs];
    struct ncclXmlNode* physNetNodes[kFakeDevs];
    int                 placedDevs[kFakeDevs];

    ~FakeNetTopo() { free(xml); }

    // onePerPort: all entries are ports of a single NIC, sharing one device name.
    // Otherwise each entry is a separate single-port NIC with a name of its own.
    void Build(bool onePerPort)
    {
        ASSERT_EQ(xmlAlloc(&xml, kFakeDevs + 1), ncclSuccess);
        struct ncclXmlNode* root = nullptr;
        ASSERT_EQ(xmlAddNode(xml, nullptr, "system", &root), ncclSuccess);

        memset(propsList, 0, sizeof(propsList));
        memset(placedDevs, 0, sizeof(placedDevs));
        for(int dev = 0; dev < kFakeDevs; dev++)
        {
            snprintf(names[dev], sizeof(names[dev]), "testnic%d", onePerPort ? 0 : dev);
            propsList[dev].name = names[dev];
            propsList[dev].port = onePerPort ? dev + 1 : 1;  // IB port numbers are 1-based
            ASSERT_EQ(xmlAddNode(xml, root, "net", &physNetNodes[dev]), ncclSuccess);
        }
    }
};

int makeVDeviceCalls = 0;

// Both cases must be rejected on the device count before any vNIC is built, so this only
// exists to make a regression fail loudly instead of calling through a null pointer.
ncclResult_t CountingMakeVDevice(int* d, ncclNetVDeviceProps_t* /*props*/)
{
    makeVDeviceCalls++;
    *d = 0;
    return ncclSuccess;
}

}  // namespace

// Case 1: a single pattern that matches more devices than ncclNetVDeviceProps_t::devs[] holds
// must be rejected inside the match loop, before the write, rather than overflowing the array.
TEST(TopoNicFusionTests, ForceMerge_PatternMatchingMoreDevsThanArray_Rejected)
{
    FakeNetTopo topo;
    ASSERT_NO_FATAL_FAILURE(topo.Build(/*onePerPort=*/true));

    struct ncclTopoNetInfo netInfo;
    memset(&netInfo, 0, sizeof(netInfo));
    netInfo.maxDevsPerNic = NCCL_NET_MAX_DEVS_PER_NIC;
    netInfo.makeVDevice   = CountingMakeVDevice;
    netInfo.forceMerge    = topo.names[0];  // no ":port", so it matches every port

    makeVDeviceCalls = 0;
    EXPECT_EQ(ncclTopoForceMerge(topo.xml, &netInfo, topo.placedDevs, topo.propsList, topo.physNetNodes, kFakeDevs),
              ncclInvalidUsage);
    EXPECT_EQ(makeVDeviceCalls, 0) << "An over-limit group must be rejected before the merge";
}

// Case 2: a group listing more NICs than a vNIC can hold must be rejected, not truncated to
// the limit and merged.
TEST(TopoNicFusionTests, ForceMerge_GroupListingMoreDevsThanLimit_Rejected)
{
    FakeNetTopo topo;
    ASSERT_NO_FATAL_FAILURE(topo.Build(/*onePerPort=*/false));

    std::string forceMerge;
    for(int dev = 0; dev < kFakeDevs; dev++)
    {
        if(dev > 0) forceMerge += ',';
        forceMerge += topo.names[dev];
    }

    struct ncclTopoNetInfo netInfo;
    memset(&netInfo, 0, sizeof(netInfo));
    netInfo.maxDevsPerNic = NCCL_NET_MAX_DEVS_PER_NIC;
    netInfo.makeVDevice   = CountingMakeVDevice;
    netInfo.forceMerge    = forceMerge.c_str();

    makeVDeviceCalls = 0;
    EXPECT_EQ(ncclTopoForceMerge(topo.xml, &netInfo, topo.placedDevs, topo.propsList, topo.physNetNodes, kFakeDevs),
              ncclInvalidUsage);
    EXPECT_EQ(makeVDeviceCalls, 0) << "A group of " << kFakeDevs << " NICs must not be merged as "
                                  << NCCL_NET_MAX_DEVS_PER_NIC;
}

}  // namespace RcclUnitTesting
