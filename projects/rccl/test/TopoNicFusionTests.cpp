/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for NIC fusion input validation in ncclTopoForceMerge()
// (src/graph/topo.cc), driven with a hand-built propsList (no real NICs, GPUs,
// or net plugin).
//
// The case pinned here is the NCCL_NET_FORCE_MERGE pattern that matches more
// physical devices than ncclNetVDeviceProps_t::devs[] can hold. parseStringList()
// caps the number of parsed patterns at NCCL_NET_MAX_DEVS_PER_NIC, but a single
// pattern written without ":port" matches every port of a multi-port NIC
// (matchPort() accepts any port against -1), so the number of matched devices is
// not bounded by the number of patterns. Without a bound inside the match loop,
// vProps.devs[vProps.ndevs++] runs off the end of the stack array.
//
// IMPORTANT -- on an ordinary build this test cannot tell the fix from its absence,
// and must not be deleted as vacuous on that basis: the pre-existing
// "vProps.ndevs != nUserIfs" check fails right after the overflowing writes, so the
// result is ncclInvalidUsage either way. It earns its keep under
// BUILD_ADDRESS_SANITIZER=ON, where the pre-fix code aborts on the store into devs[]
// and the fixed code returns cleanly.
//
// Debug-only target (rccl-UnitTestsFixturesDebug): ncclTopoForceMerge has hidden
// visibility in Release, so it is only linkable from the Debug fixtures binary.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "graph/topo.h"  // struct ncclTopoNetInfo, ncclTopoForceMerge()
#include "graph/xml.h"   // struct ncclXml, xmlAlloc(), xmlAddNode()
#include "nccl_net.h"    // ncclNetProperties_t, NCCL_NET_MAX_DEVS_PER_NIC

namespace RcclUnitTesting
{

namespace
{

// One multi-port NIC exposed as one propsList entry per port, all sharing the
// same device name. One more port than devs[] can hold, so a port-less pattern
// overruns the array by exactly one entry.
constexpr int kPhysDevs = NCCL_NET_MAX_DEVS_PER_NIC + 1;

// Any name works: it is both the fake device name and the pattern matched against it.
const char* kFakeNicName = "testnic0";

}  // namespace

// A single NCCL_NET_FORCE_MERGE pattern matching more devices than
// ncclNetVDeviceProps_t::devs[] holds must be rejected inside the match loop,
// before the write, rather than overflowing the stack array.
TEST(TopoNicFusionTests, ForceMerge_PatternMatchingMoreDevsThanArray_Rejected)
{
    // The vNIC-construction arguments (xml, physNetNodes, netInfo.makeVDevice)
    // are not exercised: the call must bail on the device count before it
    // reaches ncclTopoMakeVnic(). They are still built as real, valid objects so
    // the test itself does not rely on that to stay defined.
    struct ncclXml* xml = nullptr;
    ASSERT_EQ(xmlAlloc(&xml, kPhysDevs + 1), ncclSuccess);
    struct ncclXmlNode* root = nullptr;
    ASSERT_EQ(xmlAddNode(xml, nullptr, "system", &root), ncclSuccess);

    char                names[kPhysDevs][16];
    ncclNetProperties_t propsList[kPhysDevs];
    struct ncclXmlNode* physNetNodes[kPhysDevs];
    int                 placedDevs[kPhysDevs];

    memset(propsList, 0, sizeof(propsList));
    memset(placedDevs, 0, sizeof(placedDevs));
    for(int dev = 0; dev < kPhysDevs; dev++)
    {
        snprintf(names[dev], sizeof(names[dev]), "%s", kFakeNicName);
        propsList[dev].name = names[dev];
        propsList[dev].port = dev + 1;  // IB port numbers are 1-based
        ASSERT_EQ(xmlAddNode(xml, root, "net", &physNetNodes[dev]), ncclSuccess);
    }

    struct ncclTopoNetInfo netInfo;
    memset(&netInfo, 0, sizeof(netInfo));
    netInfo.maxDevsPerNic = NCCL_NET_MAX_DEVS_PER_NIC;
    netInfo.forceMerge    = kFakeNicName;

    EXPECT_EQ(ncclTopoForceMerge(xml, &netInfo, placedDevs, propsList, physNetNodes, kPhysDevs),
              ncclInvalidUsage);

    free(xml);
}

}  // namespace RcclUnitTesting
