/*
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include <math.h>
#include <limits.h>

#include "hsakmt/linux/kfd_ioctl.h"
#include "KFDRASTest.hpp"
#include "PM4Queue.hpp"

#define AMDGPU_DEBUGFS_NODES "/sys/kernel/debug/dri/"
#define RAS_CONTROL "ras/ras_ctrl"
#define DRM_RENDER_NUMBER 64

void KFDRASTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    char path[256], name[128], tmp[128];
    int renderNode, minor, i;
    FILE *pDriMinor, *pDriPrimary;
    uint32_t rasFeatures = 0;
    HsaEventDescriptor eventDesc;

    m_pRasEvent = NULL;
    m_setupStatus = false;

    m_defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

    renderNode = KFDBaseComponentTest::FindDRMRenderNode(m_defaultGPUNode);
    if (renderNode < 0) {
        LOG() << "Skipping test: Could not find render node for default GPU." << std::endl;
        throw;
    }

    amdgpu_query_info(m_RenderNodes[renderNode].device_handle,
                AMDGPU_INFO_RAS_ENABLED_FEATURES,
                sizeof(uint32_t), &rasFeatures);
    if (!(rasFeatures &
            (AMDGPU_INFO_RAS_ENABLED_SDMA |
             AMDGPU_INFO_RAS_ENABLED_UMC |
             AMDGPU_INFO_RAS_ENABLED_GFX))) {
        LOG() << "Skipping test: GPU doesn't support RAS features!" << std::endl;
        throw;
    }

    minor = renderNode + 128;

    snprintf(path, sizeof(path), "%s%d/%s", AMDGPU_DEBUGFS_NODES, minor, "name");
    pDriMinor = fopen(path, "r");
    if (!pDriMinor) {
        LOG() << "Skipping test: DRM render debugfs node requires root access!" << std::endl;
        throw;
    }

    memset(name, 0, sizeof(name));
    fread(name, sizeof(name), 1, pDriMinor);

    fclose(pDriMinor);

    for (i = 0; i < DRM_RENDER_NUMBER; i++) {
        snprintf(path, sizeof(path), "%s%d/%s", AMDGPU_DEBUGFS_NODES, i, "name");
        pDriPrimary = fopen(path, "r");
        if (!pDriPrimary)
            continue;
        memset(tmp, 0, sizeof(tmp));
        fread(tmp, sizeof(tmp), 1, pDriPrimary);
        if (!strcmp(name, tmp)) {
            fclose(pDriPrimary);
            break;
        }
        fclose(pDriPrimary);
    }

    if (i == DRM_RENDER_NUMBER) {
        LOG() << "Skipping test: Could not find the debugfs node!" << std::endl;
        throw;
    }

    snprintf(path, sizeof(path), "%s%d/%s", AMDGPU_DEBUGFS_NODES, i, RAS_CONTROL);
    m_pFile = fopen(path, "w");
    if (!m_pFile) {
        LOG() << "Skipping test: RAS error injection requires root access!" << std::endl;
        throw;
    }

    eventDesc.EventType = HSA_EVENTTYPE_MEMORY;
    eventDesc.NodeId = m_defaultGPUNode;
    eventDesc.SyncVar.SyncVar.UserData = NULL;
    eventDesc.SyncVar.SyncVarSize = 0;

    ASSERT_SUCCESS(hsaKmtCreateEvent(&eventDesc, true, false, &m_pRasEvent));

    m_setupStatus = true;

    ROUTINE_END
}

void KFDRASTest::TearDown() {
    ROUTINE_START

    if (m_pRasEvent != NULL) {
        EXPECT_SUCCESS(hsaKmtDestroyEvent(m_pRasEvent));
    }

    fclose(m_pFile);

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

TEST_F(KFDRASTest, DISABLED_BasicTest) {
    TEST_START(TESTPROFILE_RUNALL);

    if (!m_setupStatus) {
        return;
    }

    // write an uncorrectable error injection at address 0 as value 0
    fwrite("inject umc ue 0 0", sizeof(char), 17, m_pFile);
    fflush(m_pFile);

    EXPECT_SUCCESS(hsaKmtWaitOnEvent(m_pRasEvent, g_TestTimeOut));

    EXPECT_EQ(1, m_pRasEvent->EventData.EventData.MemoryAccessFault.Failure.ErrorType);

    TEST_END;
}

TEST_F(KFDRASTest, DISABLED_MixEventsTest) {
    TEST_START(TESTPROFILE_RUNALL);

    if (!m_setupStatus) {
        return;
    }

    PM4Queue queue;
    HsaEvent* pHsaEvent;

    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, m_defaultGPUNode, &pHsaEvent));
    ASSERT_NE(0, pHsaEvent->EventData.HWData2);

    ASSERT_SUCCESS(queue.Create(m_defaultGPUNode));

    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
            pHsaEvent->EventData.HWData2, pHsaEvent->EventId));

    queue.Wait4PacketConsumption();

    EXPECT_SUCCESS(hsaKmtWaitOnEvent(pHsaEvent, g_TestTimeOut));

    fwrite("inject umc ue 0 0", sizeof(char), 17, m_pFile);
    fflush(m_pFile);

    EXPECT_SUCCESS(hsaKmtWaitOnEvent(m_pRasEvent, g_TestTimeOut));

    EXPECT_EQ(1, m_pRasEvent->EventData.EventData.MemoryAccessFault.Failure.ErrorType);

    EXPECT_SUCCESS(queue.Destroy());
    EXPECT_SUCCESS(hsaKmtDestroyEvent(pHsaEvent));

    TEST_END;
}
