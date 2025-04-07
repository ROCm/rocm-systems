/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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
#ifndef __KFD_BASE_COMPONENT_TEST__H__
#define __KFD_BASE_COMPONENT_TEST__H__

#include <gtest/gtest.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <xf86drm.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <sys/param.h>
#include "hsakmt/hsakmt.h"
#include "OSWrapper.hpp"
#include "KFDTestUtil.hpp"
#include "Assemble.hpp"
#include "ShaderStore.hpp"

#define MAX_GPU 64

typedef struct _KFDTEST_PARAMETERS
{
    void*   pTestObject;
    int     gpuNode;
} KFDTEST_PARAMETERS;

typedef void (* Test_Function)(KFDTEST_PARAMETERS*);

typedef struct _KFDTESTGPU_PARAMETERS
{
    KFDTEST_PARAMETERS*   pKFDTest_Parameters;
    Test_Function         pTest_Function;
} KFDTEST_GPUPARAMETERS;

//  @class KFDBaseComponentTest
class KFDBaseComponentTest : public testing::Test {
 public:
    KFDBaseComponentTest(void) { m_MemoryFlags.Value = 0; }
    ~KFDBaseComponentTest(void) {}

    HSAuint64 GetSysMemSize();
    HSAuint64 GetVramSize(int gpuNode);
#define MAX_RENDER_NODES 64
    struct {
        int fd;
        uint32_t major_version;
        uint32_t minor_version;
        amdgpu_device_handle device_handle;
        uint32_t bdf;
    } m_RenderNodes[MAX_RENDER_NODES];

// @brief Finds DRM Render node corresponding to gpuNode
// @return DRM Render Node if successful or -1 on failure
    int FindDRMRenderNode(int gpuNode);
    unsigned int GetFamilyIdFromNodeId(unsigned int nodeId);
    Assembler* GetAssemblerFromNodeId(unsigned int nodeId);
    bool NeedCwsrWA(unsigned int nodeId);
    bool NeedNonPagedWptr(unsigned int nodeId);
    unsigned int GetFamilyIdFromDefaultNode(){ return m_FamilyId; }

    // @brief Executed before the first test that uses KFDBaseComponentTest.
    static  void SetUpTestCase();
    // @brief Executed after the last test from KFDBaseComponentTest.
    static  void TearDownTestCase();

    HsaVersionInfo*  Get_Version();
    HsaNodeInfo* Get_NodeInfo();
    HsaMemFlags& GetHsaMemFlags();
    bool SVMAPISupported_GPU(unsigned int nodeId);

    inline unsigned int Get_NumCpQueues(int gpuIndex){
        return m_numCpQueues_GPU[gpuIndex];
    }

    inline unsigned int Get_NumSdmaEngines(int gpuIndex){
        return m_numSdmaEngines_GPU[gpuIndex];
    }

    inline unsigned int Get_NumSdmaSdmaQueuesPerEngine(int gpuIndex){
        return m_numSdmaQueuesPerEngine_GPU[gpuIndex];
    }

    inline unsigned int Get_NumSdmaSdmaXgmiEngines(int gpuIndex){
        return m_numSdmaXgmiEngines_GPU[gpuIndex];
    }

    HSAKMT_STATUS KFDTestMultiGPU(Test_Function test_function,
				    unsigned int gpu_num);

    HSAKMT_STATUS KFDTest_Launch(Test_Function test_function);

 protected:
    HsaVersionInfo  m_VersionInfo;
    HsaSystemProperties m_SystemProperties;
    unsigned int m_FamilyId;
    unsigned int m_numCpQueues;
    unsigned int m_numSdmaEngines;
    unsigned int m_numSdmaXgmiEngines;
    unsigned int m_numSdmaQueuesPerEngine;
    HsaMemFlags m_MemoryFlags;
    HsaNodeInfo m_NodeInfo;
    HSAint32 m_xnack;
    Assembler* m_pAsm;

    Assembler* m_pAsmGPU[MAX_GPU];

    unsigned int m_numCpQueues_GPU[MAX_GPU];
    unsigned int m_numSdmaEngines_GPU[MAX_GPU];
    unsigned int m_numSdmaXgmiEngines_GPU[MAX_GPU];
    unsigned int m_numSdmaQueuesPerEngine_GPU[MAX_GPU];

    // @brief Executed before every test that uses KFDBaseComponentTest class and sets all common settings for the tests.
    virtual void SetUp();
    // @brief Executed after every test that uses KFDBaseComponentTest class.
    virtual void TearDown();

    /* TO DO: check all gpu support svm api */
    bool SVMAPISupported() {
        bool supported = m_NodeInfo.HsaDefaultGPUNodeProperties()
                        ->Capability.ui32.SVMAPISupported;
        if (!supported)
            LOG() << "SVM API not supported" << std::endl;
        return supported;
    }

    // Set xnack_override to -1 if parameter is not passed in, to avoid unnecessary code churn
    void SVMSetXNACKMode(int xnack_override = -1) {
        if (!SVMAPISupported())
            return;

        m_xnack = -1;
        HSAKMT_STATUS ret = hsaKmtGetXNACKMode(&m_xnack);
        if (ret != HSAKMT_STATUS_SUCCESS) {
            LOG() << "Failed " << ret << " to get XNACK mode" << std::endl;
            return;
        }

        HSAint32 xnack_on = -1;
        char *hsa_xnack = getenv("HSA_XNACK");

        // HSA_XNACK takes priority over kfdtest parameters
        if (hsa_xnack)
                xnack_on = strncmp(hsa_xnack, "0", 1);
        else if (xnack_override > -1)
                xnack_on = xnack_override;
        else
                return;

	// No need to set XNACK if it's already the current value
	if (xnack_on == m_xnack)
		return;

        ret = hsaKmtSetXNACKMode(xnack_on);
        if (ret != HSAKMT_STATUS_SUCCESS)
            LOG() << "Failed " << ret << " to set XNACK mode " << xnack_on << std::endl;
        else
            LOG() << "Setting XNACK mode to " << xnack_on << std::endl;
    }

    void SVMRestoreXNACKMode() {
        if (!SVMAPISupported())
             return;

        if (m_xnack == -1)
            return;

        hsaKmtSetXNACKMode(m_xnack);
    }
};

extern KFDBaseComponentTest* g_baseTest;
#endif  //  __KFD_BASE_COMPONENT_TEST__H__
