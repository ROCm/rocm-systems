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

#include <syslog.h>

#include "KFDBaseComponentTest.hpp"
#include "KFDTestUtil.hpp"

extern unsigned int g_TestGPUsNum;
extern int g_TestNodeId;
extern std::vector<int> g_SelectedNodes;
extern std::string g_ConcurrentNodes;

void KFDBaseComponentTest::SetUpTestCase() {
}

void KFDBaseComponentTest::TearDownTestCase() {
}

void KFDBaseComponentTest::SetUp() {
    ROUTINE_START

    ASSERT_SUCCESS(hsaKmtOpenKFD());
    EXPECT_SUCCESS(hsaKmtGetVersion(&m_VersionInfo));
    memset( &m_SystemProperties, 0, sizeof(m_SystemProperties) );
    memset(m_RenderNodes, 0, sizeof(m_RenderNodes));

    /** In order to be correctly testing the KFD interfaces and ensure
     *  that the KFD acknowledges relevant node parameters
     *  for the rest of the tests and used for more specific topology tests,
     *  call to GetSystemProperties for a system snapshot of the topology here
     */
    ASSERT_SUCCESS(hsaKmtAcquireSystemProperties(&m_SystemProperties));
    ASSERT_GT(m_SystemProperties.NumNodes, HSAuint32(0)) << "HSA has no nodes.";

    m_NodeInfo.Init(m_SystemProperties.NumNodes);

    // setting memory flags with default values , can be modified according to needs
    m_MemoryFlags.ui32.NonPaged = 0;                         // Paged
    m_MemoryFlags.ui32.CachePolicy = HSA_CACHING_NONCACHED;  // Non cached
    m_MemoryFlags.ui32.ReadOnly = 0;                         // Read/Write
    m_MemoryFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;         // 4KB page
    m_MemoryFlags.ui32.HostAccess = 1;                       // Host accessible
    m_MemoryFlags.ui32.NoSubstitute = 0;                     // Fall back to node 0 if needed
    m_MemoryFlags.ui32.GDSMemory = 0;
    m_MemoryFlags.ui32.Scratch = 0;

    /* nodeProperties is default gpu property, keep it to support old test method */
    const HsaNodeProperties *nodeProperties = m_NodeInfo.HsaDefaultGPUNodeProperties();
    ASSERT_NOTNULL(nodeProperties) << "failed to get HSA default GPU Node properties";

    /* m_FamilyId is default gpu family id, keep it to support old test method */
    m_FamilyId = FamilyIdFromNode(nodeProperties);

    /* these values are for default gpu, keep them to support old test method */
    GetHwQueueInfo(nodeProperties, &m_numCpQueues, &m_numSdmaEngines,
                    &m_numSdmaXgmiEngines, &m_numSdmaQueuesPerEngine);

    g_baseTest = this;

    /* m_pAsm is default gpu assembler, keep it to support old test method */
    m_pAsm = new Assembler(GetGfxVersion(nodeProperties));
    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    int gpuNode;
    for (int i = 0; i < gpuNodes.size(); i++) {
        gpuNode = gpuNodes.at(i);
        const HsaNodeProperties *nodeProperties = m_NodeInfo.GetNodeProperties(gpuNode);

        m_pAsmGPU[i] = new Assembler(GetGfxVersion(nodeProperties));
        GetHwQueueInfo(nodeProperties, &m_numCpQueues_GPU[i], &m_numSdmaEngines_GPU[i],
                    &m_numSdmaXgmiEngines_GPU[i], &m_numSdmaQueuesPerEngine_GPU[i]);
    }

    if (!g_ConcurrentNodes.empty()) {
        std::set<int> uniqueIndices;
        size_t start = 0, end = 0;

        while ((end = g_ConcurrentNodes.find(',', start)) != std::string::npos) {
            std::string token = g_ConcurrentNodes.substr(start, end - start);
            if (!token.empty()) {
                int node = std::stoi(token); 
            
                if (std::find(gpuNodes.begin(), gpuNodes.end(), node) != gpuNodes.end()) 
                    uniqueIndices.insert(node);
                else 
                    LOG() << "Node " << node << " is not a GPU node. Skipping." << std::endl;
            }
            start = end + 1;
        }
        
        if (start < g_ConcurrentNodes.size()) {
            int node = std::stoi(g_ConcurrentNodes.substr(start));
            if (std::find(gpuNodes.begin(), gpuNodes.end(), node) != gpuNodes.end()) {
                uniqueIndices.insert(node);
            } else {
                LOG() << "Node " << node << " is not a GPU node. Skipping." << std::endl;
            }
        }

        g_SelectedNodes.assign(uniqueIndices.begin(), uniqueIndices.end());
        g_TestGPUsNum = static_cast<unsigned int>(g_SelectedNodes.size());

    } else if (g_TestGPUsNum > 0) {
        g_SelectedNodes = gpuNodes;
    } 

    /* adjust g_TestGPUsNum not above MAX_GPU and gpu number at system */
    g_TestGPUsNum = std::min(g_TestGPUsNum, (unsigned int)gpuNodes.size());

    if (!g_SelectedNodes.empty())
        g_SelectedNodes.resize(g_TestGPUsNum);

    const testing::TestInfo* curr_test_info =
                ::testing::UnitTest::GetInstance()->current_test_info();

    openlog("KFDTEST", LOG_CONS , LOG_USER);

    if (g_TestGPUsNum == 1) {
        syslog(LOG_INFO, "[Test on Node#%03d] "
                    "STARTED ========== %s.%s ==========",
                    g_TestNodeId > 0 ? 
                            m_NodeInfo.HsaDefaultGPUNode() : g_SelectedNodes[0],
                    curr_test_info->test_case_name(), curr_test_info->name());
    } else {    
        syslog(LOG_INFO, "[Test on %03d Node(s)] "
                    "STARTED ========== %s.%s ==========",
                    g_TestGPUsNum,
                    curr_test_info->test_case_name(), curr_test_info->name());
    }

    ROUTINE_END
}

void KFDBaseComponentTest::TearDown() {
    ROUTINE_START

    for (int i = 0; i < MAX_RENDER_NODES; i++) {
        if (m_RenderNodes[i].fd <= 0)
            continue;

        amdgpu_device_deinitialize(m_RenderNodes[i].device_handle);
        drmClose(m_RenderNodes[i].fd);
    }

    EXPECT_SUCCESS(hsaKmtReleaseSystemProperties());
    EXPECT_SUCCESS(hsaKmtCloseKFD());
    g_baseTest = NULL;

    if (m_pAsm)
        delete m_pAsm;
    m_pAsm = nullptr;

    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    for (int i = 0; i < gpuNodes.size(); i++) {
        if ( m_pAsmGPU[i]) {
            delete  m_pAsmGPU[i];
            m_pAsmGPU[i] = NULL;
        }
    }

    const testing::TestInfo* curr_test_info =
                ::testing::UnitTest::GetInstance()->current_test_info();

    if (curr_test_info->result()->Passed())
        if (g_TestGPUsNum == 1)
            syslog(LOG_INFO, "[Test on Node#%03d] PASSED"
                             "  ========== %s.%s ==========",
                g_TestNodeId > 0 ? 
                            m_NodeInfo.HsaDefaultGPUNode() : g_SelectedNodes[0],
                curr_test_info->test_case_name(), curr_test_info->name());
        else
            syslog(LOG_INFO, "[Tested on %03d Node(s)] PASSED"
                             "  ========== %s.%s ==========",
                g_TestGPUsNum,
                curr_test_info->test_case_name(), curr_test_info->name());

    else
        if (g_TestGPUsNum == 1)
            syslog(LOG_WARNING, "[Test on Node#%03d] FAILED"
                                 "  ========== %s.%s ==========",
                g_TestNodeId > 0 ? 
                            m_NodeInfo.HsaDefaultGPUNode() : g_SelectedNodes[0],
                curr_test_info->test_case_name(), curr_test_info->name());
        else
            syslog(LOG_WARNING, "[Test on %03d Node(s)] FAILED"
                                 "  ========== %s.%s ==========",
                g_TestGPUsNum,
                curr_test_info->test_case_name(), curr_test_info->name());

    closelog();

    m_NodeInfo.Delete();
    ROUTINE_END
}

HSAuint64 KFDBaseComponentTest::GetSysMemSize() {
    const HsaNodeProperties *nodeProps;
    HsaMemoryProperties cpuMemoryProps;
    HSAuint64 systemMemSize = 0;

    /* Find System Memory size */
    for (unsigned node = 0; node < m_SystemProperties.NumNodes; node++) {
        nodeProps = m_NodeInfo.GetNodeProperties(node);
        if (nodeProps != NULL && nodeProps->NumCPUCores > 0 && nodeProps->NumMemoryBanks > 0) {
            /* For NUMA nodes, memory is distributed among different nodes.
             * Compute total system memory size. KFD driver also computes
             * the system memory (si_meminfo) similarly
             */
            EXPECT_SUCCESS(hsaKmtGetNodeMemoryProperties(node, 1, &cpuMemoryProps));
            systemMemSize += cpuMemoryProps.SizeInBytes;
        }
    }

    return systemMemSize;
}

HSAuint64 KFDBaseComponentTest::GetVramSize(int gpuNode) {
    const HsaNodeProperties *nodeProps;

    /* Find framebuffer size */
    nodeProps = m_NodeInfo.GetNodeProperties(gpuNode);
    EXPECT_NE((const HsaNodeProperties *)NULL, nodeProps);
    HSAuint32 numBanks = nodeProps->NumMemoryBanks;
    HsaMemoryProperties memoryProps[numBanks];
    EXPECT_SUCCESS(hsaKmtGetNodeMemoryProperties(gpuNode, numBanks, memoryProps));
    unsigned bank;
    for (bank = 0; bank < numBanks; bank++) {
        if (memoryProps[bank].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE
                || memoryProps[bank].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
            return memoryProps[bank].SizeInBytes;
    }

    return 0;
}

unsigned int KFDBaseComponentTest::GetFamilyIdFromNodeId(unsigned int nodeId)
{
    return  FamilyIdFromNode(m_NodeInfo.GetNodeProperties(nodeId));
}

Assembler* KFDBaseComponentTest::GetAssemblerFromNodeId(unsigned int nodeId)
{
    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(nodeId);

    if (gpuIndex < 0)
        return NULL;

    return m_pAsmGPU[gpuIndex];
}

bool KFDBaseComponentTest::SVMAPISupported_GPU(unsigned int gpuNode) {

    bool supported = m_NodeInfo.GetNodeProperties(gpuNode)
                         ->Capability.ui32.SVMAPISupported;

    if (!supported)
        LOG() << "SVM API not supported on gpuNode" << gpuNode << std::endl;

    return supported;
}


/*
 * Some asics need CWSR workround for DEGFX11_12113
 */
bool KFDBaseComponentTest::NeedCwsrWA(unsigned int nodeId)
{
    bool needCwsrWA = false;
    const HsaNodeProperties *props = m_NodeInfo.GetNodeProperties(nodeId);

    needCwsrWA = props->EngineId.ui32.Major == 11 &&
                  props->EngineId.ui32.Minor == 0 &&
                  (props->EngineId.ui32.Stepping == 0 ||
                   props->EngineId.ui32.Stepping == 1 ||
                   props->EngineId.ui32.Stepping == 2 ||
                   props->EngineId.ui32.Stepping == 5 ||
                   (props->EngineId.ui32.Stepping == 3 && props->NumArrays > 1));

    return needCwsrWA;
}

bool KFDBaseComponentTest::NeedNonPagedWptr(unsigned int nodeId)
{
    return GetFamilyIdFromNodeId(nodeId) >= FAMILY_GFX11;
}

int KFDBaseComponentTest::FindDRMRenderNode(int gpuNode) {
    HsaNodeProperties *nodeProperties;
    _HSAKMT_STATUS status;

    nodeProperties = new HsaNodeProperties();

    status = hsaKmtGetNodeProperties(gpuNode, nodeProperties);
    EXPECT_SUCCESS(status) << "Node index: " << gpuNode << "hsaKmtGetNodeProperties returned status " << status;

    if (status != HSAKMT_STATUS_SUCCESS) {
        delete nodeProperties;
        return -EINVAL;
    }

    int minor = nodeProperties->DrmRenderMinor;
    if (minor < 128) {
        LOG() << "Failed to get minor number " << minor << std::endl;
        return -EINVAL;
    }

    int index = minor - 128;

    if (m_RenderNodes[index].fd == 0) {
        m_RenderNodes[index].fd = drmOpenRender(minor);

        if (m_RenderNodes[index].fd < 0) {
            LOG() << "Failed to open render node" << std::endl;
            return -EINVAL;
        }

        if (amdgpu_device_initialize(m_RenderNodes[index].fd,
                &m_RenderNodes[index].major_version,
                &m_RenderNodes[index].minor_version,
                &m_RenderNodes[index].device_handle) != 0) {
            drmClose(m_RenderNodes[index].fd);
            m_RenderNodes[index].fd = 0;
            LOG() << "Failed to initialize amdgpu device" << std::endl;
            return -EINVAL;
        }
    }

    return index;
}

HsaVersionInfo* KFDBaseComponentTest::Get_Version() {
    return &m_VersionInfo;
}

HsaNodeInfo* KFDBaseComponentTest::Get_NodeInfo() {
    return &m_NodeInfo;
}

HsaMemFlags& KFDBaseComponentTest::GetHsaMemFlags() {
    return m_MemoryFlags;
}

static void* KFDTest_GPU(void* ptr) {

    KFDTEST_GPUPARAMETERS* pKFDTest_GPUParameters = (KFDTEST_GPUPARAMETERS*)ptr;

    Test_Function test_function        = pKFDTest_GPUParameters->pTest_Function;
    KFDTEST_PARAMETERS* pTestParamters = pKFDTest_GPUParameters->pKFDTest_Parameters;

    try {

        test_function(pTestParamters);

    } catch (...) {
        LOG() << "test failed at gpu" << pTestParamters->gpuNode << std::endl;
    }

    pthread_exit(NULL);
}

HSAKMT_STATUS KFDBaseComponentTest::KFDTestMultiGPU(
                                            std::function<void(int)> test_func, 
                                            const std::vector<int>& gpuNodes,
                                            unsigned int gpu_num) {
    HSAKMT_STATUS r = HSAKMT_STATUS_SUCCESS;
    std::vector<std::thread> threads;
    std::atomic<bool> test_failed(false);
    threads.reserve(gpu_num);

    if (gpuNodes.empty()) return HSAKMT_STATUS_SUCCESS;

    for (int i = 0; i < gpu_num; i++) {
        int gpu_node = gpuNodes.at(i);

        try {
            threads.emplace_back([test_func, gpu_node, &test_failed]() {
                const testing::TestInfo* test_info_before = 
                    ::testing::UnitTest::GetInstance()->current_test_info();
                bool had_failure_before = test_info_before->result()->Failed();
                
                test_func(gpu_node);
                
                const testing::TestInfo* test_info_after = 
                    ::testing::UnitTest::GetInstance()->current_test_info();
                if (!had_failure_before && test_info_after->result()->Failed()) {
                    LOG() << "Test failed at gpu " << gpu_node << std::endl;
                    test_failed = true;
                }
            });
        } catch (const std::system_error& e) {
            std::cout << "Thread creation for gpu node failed : " << gpu_node
                      << " " << e.what() << std::endl;
            r = HSAKMT_STATUS_ERROR;
            break;
        }
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        try {
            if (thread.joinable()) {
                thread.join();
            }
        } catch (const std::system_error& e) {
            std::cout << "thread join failed: " << e.what() << std::endl;
            r = HSAKMT_STATUS_ERROR;
        }
    }

    if (test_failed)
        r = HSAKMT_STATUS_ERROR;

    return r;
}

HSAKMT_STATUS KFDBaseComponentTest::KFDTestLaunch(std::function<void(int)> test_func) {
    /* test on default GPU only */
    if (g_TestNodeId >= 0) {
        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
        if (defaultGPUNode < 0) {
            LOG() << "defaultGPUNode is invalid." << defaultGPUNode << std::endl;
            return HSAKMT_STATUS_INVALID_PARAMETER;
        }

        const testing::TestInfo* test_info_before = 
            ::testing::UnitTest::GetInstance()->current_test_info();
        bool had_failure_before = test_info_before->result()->Failed();
        
        test_func(defaultGPUNode);
        
        const testing::TestInfo* test_info_after = 
            ::testing::UnitTest::GetInstance()->current_test_info();
        if (!had_failure_before && test_info_after->result()->Failed()) {
            LOG() << "Test failed at gpu " << defaultGPUNode << std::endl;
            return HSAKMT_STATUS_ERROR;
        }

        return HSAKMT_STATUS_SUCCESS;
    }

    /* run test_function on all selected GPUs */
    HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;
    err = KFDTestMultiGPU(test_func, g_SelectedNodes, g_TestGPUsNum);

    return err;
}
