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

#include "KFDTestUtil.hpp"
#include <stdlib.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <algorithm>
#include <vector>
#include "BaseQueue.hpp"
#include "Dispatch.hpp"
#include "SDMAPacket.hpp"

void WaitUntilInput() {
    char dummy;
    printf("Press enter to continue: ");
    do {
        scanf("%c", &dummy);
    } while (dummy != 10); // enter key's ascii value is 10
}

/* fscanf_dec - read a file whose content is a decimal number
 *      @file [IN ] file to read
 *      @num [OUT] number in the file
 *
 * It is copied from the same function in libhsakmt
 */
HSAKMT_STATUS fscanf_dec(const char *file, uint32_t *num)
{
    FILE *fd;
    HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

    fd = fopen(file, "r");
    if (!fd) {
        LOG() << "Failed to open " << file << std::endl;
        return HSAKMT_STATUS_INVALID_PARAMETER;
    }
    if (fscanf(fd, "%u", num) != 1) {
        LOG() << "Failed to parse as a decimal: " << file << std::endl;;
        ret = HSAKMT_STATUS_ERROR;
    }

    fclose(fd);
    return ret;
}

uint64_t RoundToPowerOf2(uint64_t val) {
    val--;

    /* Shift with amount larger than the bit width can result in
     * undefined behavior by compiler for release builds.
     * Shift till 32 bit only which is less than bit width of val.
     */
    for (int i = 1; i <= 32; i *= 2)
        val |= val >> i;

    val++;

    return val;
}

bool WaitOnValue(const volatile unsigned int *buf, unsigned int value, unsigned int timeOut) {
    while (timeOut > 0 && *buf != value) {
        Delay(1);

        if (timeOut != HSA_EVENTTIMEOUT_INFINITE)
            timeOut--;
    }

    return *buf == value;
}

void SplitU64(const HSAuint64 value, unsigned int& rLoPart, unsigned int& rHiPart) {
    rLoPart = static_cast<unsigned int>(value);
    rHiPart = static_cast<unsigned int>(value >> 32);
}

bool CheckEmuModeEnabled()
{
    uint32_t emu_mode = false;
    fscanf_dec("/sys/module/amdgpu/parameters/emu_mode", &emu_mode);
    return (emu_mode != 0);
}

bool GetHwCapabilityHWS() {
    unsigned int value = 0;
    bool valExists = ReadDriverConfigValue(CONFIG_HWS, value);

    /* HWS is enabled by default */
    return ( (!valExists) || ( value > 0));
}

HSAKMT_STATUS CreateQueueTypeEvent(
    bool                ManualReset,            // IN
    bool                IsSignaled,             // IN
    unsigned int        NodeId,                 // IN
    HsaEvent**          Event                   // OUT
    ) {
    HsaEventDescriptor Descriptor;

// TODO: Create per-OS header with this sort of definitions
#ifdef _WIN32
    Descriptor.EventType = HSA_EVENTTYPE_QUEUE_EVENT;
#else
    Descriptor.EventType = HSA_EVENTTYPE_SIGNAL;
#endif
    Descriptor.SyncVar.SyncVar.UserData = (void*)0xABCDABCD;
    Descriptor.NodeId = NodeId;

    return hsaKmtCreateEvent(&Descriptor, ManualReset, IsSignaled, Event);
}

HSAKMT_STATUS CreateHWExceptionEvent(
    bool                ManualReset,            // IN
    bool                IsSignaled,             // IN
    unsigned int        NodeId,                 // IN
    HsaEvent**          Event                   // OUT
    ) {
    HsaEventDescriptor Descriptor;

    Descriptor.EventType = HSA_EVENTTYPE_HW_EXCEPTION;
    Descriptor.SyncVar.SyncVar.UserData = (void*)0xABCDABCD;
    Descriptor.NodeId = NodeId;

    return hsaKmtCreateEvent(&Descriptor, ManualReset, IsSignaled, Event);
}

static bool hsakmt_is_dgpu_dev = false;

bool hsakmt_is_dgpu() {
    return hsakmt_is_dgpu_dev;
}

bool hasPciAtomicsSupport(int node) {
    /* If we can't get Node Properties, assume a lack of Atomics support */
    HsaNodeProperties *pNodeProperties = new HsaNodeProperties();
    if (hsaKmtGetNodeProperties(node, pNodeProperties)) {
        LOG() << "Unable to get Node Properties for node " << node << std::endl;
        return false;
    }

    /* APUs don't have IO Links, but support Atomic Ops by default */
    if (pNodeProperties->NumCPUCores && pNodeProperties->NumFComputeCores)
        return true;

    HsaIoLinkProperties *IolinkProperties = new HsaIoLinkProperties[pNodeProperties->NumIOLinks];
    if (hsaKmtGetNodeIoLinkProperties(node, pNodeProperties->NumIOLinks, IolinkProperties)) {
        LOG() << "Unable to get Node IO Link Information for node " << node << std::endl;
        return false;
    }

    /* Make sure we're checking GPU-to-CPU connection here */
    for (int linkId = 0; linkId < pNodeProperties->NumIOLinks; linkId++) {
        /* Make sure it's a CPU */
        HsaNodeProperties *linkProps = new HsaNodeProperties();
        if (hsaKmtGetNodeProperties(IolinkProperties[linkId].NodeTo, linkProps)) {
            LOG() << "Unable to get connected device's IO Link information" << std::endl;
            return false;
        }
        if (linkProps->NumCPUCores) {
            /* IOLink flags are only valid if Override flag is set */
            return (IolinkProperties[linkId].Flags.ui32.Override &&
                   !IolinkProperties[linkId].Flags.ui32.NoAtomics32bit &&
                   !IolinkProperties[linkId].Flags.ui32.NoAtomics64bit);
        }
    }

    return false;
}

unsigned int FamilyIdFromNode(const HsaNodeProperties *props) {
    unsigned int familyId = FAMILY_UNKNOWN;

    switch (props->EngineId.ui32.Major) {
    case 7:
        if (props->EngineId.ui32.Minor == 0) {
            if (props->EngineId.ui32.Stepping == 0)
                familyId = FAMILY_KV;
            else
                familyId = FAMILY_CI;
        }
        break;
    case 8:
        familyId = FAMILY_VI;
        if (props->EngineId.ui32.Stepping == 1)
            familyId = FAMILY_CZ;
        break;
    case 9:
        familyId = FAMILY_AI;
        if (props->EngineId.ui32.Minor >= 4)
            familyId = FAMILY_AV;
        else if (props->EngineId.ui32.Stepping == 2)
            familyId = FAMILY_RV;
        else if (props->EngineId.ui32.Stepping == 8)
            familyId = FAMILY_AR;
        else if (props->EngineId.ui32.Stepping == 10)
            familyId = FAMILY_AL;
        break;
    case 10:
        familyId = FAMILY_NV;
        break;
    case 11:
        familyId = FAMILY_GFX11;
        break;
    case 12:
        familyId = FAMILY_GFX12;
	break;
    }

    if (props->NumCPUCores && props->NumFComputeCores)
        hsakmt_is_dgpu_dev = false;
    else
        hsakmt_is_dgpu_dev = true;

    return familyId;
}

void GetHwQueueInfo(const HsaNodeProperties *props,
                 unsigned int *p_num_cp_queues,
                 unsigned int *p_num_sdma_engines,
                 unsigned int *p_num_sdma_xgmi_engines,
                 unsigned int *p_num_sdma_queues_per_engine) {
    if (p_num_sdma_engines)
        *p_num_sdma_engines = props->NumSdmaEngines;

    if (p_num_sdma_xgmi_engines)
        *p_num_sdma_xgmi_engines = props->NumSdmaXgmiEngines;

    if (p_num_sdma_queues_per_engine)
        *p_num_sdma_queues_per_engine = props->NumSdmaQueuesPerEngine;

    if (p_num_cp_queues)
        *p_num_cp_queues = props->NumCpQueues;
}

bool isTonga(const HsaNodeProperties *props) {
    /* Tonga has some workarounds in the thunk that cause certain failures */
    if (props->EngineId.ui32.Major == 8 && props->EngineId.ui32.Stepping == 2) {
        return true;
    }

    return false;
}

const uint32_t GetGfxVersion(const HsaNodeProperties *props) {
    return ((props->EngineId.ui32.Major << 16) |
            (props->EngineId.ui32.Minor <<  8) |
            (props->EngineId.ui32.Stepping));
}

HSAuint64 GetSystemTickCountInMicroSec() {
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec * 1000000ULL + t.tv_usec;
}

const HsaMemoryBuffer HsaMemoryBuffer::Null;

HsaMemoryBuffer::HsaMemoryBuffer(HSAuint64 size, unsigned int node, bool zero, bool isLocal, bool isExec,
                                 bool isScratch, bool isReadOnly, bool isUncached, bool NonPaged)
    :m_Size(size),
    m_pUser(NULL),
    m_pBuf(NULL),
    m_Local(isLocal),
    m_Node(node) {
    m_Flags.Value = 0;

    HsaMemMapFlags mapFlags = {0};
    bool map_specific_gpu = (node && !isScratch);

    if (isScratch) {
        m_Flags.ui32.Scratch = 1;
        m_Flags.ui32.HostAccess = 1;
    } else {
        m_Flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;

        if (isLocal) {
            m_Flags.ui32.HostAccess = 0;
            m_Flags.ui32.NonPaged = 1;
            m_Flags.ui32.CoarseGrain = 1;
            EXPECT_EQ(isUncached, 0) << "Uncached flag is relevant only for system or host memory";
        } else {
            m_Flags.ui32.HostAccess = 1;
            m_Flags.ui32.NonPaged = NonPaged ? 1 : 0;
            m_Flags.ui32.CoarseGrain = 0;
            m_Flags.ui32.NoNUMABind = 1;
            m_Flags.ui32.Uncached = isUncached;
        }

        if (isExec)
            m_Flags.ui32.ExecuteAccess = 1;
    }
    if (isReadOnly)
        m_Flags.ui32.ReadOnly = 1;

    if (zero)
        EXPECT_EQ(m_Flags.ui32.HostAccess, 1);

    EXPECT_SUCCESS(hsaKmtAllocMemory(m_Node, m_Size, m_Flags, &m_pBuf));
    if (hsakmt_is_dgpu()) {
        if (map_specific_gpu)
            EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes(m_pBuf, m_Size, NULL, mapFlags, 1, &m_Node));
        else
            EXPECT_SUCCESS(hsaKmtMapMemoryToGPU(m_pBuf, m_Size, NULL));
        m_MappedNodes = 1 << m_Node;
    }

    if (zero && !isLocal)
        Fill(0);
}

HsaMemoryBuffer::HsaMemoryBuffer(void *addr, HSAuint64 size):
    m_Size(size),
    m_pUser(addr),
    m_pBuf(NULL),
    m_Local(false),
    m_Node(0) {
    HSAuint64 gpuva = 0;
    EXPECT_SUCCESS(hsaKmtRegisterMemory(m_pUser, m_Size));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPU(m_pUser, m_Size, &gpuva));
    m_pBuf = gpuva ? (void *)gpuva : m_pUser;
}

HsaMemoryBuffer::HsaMemoryBuffer()
    :m_Size(0),
    m_pBuf(NULL) {
}

void HsaMemoryBuffer::Fill(unsigned char value, HSAuint64 offset, HSAuint64 size) {
    HSAuint32 uiValue;

    EXPECT_EQ(m_Local, 0) << "Local Memory. Call Fill(HSAuint32 value, BaseQueue& baseQueue)";

    size = size ? size : m_Size;
    ASSERT_TRUE(size + offset <= m_Size) << "Buffer Overflow" << std::endl;

    if (m_pUser != NULL)
        memset(reinterpret_cast<char *>(m_pUser) + offset, value, size);
    else if (m_pBuf != NULL)
        memset(reinterpret_cast<char *>(m_pBuf) + offset, value, size);
    else
        ASSERT_TRUE(0) << "Invalid HsaMemoryBuffer";
}

/* Fill CPU accessible buffer with the value. */
void HsaMemoryBuffer::Fill(HSAuint32 value, HSAuint64 offset, HSAuint64 size) {
    HSAuint64 i;
    HSAuint32 *ptr = NULL;

    EXPECT_EQ(m_Local, 0) << "Local Memory. Call Fill(HSAuint32 value, BaseQueue& baseQueue)";
    size = size ? size : m_Size;
    EXPECT_EQ((size & (sizeof(HSAuint32) - 1)), 0) << "Not word aligned. Call Fill(unsigned char)";
    ASSERT_TRUE(size + offset <= m_Size) << "Buffer Overflow" << std::endl;

    if (m_pUser != NULL)
        ptr = reinterpret_cast<HSAuint32 *>(reinterpret_cast<char *>(m_pUser) + offset);
    else if (m_pBuf != NULL)
        ptr = reinterpret_cast<HSAuint32 *>(reinterpret_cast<char *>(m_pBuf) + offset);

    ASSERT_NOTNULL(ptr);

    for (i = 0; i < size / sizeof(HSAuint32); i++)
        ptr[i] = value;
}

/* Fill GPU only accessible Local memory with @value using SDMA Constant Fill Command */
void HsaMemoryBuffer::Fill(HSAuint32 value, BaseQueue& baseQueue, HSAuint64 offset, HSAuint64 size) {
    HsaEvent* event = NULL;

    EXPECT_NE(m_Local, 0) << "Not Local Memory. Call Fill(HSAuint32 value)";

    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, m_Node, &event));
    ASSERT_EQ(baseQueue.GetQueueType(), HSA_QUEUE_SDMA) << "Only SDMA queues supported";

    size = size ? size : m_Size;
    ASSERT_TRUE(size + offset <= m_Size) << "Buffer Overflow" << std::endl;

    baseQueue.PlacePacket(SDMAFillDataPacket(baseQueue.GetFamilyId(),
                                (reinterpret_cast<void *>(this->As<char*>() + offset)), value, size));
    baseQueue.PlacePacket(SDMAFencePacket(baseQueue.GetFamilyId(),
                                reinterpret_cast<void*>(event->EventData.HWData2), event->EventId));
    baseQueue.PlaceAndSubmitPacket(SDMATrapPacket(event->EventId));
    EXPECT_SUCCESS(hsaKmtWaitOnEvent(event, g_TestTimeOut));

    hsaKmtDestroyEvent(event);
}

/* Check if HsaMemoryBuffer[location] has the pattern specified.
 * Return TRUE if correct pattern else return FALSE
 * HsaMemoryBuffer has to be CPU accessible
 */
bool HsaMemoryBuffer::IsPattern(HSAuint64 location, HSAuint32 pattern) {
    HSAuint32 *ptr = NULL;

    EXPECT_EQ(m_Local, 0) << "Local Memory. Call IsPattern(..baseQueue& baseQueue)";

    if (location >= m_Size) /* Out of bounds */
        return false;

    if (m_pUser != NULL)
        ptr = reinterpret_cast<HSAuint32 *>(m_pUser);
    else if (m_pBuf != NULL)
        ptr = reinterpret_cast<HSAuint32 *>(m_pBuf);
    else
        return false;

    if (ptr)
        return (ptr[location/sizeof(HSAuint32)] == pattern);

    return false;
}

/* Check if HsaMemoryBuffer[location] has the pattern specified.
 * Return TRUE if correct pattern else return FALSE
 * HsaMemoryBuffer is supposed to be only GPU accessible
 * Use @baseQueue to copy the HsaMemoryBuffer[location] to stack and check the value
 */

bool HsaMemoryBuffer::IsPattern(HSAuint64 location, HSAuint32 pattern, BaseQueue& baseQueue, volatile HSAuint32 *tmp) {
    HsaEvent* event = NULL;
    int ret;

    EXPECT_NE(m_Local, 0) << "Not Local Memory. Call IsPattern(HSAuint64 location, HSAuint32 pattern)";
    EXPECT_EQ(baseQueue.GetQueueType(), HSA_QUEUE_SDMA) << "Only SDMA queues supported";

    if (location >= m_Size) /* Out of bounds */
        return false;

    ret = CreateQueueTypeEvent(false, false, m_Node, &event);
    if (ret)
        return false;

    *tmp = ~pattern;
    baseQueue.PlacePacket(SDMACopyDataPacket(baseQueue.GetFamilyId(), (void *)tmp,
            reinterpret_cast<void *>(this->As<HSAuint64>() + location),
            sizeof(HSAuint32)));
    baseQueue.PlacePacket(SDMAFencePacket(baseQueue.GetFamilyId(), reinterpret_cast<void*>(event->EventData.HWData2),
            event->EventId));
    baseQueue.PlaceAndSubmitPacket(SDMATrapPacket(event->EventId));

    ret = hsaKmtWaitOnEvent(event, g_TestTimeOut);
    hsaKmtDestroyEvent(event);
    if (ret)
        return false;

    return WaitOnValue(tmp, pattern);
}

unsigned int HsaMemoryBuffer::Size() {
    return m_Size;
}

HsaMemFlags HsaMemoryBuffer::Flags() {
    return m_Flags;
}

unsigned int HsaMemoryBuffer::Node() const {
    return m_Node;
}

int HsaMemoryBuffer::MapMemToNodes(unsigned int *nodes, unsigned int nodes_num) {
    HsaMemMapFlags mapFlags = {0};
    int ret, bit;

    ret = hsaKmtMapMemoryToGPUNodes(m_pBuf, m_Size, NULL, mapFlags, nodes_num, nodes);
    if (ret != 0) {
        return ret;
    }

    for (unsigned int i = 0; i < nodes_num; i++) {
        bit = 1 << nodes[i];
        m_MappedNodes |= bit;
    }

    return 0;
}

int HsaMemoryBuffer::UnmapMemToNodes(unsigned int *nodes, unsigned int nodes_num) {
    int ret, bit;

    ret = hsaKmtUnmapMemoryToGPU(m_pBuf);
    if (ret)
        return ret;

    for (unsigned int i = 0; i < nodes_num; i++) {
        bit = 1 << nodes[i];
        m_MappedNodes &= ~bit;
    }

    return 0;
}

void HsaMemoryBuffer::UnmapAllNodes() {
    unsigned int *Arr, size, i, j;
    int bit;

    size = 0;
    for (i = 0; i < 8; i++) {
        bit = 1 << i;
        if (m_MappedNodes & bit)
            size++;
    }

    Arr = (unsigned int *)malloc(sizeof(unsigned int) * size);
    if (!Arr)
        return;

    for (i = 0, j =0; i < 8; i++) {
        bit = 1 << i;
        if (m_MappedNodes & bit)
            Arr[j++] = i;
    }

    /*
     * TODO: When thunk is updated, use hsaKmtRegisterToNodes. Then nodes will be used
     */
    hsaKmtUnmapMemoryToGPU(m_pBuf);

    m_MappedNodes = 0;

    free(Arr);
}

HsaMemoryBuffer::~HsaMemoryBuffer() {
    if (m_pUser != NULL) {
        hsaKmtUnmapMemoryToGPU(m_pUser);
        hsaKmtDeregisterMemory(m_pUser);
    } else if (m_pBuf != NULL) {
        if (hsakmt_is_dgpu()) {
            if (m_MappedNodes) {
                hsaKmtUnmapMemoryToGPU(m_pBuf);
            }
        }
        hsaKmtFreeMemory(m_pBuf, m_Size);
    }
    m_pBuf = NULL;
}

HsaInteropMemoryBuffer::HsaInteropMemoryBuffer(HSAuint64 device_handle, HSAuint64 buffer_handle,
                                               HSAuint64 size, unsigned int node)
    :m_Size(0),
     m_pBuf(NULL),
     m_graphic_handle(0),
     m_Node(node) {
    HSAuint64 flat_address;
    EXPECT_SUCCESS(hsaKmtMapGraphicHandle(m_Node, device_handle, buffer_handle, 0, size, &flat_address));
    m_pBuf = reinterpret_cast<void*>(flat_address);
}

HsaInteropMemoryBuffer::~HsaInteropMemoryBuffer() {
    hsaKmtUnmapGraphicHandle(m_Node, (HSAuint64)m_pBuf, m_Size);
}


HsaNodeInfo::HsaNodeInfo() {
}

/* Init - Get and store information about all the HSA nodes from the Thunk Library.
 * @NumOfNodes - Number to system nodes returned by hsaKmtAcquireSystemProperties
 * @Return - false: if no node information is available
 */
bool HsaNodeInfo::Init(int NumOfNodes) {
    HsaNodeProperties *nodeProperties;
    _HSAKMT_STATUS status;
    bool ret = false;

    for (int i = 0; i < NumOfNodes; i++) {
        nodeProperties = new HsaNodeProperties();

        status = hsaKmtGetNodeProperties(i, nodeProperties);
        /* This is not a fatal test (not using assert), since even when it fails for one node
         * we want to get information regarding others.
         */
        EXPECT_SUCCESS(status) << "Node index: " << i << "hsaKmtGetNodeProperties returned status " << status;

        if (status == HSAKMT_STATUS_SUCCESS) {
            m_HsaNodeProps.push_back(nodeProperties);
            ret = true;  // Return true if atleast one information is available

            if (nodeProperties->NumFComputeCores)
                m_NodesWithGPU.push_back(i);
            else
                m_NodesWithoutGPU.push_back(i);
        } else {
            delete nodeProperties;
        }
    }

    return ret;
}

void HsaNodeInfo::Delete() {
    const HsaNodeProperties *nodeProperties;

    for (unsigned int i = 0; i < m_HsaNodeProps.size(); i++)
        delete m_HsaNodeProps.at(i);

    m_HsaNodeProps.clear();
    m_NodesWithGPU.clear();
    m_NodesWithoutGPU.clear();
}

HsaNodeInfo::~HsaNodeInfo() {
    Delete();
}

const std::vector<int>& HsaNodeInfo::GetNodesWithGPU() const {
    return m_NodesWithGPU;
}

const HsaNodeProperties* HsaNodeInfo::GetNodeProperties(int NodeNum) const {
    return m_HsaNodeProps.at(NodeNum);
}

const int HsaNodeInfo::HsaGPUindexFromGpuNode(int gpuNodeId) const {
    if (m_NodesWithGPU.size() == 0)
        return -1;

    for (unsigned int i = 0; i < m_NodesWithGPU.size(); i++) {
        if (gpuNodeId == m_NodesWithGPU.at(i))
            return i;
    }

    return -1;
}

const HsaNodeProperties* HsaNodeInfo::HsaDefaultGPUNodeProperties() const {
    int NodeNum = HsaDefaultGPUNode();
    if (NodeNum < 0)
        return NULL;
    return GetNodeProperties(NodeNum);
}

const int HsaNodeInfo::HsaDefaultGPUNode() const {
    if (m_NodesWithGPU.size() == 0)
        return -1;

    if (g_TestNodeId >= 0) {
        // Check if this is a valid Id, if so use this else use first available
        for (unsigned int i = 0; i < m_NodesWithGPU.size(); i++) {
            if (g_TestNodeId == m_NodesWithGPU.at(i))
                return g_TestNodeId;
        }
    }

    return m_NodesWithGPU.at(0);
}

void HsaNodeInfo::PrintNodeInfo() const {
    const HsaNodeProperties *nodeProperties;

    for (unsigned int i = 0; i < m_HsaNodeProps.size(); i++) {
        nodeProperties = m_HsaNodeProps.at(i);

        LOG() << "***********************************" << std::endl;
        LOG() << "Node " << i << std::endl;
        LOG() << "NumCPUCores=\t" << nodeProperties->NumCPUCores << std::endl;
        LOG() << "NumFComputeCores=\t" << nodeProperties->NumFComputeCores << std::endl;
        LOG() << "NumMemoryBanks=\t" << nodeProperties->NumMemoryBanks << std::endl;
        LOG() << "VendorId=\t" << nodeProperties->VendorId << std::endl;
        LOG() << "DeviceId=\t" << nodeProperties->DeviceId << std::endl;
        LOG() << "***********************************" << std::endl;
    }

    LOG() << "Default GPU NODE " << HsaDefaultGPUNode() << std::endl;
}

const bool HsaNodeInfo::IsGPUNodeLargeBar(int node) const {
    const HsaNodeProperties *pNodeProperties;

    pNodeProperties = GetNodeProperties(node);
    if (pNodeProperties) {
        HsaMemoryProperties *memoryProperties =
                new HsaMemoryProperties[pNodeProperties->NumMemoryBanks];
        EXPECT_SUCCESS(hsaKmtGetNodeMemoryProperties(node,
                       pNodeProperties->NumMemoryBanks, memoryProperties));
        for (unsigned bank = 0; bank < pNodeProperties->NumMemoryBanks; bank++)
            if (memoryProperties[bank].HeapType ==
                                HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC) {
                delete [] memoryProperties;
                return true;
            }
        delete [] memoryProperties;
    }

    return false;
}

const bool HsaNodeInfo::IsAppAPU(int node) const {
    const HsaNodeProperties *pNodeProperties = GetNodeProperties(node);

    /*  CPU with compute cores is small APU, not AppAPU */
    if (pNodeProperties->NumCPUCores && pNodeProperties->NumFComputeCores)
        return false;

    HsaIoLinkProperties *IolinkProperties = new HsaIoLinkProperties[pNodeProperties->NumIOLinks];
    if (hsaKmtGetNodeIoLinkProperties(node, pNodeProperties->NumIOLinks, IolinkProperties)) {
        LOG() << "Unable to get Node IO Link Information for node " << node << std::endl;
        delete [] IolinkProperties;
        return false;
    }

    /* Checking GPU-to-CPU connection weight */
    for (int linkId = 0; linkId < pNodeProperties->NumIOLinks; linkId++) {
        HsaNodeProperties linkProps;

        if (hsaKmtGetNodeProperties(IolinkProperties[linkId].NodeTo, &linkProps)) {
            LOG() << "Unable to get connected device's IO Link information" << std::endl;
            break;
        }

        /* If it's GPU-CPU link with connection weight KFD_CRAT_INTRA_SOCKET_WEIGHT 13 */
        if (linkProps.NumCPUCores && IolinkProperties[linkId].Weight == 13) {
            delete [] IolinkProperties;
            return true;
        }
    }
    delete [] IolinkProperties;
    return false;
}

const bool HsaNodeInfo::IsPeerAccessibleByNode(int peer, int node) const {
    const HsaNodeProperties *pNodeProperties;

    pNodeProperties = GetNodeProperties(node);
    if (pNodeProperties) {
        HsaIoLinkProperties p2pLinksProperties[pNodeProperties->NumIOLinks];
        EXPECT_SUCCESS(hsaKmtGetNodeIoLinkProperties(node,
					pNodeProperties->NumIOLinks, p2pLinksProperties));

        for (unsigned link = 0; link < pNodeProperties->NumIOLinks; link++)
            if (p2pLinksProperties[link].NodeTo == peer)
                return true;
    }

    return false;
}

const int HsaNodeInfo::FindLargeBarGPUNode() const {
    const std::vector<int> gpuNodes = GetNodesWithGPU();

    for (unsigned i = 0; i < gpuNodes.size(); i++)
        if (IsGPUNodeLargeBar(gpuNodes.at(i)))
            return gpuNodes.at(i);

    return -1;
}

const bool HsaNodeInfo::AreGPUNodesXGMI(int node0, int node1) const {
    const HsaNodeProperties *pNodeProperties0 = GetNodeProperties(node0);
    const HsaNodeProperties *pNodeProperties1 = GetNodeProperties(node1);

    if ((pNodeProperties0->HiveID != 0) && (pNodeProperties1->HiveID != 0) &&
        (pNodeProperties0->HiveID == pNodeProperties1->HiveID))
        return true;

    return false;
}

int HsaNodeInfo::FindAccessiblePeers(std::vector<int> *peers,
		                             HSAuint32 node) const {
    peers->push_back(node);

    for (unsigned i = 0; i < m_NodesWithGPU.size(); i++) {
        if (m_NodesWithGPU.at(i) == node)
            continue;

        if (IsPeerAccessibleByNode(m_NodesWithGPU.at(i), node))
            peers->push_back(m_NodesWithGPU.at(i));
    }
    return peers->size();
}

const bool HsaNodeInfo::IsNodeXGMItoCPU(int node) const {
    const HsaNodeProperties *pNodeProperties;
    bool ret = false;

    pNodeProperties = GetNodeProperties(node);
    if (pNodeProperties && pNodeProperties->NumIOLinks) {
        HsaIoLinkProperties  *IolinkProperties =  new HsaIoLinkProperties[pNodeProperties->NumIOLinks];
        EXPECT_SUCCESS(hsaKmtGetNodeIoLinkProperties(node, pNodeProperties->NumIOLinks, IolinkProperties));

        for (int linkId = 0; linkId < pNodeProperties->NumIOLinks; linkId++) {
            EXPECT_EQ(node, IolinkProperties[linkId].NodeFrom);
            const HsaNodeProperties *pNodeProperties0 =
                    GetNodeProperties(IolinkProperties[linkId].NodeTo);
            if (pNodeProperties0->NumFComputeCores == 0 &&
                    IolinkProperties[linkId].IoLinkType == HSA_IOLINK_TYPE_XGMI)
                ret = true;
        }
        delete [] IolinkProperties;
    }

    return ret;
}

HSAKMT_STATUS RegisterSVMRange(HSAuint32 GPUNode, void *MemoryAddress,
                               HSAuint64 SizeInBytes, HSAuint32 PrefetchNode,
                               HSAuint32 SVMFlags) {
    HSA_SVM_ATTRIBUTE *attrs;
    HSAuint64 s_attr;
    HSAuint32 nattr;
    HSAKMT_STATUS r;

    nattr = 4;
    s_attr = sizeof(*attrs) * nattr;
    attrs = (HSA_SVM_ATTRIBUTE *)alloca(s_attr);

    attrs[0].type = HSA_SVM_ATTR_PREFETCH_LOC;
    attrs[0].value = PrefetchNode;
    attrs[1].type = HSA_SVM_ATTR_PREFERRED_LOC;
    attrs[1].value = PrefetchNode;
    attrs[2].type = HSA_SVM_ATTR_SET_FLAGS;
    attrs[2].value = SVMFlags;
    attrs[3].type = HSA_SVM_ATTR_ACCESS;
    attrs[3].value = GPUNode;

    r = hsaKmtSVMSetAttr(MemoryAddress, SizeInBytes, nattr, attrs);
    if (r)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS SVMRangeGetPrefetchNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                      HSAuint32 *PrefetchNode) {
    HSA_SVM_ATTRIBUTE attr;
    int r;

    attr.type = HSA_SVM_ATTR_PREFETCH_LOC;
    attr.value = 0;

    r = hsaKmtSVMGetAttr(MemoryAddress, SizeInBytes, 1, &attr);
    if (r)
        return HSAKMT_STATUS_ERROR;

    *PrefetchNode = attr.value;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS SVMRangePrefetchToNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                           HSAuint32 PrefetchNode) {
    HSA_SVM_ATTRIBUTE attr;
    int r;

    attr.type = HSA_SVM_ATTR_PREFETCH_LOC;
    attr.value = PrefetchNode;

    r = hsaKmtSVMSetAttr(MemoryAddress, SizeInBytes, 1, &attr);
    if (r)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS SVMRangeMapToNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                           HSAuint32 NodeID) {
    HSA_SVM_ATTRIBUTE attr;
    int r;

    attr.type = HSA_SVM_ATTR_ACCESS;
    attr.value = NodeID;

    r = hsaKmtSVMSetAttr(MemoryAddress, SizeInBytes, 1, &attr);
    if (r)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS SVMRangeMapInPlaceToNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                           HSAuint32 NodeID) {
    HSA_SVM_ATTRIBUTE attr;
    int r;

    attr.type = HSA_SVM_ATTR_ACCESS_IN_PLACE;
    attr.value = NodeID;

    r = hsaKmtSVMSetAttr(MemoryAddress, SizeInBytes, 1, &attr);
    if (r)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS SVMRangSetGranularity(void *MemoryAddress, HSAuint64 SizeInBytes,
                                    HSAuint32 Granularity) {
    HSA_SVM_ATTRIBUTE attr;
    int r;

    attr.type = HSA_SVM_ATTR_GRANULARITY;
    attr.value = Granularity;

    r = hsaKmtSVMSetAttr(MemoryAddress, SizeInBytes, 1, &attr);
    if (r)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HsaSVMRange::HsaSVMRange(HSAuint64 size, HSAuint32 GPUNode) :
    HsaSVMRange(NULL, size, GPUNode, 0) {}

HsaSVMRange::HsaSVMRange(HSAuint64 size) :
    HsaSVMRange(NULL, size, 0, 0, true) {}

HsaSVMRange::HsaSVMRange(HSAuint64 size, HSAuint32 GPUNode, HSAuint32 PrefetchNode) :
    HsaSVMRange(NULL, size, GPUNode, PrefetchNode) {}

HsaSVMRange::HsaSVMRange(void *addr, HSAuint64 size, HSAuint32 GPUNode, HSAuint32 PrefetchNode,
                         bool noRegister, bool isLocal, bool isExec, bool isReadOnly):
    m_Size(size),
    m_pUser(addr),
    m_Local(isLocal),
    m_Node(PrefetchNode),
    m_SelfAllocated(false) {
    if (!m_pUser) {
        m_pUser = mmap(0, m_Size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        EXPECT_NE(MAP_FAILED, m_pUser);
        m_SelfAllocated = true;
    }

    if (m_Local)
        m_Flags = HSA_SVM_FLAG_HOST_ACCESS;
    else
        m_Flags = HSA_SVM_FLAG_HOST_ACCESS | HSA_SVM_FLAG_COHERENT;

    if (isReadOnly)
        m_Flags |= HSA_SVM_FLAG_GPU_RO;
    if (isExec)
        m_Flags |= HSA_SVM_FLAG_GPU_EXEC;

    if (!noRegister)
        EXPECT_SUCCESS(RegisterSVMRange(GPUNode, m_pUser, m_Size, PrefetchNode, m_Flags));
}

HsaSVMRange::~HsaSVMRange() {
    if (m_pUser != NULL) {
        if (m_SelfAllocated)
            munmap(m_pUser, m_Size);
        m_pUser = NULL;
    }
}

void HsaSVMRange::Fill(HSAuint32 value, HSAuint64 offset, HSAuint64 size) {
    HSAuint64 i;
    HSAuint32 *ptr = NULL;

    size = size ? size : m_Size;
    EXPECT_EQ((size & (sizeof(HSAuint32) - 1)), 0) << "Not word aligned. Call Fill(unsigned char)";
    ASSERT_TRUE(size + offset <= m_Size) << "Buffer Overflow" << std::endl;

    if (m_pUser != NULL)
        ptr = reinterpret_cast<HSAuint32 *>(reinterpret_cast<char *>(m_pUser) + offset);

    ASSERT_NOTNULL(ptr);

    for (i = 0; i < size / sizeof(HSAuint32); i++)
        ptr[i] = value;
}
