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

#ifndef __KFD__TEST__UTIL__H__
#define __KFD__TEST__UTIL__H__

#include <gtest/gtest.h>
#include <vector>
#include "OSWrapper.hpp"
#include "GoogleTestExtension.hpp"
#include "hsakmt/hsakmt.h"

class BaseQueue;
#define ARRAY_SIZE(_x) (sizeof(_x)/sizeof(_x[0]))
#define ALIGN_UP(x, align) (((uint64_t)(x) + (align) - 1) & ~(uint64_t)((align)-1))
#define CounterToNanoSec(x) ((x) * 1000 / (hsakmt_is_dgpu() ? 27 : 100))

void WaitUntilInput();
HSAKMT_STATUS fscanf_dec(const char *file, uint32_t *num);
uint64_t RoundToPowerOf2(uint64_t val);

// @brief: waits until the value is written to the buffer or until time out if received through args
bool WaitOnValue(const volatile unsigned int *buf, unsigned int value, unsigned int timeOut = g_TestTimeOut);

void SplitU64(const HSAuint64 value, unsigned int& rLoPart, unsigned int& rHiPart);

bool CheckEmuModeEnabled();

bool GetHwCapabilityHWS();

HSAKMT_STATUS CreateQueueTypeEvent(bool ManualReset, bool IsSignaled, unsigned int NodeId, HsaEvent** Event);
HSAKMT_STATUS CreateHWExceptionEvent(bool ManualReset, bool IsSignaled, unsigned int NodeId, HsaEvent** Event);

bool hsakmt_is_dgpu();
bool isTonga(const HsaNodeProperties *props);
bool hasPciAtomicsSupport(int node);
unsigned int FamilyIdFromNode(const HsaNodeProperties *props);
const uint32_t GetGfxVersion(const HsaNodeProperties *props);

void GetHwQueueInfo(const HsaNodeProperties *props,
                 unsigned int *p_num_cp_queues,
                 unsigned int *p_num_sdma_engines,
                 unsigned int *p_num_sdma_xgmi_engines,
                 unsigned int *p_num_sdma_queues_per_engine);

HSAuint64 GetSystemTickCountInMicroSec();

class HsaMemoryBuffer {
 public:
    static const HsaMemoryBuffer Null;

 public:
    HsaMemoryBuffer(HSAuint64 size, unsigned int node, bool zero = true, bool isLocal = false,
                    bool isExec = false, bool isScratch = false, bool isReadOnly = false, bool isUncached = false, bool NonPaged = false);
    HsaMemoryBuffer(void *addr, HSAuint64 size);
    template<typename RetType>
    RetType As() {
        return reinterpret_cast<RetType>(m_pBuf);
    }

    template<typename RetType>
    const RetType As() const {
        return reinterpret_cast<const RetType>(m_pBuf);
    }

    /* Fill @size bytes of buffer with @value starting from @offset
     * If @size is 0, the whole buffer is filled with @value
     */
    void Fill(unsigned char value, HSAuint64 offset = 0, HSAuint64 size = 0);
    void Fill(HSAuint32 value, HSAuint64 offset = 0, HSAuint64 size = 0);
    void Fill(int value, HSAuint64 offset = 0, HSAuint64 size = 0) {
              Fill((HSAuint32)value, offset, size);
    }
    void Fill(HSAuint32 value, BaseQueue& baseQueue,
              HSAuint64 offset = 0, HSAuint64 size = 0);

    bool IsPattern(HSAuint64 location, HSAuint32 pattern);
    bool IsPattern(HSAuint64 location, HSAuint32 pattern,
                   BaseQueue& baseQueue, volatile HSAuint32 *tmp);

    unsigned int Size();
    HsaMemFlags Flags();
    unsigned int Node() const;

    int MapMemToNodes(unsigned int *nodes, unsigned int nodes_num);
    int UnmapMemToNodes(unsigned int *nodes, unsigned int nodes_num);

    void *GetUserPtr() { return m_pUser; }
    bool isLocal() { return m_Local; }
    ~HsaMemoryBuffer();

 private:
    // Disable copy
    HsaMemoryBuffer(const HsaMemoryBuffer&);
    const HsaMemoryBuffer& operator=(const HsaMemoryBuffer&);

    void UnmapAllNodes();
    HsaMemoryBuffer();

 private:
    HsaMemFlags m_Flags;
    HSAuint64 m_Size;
    void* m_pUser;
    void* m_pBuf;
    bool m_Local;
    unsigned int m_Node;
    HSAuint64 m_MappedNodes;
};
HSAKMT_STATUS RegisterSVMRange(HSAuint32 GPUNode, void *MemoryAddress,
                               HSAuint64 SizeInBytes, HSAuint32 PrefetchNode,
                               HSAuint32 SVMFlags);
HSAKMT_STATUS SVMRangeGetPrefetchNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                      HSAuint32 *PrefetchNode);
HSAKMT_STATUS SVMRangePrefetchToNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                     HSAuint32 PrefetchNode);
HSAKMT_STATUS SVMRangeMapToNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                     HSAuint32 NodeID);
HSAKMT_STATUS SVMRangeMapInPlaceToNode(void *MemoryAddress, HSAuint64 SizeInBytes,
                                     HSAuint32 NodeID);
HSAKMT_STATUS SVMRangSetGranularity(void *MemoryAddress, HSAuint64 SizeInBytes,
                                    HSAuint32 Granularity);

class HsaSVMRange {
 public:
    HsaSVMRange(HSAuint64 size, HSAuint32 GPUNode);
    HsaSVMRange(HSAuint64 size, HSAuint32 GPUNode, HSAuint32 PreferredNode);
    HsaSVMRange(HSAuint64 size);
    HsaSVMRange(void *addr, HSAuint64 size, HSAuint32 GPUNode, HSAuint32 PreferredNode = 0,
                bool noRegister = false, bool isLocal = false, bool isExec = false,
                bool isReadOnly = false);
    template<typename RetType>
    RetType As() {
        return reinterpret_cast<RetType>(m_pUser);
    }

    template<typename RetType>
    const RetType As() const {
        return reinterpret_cast<const RetType>(m_pUser);
    }
    ~HsaSVMRange();

    void Fill(HSAuint32 value, HSAuint64 offset = 0, HSAuint64 size = 0);

 private:
    HSAuint32 m_Flags;
    HSAuint64 m_Size;
    void* m_pUser;
    bool m_SelfAllocated;
    bool m_Local;
    unsigned int m_Node;
};

class HsaInteropMemoryBuffer {
 public:
    HsaInteropMemoryBuffer(HSAuint64 device_handle, HSAuint64 buffer_handle, HSAuint64 size, unsigned int node);

    template<typename RetType>
    RetType As() {
        return reinterpret_cast<RetType>(m_pBuf);
    }

    template<typename RetType>
    const RetType As() const {
        return reinterpret_cast<const RetType>(m_pBuf);
    }

    unsigned int Size();

    ~HsaInteropMemoryBuffer();

 private:
    // Disable copy
    HsaInteropMemoryBuffer(const HsaInteropMemoryBuffer&);
    const HsaInteropMemoryBuffer& operator=(const HsaInteropMemoryBuffer&);

 private:
    HSAuint64 m_Size;
    void* m_pBuf;
    HSAuint64 m_graphic_handle;
    unsigned int m_Node;
};

// @class HsaNodeInfo - Gather and store all HSA node information from Thunk.
class HsaNodeInfo {
    // List containing HsaNodeProperties of all Nodes available
    std::vector<HsaNodeProperties*> m_HsaNodeProps;

    // List of HSA Nodes that contain a GPU. This includes both APU and dGPU
    std::vector<int> m_NodesWithGPU;

    // List of HSA Nodes with CPU only
    std::vector<int> m_NodesWithoutGPU;

 public:
    HsaNodeInfo();
    ~HsaNodeInfo();

    bool Init(int NumOfNodes);
    void Delete();

    /* This function should be deprecated soon. This for transistion purpose only
     * Currently, KfdTest is designed to test only ONE node. This function acts
     * as transition.
     */
    const HsaNodeProperties* HsaDefaultGPUNodeProperties() const;
    const int HsaDefaultGPUNode() const;

    /* TODO: Use the following two functions to support multi-GPU.
     * const std::vector<int>& GpuNodes = GetNodesWithGPU()
     * for (..GpuNodes.size()..) GetNodeProperties(GpuNodes.at(i))
     */
    const std::vector<int>& GetNodesWithGPU() const;

    /* get gpu index from gpuNodeID */
    const int HsaGPUindexFromGpuNode(int gpuNodeId) const;

    // @param node index of the node we are looking at
    // @param nodeProperties HsaNodeProperties returned
    const HsaNodeProperties* GetNodeProperties(int NodeNum) const;

    void PrintNodeInfo() const;
    const bool IsGPUNodeLargeBar(int node) const;
    const bool IsAppAPU(int node) const;
    const bool IsPeerAccessibleByNode(int peer, int node) const;
    // @brief Find the first available Large-BAR GPU node
    // @return: Node ID if successful or -1
    const int FindLargeBarGPUNode() const;
    const bool AreGPUNodesXGMI(int node0, int node1) const;
    int FindAccessiblePeers(std::vector<int> *peers,
                                        HSAuint32 node) const;
    /* @brief: to determine if the node is XGMI-linked to CPU
     * @param: node index of the node we are looking at
     * @return: bool true or false
     */
    const bool IsNodeXGMItoCPU(int node) const;
};

#endif  // __KFD__TEST__UTIL__H__
