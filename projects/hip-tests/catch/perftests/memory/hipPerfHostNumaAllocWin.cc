/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipHostMalloc
 * @{
 * @ingroup hipHostMalloc
 * `hipHostMalloc(T** ptr, size_t size, unsigned int flags)` -
 * Allocate pinned host buffer.
 */
#include <hip_test_common.hh>
#include <windows.h>
#include <processtopologyapi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <psapi.h>
#include <tchar.h>

SIZE_T allocSize = 1024 * 1024; // 1 MB
DWORD  pageSize = 0;
struct NumaNodeInfo {
  WORD nodeNumber;
  WORD groupNumber;
  KAFFINITY mask;
  ULONGLONG freeBytes;
};

enum class MallocType {
  hostMallocType = 0,
  hiphostMallocType = 1,
};

struct ThreadPara {
  ThreadPara(NumaNodeInfo* node_ = nullptr, MallocType mallocType_ = MallocType::hostMallocType,
                 unsigned int flags_ = 0, int deviceId_= -1) {
    node = node_;
    mallocType = mallocType_;
    flags = flags_;
    deviceId = deviceId_;
  }
  NumaNodeInfo *node;
  MallocType mallocType;
  unsigned int flags; // flags to allocate buffer
  int deviceId; // for MGPUs test
};

bool checkNumaNodeInfo(const PVOID buffer, const SIZE_T bufferSize, const WORD nodeNumber,
                      bool pinned = false) {
  DWORD_PTR startPage = ((DWORD_PTR)buffer) / pageSize;
  DWORD_PTR endPage = ((DWORD_PTR)buffer + bufferSize - 1) / pageSize;
  DWORD_PTR numPages = (endPage - startPage) + 1;
  PCHAR startPtr = (PCHAR)(pageSize * startPage);
  PPSAPI_WORKING_SET_EX_INFORMATION wsInfo = static_cast<PPSAPI_WORKING_SET_EX_INFORMATION>(
                                    malloc(numPages * sizeof(PSAPI_WORKING_SET_EX_INFORMATION)));

  if (wsInfo == NULL) {
    std::cerr <<"Could not allocate array of PSAPI_WORKING_SET_EX_INFORMATION structures\n";
    return false;
  }

  for (DWORD_PTR i = 0; i < numPages; i++) {
    wsInfo[i].VirtualAddress = startPtr + i * pageSize;
  }

  BOOL bResult = QueryWorkingSetEx(GetCurrentProcess(), wsInfo,
                                    (DWORD)numPages * sizeof(PSAPI_WORKING_SET_EX_INFORMATION));

  if (!bResult) {
    std::cerr <<"QueryWorkingSetEx failed: " << GetLastError() << "\n";
    free(wsInfo);
    return false;
  }
  bool ret = true;
  for (DWORD_PTR i = 0; i < numPages; i++) {
    BOOL  IsValid = wsInfo[i].VirtualAttributes.Valid;
    DWORD Node = wsInfo[i].VirtualAttributes.Node;
    if (pinned) {
      if (!IsValid ) {
        std::cerr << "Page " << i << " is invalid\n";
        ret = false;
        break;
      } else if (nodeNumber != Node) {
        std::cerr << "Page " << i << " has node " << Node << " not matching expected " << nodeNumber << "\n";
        ret = false;
        break;
      }
    } else if (IsValid && nodeNumber != Node) {
        // maybe IsValid = false for unpinned
      std::cerr << "Page " << i << " has node " << Node << " not matching expected " << nodeNumber << "\n";
      ret = false;
      break;
    }
  }
  free(wsInfo);
  return ret;
}

void enumerateNumaNodes(std::vector<NumaNodeInfo> &nodes) {
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationNumaNode, nullptr, &len);
  std::vector<BYTE> buffer(len);
  if (!GetLogicalProcessorInformationEx(RelationNumaNode,
    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
    &len)) {
    std::cerr << "GetLogicalProcessorInformationEx failed. Error: " << GetLastError() << "\n";
    return;
  }

  BYTE* ptr = buffer.data();
  while (ptr < buffer.data() + len) {
    auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
    if (info->Relationship == RelationNumaNode) {
      NUMA_NODE_RELATIONSHIP *numaRelation = &info->NumaNode;
      NumaNodeInfo node{};
      node.nodeNumber = static_cast<WORD>(numaRelation->NodeNumber);
      if (!GetNumaAvailableMemoryNodeEx(node.nodeNumber, &node.freeBytes)) {
        std::cerr << "GetNumaAvailableMemoryNodeEx(" << node.nodeNumber <<
            ") failed with Error: " << GetLastError() << "\n";
        continue;
      }
      if (numaRelation->GroupCount == 0) {
        // Before Windows 20H2
        node.groupNumber = numaRelation->GroupMask.Group;
        node.mask = numaRelation->GroupMask.Mask;
        nodes.push_back(node);
      } else {
        // Since Windows 20H2.
        GROUP_AFFINITY *groupMasks = numaRelation->GroupMasks;
        for (int i = 0; i < numaRelation->GroupCount; i++) {
          node.groupNumber = groupMasks[i].Group;
          node.mask = groupMasks[i].Mask;
          nodes.push_back(node);
        }
      }
    }
    ptr += info->Size;
  }
}

static DWORD WINAPI workerThread(LPVOID lpParam) {
  ThreadPara *threadPara = reinterpret_cast<ThreadPara*>(lpParam);
  NumaNodeInfo* node = threadPara->node;

  PROCESSOR_NUMBER procNumber{};
  GetCurrentProcessorNumberEx(&procNumber);
  std::cout << "Thread is running on processor number: " << static_cast<int>(procNumber.Number)
      << " in group: " << procNumber.Group << std::endl;

  USHORT runNode = -1;
  if (GetNumaProcessorNodeEx(&procNumber, &runNode)) {
    std::cout << "Thread is running on NUMA node " << runNode << "\n";
  }
  else {
    std::cerr << "Failed to get NUMA node. Error: " << GetLastError() << std::endl;
    return -1;
  }

  if (static_cast<USHORT>(node->nodeNumber) != runNode) {
    std::cerr << "runNode " << runNode << "not matching node->nodeNumber " <<
        node->nodeNumber <<"\n";
    return -1;
  }

  if (threadPara->deviceId >= 0) {
    HIP_CHECK(hipSetDevice(threadPara->deviceId)); // doesn't matter AMD_CPU_AFFINITY is 1 or 0
  }
  void* pMem = nullptr;
  switch (threadPara->mallocType) {
    case MallocType::hostMallocType:
      // Place holder
      pMem = VirtualAllocExNuma(GetCurrentProcess(),
        nullptr,
        allocSize,
        threadPara->flags,
        PAGE_READWRITE,
        node->nodeNumber);
      break;
    case MallocType::hiphostMallocType:
      HIP_CHECK(hipHostMalloc(&pMem, allocSize, threadPara->flags));
      break;
    default:
      return -1;
  }

  if (!pMem) {
    std::cerr << "NUMA allocation failed for thread "
          << GetCurrentThreadId() << "on NUMA node " << node->nodeNumber <<
          " with Error: " << GetLastError() << "\n";
    return -1;
  }

  memset(pMem, 0xCD, allocSize);
  bool ret = checkNumaNodeInfo(pMem, allocSize, node->nodeNumber,
                    threadPara->mallocType !=  MallocType::hostMallocType);

  switch (threadPara->mallocType) {
    case MallocType::hostMallocType:
      VirtualFree(pMem, 0, MEM_RELEASE);
      break;
    case MallocType::hiphostMallocType:
      HIP_CHECK(hipHostFree(pMem));
      break;
    default:
      return -1;
  }
  return ret ? 0 : -1;
}

static void runTestPrefered(std::vector<NumaNodeInfo> &nodes, MallocType type, unsigned int flags,
    const char *description) {
  int gpuCount = 0;
  HIP_CHECK(hipGetDeviceCount(&gpuCount));
  std::cout << std::dec;
  std::vector<HANDLE> threadHandles;
  std::vector<ThreadPara> paras;
  paras.reserve(gpuCount * nodes.size());
  int index = 0;
  for (int dev = 0; dev < gpuCount; dev++) {
    int numaNode = -1;
    HIP_CHECK(hipDeviceGetAttribute(&numaNode, hipDeviceAttributeHostNumaId, dev));
    if (numaNode == -1) {
      continue; // Impossible here
    }
    for (auto& node : nodes) {
      if (numaNode != node.nodeNumber) {
        continue;
      }
      if (node.freeBytes < allocSize) {
        std::cerr << "node.freeBytes " << node.freeBytes <<" < allocSize " << allocSize << "\n";
        continue;
      }
      // For best perf, we prefer creating a thread on the host numa node of the gpu device.
      auto& ref = paras.emplace_back(&node, type, flags, dev);
      HANDLE hThread = CreateThread(nullptr, 0, workerThread, &ref, CREATE_SUSPENDED, nullptr);
      if (!hThread) {
        std::cerr << "Thread creation failed. Error: " << GetLastError() << "\n";
        continue;
      }
      GROUP_AFFINITY ga = {};
      ga.Group = node.groupNumber;
      ga.Mask = node.mask;
      GROUP_AFFINITY prev = {};
      if (!SetThreadGroupAffinity(hThread, &ga, &prev)) {
        std::cerr << "SetThreadGroupAffinity failed. Error: " << GetLastError() << "\n";
        CloseHandle(hThread);
        continue;
      }
      std::cout << "dev " << dev << ", thread " << index++ << ": Group: " << ga.Group <<
          ", Mask: " << std::hex << ga.Mask << "; prev: Group: " << std::dec << prev.Group <<
          ", Mask: " << std::hex << prev.Mask << std::dec <<"\n";
      ResumeThread(hThread);
      threadHandles.push_back(hThread);
      // A single NUMA node can span multiple processor groups on systems with more than 64 processors,
      // so we will continue searching for next node of the same nodeNumber.
    }
  }

  // Wait for all threads
  WaitForMultipleObjects((DWORD)threadHandles.size(), threadHandles.data(), TRUE, INFINITE);
  bool result = true;
  for (auto h : threadHandles) {
    DWORD exitCode = 0;
    if (GetExitCodeThread(h, &exitCode)) {
      result &= (exitCode == 0);
    } else {
      result = false;
    }
    CloseHandle(h);
  }

  std::cout << description << (result ? " passed\n" : " failed\n");
  REQUIRE(result);
}

/* Test memory allocation on preferred host numa node on each CPU */
TEST_CASE("Perf_hipPerfHostNumaAlloc_test_preferred_host_numa_node_on_each_GPU") {
  std::vector<NumaNodeInfo> nodes;
  enumerateNumaNodes(nodes);
  if (nodes.empty()) {
    std::cerr << "No NUMA nodes found.\n";
    REQUIRE(false);
  }
  SYSTEM_INFO systemInfo;
  GetSystemInfo(&systemInfo);
  pageSize = systemInfo.dwPageSize;
  std::cout << "logic processor count " << systemInfo.dwNumberOfProcessors
      << ", page size " << pageSize << "\n";
  int numaNode = -1;
  HIP_CHECK(hipDeviceGetAttribute(&numaNode, hipDeviceAttributeHostNumaId, 0));
  if (numaNode == -1) {
    HipTest::HIP_SKIP_TEST("Host NUMA isn't supported hence skipping the test...\n");
    return;
  }
  HIP_CHECK(hipSetDevice(0));
  // In windows, it is the same with / without hipHostMallocNumaUser
  runTestPrefered(nodes,
      MallocType::hiphostMallocType, hipHostMallocDefault | hipHostMallocNumaUser,
      "hiphostMalloc(hipHostMallocDefault | hipHostMallocNumaUser) on preferred numa node");
  runTestPrefered(nodes,
      MallocType::hiphostMallocType, hipHostAllocMapped | hipHostMallocNumaUser,
      "hiphostMalloc(hipHostAllocMapped | hipHostMallocNumaUser) on preferred numa node");
  runTestPrefered(nodes,
      MallocType::hostMallocType, MEM_RESERVE | MEM_COMMIT,
      "VirtualAllocExNuma(MEM_RESERVE | MEM_COMMIT) on preferred numa node");
}
/**
 * End doxygen group hipHostMalloc.
 * @}
 */
