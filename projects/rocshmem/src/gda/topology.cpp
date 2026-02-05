/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *****************************************************************************/

#include "topology.hpp"
#include "ibv_wrapper.hpp"
#include "numa_wrapper.hpp"

using namespace rocshmem;

namespace rocshmem
{

  const char* GidPriorityStr[] = {
    "RoCEv1 Link-local",
    "RoCEv2 Link-local",
    "RoCEv1 IPv6",
    "RoCEv2 IPv6",
    "RoCEv1 IPv4-mapped IPv6",
    "RoCEv2 IPv4-mapped IPv6"
  };

  // Check that CPU memory array of numBytes has been allocated on targetId NUMA node
  static int CheckPages(char* array, size_t numBytes, int targetId)
  {
    size_t const pageSize = getpagesize();
    size_t const numPages = (numBytes + pageSize - 1) / pageSize;

    std::vector<void *> pages(numPages);
    std::vector<int> status(numPages);

    pages[0] = array;
    for (int i = 1; i < numPages; i++) {
      pages[i] = (char*)pages[i-1] + pageSize;
    }

    long const retCode = numa.move_pages(0, numPages, pages.data(), NULL, status.data(), 0);
    if (retCode) {
      fprintf(stderr,"Unable to collect page table information for allocated memory. "
              "Ensure NUMA library is installed properly");
      return -1;
    }

    size_t mistakeCount = 0;
    for (size_t i = 0; i < numPages; i++) {
      if (status[i] < 0) {
        fprintf(stderr, "Unexpected page status (%d) for page %zu", status[i], i);
        return -1;
      }
      if (status[i] != targetId) mistakeCount++;
    }
    if (mistakeCount > 0) {
      fprintf(stderr, "%lu out of %lu pages for memory allocation were not on NUMA node %d."
              " This could be due to hardware memory issues, or the use of numa-rebalancing daemons such as numad",
              mistakeCount, numPages, targetId);
      return -1;
    }
    return ROCSHMEM_SUCCESS;
  }

  // Allocate memory
  static int AllocateMemory(MemDevice memDevice, size_t numBytes, void** memPtr)
  {
    if (numBytes == 0) {
      fprintf(stderr, "Unable to allocate 0 bytes");
      return -1;
    }
    *memPtr = nullptr;

    MemType const& memType = memDevice.memType;

    if (IsCpuMemType(memType)) {
      // Set numa policy prior to call to hipHostMalloc
      numa.set_preferred(memDevice.memIndex);

      // Allocate host-pinned memory (should respect NUMA mem policy)
      CHECK_HIP(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser | hipHostMallocNonCoherent));

      // Check that the allocated pages are actually on the correct NUMA node
      memset(*memPtr, 0, numBytes);
      ERR_CHECK(CheckPages((char*)*memPtr, numBytes, memDevice.memIndex));
      // Reset to default numa mem policy
      numa.set_preferred(-1);
    } else if (IsGpuMemType(memType)) {
      int prev_dev;
      CHECK_HIP(hipGetDevice(&prev_dev));

      // Switch to the appropriate GPU
      CHECK_HIP(hipSetDevice(memDevice.memIndex));

      // Allocate GPU memory on appropriate device
      CHECK_HIP(hipMalloc((void**)memPtr, numBytes));

      // Clear the memory
      CHECK_HIP(hipMemset(*memPtr, 0, numBytes));
      CHECK_HIP(hipDeviceSynchronize());

      // Reset to original GPU
      CHECK_HIP(hipSetDevice(prev_dev));
    } else {
      printf("Unsupported memory type (%d)", memType);
      return -1;
    }
    return ROCSHMEM_SUCCESS;
  }

  // Deallocate memory
  static int DeallocateMemory(MemType memType, void *memPtr, size_t const bytes)
  {
    // Avoid deallocating nullptr
    if (memPtr == nullptr) {
      fprintf(stderr, "Attempted to free null pointer for %lu bytes", bytes);
      return -1;
    }

    switch (memType) {
    case MEM_CPU:
      {
        CHECK_HIP(hipHostFree(memPtr));
        break;
      }
    case MEM_GPU:
      {
        CHECK_HIP(hipFree(memPtr));
        break;
      }
    default:
      fprintf(stderr, "Attempting to deallocate unrecognized memory type (%d)", memType);
      return -1;
    }
    return ROCSHMEM_SUCCESS;
  }


  // HSA-related functions
  //========================================================================================

  static int GetHsaAgent(ExeDevice const& exeDevice, hsa_agent_t& agent)
  {
    static bool isInitialized = false;
    static std::vector<hsa_agent_t> cpuAgents;
    static std::vector<hsa_agent_t> gpuAgents;

    int const& exeIndex = exeDevice.exeIndex;
    int const numCpus   = GetNumDevices(EXE_CPU);
    int const numGpus   = GetNumDevices(EXE_GPU);

    // Initialize results on first use
    if (!isInitialized) {
      hsa_amd_pointer_info_t info;
      info.size = sizeof(info);

      int err;
      int32_t* tempBuffer;

      // Index CPU agents
      cpuAgents.clear();
      for (int i = 0; i < numCpus; i++) {
        ERR_CHECK(AllocateMemory({MEM_CPU, i}, 1024, (void**)&tempBuffer));
        CHECK_HSA(hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL));
        cpuAgents.push_back(info.agentOwner);
        ERR_CHECK(DeallocateMemory(MEM_CPU, tempBuffer, 1024));
      }

      // Index GPU agents
      gpuAgents.clear();
      for (int i = 0; i < numGpus; i++) {
        ERR_CHECK(AllocateMemory({MEM_GPU, i}, 1024, (void**)&tempBuffer));
        CHECK_HSA(hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL));
        gpuAgents.push_back(info.agentOwner);
        ERR_CHECK(DeallocateMemory(MEM_GPU, tempBuffer, 1024));
      }
      isInitialized = true;
    }

    switch (exeDevice.exeType) {
    case EXE_CPU:
      if (exeIndex < 0 || exeIndex >= numCpus) {
        fprintf(stderr, "CPU index must be between 0 and %d inclusively", numCpus - 1);
        return -1;
      }
      agent = cpuAgents[exeDevice.exeIndex];
      break;
    case EXE_GPU:
      if (exeIndex < 0 || exeIndex >= numGpus) {
        fprintf(stderr, "GPU index must be between 0 and %d inclusively", numGpus - 1);
        return -1;
      }
      agent = gpuAgents[exeIndex];
      break;
    default:
      fprintf(stderr, "Attempting to get HSA agent of unknown or unsupported executor type (%d)",
             exeDevice.exeType);
      return -1;
    }
    return ROCSHMEM_SUCCESS;
  }

  // Get the hsa_agent_t associated with a MemDevice
  static int GetHsaAgent(MemDevice const& memDevice, hsa_agent_t& agent)
  {
    if (IsCpuMemType(memDevice.memType)) return GetHsaAgent({EXE_CPU, memDevice.memIndex}, agent);
    if (IsGpuMemType(memDevice.memType)) return GetHsaAgent({EXE_GPU, memDevice.memIndex}, agent);

    fprintf(stderr, "Unable to get HSA agent for memDevice (%d,%d)",
           memDevice.memType, memDevice.memIndex);
    return -1;
  }

  // Structure to track PCIe topology
  struct PCIeNode
  {
    std::string        address;                   ///< PCIe address for this PCIe node
    std::string        description;               ///< Description for this PCIe node
    std::set<PCIeNode> children;                  ///< Children PCIe nodes

    // Default constructor
    PCIeNode() : address(""), description("") {}

    // Constructor
    PCIeNode(std::string const& addr) : address(addr) {}

    // Constructor
    PCIeNode(std::string const& addr, std::string const& desc)
      :address(addr), description(desc) {}

    // Comparison operator for std::set
    bool operator<(PCIeNode const& other) const {
      return address < other.address;
    }
  };

  // Structure to track information about IBV devices
  struct IbvDevice
  {
    ibv_device* devicePtr;
    std::string name;
    std::string busId;
    bool        hasActivePort;
    int         numaNode;
    int         gidIndex;
    std::string gidDescriptor;
    bool        isRoce;
  };

  // Function to collect information about IBV devices
  //========================================================================================
  static bool IsConfiguredGid(union ibv_gid const& gid)
  {
    const struct in6_addr *a = (struct in6_addr *) gid.raw;
    int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
    if (((a->s6_addr32[0] | trailer) == 0UL) ||
        ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
      return false;
    }
    return true;
  }

  static bool LinkLocalGid(union ibv_gid const& gid)
  {
    const struct in6_addr *a = (struct in6_addr *) gid.raw;
    if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
      return true;
    }
    return false;
  }

  static int GetRoceVersionNumber(struct ibv_context* const& context,
                                  int const&  portNum,
                                  int const&  gidIndex,
                                  int&        version)
  {
    char const* deviceName = ibv.get_device_name(context->device);
    char gidRoceVerStr[16]      = {};
    char roceTypePath[PATH_MAX] = {};
    sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
            deviceName, portNum, gidIndex);

    int fd = open(roceTypePath, O_RDONLY);
    if (fd == -1) {
      fprintf(stderr, "Failed while opening RoCE file path (%s)", roceTypePath);
      return -1;
    }

    int ret = read(fd, gidRoceVerStr, 15);
    close(fd);

    if (ret == -1) {
      fprintf(stderr, "Failed while reading RoCE version");
      return -1;
    }

    if (strlen(gidRoceVerStr)) {
      if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0
          || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
        version = 1;
      }
      else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
        version = 2;
      }
    }
    return ROCSHMEM_SUCCESS;
  }

  static bool IsIPv4MappedIPv6(const union ibv_gid &gid)
  {
    // look for ::ffff:x.x.x.x format
    // From Broadcom documentation
    // https://techdocs.broadcom.com/us/en/storage-and-ethernet-connectivity/ethernet-nic-controllers/bcm957xxx/adapters/frequently-asked-questions1.html
    // "The IPv4 address is really an IPv4 address mapped into the IPv6 address space.
    // This can be identified by 80 “0” bits, followed by 16 “1” bits (“FFFF” in hexadecimal)
    // followed by the original 32-bit IPv4 address."
    return (gid.global.subnet_prefix == 0    &&
            gid.raw[8]               == 0    &&
            gid.raw[9]               == 0    &&
            gid.raw[10]              == 0xff &&
            gid.raw[11]              == 0xff);
  }

  static int GetGidIndex(struct ibv_context*          context,
                         int const&                   gidTblLen,
                         int const&                   portNum,
                         std::pair<int, std::string>& gidInfo)
  {
    if(gidInfo.first >= 0) return ROCSHMEM_SUCCESS; // honor user choice
    union ibv_gid gid;

    GidPriority highestPriority = GidPriority::UNKNOWN;
    int gidIndex = -1;

    for (int i = 0; i < gidTblLen; ++i) {
      IBV_CALL(ibv.query_gid, context, portNum, i, &gid);
      if (!IsConfiguredGid(gid)) continue;
      int gidCurrRoceVersion;
      if(GetRoceVersionNumber(context, portNum, i, gidCurrRoceVersion) != ROCSHMEM_SUCCESS) continue;
      GidPriority currPriority;
      if (IsIPv4MappedIPv6(gid)) {
        currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_IPV4 : GidPriority::ROCEV1_IPV4;
      } else if (!LinkLocalGid(gid)) {
        currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_IPV6 : GidPriority::ROCEV1_IPV6;
      } else {
        currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_LINK_LOCAL : GidPriority::ROCEV1_LINK_LOCAL;
      }
      if(currPriority > highestPriority) {
        highestPriority = currPriority;
        gidIndex = i;
      }
    }

    if (highestPriority == GidPriority::UNKNOWN) {
      gidInfo.first = -1;
      fprintf(stderr, "Failed to auto-detect a valid GID index. Try setting it manually through IB_GID_INDEX");
      return -1;
    }
    gidInfo.first = gidIndex;
    gidInfo.second = GidPriorityStr[highestPriority];
    return ROCSHMEM_SUCCESS;
  }

  static vector<IbvDevice>& GetIbvDeviceList()
  {
    static bool isInitialized = false;
    static vector<IbvDevice> ibvDeviceList = {};

    // Build list on first use
    if (!isInitialized) {

      // Query the number of IBV devices
      int numIbvDevices = 0;
      ibv_device** deviceList = ibv.get_device_list(&numIbvDevices);
      CHECK_NNULL(deviceList, "ibv_get_device_list");

      if (numIbvDevices > 0) {
        // Loop over each device to collect information
        for (int i = 0; i < numIbvDevices; i++) {
          IbvDevice ibvDevice;
          ibvDevice.devicePtr = deviceList[i];
          ibvDevice.name = deviceList[i]->name;
          ibvDevice.hasActivePort = false;
          {
            struct ibv_context *context = ibv.open_device(ibvDevice.devicePtr);
            if (context) {
              struct ibv_device_attr deviceAttr;
              if (!ibv.query_device(context, &deviceAttr)) {
                int activePort;
                ibvDevice.gidIndex = -1;
                for (int port = 1; port <= deviceAttr.phys_port_cnt; ++port) {
                  struct ibv_port_attr portAttr;
                  if (ibv.query_port(context, port, &portAttr)) continue;
                  if (portAttr.state == IBV_PORT_ACTIVE) {
                    activePort = port;
                    ibvDevice.hasActivePort = true;
                    if(portAttr.link_layer == IBV_LINK_LAYER_ETHERNET) {
                      ibvDevice.isRoce = true;
                      std::pair<int, std::string> gidInfo (-1, "");
                      auto res = GetGidIndex(context, portAttr.gid_tbl_len, activePort, gidInfo);
                      if (res == ROCSHMEM_SUCCESS) {
                        ibvDevice.gidIndex = gidInfo.first;
                        ibvDevice.gidDescriptor = gidInfo.second;
                      }
                    }
                    break;
                  }
                }
              }
              ibv.close_device(context);
            }
          }
          ibvDevice.busId = "";
          {
            std::string device_path(ibvDevice.devicePtr->dev_path);
            if (std::filesystem::exists(device_path)) {
              std::string pciPath = std::filesystem::canonical(device_path + "/device").string();
              std::size_t pos = pciPath.find_last_of('/');
              if (pos != std::string::npos) {
                ibvDevice.busId = pciPath.substr(pos + 1);
              }
            }
          }

          // Get nearest numa node for this device
          ibvDevice.numaNode = -1;
          std::filesystem::path devicePath = "/sys/bus/pci/devices/" + ibvDevice.busId + "/numa_node";
          std::string canonicalPath = std::filesystem::canonical(devicePath).string();

          if (std::filesystem::exists(canonicalPath)) {
            std::ifstream file(canonicalPath);
            if (file.is_open()) {
              std::string numaNodeStr;
              std::getline(file, numaNodeStr);
              int numaNodeVal;
              if (sscanf(numaNodeStr.c_str(), "%d", &numaNodeVal) == 1)
                ibvDevice.numaNode = numaNodeVal;
              file.close();
            }
          }
          ibvDeviceList.push_back(ibvDevice);
        }
      } else {
        fprintf(stderr, "[Error] No visible InfiniBand devices found.\n");
        exit(1);
      }
      ibv.free_device_list(deviceList);
      isInitialized = true;
    }
    return ibvDeviceList;
  }

  // PCIe-related functions
  //========================================================================================

  // Prints off PCIe tree
  static void PrintPCIeTree(PCIeNode    const& node,
                            std::string const& prefix = "",
                            bool               isLast = true)
  {
    if (!node.address.empty()) {
      printf("%s%s%s", prefix.c_str(), (isLast ? "└── " : "├── "), node.address.c_str());
      if (!node.description.empty()) {
        printf("(%s)", node.description.c_str());
      }
      printf("\n");
    }
    auto const& children = node.children;
    for (auto it = children.begin(); it != children.end(); ++it) {
      PrintPCIeTree(*it, prefix + (isLast ? "    " : "│   "), std::next(it) == children.end());
    }
  }

  // Inserts nodes along pcieAddress down a tree starting from root
  static int InsertPCIePathToTree(std::string const& pcieAddress,
                                  std::string const& description,
                                  PCIeNode&          root)
  {
    std::filesystem::path devicePath = "/sys/bus/pci/devices/" + pcieAddress;
    std::string canonicalPath = std::filesystem::canonical(devicePath).string();

    if (!std::filesystem::exists(devicePath)) {
      fprintf(stderr, "Device path %s does not exist", devicePath.c_str());
      return -1;
    }

    std::istringstream iss(canonicalPath);
    std::string token;

    PCIeNode* currNode = &root;
    while (std::getline(iss, token, '/')) {
      auto it = (currNode->children.insert(PCIeNode(token))).first;
      currNode = const_cast<PCIeNode*>(&(*it));
    }
    currNode->description = description;

    return ROCSHMEM_SUCCESS;
  }

  // Returns root node for PCIe tree.  Constructed on first use
  static PCIeNode* GetPCIeTreeRoot()
  {
    static bool isInitialized = false;
    static PCIeNode pcieRoot;

    // Build PCIe tree on first use
    if (!isInitialized) {
      // Add NICs to the tree
      int numNics = rocshmem::GetNumDevices(rocshmem::EXE_NIC);
      auto const& ibvDeviceList = rocshmem::GetIbvDeviceList();
      for (IbvDevice const& ibvDevice : ibvDeviceList) {
        if (!ibvDevice.hasActivePort || ibvDevice.busId == "") continue;
        InsertPCIePathToTree(ibvDevice.busId, ibvDevice.name, pcieRoot);
      }

      // Add GPUs to the tree
      int numGpus = rocshmem::GetNumDevices(rocshmem::EXE_GPU);
      for (int i = 0; i < numGpus; ++i) {
        char hipPciBusId[64];
        if (hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i) == hipSuccess) {
          InsertPCIePathToTree(hipPciBusId, "GPU " + std::to_string(i), pcieRoot);
        }
      }
#ifdef VERBS_DEBUG
      PrintPCIeTree(pcieRoot);
#endif
      isInitialized = true;
    }
    return &pcieRoot;
  }

  // Finds the lowest common ancestor in PCIe tree between two nodes
  static PCIeNode const* GetLcaBetweenNodes(PCIeNode    const* root,
                                            std::string const& node1Address,
                                            std::string const& node2Address)
  {
    if (!root || root->address == node1Address || root->address == node2Address)
      return root;

    PCIeNode const* lcaFound1 = nullptr;
    PCIeNode const* lcaFound2 = nullptr;

    // Recursively iterate over children
    for (auto const& child : root->children) {
      PCIeNode const* lca = GetLcaBetweenNodes(&child, node1Address, node2Address);
      if (!lca) continue;
      if (!lcaFound1) {
        // First time found
        lcaFound1 = lca;
      } else {
        // Second time found
        lcaFound2 = lca;
        break;
      }
    }

    // If two children were found, then current node is the lowest common ancestor
    return (lcaFound1 && lcaFound2) ? root : lcaFound1;
  }

  // Gets the depth of an node in the PCIe tree
  static int GetLcaDepth(std::string const&     targetBusID,
                         PCIeNode const* const& node,
                         int                    depth = 0)
  {
    if (!node) return -1;
    if (targetBusID == node->address) return depth;

    for (auto const& child : node->children) {
      int distance = GetLcaDepth(targetBusID, &child, depth + 1);
      if (distance != -1)
        return distance;
    }
    return -1;
  }

  // Function to extract the bus number from a PCIe address (domain:bus:device.function)
  static int ExtractBusNumber(std::string const& pcieAddress)
  {
    int domain, bus, device, function;
    char delimiter;

    std::istringstream iss(pcieAddress);
    iss >> std::hex >> domain >> delimiter >> bus >> delimiter >> device >> delimiter >> function;
    if (iss.fail()) {
#ifdef VERBS_DEBUG
      printf("Invalid PCIe address format: %s\n", pcieAddress.c_str());
#endif
      return -1;
    }
    return bus;
  }

  // Function to compute the distance between two bus IDs
  static int GetBusIdDistance(std::string const& pcieAddress1,
                              std::string const& pcieAddress2)
  {
    int bus1 = ExtractBusNumber(pcieAddress1);
    int bus2 = ExtractBusNumber(pcieAddress2);
    return (bus1 < 0 || bus2 < 0) ? -1 : std::abs(bus1 - bus2);
  }

  // Given a target busID and a set of candidate devices, returns a set of indices
  // that is "closest" to the target
  static std::set<int> GetNearestDevicesInTree(std::string              const& targetBusId,
                                               std::vector<std::string> const& candidateBusIdList)
  {
    int maxDepth = -1;
    int minDistance = std::numeric_limits<int>::max();
    std::set<int> matches = {};

    // Loop over the candidates to find the ones with the lowest common ancestor (LCA)
    for (int i = 0; i < candidateBusIdList.size(); i++) {
      std::string const& candidateBusId = candidateBusIdList[i];
      if (candidateBusId == "") continue;
      PCIeNode const* lca = GetLcaBetweenNodes(GetPCIeTreeRoot(), targetBusId, candidateBusId);
      if (!lca) continue;

      int depth = GetLcaDepth(lca->address, GetPCIeTreeRoot());
      int currDistance = GetBusIdDistance(targetBusId, candidateBusId);

      // When more than one LCA match is found, choose the one with smallest busId difference
      // NOTE: currDistance could be -1, which signals problem with parsing, however still
      //       remains a valid "closest" candidate, so is included
      if (depth > maxDepth || (depth == maxDepth && depth >= 0 && currDistance < minDistance)) {
        maxDepth = depth;
        matches.clear();
        matches.insert(i);
        minDistance = currDistance;
      } else if (depth == maxDepth && depth >= 0 && currDistance == minDistance) {
        matches.insert(i);
      }
    }
    return matches;
  }

  int GetNumDevices(DeviceType exeType)
  {
    switch (exeType) {
    case rocshmem::EXE_CPU:
      return numa.num_configured_nodes();
    case rocshmem::EXE_GPU:
      {
        int numDetectedGpus = 0;
        hipError_t status = hipGetDeviceCount(&numDetectedGpus);
        if (status != hipSuccess) numDetectedGpus = 0;
        return numDetectedGpus;
      }
    case rocshmem::EXE_NIC:
      {
        return GetIbvDeviceList().size();
      }
    default:
      return 0;
    }
  }

  int GetClosestCpuNumaToGpu(int gpuIndex)
  {
    hsa_agent_t gpuAgent;
    ERR_CHECK(GetHsaAgent({EXE_GPU, gpuIndex}, gpuAgent));

    hsa_agent_t closestCpuAgent;
    if (hsa_agent_get_info(gpuAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU, &closestCpuAgent)
        == HSA_STATUS_SUCCESS) {
      int numCpus = GetNumDevices(EXE_CPU);
      for (int i = 0; i < numCpus; i++) {
        hsa_agent_t cpuAgent;
        ERR_CHECK(GetHsaAgent({EXE_CPU, i}, cpuAgent));
        if (cpuAgent.handle == closestCpuAgent.handle) return i;
      }
    }
    return -1;
  }

  int GetClosestCpuNumaToNic(int nicIndex)
  {
    int numNics = GetNumDevices(rocshmem::EXE_NIC);
    if (nicIndex < 0 || nicIndex >= numNics) return -1;
    return GetIbvDeviceList()[nicIndex].numaNode;
  }


  int GetClosestNicToGpu(int gpuIndex, const char** dev_name)
  {
    static bool isInitialized = false;
    static std::vector<int> closestNicId;
    static auto const& ibvDeviceList = GetIbvDeviceList();

    int numGpus = GetNumDevices(rocshmem::EXE_GPU);
    if (gpuIndex < 0 || gpuIndex >= numGpus) return -1;

    // Build closest NICs per GPU on first use
    if (!isInitialized) {
      closestNicId.resize(numGpus, -1);

      // Build up list of NIC bus addresses
      std::vector<std::string> ibvAddressList;
      for (auto const& ibvDevice : ibvDeviceList)
        ibvAddressList.push_back(ibvDevice.hasActivePort ? ibvDevice.busId : "");

      // Track how many times a device has been assigned as "closest"
      // This allows distributed work across devices using multiple ports (sharing the same busID)
      // NOTE: This isn't necessarily optimal, but likely to work in most cases involving multi-port
      // Counter example:
      //
      //  G0 prefers (N0,N1), picks N0
      //  G1 prefers (N1,N2), picks N1
      //  G2 prefers N0,      picks N0
      //
      //  instead of G0->N1, G1->N2, G2->N0

      std::vector<int> assignedCount(ibvDeviceList.size(), 0);

      // Loop over each GPU to find the closest NIC(s) based on PCIe address
      for (int i = 0; i < numGpus; i++) {
        // Collect PCIe address for the GPU
        char hipPciBusId[64];
        hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i);
        if (err != hipSuccess) {
#ifdef VERBS_DEBUG
          printf("Failed to get PCI Bus ID for HIP device %d: %s\n", i, hipGetErrorString(err));
#endif
          closestNicId[i] = -1;
          continue;
        }

        // Find closest NICs
        std::set<int> closestNicIdxs = GetNearestDevicesInTree(hipPciBusId, ibvAddressList);

        // Pick the least-used NIC to assign as closest
        int closestIdx = -1;
        for (auto idx : closestNicIdxs) {
          if (closestIdx == -1 || assignedCount[idx] < assignedCount[closestIdx])
            closestIdx = idx;
        }

        // The following will only use distance between bus IDs
        // to determine the closest NIC to GPU if the PCIe tree approach fails
        if (closestIdx < 0) {
#ifdef VERBS_DEBUG
          printf("[WARN] Falling back to PCIe bus ID distance to determine proximity\n");
#endif

          int minDistance = std::numeric_limits<int>::max();
          for (int j = 0; j < ibvDeviceList.size(); j++) {
            if (ibvDeviceList[j].busId != "") {
              int distance = GetBusIdDistance(hipPciBusId, ibvDeviceList[j].busId);
              if (distance < minDistance && distance >= 0) {
                minDistance = distance;
                closestIdx = j;
              }
            }
          }
        }
        closestNicId[i] = closestIdx;
        if (closestIdx != -1) assignedCount[closestIdx]++;
      }
      isInitialized = true;
    }

    DPRINTF("GPU Device id: %d closest NIC id : %d name: %s\n", gpuIndex, closestNicId[gpuIndex],
           ibvDeviceList[closestNicId[gpuIndex]].name.c_str());
    if (dev_name != nullptr) {
      *dev_name = strdup(ibvDeviceList[closestNicId[gpuIndex]].name.c_str());
    }

    return closestNicId[gpuIndex];
  }

  static int RemappedCpuIndex(int origIdx)
  {
    static std::vector<int> remappingCpu;

    // Build CPU remapping on first use
    // Skip numa nodes that are not configured
    if (remappingCpu.empty()) {
      for (int node = 0; node <= numa.max_node(); node++)
        if (numa.bitmask_isbitset(numa.get_mems_allowed(), node))
          remappingCpu.push_back(node);
    }
    return remappingCpu[origIdx];
  }

  static void PrintNicToGPUTopo(bool outputToCsv)
  {
    printf(" NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | GID Index | GID Descriptor\n");
    if(!outputToCsv)
      printf("-----+-------------+--------+--------------+------+----------------+-----------+-------------------\n");

    int numGpus = rocshmem::GetNumDevices(rocshmem::EXE_GPU);
    auto const& ibvDeviceList = rocshmem::GetIbvDeviceList();
    for (int i = 0; i < ibvDeviceList.size(); i++) {

      std::string closestGpusStr = "";
      for (int j = 0; j < numGpus; j++) {
        if (rocshmem::GetClosestNicToGpu(j, nullptr) == i) {
          if (closestGpusStr != "") closestGpusStr += ",";
          closestGpusStr += std::to_string(j);
        }
      }

      printf(" %-3d | %-11s | %-6s | %-12s | %-4d | %-14s | %-9s | %-20s\n",
             i, ibvDeviceList[i].name.c_str(),
             ibvDeviceList[i].hasActivePort ? "Yes" : "No",
             ibvDeviceList[i].busId.c_str(),
             ibvDeviceList[i].numaNode,
             closestGpusStr.c_str(),
             ibvDeviceList[i].isRoce && ibvDeviceList[i].hasActivePort?  std::to_string(ibvDeviceList[i].gidIndex).c_str() : "N/A",
             ibvDeviceList[i].isRoce && ibvDeviceList[i].hasActivePort?  ibvDeviceList[i].gidDescriptor.c_str() : "N/A"
             );
    }
    printf("\n");
  }

  void DisplayTopology(bool outputToCsv)
  {
    int numCpus = rocshmem::GetNumDevices(rocshmem::EXE_CPU);
    int numGpus = rocshmem::GetNumDevices(rocshmem::EXE_GPU);
    int numNics = rocshmem::GetNumDevices(rocshmem::EXE_NIC);
    char sep = (outputToCsv ? ',' : '|');

    if (outputToCsv) {
      printf("NumCpus,%d\n", numCpus);
      printf("NumGpus,%d\n", numGpus);
      printf("NumNics,%d\n", numNics);
    } else {
      printf("\nDetected Topology:\n");
      printf("==================\n");
      printf("  %d configured CPU NUMA node(s) [%d total]\n", numCpus, numa.max_node() + 1);
      printf("  %d GPU device(s)\n", numGpus);
      printf("  %d Supported NIC device(s)\n", numNics);
    }

    // Print out detected CPU topology
    printf("\n            %c", sep);
    for (int j = 0; j < numCpus; j++)
      printf("NUMA %02d%c", j, sep);
    printf(" #Cpus %c Closest GPU(s)\n", sep);

    if (!outputToCsv) {
      printf("------------+");
      for (int j = 0; j <= numCpus; j++)
        printf("-------+");
      printf("---------------\n");
    }

    for (int i = 0; i < numCpus; i++) {
      int nodeI = RemappedCpuIndex(i);
      printf("NUMA %02d (%02d)%c", i, nodeI, sep);
      for (int j = 0; j < numCpus; j++) {
        int nodeJ = RemappedCpuIndex(j);
        int numaDist = numa.distance(nodeI, nodeJ);
        printf(" %5d %c", numaDist, sep);
      }

      int numCpuCores = 0;
      for (int j = 0; j < numa.num_configured_cpus(); j++)
        if (numa.node_of_cpu(j) == nodeI) numCpuCores++;
      printf(" %5d %c", numCpuCores, sep);

      for (int j = 0; j < numGpus; j++) {
        if (rocshmem::GetClosestCpuNumaToGpu(j) == nodeI) {
          printf(" %d", j);
        }
      }
      printf("\n");
    }
    printf("\n");

    // Print out detected NIC topology
    PrintNicToGPUTopo(outputToCsv);
  }
}
