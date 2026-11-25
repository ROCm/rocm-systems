/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// TopoCpuArch: ncclTopoGetXmlCpuArch() (src/graph/topo.cc, NCCL 2.32.3 a14bb946) takes the CPU
// architecture from the first <cpu> of the topology XML that has an "arch" attribute. It used to
// read only the first <cpu> and fail comm init when that one had none. RCCL fills "arch" in only
// on CPUs it reaches from the GPU or a PCI NIC, so a <cpu> listed in NCCL_TOPO_FILE can lack it.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

// A mkstemp() file, removed when this goes out of scope.
struct ScopedTempFile {
  explicit ScopedTempFile(const char* pathTemplate) : path(pathTemplate) {
    int fd = mkstemp(path.data());
    valid = fd >= 0;
    if (valid) close(fd);
  }
  ~ScopedTempFile() {
    if (valid) unlink(path.c_str());
  }
  std::string path;
  bool valid = false;
};

// The leading <cpu> (numaid 98, no real NUMA node) holds nothing, so RCCL never fills it in and
// trims it before building the system. lo sits under a complete second <cpu>, where RCCL finds it
// by name: a NIC without a PCI device, like lo, would otherwise be attached to the first <cpu>.
std::string topoXml(const char* leadingCpuAttrs) {
  return std::string("<system version=\"2\">\n  <cpu numaid=\"98\"") + leadingCpuAttrs + "/>\n" +
         "  <cpu numaid=\"99\" arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"25\" modelid=\"17\">\n"
         "    <nic>\n      <net name=\"lo\"/>\n    </nic>\n  </cpu>\n</system>\n";
}

// Inits a 1-rank communicator on a topology file whose leading <cpu> carries leadingCpuAttrs.
void initWithLeadingCpu(const char* leadingCpuAttrs) {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
    GTEST_SKIP() << "requires at least one GPU";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  ScopedTempFile topoFile("/tmp/rccl_topo_cpuarch_XXXXXX");
  ScopedTempFile dumpFile("/tmp/rccl_topo_cpuarch_dump_XXXXXX");
  ASSERT_TRUE(topoFile.valid && dumpFile.valid) << "failed to create temp files";
  std::ofstream(topoFile.path) << topoXml(leadingCpuAttrs);

  // Socket on lo is then the only NIC: AMD GIN backends and external plugins can report devices
  // without a PCI path, which would land on the leading <cpu>.
  const char* env[][2] = {{"NCCL_TOPO_FILE", topoFile.path.c_str()},
                          {"NCCL_TOPO_DUMP_FILE", dumpFile.path.c_str()},
                          {"NCCL_NET", "Socket"},      {"NCCL_SOCKET_IFNAME", "lo"},
                          {"NCCL_IB_DISABLE", "1"},    {"NCCL_GIN_ENABLE", "0"},
                          {"NCCL_NET_PLUGIN", "none"}, {"NCCL_RMA_PLUGIN", "none"}};
  for (auto& kv : env) ASSERT_EQ(setenv(kv[0], kv[1], 1), 0) << kv[0];

  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ncclComm_t comm = nullptr;
  ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess)
      << "comm init failed on a topology whose leading <cpu> is <cpu numaid=\"98\"" << leadingCpuAttrs << "/>";

  // Only the file has numaid 99, so this rules out a pass on an auto-detected topology.
  std::stringstream dump;
  dump << std::ifstream(dumpFile.path).rdbuf();
  EXPECT_NE(dump.str().find("numaid=\"99\""), std::string::npos) << "RCCL did not load NCCL_TOPO_FILE";
  EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

} // namespace

TEST(TopoCpuArch, InitScansPastLeadingCpuWithoutArch) {
  RUN_ISOLATED_TEST("TopoCpuArch.InitScansPastLeadingCpuWithoutArch", []() { initWithLeadingCpu(""); });
}

// Control: the same topology with "arch" on the leading <cpu>, which a first-<cpu>-only lookup accepts too.
TEST(TopoCpuArch, InitWithArchOnLeadingCpu) {
  RUN_ISOLATED_TEST("TopoCpuArch.InitWithArchOnLeadingCpu", []() { initWithLeadingCpu(" arch=\"x86_64\""); });
}

} // namespace RcclUnitTesting
