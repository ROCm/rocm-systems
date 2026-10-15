/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Topology fixtures for rccl-UnitTestsMicroTopo: NCCL_TOPO_FILE-style XML built in the test, parsed and
// turned into a ncclTopoSystem by the real xml.cc/topo.cc, so paths and search run on real input.

#ifndef RCCL_TEST_HOST_TOPO_SCENE_H_
#define RCCL_TEST_HOST_TOPO_SCENE_H_

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "fakes/graph_fakes.h"

#include "comm.h"
#include "graph.h"
#include "graph/topo.h"
#include "graph/xml.h"

// Defined in topo-test.cc, which owns topo.cc's NCCL_NETDEVS_POLICY statics.
void TopoSetNetDevsPolicy(enum netDevsPolicy policy, int num);

namespace topo_scene {

constexpr int kXmlNodes = 512;

struct FreeDeleter {
  void operator()(void* p) const { free(p); }
};
struct SystemDeleter {
  void operator()(ncclTopoSystem* s) const { ncclTopoFree(s); }
};
using XmlPtr = std::unique_ptr<ncclXml, FreeDeleter>;
using SystemPtr = std::unique_ptr<ncclTopoSystem, SystemDeleter>;

inline XmlPtr NewXml(int maxNodes = kXmlNodes) {
  ncclXml* xml = nullptr;
  EXPECT_EQ(ncclSuccess, xmlAlloc(&xml, maxNodes));
  return XmlPtr(xml);
}

// A file in /tmp holding `text`, removed when the object goes away.
class TempFile {
 public:
  explicit TempFile(const std::string& text = "") {
    char tmpl[] = "/tmp/rccl-topo-XXXXXX";
    int fd = mkstemp(tmpl);
    path_ = tmpl;
    if (fd >= 0) {
      if (!text.empty()) EXPECT_EQ((ssize_t)text.size(), write(fd, text.data(), text.size()));
      close(fd);
    }
  }
  ~TempFile() { unlink(path_.c_str()); }
  const char* path() const { return path_.c_str(); }
  std::string Read() const {
    std::string out;
    FILE* f = fopen(path_.c_str(), "r");
    if (f == nullptr) return out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
  }

 private:
  std::string path_;
};

inline ncclResult_t ParseXml(const std::string& text, ncclXml* xml) {
  TempFile file(text);
  return ncclTopoGetXmlFromFile(file.path(), xml, 1);
}

inline XmlPtr MustParseXml(const std::string& text, int maxNodes = kXmlNodes) {
  XmlPtr xml = NewXml(maxNodes);
  EXPECT_EQ(ncclSuccess, ParseXml(text, xml.get())) << text;
  return xml;
}

// --- XML text builders. Extra attributes are passed through verbatim. The parser reads an empty attribute
// for every stray space, so attributes are joined with exactly one.
inline std::string Extra(const std::string& attrs) { return attrs.empty() ? "" : " " + attrs; }

inline std::string System(const std::string& body) { return "<system version=\"2\">\n" + body + "</system>\n"; }

constexpr const char* kAmdCpu = "arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"25\" modelid=\"1\"";
constexpr const char* kRomeCpu = "arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"143\" modelid=\"49\"";
constexpr const char* kIntelCpu = "arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0x55\"";

inline std::string Cpu(int numa, const std::string& body, const std::string& attrs = "", const char* ids = kAmdCpu) {
  return "<cpu numaid=\"" + std::to_string(numa) + "\" " + ids + " affinity=\"0000000f\"" + Extra(attrs) + ">\n" + body +
         "</cpu>\n";
}

inline std::string Pci(const std::string& busId, const std::string& body, const std::string& cls = "0x060400",
                       const std::string& attrs = "") {
  return "<pci busid=\"" + busId + "\" class=\"" + cls + "\" vendor=\"0x1022\" device=\"0x14ab\" " +
         "subsystem_vendor=\"0x1022\" subsystem_device=\"0x14ab\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\"" +
         Extra(attrs) + ">\n" + body + "</pci>\n";
}

inline std::string Xgmi(const std::string& target, int count = 1, const std::string& tclass = "0x120000") {
  return "<xgmi target=\"" + target + "\" count=\"" + std::to_string(count) + "\" tclass=\"" + tclass + "\"/>\n";
}

inline std::string Gpu(int dev, int rank, const std::string& links = "", const std::string& attrs = "",
                       const std::string& gcn = "gfx942") {
  return "<gpu dev=\"" + std::to_string(dev) + "\" sm=\"304\" gcn=\"" + gcn + "\" arch=\"38911\" rank=\"" +
         std::to_string(rank) + "\" gdr=\"1\"" + Extra(attrs) + ">\n" + links + "</gpu>\n";
}

inline std::string GpuPci(const std::string& busId, int dev, int rank, const std::string& links = "",
                          const std::string& gpuAttrs = "", const std::string& gcn = "gfx942") {
  return "<pci busid=\"" + busId + "\" class=\"0x120000\" vendor=\"0x1002\" device=\"0x74a1\" " +
         "subsystem_vendor=\"0x1002\" subsystem_device=\"0x74a1\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\">\n" +
         Gpu(dev, rank, links, gpuAttrs, gcn) + "</pci>\n";
}

inline std::string Net(const std::string& name, int dev, const std::string& attrs = "") {
  return "<net name=\"" + name + "\" dev=\"" + std::to_string(dev) + "\" speed=\"400000\" port=\"1\" " +
         "latency=\"0.000000\" guid=\"0x" + std::to_string(100 + dev) + "\" maxconn=\"131072\" gdr=\"1\" " +
         "rail=\"" + std::to_string(dev) + "\" plane=\"0\"" + Extra(attrs) + "/>\n";
}

// A net/gin/rma device named <tag><dev>, with explicit rail and plane.
inline std::string RailNet(const std::string& tag, int dev, int rail = 0, int plane = 0, int speed = 400000) {
  return "<" + tag + " name=\"" + tag + std::to_string(dev) + "\" dev=\"" + std::to_string(dev) + "\" speed=\"" +
         std::to_string(speed) + "\" guid=\"0x" + std::to_string(dev + 1) + "\" gdr=\"1\" rail=\"" +
         std::to_string(rail) + "\" plane=\"" + std::to_string(plane) + "\"/>\n";
}

inline std::string NicPci(const std::string& busId, const std::string& nets) {
  return "<pci busid=\"" + busId + "\" class=\"0x020000\" vendor=\"0x15b3\" device=\"0x1021\" " +
         "subsystem_vendor=\"0x15b3\" subsystem_device=\"0x0023\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\">\n" +
         "<nic>\n" + nets + "</nic>\n</pci>\n";
}

// Bus id "0000:BB:00.0" of GPU g, NIC n in the canned systems below.
inline std::string GpuBus(int g) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0000:%02x:00.0", 0x10 + 0x10 * g);
  return buf;
}
inline std::string NicBus(int n) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0000:%02x:00.0", 0x11 + 0x10 * n);
  return buf;
}
inline std::string SwitchBus(int g) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0000:%02x:00.0", 0x0f + 0x10 * g);
  return buf;
}

// nGpus gfx942 GPUs, all-to-all XGMI, split evenly over nCpus NUMA nodes. With nics, each GPU shares a
// PCI switch with its own NIC.
inline std::string XgmiSystem(int nGpus, int nCpus = 1, bool nics = false, const std::string& gcn = "gfx942",
                              const std::string& cpuAttrs = "") {
  std::string body;
  int perCpu = nGpus / nCpus;
  for (int c = 0; c < nCpus; c++) {
    std::string cpuBody;
    for (int g = c * perCpu; g < (c + 1) * perCpu; g++) {
      std::string links;
      for (int p = 0; p < nGpus; p++)
        if (p != g) links += Xgmi(GpuBus(p));
      std::string sw = GpuPci(GpuBus(g), g, g, links, "", gcn);
      if (nics) sw += NicPci(NicBus(g), Net("mlx5_" + std::to_string(g), g));
      cpuBody += Pci(SwitchBus(g), sw);
    }
    body += Cpu(c, cpuBody, cpuAttrs);
  }
  return System(body);
}

inline SystemPtr BuildSystem(ncclXml* xml, ncclResult_t expect = ncclSuccess) {
  ncclTopoSystem* system = nullptr;
  EXPECT_EQ(expect, ncclTopoGetSystemFromXml(xml, &system, 0));
  return SystemPtr(system);
}

inline SystemPtr BuildSystem(const std::string& text, ncclResult_t expect = ncclSuccess) {
  return BuildSystem(MustParseXml(text).get(), expect);
}

// Builds the system as init.cc does before computing its paths, without a communicator.
inline SystemPtr BuildWithPaths(const std::string& text) {
  SystemPtr system = BuildSystem(text);
  if (system) {
    system->nRanks = system->nodes[GPU].count;
    system->netGdrLevel = -2;
    EXPECT_EQ(ncclSuccess, ncclTopoComputePaths(system.get(), nullptr));
  }
  return system;
}

inline int GpuIndexOfRank(ncclTopoSystem* system, int rank) {
  for (int g = 0; g < system->nodes[GPU].count; g++)
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) return g;
  return -1;
}

}  // namespace topo_scene

#endif  // RCCL_TEST_HOST_TOPO_SCENE_H_
