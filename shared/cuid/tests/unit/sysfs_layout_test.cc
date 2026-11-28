// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Discovery reads fixed paths under /sys. Each cuidtst case here builds a
// sysfs tree laid out as amdgpu, KFD and the PCI core lay it out, and reruns
// this binary in a bwrap sandbox with that tree mounted over /sys, where the
// matching cuidtstSysfsLayout case runs against the public API. Without bwrap
// or user namespaces the cuidtst case is skipped; outside the sandbox the
// cuidtstSysfsLayout case is.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "include/amd_cuid.h"
#include "src/cuid_util.h"
#include "test_common.h"

namespace {

constexpr char kSandboxEnv[] = "CUID_SYSFS_LAYOUT_TEST";
constexpr char kMarker[] = "/.cuid-sysfs-layout";

bool InSandbox() {
  return std::getenv(kSandboxEnv) && access((std::string("/sys") + kMarker).c_str(), F_OK) == 0;
}

class SysTree {
 public:
  SysTree() : dir_("cuid_sysfs_layout_"), root_(dir_.path()) {
    Write(kMarker, "");
    for (const char* driver : {"amdgpu", "amdxdna"}) Dir(std::string("/bus/pci/drivers/") + driver);
    Dir("/class/drm");
  }

  const std::string& root() const { return root_; }

  void Write(const std::string& path, const std::string& text) const {
    Dir(path.substr(0, path.find_last_of('/')));
    std::ofstream(root_ + path, std::ios::binary | std::ios::trunc) << text;
  }

  void Dir(const std::string& path) const {
    const std::string full = root_ + path;
    for (size_t at = full.find('/', 1); at != std::string::npos; at = full.find('/', at + 1))
      mkdir(full.substr(0, at).c_str(), 0755);
    mkdir(full.c_str(), 0755);
  }

  // Targets are absolute paths as the sandbox sees them.
  void Link(const std::string& path, const std::string& target) const {
    Dir(path.substr(0, path.find_last_of('/')));
    symlink(("/sys" + target).c_str(), (root_ + path).c_str());
  }

  std::string Pci(const std::string& bdf, uint16_t vendor, uint16_t device, uint32_t pci_class,
                  const char* driver) const {
    const std::string dev = "/devices/pci0000:00/0000:00:01.0/" + bdf;
    char text[16];
    std::snprintf(text, sizeof(text), "0x%04x\n", vendor);
    Write(dev + "/vendor", text);
    std::snprintf(text, sizeof(text), "0x%04x\n", device);
    Write(dev + "/device", text);
    Write(dev + "/revision", "0x00\n");
    std::snprintf(text, sizeof(text), "0x%06x\n", pci_class);
    Write(dev + "/class", text);
    if (driver) Link(dev + "/driver", std::string("/bus/pci/drivers/") + driver);
    Link("/bus/pci/devices/" + bdf, dev);
    return dev;
  }

  void Drm(const std::string& device, int render, int card) const {
    for (const std::string& name :
         {"renderD" + std::to_string(render), "card" + std::to_string(card)}) {
      const std::string node = device + "/drm/" + name;
      Dir(node);
      Link(node + "/device", device);
      Link("/class/drm/" + name, node);
    }
  }

  // KFD node 0 is a CPU; GPU node i + 1 renders minor `nodes[i].first` with
  // location_id `nodes[i].second`.
  void Kfd(const std::vector<std::pair<int, int>>& nodes) const {
    Write("/class/kfd/kfd/topology/generation_id", "1\n");
    Write("/class/kfd/kfd/topology/nodes/0/properties",
          "location_id 0\ndomain 0\ndrm_render_minor 0\n");
    for (size_t i = 0; i < nodes.size(); ++i)
      Write("/class/kfd/kfd/topology/nodes/" + std::to_string(i + 1) + "/properties",
            "location_id " + std::to_string(nodes[i].second) + "\ndomain 0\ndrm_render_minor " +
                std::to_string(nodes[i].first) + "\n");
  }

  // An MI300X-like GPU at 0000:f5:00.0 in `mode` with `n` partitions, as a
  // stock amdgpu lays it out: partition 0 renders through the PCI function,
  // the rest through amdgpu_xcp_<k> platform devices. `xcp` adds the
  // per-partition kobject amdgpu has had since v6.16.
  std::string PartitionedGpu(int n, const char* mode, bool xcp, bool kfd = true) const {
    const std::string dev = Pci("0000:f5:00.0", 0x1002, 0x74a1, 0x120000, "amdgpu");
    Write(dev + "/current_compute_partition", std::string(mode) + "\n");
    Drm(dev, 128, 1);
    std::vector<std::pair<int, int>> nodes{{128, 0xf500}};
    if (xcp) Dir(dev + "/xcp");
    for (int k = 1; k < n; ++k) {
      const std::string platform = "/devices/platform/amdgpu_xcp_" + std::to_string(k);
      Drm(platform, 128 + k, 1 + k);
      if (xcp) Dir(platform + "/xcp");
      nodes.emplace_back(128 + k, 0xf500 | k);
    }
    if (kfd) Kfd(nodes);
    return dev;
  }

 private:
  ScopedTempDir dir_;
  const std::string root_;
};

// The lifecycle harness links this suite with a shim that replaces NPU, NIC,
// CPU and Platform discovery and exits any process whose /sys lacks the
// harness's own fixture marker, so a sandboxed rerun there can only fail.
bool UnderLifecycleShim() { return access("/sys/.cuid-path-fixture", F_OK) == 0; }

bool BwrapWorks(const std::string& root) {
  const std::string probe =
      "bwrap --dev-bind / / --bind '" + root + "' /sys test -e /sys" + kMarker + " >/dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

// Runs cuidtstSysfsLayout.`name` with `tree` over /sys; its output is shown
// only when it fails.
void RunInSandbox(const SysTree& tree, const char* name) {
  char exe[4096] = {};
  ASSERT_GT(readlink("/proc/self/exe", exe, sizeof(exe) - 1), 0);
  const std::string log = tree.root() + ".log";
  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0) {
      dup2(fd, 1);
      dup2(fd, 2);
    }
    setenv(kSandboxEnv, "1", 1);
    const std::string filter = std::string("--gtest_filter=cuidtstSysfsLayout.") + name;
    execlp("bwrap", "bwrap", "--dev-bind", "/", "/", "--bind", tree.root().c_str(), "/sys", exe,
           filter.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  std::ifstream in(log);
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  unlink(log.c_str());
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << output;
  EXPECT_NE(output.find("[  PASSED  ] 1 test"), std::string::npos) << output;
}

#define RUN_IN_SANDBOX(tree, name)                                                           \
  do {                                                                                       \
    if (UnderLifecycleShim()) GTEST_SKIP() << "discovery is replaced by the lifecycle shim"; \
    if (!BwrapWorks((tree).root())) GTEST_SKIP() << "bwrap cannot mount over /sys";          \
    RunInSandbox((tree), name);                                                              \
  } while (0)

#define REQUIRE_SANDBOX()                                                            \
  do {                                                                               \
    if (!InSandbox()) GTEST_SKIP() << "runs only inside its cuidtst case's sandbox"; \
  } while (0)

std::string Uuid(const amdcuid_id_t& id) { return CuidUtilities::get_cuid_as_string(&id); }

// The temporary CUID of 0000:f5:00.0 1002:74a1 rev 00 with `unit_id`.
std::string Expected(uint16_t unit_id) {
  CuidUtilities::AuxiliaryInput aux;
  aux.format = CuidUtilities::kAuxFormatPcie;
  CuidUtilities::set_pcie_routing(aux, "0000:f5:00.0");
  aux.device_id = 0x74A1;
  aux.vendor_id = 0x1002;
  aux.component_type = AMDCUID_DEVICE_TYPE_GPU;
  aux.unit_id = unit_id;
  amdcuid_derived_id derived = {};
  EXPECT_EQ(CuidUtilities::temporary_cuid(aux, &derived), AMDCUID_STATUS_SUCCESS);
  return CuidUtilities::get_cuid_as_string(&derived.UUIDv8_representation);
}

// Every handle of `type`, by its CUID.
std::map<std::string, amdcuid_id_t> Enumerate(amdcuid_device_type_t type) {
  std::map<std::string, amdcuid_id_t> found;
  uint32_t count = 0;
  amdcuid_get_all_handles(nullptr, &count);
  std::vector<amdcuid_id_t> handles(count + 8);
  count = static_cast<uint32_t>(handles.size());
  EXPECT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  for (uint32_t i = 0; i < count; ++i) {
    amdcuid_device_type_t t;
    uint32_t size = sizeof(t);
    if (amdcuid_query_device_property(handles[i], AMDCUID_QUERY_DEVICE_TYPE, &t, &size) !=
            AMDCUID_STATUS_SUCCESS ||
        t != type)
      continue;
    amdcuid_id_t cuid;
    size = sizeof(cuid);
    if (amdcuid_query_device_property(handles[i], AMDCUID_QUERY_DERIVED_CUID, &cuid, &size) ==
        AMDCUID_STATUS_SUCCESS)
      found[Uuid(cuid)] = handles[i];
  }
  return found;
}

// The CUID of the GPU that `path` names, or the lookup's error.
std::string LookUp(const std::string& path) {
  amdcuid_id_t handle;
  const auto status =
      amdcuid_get_handle_by_dev_path(path.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle);
  if (status != AMDCUID_STATUS_SUCCESS) return amdcuid_status_to_string(status);
  amdcuid_id_t cuid;
  uint32_t size = sizeof(cuid);
  const auto query =
      amdcuid_query_device_property(handle, AMDCUID_QUERY_DERIVED_CUID, &cuid, &size);
  return query == AMDCUID_STATUS_SUCCESS ? Uuid(cuid) : amdcuid_status_to_string(query);
}

bool HaveMachineId() {
  uint8_t machine_id[16];
  return CuidUtilities::host_machine_id(machine_id) == AMDCUID_STATUS_SUCCESS;
}

// The GPU as a whole and partition k of n, UnitID n*(n-1)/2 + 1 + k, each by
// enumeration and each partition by its render node, except that the
// partitions in `none` have no CUID.
void ExpectPartitions(int n, const std::set<int>& none = {}) {
  if (!HaveMachineId()) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  std::set<std::string> expected{Expected(0)};
  std::set<std::string> any{Expected(0)};
  for (int k = 0; k < n; ++k) {
    const std::string cuid = Expected(CuidUtilities::partition_unit_id(n, k));
    any.insert(cuid);
    if (!none.count(k)) expected.insert(cuid);
  }
  std::set<std::string> found;
  for (const auto& entry : Enumerate(AMDCUID_DEVICE_TYPE_GPU)) found.insert(entry.first);
  EXPECT_EQ(found, expected);
  for (int k = 0; k < n; ++k) {
    const std::string render = "/sys/class/drm/renderD" + std::to_string(128 + k);
    const std::string card = "/sys/class/drm/card" + std::to_string(1 + k);
    const std::string expected_cuid = Expected(CuidUtilities::partition_unit_id(n, k));
    for (const std::string& path : {render, card, render + "/device/xcp"}) {
      if (none.count(k))
        EXPECT_EQ(any.count(LookUp(path)), 0u) << path;
      else
        EXPECT_EQ(LookUp(path), expected_cuid) << path;
    }
  }
  EXPECT_EQ(LookUp("/sys/bus/pci/devices/0000:f5:00.0"), Expected(0));
}

// The GPU as a whole, by enumeration and by each of its nodes, and nothing
// else: its render node is not a partition 0.
void ExpectWholeGpu() {
  if (!HaveMachineId()) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  std::set<std::string> found;
  for (const auto& entry : Enumerate(AMDCUID_DEVICE_TYPE_GPU)) found.insert(entry.first);
  EXPECT_EQ(found, std::set<std::string>{Expected(0)});
  for (const char* path :
       {"/sys/class/drm/renderD128", "/sys/class/drm/card1", "/sys/bus/pci/devices/0000:f5:00.0"})
    EXPECT_EQ(LookUp(path), Expected(0)) << path;
}

// Each NPU's BDF and device path, by enumeration.
std::map<std::string, std::string> EnumerateNpus() {
  std::map<std::string, std::string> found;
  for (const auto& entry : Enumerate(AMDCUID_DEVICE_TYPE_NPU)) {
    char bdf[64] = {};
    char path[256] = {};
    uint32_t size = sizeof(bdf);
    EXPECT_EQ(amdcuid_query_device_property(entry.second, AMDCUID_QUERY_BDF, bdf, &size),
              AMDCUID_STATUS_SUCCESS);
    size = sizeof(path);
    EXPECT_EQ(amdcuid_query_device_property(entry.second, AMDCUID_QUERY_DEVICE_PATH, path, &size),
              AMDCUID_STATUS_SUCCESS);
    found[bdf] = path;
  }
  return found;
}

}  // namespace

// amdgpu before v6.16 has partitions but no per-partition xcp kobject; KFD
// alone names them, as on the kernels distributions ship.
TEST(cuidtst, PartitionsWithoutAnXcpKobjectAreFoundThroughKfd) {
  for (const char* mode : {"CPX", "DPX"}) {
    SCOPED_TRACE(mode);
    SysTree tree;
    tree.PartitionedGpu(mode[0] == 'C' ? 8 : 2, mode, false);
    RUN_IN_SANDBOX(tree, mode[0] == 'C' ? "CpxWithoutXcp" : "DpxWithoutXcp");
  }
}

TEST(cuidtst, PartitionsWithAnXcpKobjectAreFoundThroughKfd) {
  SysTree tree;
  tree.PartitionedGpu(8, "CPX", true);
  RUN_IN_SANDBOX(tree, "CpxWithXcp");
}

TEST(cuidtstSysfsLayout, CpxWithoutXcp) {
  REQUIRE_SANDBOX();
  ExpectPartitions(8);
}

TEST(cuidtstSysfsLayout, DpxWithoutXcp) {
  REQUIRE_SANDBOX();
  ExpectPartitions(2);
}

TEST(cuidtstSysfsLayout, CpxWithXcp) {
  REQUIRE_SANDBOX();
  ExpectPartitions(8);
}

// KFD refuses with EPERM the properties of a partition outside the caller's
// device cgroup, and the library treats an unreadable properties file as the
// empty one written here. In CPX amdgpu lists the partitions in the parent's
// enforce_isolation, so the other partitions keep their UnitIDs.
TEST(cuidtst, CpxPartitionsBesideAHiddenOneAreCountedFromEnforceIsolation) {
  for (const bool xcp : {false, true}) {
    SCOPED_TRACE(xcp ? "xcp" : "no xcp");
    SysTree tree;
    const std::string dev = tree.PartitionedGpu(8, "CPX", xcp);
    tree.Write(dev + "/enforce_isolation", "0 0 0 1 0 0 0 0\n");
    tree.Write("/class/kfd/kfd/topology/nodes/4/properties", "");
    RUN_IN_SANDBOX(tree, "CpxPartition3Hidden");
  }
}

// Without a list that agrees with KFD, a CPX partition count is the count of
// KFD nodes, which a hidden node makes unknown: then no partition has a CUID.
TEST(cuidtst, CpxPartitionsWithoutAProvenCountHaveNoCuid) {
  const std::pair<const char*, bool> cases[] = {
      {nullptr, true}, {"0 0 0 0\n", true},          {"0 0 0 0 0 0 0 x\n", true}, {"0\n", true},
      {"\n", true},    {"0 0 0 0 0 0 0 0 \n", true}, {"0 0 0 0 0 0 0\n", false}};
  for (const auto& [listed, hidden] : cases) {
    SCOPED_TRACE(std::string(listed ? listed : "no enforce_isolation") +
                 (hidden ? ", partition 3 hidden" : ", every node read"));
    SysTree tree;
    const std::string dev = tree.PartitionedGpu(8, "CPX", false);
    if (listed) tree.Write(dev + "/enforce_isolation", listed);
    if (hidden) tree.Write("/class/kfd/kfd/topology/nodes/4/properties", "");
    RUN_IN_SANDBOX(tree, "CpxNoPartitions");
  }
}

TEST(cuidtstSysfsLayout, CpxPartition3Hidden) {
  REQUIRE_SANDBOX();
  ExpectPartitions(8, {3});
}

TEST(cuidtstSysfsLayout, CpxNoPartitions) {
  REQUIRE_SANDBOX();
  ExpectPartitions(8, {0, 1, 2, 3, 4, 5, 6, 7});
}

// While amdgpu cannot report the mode, as during a reset, the specification
// still names the device as a whole when KFD shows it as a single node, and
// when the driver publishes the device's own cuid_unit_id and no partition's.
TEST(cuidtst, WholeGpuWhoseModeCannotBeReadIsStillTheWholeGpu) {
  for (const char* mode : {"", "UNKNOWN", "unreadable"}) {
    SCOPED_TRACE(mode);
    const bool unreadable = std::string(mode) == "unreadable";
    if (unreadable && geteuid() == 0) continue;
    SysTree tree;
    const std::string dev = tree.PartitionedGpu(1, unreadable ? "SPX" : mode, true);
    if (unreadable) chmod((tree.root() + dev + "/current_compute_partition").c_str(), 0);
    RUN_IN_SANDBOX(tree, "WholeGpu");
  }
}

TEST(cuidtst, WholeGpuPublishingItsUnitIdIsTheWholeGpuWhateverTheMode) {
  SysTree tree;
  const std::string dev = tree.PartitionedGpu(1, "UNKNOWN", true, false);
  tree.Write(dev + "/cuid_unit_id", "0\n");
  RUN_IN_SANDBOX(tree, "WholeGpu");
}

TEST(cuidtstSysfsLayout, WholeGpu) {
  REQUIRE_SANDBOX();
  ExpectWholeGpu();
}

// An NPU is classified from its PCI function, so an amdxdna NPU without an
// accel node is found also when another NPU has one.
TEST(cuidtst, NpuWithoutAnAccelNodeIsFoundBesideOneWithIt) {
  SysTree tree;
  const std::string first = tree.Pci("0000:c4:00.1", 0x1022, 0x17f0, 0x118000, "amdxdna");
  tree.Dir(first + "/accel/accel0");
  tree.Link(first + "/accel/accel0/device", first);
  tree.Link("/class/accel/accel0", first + "/accel/accel0");
  tree.Pci("0000:c5:00.1", 0x1022, 0x17f0, 0x118000, "amdxdna");
  RUN_IN_SANDBOX(tree, "TwoNpus");
}

TEST(cuidtstSysfsLayout, TwoNpus) {
  REQUIRE_SANDBOX();
  if (!HaveMachineId()) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  const std::map<std::string, std::string> expected{
      {"0000:c4:00.1", "/sys/class/accel/accel0"},
      {"0000:c5:00.1", "/sys/bus/pci/devices/0000:c5:00.1"}};
  EXPECT_EQ(EnumerateNpus(), expected);
  // Each NPU's CUID is the temporary CUID of the address, IDs and UnitID the
  // library reports for it, so none of those queries may fail.
  for (const auto& entry : Enumerate(AMDCUID_DEVICE_TYPE_NPU)) {
    amdcuid_id_t rebuilt = {};
    ASSERT_TRUE(RebuiltTemporaryCuid(entry.second, rebuilt)) << entry.first;
    EXPECT_EQ(std::memcmp(&rebuilt, &entry.second, sizeof(rebuilt)), 0) << entry.first;
  }
}
