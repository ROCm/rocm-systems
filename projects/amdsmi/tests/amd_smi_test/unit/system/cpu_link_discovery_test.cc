// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amd::smi::DiscoverGpuCpuLinks() (rocm_smi_cpu_link.cc), the
// backing layer for amdsmi_get_gpu_cpu_links(). Exercises it directly against
// a mock KFD topology via the KfdNodesPathRoot test hook, the same hook used
// by kfd_numa_node_test.cc in this directory -- no GPU or node_handle/board
// fabrication involved, so there is no WSLGPUBackend short-circuit to guard
// against here (DiscoverGpuCpuLinks and its callees have no
// ENABLE_WSL_BACKEND branch; WSLGPUBackend only intercepts the amd_smi.cc
// public-API layer, which this test does not go through).

#include <gtest/gtest.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_cpu_link.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_io_link.h"
#include "test_fixture_utils.h"

namespace {

// TEMP DIAGNOSTIC -- not for commit. Runs the real `amd-smi xgmi` CLI (if
// findable) and dumps its full stdout+stderr, prefixed so it's easy to grep
// out of the CI job log alongside the DiscoverGpuCpuLinks() errno tracing in
// rocm_smi_cpu_link.cc.
std::string RunCommandCaptureOutput(const std::string& cmd) {
  std::string full_cmd = cmd + " 2>&1";
  FILE* pipe = popen(full_cmd.c_str(), "r");
  if (!pipe) return "<popen failed>";
  std::array<char, 4096> buf{};
  std::string out;
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    out += buf.data();
  }
  int status = pclose(pipe);
  out += "\n[exit status: " + std::to_string(status) + "]";
  return out;
}

void DumpXgmiCliOutputOnce(int iteration) {
  std::array<char, 4096> self_path{};
  ssize_t len = readlink("/proc/self/exe", self_path.data(), self_path.size() - 1);
  std::string exe_dir;
  if (len > 0) {
    self_path[static_cast<size_t>(len)] = '\0';
    std::string exe(self_path.data());
    auto pos = exe.find_last_of('/');
    if (pos != std::string::npos) exe_dir = exe.substr(0, pos);
  }

  std::vector<std::string> candidates = {"amd-smi xgmi"};
  if (!exe_dir.empty()) {
    // amdsmitst typically lives at <prefix>/share/amd_smi/tests/amdsmitst.
    candidates.push_back(exe_dir + "/../../../bin/amd-smi xgmi");
    candidates.push_back(exe_dir + "/../../../../bin/amd-smi xgmi");
  }

  for (const auto& cmd : candidates) {
    std::cerr << "[CPU_LINK_DEBUG][xgmi-cli][iter " << iteration << "] trying: " << cmd << "\n";
    std::string out = RunCommandCaptureOutput(cmd);
    std::cerr << "[CPU_LINK_DEBUG][xgmi-cli][iter " << iteration << "] ---- output begin ----\n"
              << out << "\n[CPU_LINK_DEBUG][xgmi-cli][iter " << iteration
              << "] ---- output end ----\n";
  }
}

// Run twice: the CI history shows this failing identically on every nightly,
// which points to something deterministic rather than a race -- but a second,
// cheap run costs nothing and would catch it if we're wrong about that.
TEST(SystemUnit, DebugDumpXgmiCliOutput) {
  DumpXgmiCliOutputOnce(1);
  DumpXgmiCliOutputOnce(2);
}

// TEMP DIAGNOSTIC -- not for commit. Dumps the real /sys/class/kfd/kfd/topology
// tree as a base64 tarball, so it can be extracted and replayed offline
// afterward (e.g. via the KfdNodesPathRoot test hook) without needing to
// re-dispatch CI.
TEST(SystemUnit, DebugDumpKfdTopologySysfs) {
  // Deliberately scoped to just the KFD topology tree -- exactly what
  // DiscoverGpuCpuLinks() reads, confirmed fast (~10ms, ~10KB) locally.
  // An earlier version of this also tried to tar each GPU's real PCI device
  // directory (to capture gpu_metrics/xgmi_port_num for offline replay), but
  // that combination made tar pathologically slow locally (60s+ timeout) for
  // reasons not fully root-caused, and isn't needed anyway: gpu_metrics is
  // already dumped via the public API in DebugDumpRawGpuMetricsAndLinkMetrics
  // below, and xgmi_port_num presence is checked directly in
  // DebugDumpDriverVersionInfo. Not worth the risk of eating the one CI run
  // we get on a tool quirk for data we already have another way.
  std::string out = RunCommandCaptureOutput(
      "timeout 20 tar czf - -C / sys/class/kfd/kfd/topology 2>/dev/null | base64 -w 0");
  std::cerr << "[CPU_LINK_DEBUG][sysfs-capture] ---- BEGIN KFD TOPOLOGY TARBALL ----\n"
            << out << "\n[CPU_LINK_DEBUG][sysfs-capture] ---- END KFD TOPOLOGY TARBALL ----\n";
}

namespace {
std::vector<int> ListNumericEntries(const std::string& dir_path) {
  std::vector<int> result;
  DIR* dir = opendir(dir_path.c_str());
  if (!dir) return result;
  struct dirent* ent;
  while ((ent = readdir(dir)) != nullptr) {
    std::string name = ent->d_name;
    if (!name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
          return std::isdigit(c);
        })) {
      result.push_back(std::stoi(name));
    }
  }
  closedir(dir);
  std::sort(result.begin(), result.end());
  return result;
}

// Mirrors OpenKFDNodeFile()/ReadKFDGpuId() exactly (rocm_smi_kfd.cc): stat()
// then open()+read(), reporting the literal errno at whichever step fails --
// this is the precise question "what errno does DiscoverGpuCpuLinks() hit",
// answered directly instead of inferred from a sysfs snapshot.
std::string ProbeFile(const std::string& path) {
  struct stat st{};
  std::ostringstream oss;
  if (stat(path.c_str(), &st) != 0) {
    oss << "stat() FAILED errno=" << errno << " (" << strerror(errno) << ")";
    return oss.str();
  }
  if (!S_ISREG(st.st_mode)) {
    oss << "stat() OK but not a regular file (mode=" << std::oct << st.st_mode << std::dec << ")";
    return oss.str();
  }
  std::ifstream f(path);
  if (!f.is_open()) {
    oss << "open() FAILED errno=" << errno << " (" << strerror(errno) << ")";
    return oss.str();
  }
  std::ostringstream content;
  content << f.rdbuf();
  std::string c = content.str();
  c.erase(std::remove(c.begin(), c.end(), '\n'), c.end());
  if (c.size() > 120) c = c.substr(0, 120) + "...(truncated)";
  oss << "OK content=[" << c << "]";
  return oss.str();
}
}  // namespace

TEST(SystemUnit, DebugProbeKfdNodeFileAccess) {
  const std::string topo_root = "/sys/class/kfd/kfd/topology/nodes";
  std::vector<int> nodes = ListNumericEntries(topo_root);
  std::cerr << "[CPU_LINK_DEBUG][file-probe] found " << nodes.size()
            << " topology nodes under " << topo_root << "\n";

  for (int n : nodes) {
    std::string node_dir = topo_root + "/" + std::to_string(n);
    std::cerr << "[CPU_LINK_DEBUG][file-probe] node=" << n
              << " gpu_id: " << ProbeFile(node_dir + "/gpu_id") << "\n";

    std::vector<int> links = ListNumericEntries(node_dir + "/io_links");
    std::cerr << "[CPU_LINK_DEBUG][file-probe] node=" << n << " has " << links.size()
              << " io_links entries\n";
    for (int l : links) {
      std::string props_path = node_dir + "/io_links/" + std::to_string(l) + "/properties";
      std::string result = ProbeFile(props_path);
      std::cerr << "[CPU_LINK_DEBUG][file-probe] node=" << n << " io_link=" << l
                << " properties: " << result << "\n";

      // Mirror DiscoverGpuCpuLinks() exactly: parse node_to out of properties,
      // then probe THAT node's gpu_id -- the literal second file read that
      // can return AMDSMI_STATUS_FILE_ERROR.
      std::ifstream pf(props_path);
      if (pf.is_open()) {
        std::string key;
        uint64_t val;
        while (pf >> key >> val) {
          if (key == "node_to") {
            std::string target_gpu_id = topo_root + "/" + std::to_string(val) + "/gpu_id";
            std::cerr << "[CPU_LINK_DEBUG][file-probe]   -> node_to=" << val
                      << " gpu_id: " << ProbeFile(target_gpu_id) << "\n";
            break;
          }
        }
      }
    }
  }
}

// TEMP DIAGNOSTIC -- not for commit. Calls the actual production function
// (amd::smi::ReadKFDGpuId(), not an approximation of it) directly against
// every topology node. ProbeFile() above uses std::ifstream the same way
// ReadKFDGpuId() does, and a local experiment showed that approach can be
// misleading: when the underlying read() is denied (confirmed via `cat`
// returning EPERM) rather than open() itself failing, std::ifstream's
// is_open() stays true and the stream silently yields empty content instead
// of surfacing the read error -- ProbeFile() reports that as "OK
// content=[]", which looks like success. Calling ReadKFDGpuId() directly
// shows what that actually becomes once the caller's own is_number() check
// rejects the resulting empty string.
TEST(SystemUnit, DebugProbeReadKFDGpuIdDirect) {
  const std::string topo_root = "/sys/class/kfd/kfd/topology/nodes";
  std::vector<int> nodes = ListNumericEntries(topo_root);
  for (int n : nodes) {
    uint64_t gpu_id = 0xDEADBEEF;  // sentinel: must change on success
    int ret = amd::smi::ReadKFDGpuId(static_cast<uint32_t>(n), &gpu_id);
    std::cerr << "[CPU_LINK_DEBUG][read-kfd-gpu-id-direct] node=" << n << " ret=" << ret;
    if (ret == 0) {
      std::cerr << " (SUCCESS) gpu_id=" << gpu_id;
    } else {
      std::cerr << " (" << strerror(ret) << ")";
    }
    std::cerr << "\n";
  }
}

// TEMP DIAGNOSTIC -- not for commit. Calls the raw public APIs directly (not
// through the CLI) for every discovered GPU, printing xgmi_read_data_acc /
// xgmi_write_data_acc / link_speed / link_width with explicit sentinel
// (UINT64_MAX / UINT16_MAX) flags -- this is what the xgmi CLI's Read/Write
// "N/A" columns ultimately come from, and lets us tell apart "PMFW never
// populates this for a GPU<->CPU link" from "amdsmi_get_link_metrics()'s
// xgmi_port_num-gated num_links hides an otherwise-valid value" without
// needing another CI run to check.
TEST(SystemUnit, DebugDumpRawGpuMetricsAndLinkMetrics) {
  amdsmi_status_t init_status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  std::cerr << "[CPU_LINK_DEBUG][raw-metrics] amdsmi_init status=" << init_status << "\n";
  if (init_status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "[CPU_LINK_DEBUG][raw-metrics] init failed, skipping\n";
    return;
  }

  amdsmi_socket_handle sockets[32];
  uint32_t socket_count = 32;
  if (amdsmi_get_socket_handles(&socket_count, sockets) != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "[CPU_LINK_DEBUG][raw-metrics] amdsmi_get_socket_handles failed\n";
    amdsmi_shut_down();
    return;
  }
  std::cerr << "[CPU_LINK_DEBUG][raw-metrics] socket_count=" << socket_count << "\n";

  for (uint32_t s = 0; s < socket_count; ++s) {
    amdsmi_processor_handle procs[32];
    uint32_t proc_count = 32;
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, procs) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (uint32_t p = 0; p < proc_count; ++p) {
      amdsmi_bdf_t bdf{};
      amdsmi_get_gpu_device_bdf(procs[p], &bdf);
      std::cerr << "[CPU_LINK_DEBUG][raw-metrics] socket=" << s << " proc=" << p
                << " bdf=" << std::hex << bdf.domain_number << ":"
                << static_cast<int>(bdf.bus_number) << ":" << static_cast<int>(bdf.device_number)
                << "." << static_cast<int>(bdf.function_number) << std::dec << "\n";

      uint32_t num_cpu_links = 0;
      amdsmi_status_t cpu_st = amdsmi_get_gpu_cpu_links(procs[p], &num_cpu_links, nullptr);
      std::cerr << "[CPU_LINK_DEBUG][raw-metrics]   amdsmi_get_gpu_cpu_links status=" << cpu_st
                << " num_cpu_links=" << num_cpu_links << "\n";

      amdsmi_link_metrics_t link_metrics{};
      amdsmi_status_t link_st = amdsmi_get_link_metrics(procs[p], &link_metrics);
      std::cerr << "[CPU_LINK_DEBUG][raw-metrics]   amdsmi_get_link_metrics status=" << link_st
                << " num_links=" << link_metrics.num_links << "\n";
      if (link_st == AMDSMI_STATUS_SUCCESS) {
        for (uint32_t i = 0; i < link_metrics.num_links && i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i) {
          const auto& l = link_metrics.links[i];
          bool read_max = l.read == std::numeric_limits<uint64_t>::max();
          bool write_max = l.write == std::numeric_limits<uint64_t>::max();
          std::cerr << "[CPU_LINK_DEBUG][raw-metrics]     link[" << i
                    << "] bit_rate=" << l.bit_rate << " max_bandwidth=" << l.max_bandwidth
                    << " link_type=" << l.link_type << " read=" << l.read
                    << (read_max ? " (SENTINEL)" : "") << " write=" << l.write
                    << (write_max ? " (SENTINEL)" : "") << "\n";
        }
      }

      amdsmi_gpu_metrics_t metrics{};
      amdsmi_status_t metrics_st = amdsmi_get_gpu_metrics_info(procs[p], &metrics);
      std::cerr << "[CPU_LINK_DEBUG][raw-metrics]   amdsmi_get_gpu_metrics_info status="
                << metrics_st << "\n";
      if (metrics_st == AMDSMI_STATUS_SUCCESS) {
        std::cerr << "[CPU_LINK_DEBUG][raw-metrics]   xgmi_link_width=" << metrics.xgmi_link_width
                  << " xgmi_link_speed=" << metrics.xgmi_link_speed << "\n";
        constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
        constexpr uint16_t kU16Max = std::numeric_limits<uint16_t>::max();
        for (int i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i) {
          uint64_t r = metrics.xgmi_read_data_acc[i];
          uint64_t w = metrics.xgmi_write_data_acc[i];
          uint16_t ls = metrics.xgmi_link_status[i];
          std::cerr << "[CPU_LINK_DEBUG][raw-metrics]     gpu_metrics.xgmi[" << i
                    << "] read=" << r << (r == kU64Max ? " (SENTINEL)" : "") << " write=" << w
                    << (w == kU64Max ? " (SENTINEL)" : "") << " link_status=" << ls
                    << (ls == kU16Max ? " (SENTINEL)" : "") << "\n";
        }
      }
    }
  }
  amdsmi_shut_down();
}

// TEMP DIAGNOSTIC -- not for commit. Driver/kernel version context for the
// "do we need a particular driver version for this?" question -- cheap to
// grab now, nothing we could add later without another CI run.
TEST(SystemUnit, DebugDumpDriverVersionInfo) {
  std::string out = RunCommandCaptureOutput(
      "echo ===uname===; uname -a; "
      // Direct container confirmation: PID 1 running as a container's init
      // (not systemd/real init) and/or a non-root cgroup path are the two
      // reliable host-vs-container signals that don't depend on inferring
      // anything from env vars some orchestrator happens to forward.
      "echo ===container_detection===; "
      "echo \"/.dockerenv exists: $([ -f /.dockerenv ] && echo yes || echo no)\"; "
      "echo \"pid1 cmdline: $(tr '\\0' ' ' < /proc/1/cmdline 2>/dev/null)\"; "
      "echo \"cgroup: $(cat /proc/self/cgroup 2>&1)\"; "
      "echo ===amdgpu_module_version===; cat /sys/module/amdgpu/version 2>&1; "
      "echo ===modinfo_amdgpu===; modinfo amdgpu 2>&1 | grep -iE 'version|vermagic'; "
      // Decisive check for the container/GPU-isolation hypothesis: if KFD
      // topology sees more GPU nodes than /dev/dri exposes device files for,
      // this container hides peer GPUs from /dev while sysfs topology (which
      // DiscoverGpuCpuLinks() walks) stays global -- exactly the condition
      // needed for it to walk into a neighbor it can't fully read. If the
      // counts match, that hypothesis is wrong.
      // ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES are consumed by the ROCR/HSA
      // userspace runtime during its own agent enumeration -- amd-smi doesn't
      // go through ROCR/HSA, it reads KFD topology sysfs and correlates to
      // DRM render nodes directly, so those env vars don't affect what it
      // sees. What matters here is the actual Docker device cgroup grant:
      // which /dev/dri/* and /dev/kfd nodes exist and are accessible at all.
      "echo ===dev_dri_listing===; ls -la /dev/dri/ 2>&1; "
      "echo ===dev_kfd===; ls -la /dev/kfd 2>&1; "
      "echo ===node_count_vs_dev_count===; "
      "echo \"kfd topology nodes: $(ls /sys/class/kfd/kfd/topology/nodes/ 2>/dev/null | wc -l)\"; "
      "echo \"/dev/dri renderD* nodes: $(ls /dev/dri/ 2>/dev/null | grep -c '^renderD')\"; "
      "echo \"/dev/dri card* nodes: $(ls /dev/dri/ 2>/dev/null | grep -c '^card')\"; "
      "echo \"gpu_id!=0 topology nodes (real GPUs per KFD): \"; "
      "for n in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do "
      "  v=$(cat \"$n\" 2>/dev/null); "
      "  [ -n \"$v\" ] && [ \"$v\" != \"0\" ] && echo \"  $n = $v\"; "
      "done; "
      "echo ===xgmi_port_num_presence===; "
      // Several /sys/class/drm/cardN-<connector> entries resolve to the same
      // underlying GPU device dir; dedupe so each real device prints once.
      "for d in /sys/class/drm/card*/device; do "
      "  readlink -f \"$d\" 2>/dev/null; "
      "done | sort -u | while read -r r; do "
      "  if [ -f \"$r/xgmi_port_num\" ]; then "
      "    echo \"$r: xgmi_port_num=$(cat $r/xgmi_port_num 2>&1)\"; "
      "  else "
      "    echo \"$r: xgmi_port_num MISSING\"; "
      "  fi; "
      "  ls \"$r\" 2>&1 | grep -i xgmi | tr '\\n' ' '; echo; "
      "done; "
      "echo ===dmesg_amdgpu_xgmi_kfd===; "
      "(dmesg 2>&1 || sudo -n dmesg 2>&1) | grep -iE 'amdgpu|xgmi|kfd|pcie.*error' | tail -200 "
      "|| echo 'dmesg unavailable (no permission, neither plain nor sudo -n worked)'");
  std::cerr << "[CPU_LINK_DEBUG][driver-info] ---- BEGIN ----\n"
            << out << "\n[CPU_LINK_DEBUG][driver-info] ---- END ----\n";
}

// MI450-like topology: GPU node 1 has a direct XGMI io_link to CPU node 0.
TEST(SystemUnit, DiscoverGpuCpuLinksFindsDirectXgmiCpuLink) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/0, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_XGMI);
}

// GPU node 1's only io_link neighbor is peer GPU node 2 (not a CPU) -- no CPU
// link exists, which must be reported as an empty, successful result.
TEST(SystemUnit, DiscoverGpuCpuLinksEmptyWhenOnlyPeerGpuNeighbor) {
  FakeSysfsTree tree;
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteGpuNode(tree, /*node_idx=*/2, /*card_idx=*/1, /*gpu_id=*/23456);  // peer GPU, not a CPU node
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);
  EXPECT_TRUE(links.empty());
}

// GPU node 1 has both a peer-GPU XGMI neighbor and a direct CPU PCIe
// neighbor -- only the CPU link must be reported.
TEST(SystemUnit, DiscoverGpuCpuLinksIgnoresNonCpuNeighbors) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteGpuNode(tree, /*node_idx=*/2, /*card_idx=*/1, /*gpu_id=*/23456);  // peer GPU, not a CPU node
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/1, /*node_to=*/0, amd::smi::IOLINK_TYPE_PCIEXPRESS,
              /*weight=*/20);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_PCIEXPRESS);
}

// GPU node 1 has two io_link neighbors: a real CPU (node 0) and a peer whose
// gpu_id file exists but can't be read as a clean number (node 2) -- e.g. a
// real GPU peer whose sysfs read is permission-denied by a container/device
// cgroup restriction, which ReadKFDGpuId() observes as empty content (not a
// missing file) and reports as ENXIO, not ENOENT. The unreadable neighbor's
// identity can't be determined, so it must be skipped -- neither
// misclassified as a CPU link nor treated as a hard failure for the whole
// query.
TEST(SystemUnit, DiscoverGpuCpuLinksSkipsNeighborWithUnreadableGpuId) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  tree.WriteFile("2/gpu_id", "");  // exists but empty: ReadKFDGpuId() -> ENXIO
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/0, amd::smi::IOLINK_TYPE_PCIEXPRESS,
              /*weight=*/20);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/1, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_PCIEXPRESS);
}

}  // namespace
