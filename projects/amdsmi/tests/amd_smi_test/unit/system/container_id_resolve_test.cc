// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for amd::smi::ResolveContainerId, the whole policy fdinfo.cc applies
// to a /proc/<pid>/cgroup file. The per-extractor tests check the primitives in
// isolation; these rows are complete cgroup files captured from real runtimes,
// so they pin the value that actually reaches amdsmi_proc_info_t.container_name.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "container_id_test_util.h"
#include "guarded_buffer.h"

namespace {

std::string Resolve(const std::vector<std::string>& lines) {
  char buf[AMDSMI_MAX_STRING_LENGTH] = {0};
  amd::smi::ResolveContainerId(lines, buf, sizeof(buf));
  return std::string(buf);
}

struct ResolveCase {
  std::vector<std::string> lines;
  std::string expected;
  const char* desc;
};

}  // namespace

// One row per container runtime, as the whole cgroup file looks on a host
// running it. The OCI runtimes all report the bare 64-char ID; LXC reports the
// container name, because LXC cgroups carry no SHA-256.
TEST(SystemUnit, ResolveContainerIdCoversSupportedRuntimes) {
  const std::string id = amdsmi_test::kDocker64;
  const std::string pod = "kubepods-burstable-pod3f5e1c2a_9b7d_4c3e_8a11_0d2b4c6e8f90.slice";
  const std::vector<ResolveCase> cases = {
      {{"0::/system.slice/docker-" + id + ".scope"}, id, "docker, systemd driver"},
      {{"0::/docker/" + id}, id, "docker, cgroupfs driver"},
      {{"0::/kubepods.slice/kubepods-burstable.slice/" + pod + "/cri-containerd-" + id + ".scope"},
       id,
       "kubernetes + containerd"},
      {{"0::/kubepods.slice/kubepods-besteffort.slice/" + pod + "/crio-" + id + ".scope"},
       id,
       "kubernetes + CRI-O"},
      {{"0::/machine.slice/libpod-" + id + ".scope"}, id, "podman"},
      {{"0::/lxc/web01"}, "web01", "LXC, classic layout"},
      {{"0::/lxc.payload.web01"}, "web01", "LXC 3+/LXD under systemd"},
      {{"0::/lxc/parent/child"}, "parent", "nested LXC reports the outer container"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(Resolve(c.lines), c.expected) << "runtime: " << c.desc;
  }
}

// A process that is not in a container must report no container, however much
// its cgroup path looks like one. Every row here produced a wrong non-empty
// value before the parser anchored its matches on path-component boundaries.
TEST(SystemUnit, ResolveContainerIdReportsNoIdOutsideAContainer) {
  const std::vector<ResolveCase> cases = {
      {{"0::/user.slice/user-1000.slice/session-3.scope"}, "", "ordinary login session"},
      {{"0::/system.slice/docker.service"}, "", "the docker daemon itself"},
      {{"0::/system.slice/containerd.service"}, "", "the containerd daemon itself"},
      {{"0::/user.slice/mydocker-notacontainer.scope"}, "", "runtime name as a substring"},
      {{"0::/not-docker-evil/aaaaaaaaaaaaaaaaaaaa"}, "", "path crafted to look like a match"},
      {{"0::/path/to/lxc-tools/extra"}, "", "'lxc-' is not a container prefix"},
      {{"0::/"}, "", "cgroup root"},
      {{""}, "", "empty line"},
      {{}, "", "empty cgroup file"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(Resolve(c.lines), c.expected) << "non-container: " << c.desc;
  }
}

// A cgroup path whose last component is exactly a runtime name leaves no bytes
// after the separator to read. The v1 rows also confirm every line is scanned,
// not just the first: on a v1 host the container appears on the controller
// lines, and which controllers are mounted varies.
TEST(SystemUnit, ResolveContainerIdScansEveryLineAndStopsAtLineEnd) {
  const std::string id = amdsmi_test::kDocker64;
  const std::vector<ResolveCase> cases = {
      {{"0::/docker"}, "", "line ends on the runtime name"},
      {{"0::/lxc"}, "", "line ends on the runtime name"},
      {{"0::/docker/"}, "", "separator with no id after it"},
      {{"11:hugetlb:/", "10:memory:/docker/" + id, "9:cpuset:/"},
       id,
       "v1 file, container on a middle line"},
      {{"11:hugetlb:/", "10:memory:/", "9:cpuset:/docker/" + id},
       id,
       "v1 file, container on the last line"},
      {{"12:pids:/init.scope", "0::/lxc.payload.web01"},
       "web01",
       "unified line after a systemd line"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(Resolve(c.lines), c.expected) << "scan case: " << c.desc;
  }
}

// /proc content is not a trusted format. Resolution must terminate and leave a
// NUL-terminated buffer for any input, and must never write outside `out_cap`.
TEST(SystemUnit, ResolveContainerIdIsTotalOverMalformedInput) {
  const std::vector<std::vector<std::string>> lines_set = {
      {"0::/docker"},
      {"0::/lxc"},
      {"docker"},
      {"lxc.payload."},
      {"/"},
      {"//////"},
      {"0::/docker/" + std::string(4096, 'a')},
      {"0::/lxc/" + std::string("\x01\x02\x7f")},
      {std::string(8192, '/')},
  };
  for (const auto& lines : lines_set) {
    amdsmi_test::GuardedBuffer<64> gb;
    const size_t n = amd::smi::ResolveContainerId(lines, gb.buf, sizeof(gb.buf));
    EXPECT_LT(n, sizeof(gb.buf)) << "line: " << lines.front().substr(0, 40);
    EXPECT_EQ(gb.buf[n], '\0') << "line: " << lines.front().substr(0, 40);
    EXPECT_TRUE(gb.CanariesIntact()) << "line: " << lines.front().substr(0, 40);
  }
}

// A zero-capacity buffer must not be written to at all.
TEST(SystemUnit, ResolveContainerIdZeroCapacityBufferIsNotWritten) {
  amdsmi_test::GuardedBuffer<1> gb;
  gb.buf[0] = 'X';
  EXPECT_EQ(amd::smi::ResolveContainerId({"0::/docker/abc123"}, gb.buf, 0), 0u);
  EXPECT_EQ(gb.buf[0], 'X');
  EXPECT_TRUE(gb.CanariesIntact());
}
