// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Parse-behavior coverage for the container-ID extractors. Each row of the
// format tables is a cgroup line a ROCm deployment actually produces, paired
// with the ID that must come out of it: this is the regression contract for
// the 16-char truncation bug.

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;
using amdsmi_test::ExtractOciIdString;

namespace {
struct ParseCase {
  std::string line;
  const char* prefix;
  std::string expected;
  const char* desc;
};
}  // namespace

// One row per runtime + host-OS cgroup format seen in ROCm deployments.
// The kubernetes/containerd rows expect "" from the prefix scan, whose prefix
// set names only LXC and Docker; ExtractOciContainerId covers them, below.
TEST(SystemUnit, ContainerIdHandlesKnownCgroupFormats) {
  const std::string kDocker64 = amdsmi_test::kDocker64;
  const std::string k8s_line =
      "12:pids:/kubepods.slice/kubepods-besteffort.slice/"
      "kubepods-besteffort-pod12345678_abcd_ef01_2345_6789abcdef01.slice/"
      "cri-containerd-" +
      kDocker64 + ".scope";
  const std::vector<ParseCase> cases = {
      {"0::/docker/" + kDocker64, "docker/", kDocker64, "docker raw cgroup v1"},
      {"0::/system.slice/docker-" + kDocker64 + ".scope", "docker-", kDocker64,
       "docker systemd scope"},
      {k8s_line, "docker/", "", "k8s+containerd not in prefix set (docker)"},
      {k8s_line, "lxc/", "", "k8s+containerd not in prefix set (lxc)"},
      {"0::/lxc/" + kDocker64, "lxc/", kDocker64, "lxc name that looks like a sha256"},
      {"0::/lxc/my-container", "lxc/", "my-container", "lxc named"},
      {"0::/lxc.payload.web", "lxc.payload.", "web", "lxc systemd payload"},
      {"0::/lxc/parent/child", "lxc/", "parent", "lxc nested stops at slash"},
      {"0::/lxc/my_app_v2", "lxc/", "my_app_v2", "lxc with underscores"},
      {"0::/docker/abc123def456", "docker/", "abc123def456", "docker short id"},
      {"0::/docker/", "docker/", "", "empty after prefix"},
      {"0::/docker/abc123", "docker/", "abc123", "short id at end of line"},
      {"0::/user.slice/user-1000.slice/user@1000.service/app.slice/docker-" + kDocker64 + ".scope",
       "docker-", kDocker64, "cgroup v2 unified hierarchy"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, c.prefix), c.expected) << "format: " << c.desc;
  }
}

// The original unanchored find("docker") matched inside "/not-docker-evil/", so
// a prefix now has to sit at a cgroup path-component boundary: '/' (or
// start-of-line) immediately before it. Docker's "/docker-<id>.scope" format
// forces "docker-" into the prefix set, which leaves a residual false positive
// on "/docker-compose-tool/" that the anchor alone cannot distinguish from a
// real entry.
TEST(SystemUnit, ContainerIdAnchorBoundaryRules) {
  const std::vector<ParseCase> cases = {
      {"0::/not-docker-evil/attackerstring", "docker-", "", "no '/' before docker -> rejected"},
      {"0::/path/to/docker-compose-tool/extra/id", "docker-", "compose-tool",
       "anchored '/docker-' succeeds; residual false-positive"},
      {"0::/path/to/lxc-tools/extra", "lxc/", "", "'lxc-' is not a container prefix"},
      {"0::xdocker/id", "docker/", "", "'x' precedes docker -> rejected"},
      {"docker/abc123", "docker/", "abc123", "start-of-line is a valid prefix"},
      {"0::/path/docker", "docker/", "", "no separator after docker -> rejected"},
      {"0::/system.slice/docker.service", "docker/", "", "docker.service is not a container"},
      {"0::/system.slice/docker.service", "docker-", "", "docker.service is not a container"},
      {"0::/Docker/abc123", "docker/", "", "case-sensitive: Docker != docker"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, c.prefix), c.expected) << "prefix case: " << c.desc;
  }
}

// Cases where there is no match to make. The arithmetic hazard is find()
// returning npos: adding strlen(prefix) to it must not underflow into an
// in-range offset, and a prefix at the very end must not be read past.
TEST(SystemUnit, ContainerIdEdgeCaseInvariants) {
  const std::vector<ParseCase> cases = {
      {"0::/kubepods/besteffort/abc", "docker/", "",
       "prefix not found, no integer underflow (docker)"},
      {"0::/kubepods/besteffort/abc", "lxc/", "", "prefix not found, no integer underflow (lxc)"},
      {"0::/docker", "docker/", "", "runtime name at exact end of line, no over-read"},
      {"0::/docker/", "docker/", "", "separator but no id"},
      {"", "docker/", "", "empty line"},
      {"0::/docker/abc", "", "", "empty prefix safely rejected"},
      {"0::/docker/first/lxc/second", "docker/", "first",
       "multiple markers resolve independently (docker)"},
      {"0::/docker/first/lxc/second", "lxc/", "second",
       "multiple markers resolve independently (lxc)"},
      {"0::/lxc/AbC_dEf-123", "lxc/", "AbC_dEf-123", "id preserves mixed case"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, c.prefix), c.expected) << "edge case: " << c.desc;
  }
}

// The expected value is the bare 64-hex ID in every case: the runtime prefix
// and the ".scope" suffix are cgroup naming, not part of the ID, and the bare
// form is what `docker inspect`, the Docker Engine API and the CRI services
// accept. Rows cover both cgroup drivers (systemd, cgroupfs) and both
// hierarchies (a unified "0::" line and a v1 controller line).
TEST(SystemUnit, ContainerIdHandlesOciRuntimeCgroupFormats) {
  const std::string id = amdsmi_test::kDocker64;
  const std::string pod_systemd =
      "kubepods-burstable-pod3f5e1c2a_9b7d_4c3e_8a11_0d2b4c6e8f90.slice";
  const std::string pod_cgroupfs = "pod3f5e1c2a-9b7d-4c3e-8a11-0d2b4c6e8f90";
  const std::vector<std::pair<std::string, std::string>> cases = {
      // Docker, cgroupfs driver — the reporter's production hosts (#7081).
      {"13:cpu,cpuacct:/docker/" + id, "docker cgroup v1 controller line"},
      {"0::/docker/" + id, "docker cgroup v2 unified line"},
      {"0::/system.slice/docker-" + id + ".scope", "docker systemd driver"},
      // Kubernetes + containerd.
      {"0::/kubepods.slice/kubepods-burstable.slice/" + pod_systemd + "/cri-containerd-" + id +
           ".scope",
       "k8s containerd systemd driver"},
      {"11:memory:/kubepods/burstable/" + pod_cgroupfs + "/" + id,
       "k8s containerd cgroupfs driver (bare id)"},
      {"0::/system.slice/containerd.service/containerd/" + id, "containerd non-systemd"},
      {"0::/k8s.io/" + id, "containerd k8s.io namespace"},
      // Kubernetes + CRI-O.
      {"0::/kubepods.slice/kubepods-besteffort.slice/" + pod_systemd + "/crio-" + id + ".scope",
       "k8s cri-o systemd driver"},
      {"9:devices:/kubepods/besteffort/" + pod_cgroupfs + "/crio-" + id, "k8s cri-o cgroupfs"},
      // Podman.
      {"0::/machine.slice/libpod-" + id + ".scope", "podman rootful"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractOciIdString(c.first), id) << "runtime format: " << c.second;
  }
}

// The SHA-256 match is anchored the same way the named-type match is: the run
// must be a whole cgroup path component (possibly after a runtime prefix), of
// exactly 64 lowercase hex digits. Anything else is not an OCI container ID.
TEST(SystemUnit, ContainerIdOciAnchorAndLengthRules) {
  const std::string id = amdsmi_test::kDocker64;
  const std::vector<ParseCase> cases = {
      {"0::/kubepods.slice/kubepods-burstable-pod3f5e1c2a_9b7d_4c3e_8a11_0d2b4c6e8f90.slice", "",
       "", "pod UID is not 64 hex"},
      {"0::/docker/" + id.substr(1), "", "", "63 hex digits rejected"},
      {"0::/docker/" + id + "0", "", "", "65 hex digits rejected"},
      {"0::/docker/x" + id, "", "", "run not preceded by '/' or '-'"},
      {"0::/docker/" + id + "x", "", "", "run not followed by '/', '.' or end"},
      {"0::/docker/ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789", "", "",
       "uppercase hex is not an OCI id"},
      {"0::/lxc/my-container", "", "", "lxc names carry no sha256"},
      {"", "", "", "empty line"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractOciIdString(c.line), c.expected) << "oci anchor case: " << c.desc;
  }
}
