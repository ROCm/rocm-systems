/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Real-world happy-path coverage for the production parser.
//
// These tests exercise the cgroup line formats that the parser must handle
// correctly in production. Each corresponds to a concrete runtime +
// host-OS combination seen in AMD ROCm deployments.
//
// Runtimes & formats covered:
//
//   - Docker raw cgroup v1     ->  0::/docker/<64-hex>
//   - Docker systemd cgroup    ->  0::/system.slice/docker-<64-hex>.scope
//   - Kubernetes + containerd  ->  12:pids:/kubepods.slice/.../cri-containerd-<id>.scope
//                                  (NOT detected by current code — type list is {lxc, docker};
//                                  documented as out-of-scope for this PR)
//   - LXC named containers     ->  0::/lxc/<name>
//   - LXC nested containers    ->  0::/lxc/<parent>/<child>
//   - LXC with underscores     ->  0::/lxc/<name_with_under>
//   - Docker short IDs         ->  0::/docker/<12-hex>
//   - Empty after prefix       ->  0::/docker/
//   - Truncated mid-ID line    ->  0::/docker/<partial>
//   - cgroup v2 unified        ->  0::/user.slice/.../docker-<id>.scope
//
// Reference: Docker's full vs. short ID constants in moby/moby:
// https://github.com/moby/moby/blob/2200f277f9f576886e90ca75929a2bb892b9ef23/client/pkg/stringid/stringid.go#L14-L15

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "fixtures.h"
#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct RealWorldCase {
  std::string line;
  const char* type_name;
  std::string expected;
  const char* desc;
};
}  // namespace

// One row per runtime + host-OS cgroup format seen in ROCm deployments.
// The kubernetes/containerd rows expect "" because the current type set in
// fdinfo.cc is {lxc, docker} — a documented gap, not a regression.
TEST(ContainerIdParser_RealWorld, HandlesKnownCgroupFormats) {
  const std::string kDocker64 = amdsmi_test::kDocker64;
  const std::string k8s_line =
      "12:pids:/kubepods.slice/kubepods-besteffort.slice/"
      "kubepods-besteffort-pod12345678_abcd_ef01_2345_6789abcdef01.slice/"
      "cri-containerd-" +
      kDocker64 + ".scope";
  const std::vector<RealWorldCase> cases = {
      {"0::/docker/" + kDocker64, "docker", kDocker64, "docker raw cgroup v1"},
      {"0::/system.slice/docker-" + kDocker64 + ".scope", "docker", kDocker64,
       "docker systemd scope"},
      {k8s_line, "docker", "", "k8s+containerd not in type set (docker)"},
      {k8s_line, "lxc", "", "k8s+containerd not in type set (lxc)"},
      {"0::/lxc/my-container", "lxc", "my-container", "lxc named"},
      {"0::/lxc/parent/child", "lxc", "parent", "lxc nested stops at slash"},
      {"0::/lxc/my_app_v2", "lxc", "my_app_v2", "lxc with underscores"},
      {"0::/docker/abc123def456", "docker", "abc123def456", "docker short id"},
      {"0::/docker/", "docker", "", "empty after prefix"},
      {"0::/docker/abc123", "docker", "abc123", "truncated line returns partial"},
      {"0::/user.slice/user-1000.slice/user@1000.service/app.slice/docker-" +
           kDocker64 + ".scope",
       "docker", kDocker64, "cgroup v2 unified hierarchy"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, c.type_name), c.expected)
        << "format: " << c.desc;
  }
}
