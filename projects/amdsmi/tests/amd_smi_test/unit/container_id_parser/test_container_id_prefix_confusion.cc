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

// CWE-20 prefix / substring confusion (anchored-match tests).
//
// The original unanchored `find("docker")` matched inside
// "/not-docker-evil/..." and "/mydocker/...", producing silent false
// positives. The parser requires '/' (or SOL) before the type name AND '/'
// or '-' after it, which eliminates mid-word hits like "xdocker" and
// "not-docker-evil".
//
// Tradeoff: the '-' suffix anchor means "/docker-compose-tool/..." DOES
// match, yielding "compose-tool" as the container ID. This is intentional
// -- Docker's systemd cgroup format is "/docker-<id>.scope", so '-' must
// be accepted as a valid separator. A path like "/docker-compose-tool/"
// is legitimate cgroup syntax and the anchored matcher cannot distinguish
// it from a real container entry without a full type-name allowlist.
// The residual false-positive is documented here and in the PR description.
//
// CWE-20:  https://cwe.mitre.org/data/definitions/20.html
// CWE-185: https://cwe.mitre.org/data/definitions/185.html

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct PrefixCase {
  std::string line;
  const char* type_name;
  std::string expected;
  const char* desc;
};
}  // namespace

// Anchoring requires '/' or start-of-line before type_name and '/' or '-'
// after it. Cases with a "compose-tool"/"tools" expected value document the
// residual false-positive inherent in the current {lxc, docker} type set.
TEST(ContainerIdParser_PrefixConfusion, AnchorBoundaryRules) {
  const std::vector<PrefixCase> cases = {
      {"0::/not-docker-evil/attackerstring", "docker", "",
       "no '/' before docker -> rejected"},
      {"0::/path/to/docker-compose-tool/extra/id", "docker", "compose-tool",
       "'/'+'-' anchor succeeds; residual false-positive"},
      {"0::/path/to/lxc-tools/extra", "lxc", "tools",
       "'/'+'-' anchor succeeds; residual false-positive"},
      {"0::xdocker/id", "docker", "", "'x' precedes docker -> rejected"},
      {"docker/abc123", "docker", "abc123", "start-of-line is a valid prefix"},
      {"0::/path/docker", "docker", "", "no '/' or '-' after docker -> rejected"},
      {"0::/Docker/abc123", "docker", "", "case-sensitive: Docker != docker"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, c.type_name), c.expected)
        << "prefix case: " << c.desc;
  }
}
