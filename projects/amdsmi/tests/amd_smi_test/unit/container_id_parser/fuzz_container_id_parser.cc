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

// Fuzz harness for amd::smi::ExtractContainerId.
//
// The table-driven suites pin down specific known-adversarial inputs; this
// harness complements them by asserting the parser's *invariants* hold for
// arbitrary byte sequences — the cgroup line is attacker-influenceable, so no
// input, however malformed, may violate memory safety or the output contract.
//
// Invariants checked for every (line, type_name, out_cap):
//   1. No out-of-bounds write        — GuardedBuffer canaries stay intact.
//   2. Length is bounded             — bytes written <= min(cap-1, MAX_ID_LEN).
//   3. Output is always NUL-terminated at the reported length.
//   4. Charset is enforced           — every output byte is [a-zA-Z0-9_-].
//   5. Zero capacity writes nothing.
//
// Dual-mode:
//   * Default (globbed into amdsmitst): runs the invariant checker over a seed
//     corpus plus a fixed-seed random sweep, as GTest cases — reproducible, no
//     external tooling required.
//   * libFuzzer (compile this file alone with -DCONTAINER_ID_PARSER_FUZZER_MAIN
//     and clang -fsanitize=fuzzer): LLVMFuzzerTestOneInput drives the same
//     checker with coverage guidance.
//
// See README.md in this directory for build instructions.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "guarded_buffer.h"
#include "container_id_test_util.h"

using amdsmi_test::GuardedBuffer;

namespace {

// Run ExtractContainerId into a canary-guarded buffer of compile-time size
// `Cap` with runtime capacity `out_cap`, then verify every safety invariant.
// Returns nullptr on success or a static description of the first violation.
template <size_t Cap>
const char* CheckWithCap(const std::string& line, const char* type_name,
                         size_t out_cap) {
  GuardedBuffer<Cap> gb;
  const size_t n = amdsmi_test::ExtractIdInto(line, type_name, gb.buf, out_cap);

  if (!gb.CanariesIntact()) return "canary smashed (out-of-bounds write)";
  if (out_cap == 0) {
    return (n == 0) ? nullptr : "nonzero write into zero-capacity buffer";
  }
  if (n > out_cap - 1) return "bytes written exceed capacity";
  if (n > static_cast<size_t>(AMDSMI_MAX_CONTAINER_ID_LENGTH)) {
    return "bytes written exceed AMDSMI_MAX_CONTAINER_ID_LENGTH";
  }
  if (gb.buf[n] != '\0') return "output not NUL-terminated at reported length";
  for (size_t i = 0; i < n; ++i) {
    if (!amd::smi::IsContainerIdChar(static_cast<unsigned char>(gb.buf[i]))) {
      return "output byte outside [a-zA-Z0-9_-]";
    }
  }
  return nullptr;
}

// Exercise both container types and a spread of output capacities (full,
// small, one-byte, zero) so the truncation and zero-capacity paths are hit.
const char* RunInvariants(const std::string& line) {
  static const char* const kTypes[] = {"docker", "lxc"};
  for (const char* type_name : kTypes) {
    if (const char* e =
            CheckWithCap<AMDSMI_MAX_STRING_LENGTH>(line, type_name,
                                                   AMDSMI_MAX_STRING_LENGTH)) {
      return e;
    }
    if (const char* e = CheckWithCap<16>(line, type_name, 16)) return e;
    if (const char* e = CheckWithCap<1>(line, type_name, 1)) return e;
    if (const char* e = CheckWithCap<1>(line, type_name, 0)) return e;
  }
  return nullptr;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const std::string line(reinterpret_cast<const char*>(data), size);
  if (const char* violation = RunInvariants(line)) {
    std::fprintf(stderr, "ExtractContainerId invariant violated: %s\n",
                 violation);
    std::abort();
  }
  return 0;
}

#if !defined(CONTAINER_ID_PARSER_FUZZER_MAIN)

#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Realistic and adversarial seeds; the random sweep prepends a valid prefix to
// half its inputs so the charset consume loop is reached with random tails.
const std::vector<std::string>& FuzzSeeds() {
  static const std::vector<std::string> seeds = {
      "",
      "0::/docker/abcdef0123456789",
      "0::/system.slice/docker-abcdef0123456789.scope",
      "0::/lxc/my-container_01",
      "0::/kubepods/besteffort/pod123/abc",
      "0::/docker/",
      "0::/not-docker-evil/payload",
      std::string("0::/docker/abc") + '\0' + "smuggled",
      "0::/docker/abc\nFAKE_LOG",
      "0::/docker/abc;rm -rf /",
      std::string("0::/docker/") + std::string(10000, 'a'),
  };
  return seeds;
}

}  // namespace

TEST(ContainerIdParser_Fuzz, InvariantsHoldOnSeedCorpus) {
  for (const auto& seed : FuzzSeeds()) {
    const char* violation = RunInvariants(seed);
    EXPECT_EQ(violation, nullptr)
        << "seed of " << seed.size()
        << " bytes: " << (violation ? violation : "");
  }
}

TEST(ContainerIdParser_Fuzz, InvariantsHoldOnRandomInputs) {
  std::mt19937 rng(0xC0FFEEu);  // fixed seed -> reproducible failures
  std::uniform_int_distribution<int> len_dist(0, 300);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  for (int iter = 0; iter < 50000; ++iter) {
    std::string line;
    const int len = len_dist(rng);
    line.reserve(static_cast<size_t>(len));
    for (int i = 0; i < len; ++i) {
      line.push_back(static_cast<char>(byte_dist(rng)));
    }
    if ((iter & 1) == 0) line.insert(0, "0::/docker/");

    const char* violation = RunInvariants(line);
    ASSERT_EQ(violation, nullptr)
        << "iter " << iter << ": " << (violation ? violation : "");
  }
}

#endif  // !CONTAINER_ID_PARSER_FUZZER_MAIN
