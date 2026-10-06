/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Packaging guard for the precompiled GPU diagnostic kernels.
//
// The unauthenticated "rdci diag" run loads precompiled HSACO code objects from
// rdc_modules/kernels/hsaco/<gfxName>/ for the Compute Queue and system memory
// subtests. When the kernels for a supported accelerator are missing, those
// subtests report Skip ("fail to open <kernel>.hsaco") and the test harness
// fails the run. This happened on gfx950 (AMD Instinct MI350X/MI355X) because
// the kernels were never packaged for that target.
//
// These are pure-logic checks: they read the committed kernel assets directly,
// so they need no GPU, no amdsmi and no rdc libraries and run anywhere. If the
// kernel source tree is not reachable (for example a packaged test run) the
// checks skip instead of failing.

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <fstream>
#include <iterator>
#include <string>

#ifndef RDC_KERNELS_HSACO_DIR
#define RDC_KERNELS_HSACO_DIR ""
#endif

namespace {

bool DirExists(const std::string& path) {
  struct stat st{};
  return !path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Kernels loaded by the diagnostic (see ComputeQueueTest and MemoryAccess).
const char* const kDiagKernels[] = {
    "binary_search_kernels.hsaco",
    "gpuReadWrite_kernels.hsaco",
};

// Assert every diagnostic kernel for gfx_name exists and is an AMDGPU code
// object built for that exact ISA.
void ExpectKernelsForTarget(const std::string& gfx_name) {
  const std::string base = RDC_KERNELS_HSACO_DIR;
  if (!DirExists(base)) {
    GTEST_SKIP() << "kernel source tree not available at '" << base << "'";
  }

  for (const char* kernel : kDiagKernels) {
    const std::string path = base + "/" + gfx_name + "/" + kernel;
    const std::string data = ReadFile(path);

    ASSERT_FALSE(data.empty()) << "missing or empty diagnostic kernel: " << path;

    // ELF magic (0x7F 'E' 'L' 'F').
    ASSERT_GE(data.size(), 4U) << path;
    EXPECT_EQ(static_cast<unsigned char>(data[0]), 0x7FU) << path;
    EXPECT_EQ(data[1], 'E') << path;
    EXPECT_EQ(data[2], 'L') << path;
    EXPECT_EQ(data[3], 'F') << path;

    // Built for the expected target, e.g. "amdgcn-amd-amdhsa--gfx950".
    const std::string isa = "amdgcn-amd-amdhsa--" + gfx_name;
    EXPECT_NE(data.find(isa), std::string::npos)
        << path << " is not a code object for " << gfx_name;
  }
}

// Regression guard for the reported gap: gfx950 diagnostic kernels must ship.
TEST(DiagKernelPackaging, Gfx950KernelsPresentAndTargeted) { ExpectKernelsForTarget("gfx950"); }

// A previously shipped target, so a wholesale packaging break is also caught
// and the checks are exercised against known-good assets.
TEST(DiagKernelPackaging, Gfx942KernelsPresentAndTargeted) { ExpectKernelsForTarget("gfx942"); }

}  // namespace
