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

#include <dirent.h>
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

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

    // Built for the expected target. The code object names it as, e.g.,
    // "amdgcn-amd-amdhsa--gfx950", or for gfx1250 as
    // "amdgpu12.50-amd-amdhsa-unknown-gfx1250".
    const bool targeted = data.find("amdgcn-amd-amdhsa--" + gfx_name) != std::string::npos ||
                          data.find("-amd-amdhsa-unknown-" + gfx_name) != std::string::npos;
    EXPECT_TRUE(targeted) << path << " is not a code object for " << gfx_name;
  }
}

// Assert every diagnostic kernel for gfx_name is a code object v4. The checks
// are written for the v4 kernel argument layout; a later version expects a
// larger layout with other hidden arguments. On gfx1250 the System memory
// check then failed, and a build with assertions enabled stops on the kernel
// argument size check.
void ExpectCodeObjectV4(const std::string& gfx_name) {
  const std::string base = RDC_KERNELS_HSACO_DIR;
  if (!DirExists(base)) {
    GTEST_SKIP() << "kernel source tree not available at '" << base << "'";
  }

  for (const char* kernel : kDiagKernels) {
    const std::string path = base + "/" + gfx_name + "/" + kernel;
    const std::string data = ReadFile(path);
    ASSERT_GE(data.size(), 9U) << "missing or truncated diagnostic kernel: " << path;
    // e_ident[EI_OSABI] is ELFOSABI_AMDGPU_HSA (64) and e_ident[EI_ABIVERSION]
    // is ELFABIVERSION_AMDGPU_HSA_V4 (2).
    EXPECT_EQ(static_cast<unsigned char>(data[7]), 64U) << path;
    EXPECT_EQ(static_cast<unsigned char>(data[8]), 2U) << path << " is not a code object v4";
  }
}

// Regression guard for the reported gap: gfx950 diagnostic kernels must ship.
TEST(DiagKernelPackaging, Gfx950KernelsPresentAndTargeted) { ExpectKernelsForTarget("gfx950"); }

// gfx1250 (AMD Instinct MI455X) kernels must ship, as code object v4.
TEST(DiagKernelPackaging, Gfx1250KernelsPresentAndTargeted) {
  ExpectKernelsForTarget("gfx1250");
  ExpectCodeObjectV4("gfx1250");
}

// A previously shipped target, so a wholesale packaging break is also caught
// and the checks are exercised against known-good assets.
TEST(DiagKernelPackaging, Gfx942KernelsPresentAndTargeted) { ExpectKernelsForTarget("gfx942"); }

// Every target's kernels must be code object v4, so a target added later cannot
// ship the compiler's default version. gfx940 and gfx941 are left out: ROCm has
// dropped them, so no current runtime loads their kernels and no current
// compiler builds them.
TEST(DiagKernelPackaging, AllTargetsUseCodeObjectV4) {
  const std::string base = RDC_KERNELS_HSACO_DIR;
  if (!DirExists(base)) {
    GTEST_SKIP() << "kernel source tree not available at '" << base << "'";
  }

  std::vector<std::string> targets;
  if (DIR* dir = opendir(base.c_str())) {
    while (const dirent* entry = readdir(dir)) {
      const std::string name = entry->d_name;
      if (name.rfind("gfx", 0) == 0 && name != "gfx940" && name != "gfx941" &&
          DirExists(base + "/" + name)) {
        targets.push_back(name);
      }
    }
    closedir(dir);
  }
  ASSERT_FALSE(targets.empty()) << "no gfx* kernel directories under " << base;

  for (const std::string& target : targets) {
    SCOPED_TRACE(target);
    ExpectCodeObjectV4(target);
  }
}

}  // namespace
