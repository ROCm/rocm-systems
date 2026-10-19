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
// so they need no GPU, no amdsmi and no rdc libraries and run anywhere. They
// read every tree that is there: the build machine's kernel sources and, for an
// installed copy of this binary, the kernels installed beside it in
// <prefix>/lib/rdc/hsaco, which is the tree the diagnostic loads from. An
// installed copy must find that second tree, and names it when it is not there,
// because a package without it is the gap these checks guard; where no tree at
// all is reachable the checks skip instead of failing.

#include <gtest/gtest.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

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

// The directory the running binary sits in, from /proc/self/exe, which is what
// names the installed kernels' tree relative to it.
std::string ExeDir() {
  char buffer[PATH_MAX] = {};
  const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (length <= 0) {
    return {};
  }
  const std::string path(buffer, static_cast<size_t>(length));
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

// Whether this binary is the installed copy, which the package puts in
// <prefix>/share/rdc/rdctst_tests; the build tree puts it elsewhere.
bool IsInstalledTestDir(const std::string& dir) {
  const std::string installed_in = "/share/rdc/rdctst_tests";
  return dir.size() >= installed_in.size() &&
         dir.compare(dir.size() - installed_in.size(), installed_in.size(), installed_in) == 0;
}

// The kernel trees to check: the build machine's sources and, where this binary is
// installed in <prefix>/share/rdc/rdctst_tests, the kernels in <prefix>/lib/rdc/hsaco
// that the installed diagnostic loads. Run from the build tree the second names a
// directory no build makes and is left out, and run from a package the first names the
// build machine's checkout; but an installed run keeps the second whether it is there or
// not, so that a package which carries no kernels fails here naming the tree it lacks.
std::vector<std::string> KernelRoots() {
  std::vector<std::string> roots;
  const std::string sources = RDC_KERNELS_HSACO_DIR;
  if (DirExists(sources)) {
    roots.push_back(sources);
  }
  const std::string exe_dir = ExeDir();
  if (!exe_dir.empty()) {
    const std::string installed = exe_dir + "/../../../lib/rdc/hsaco";
    if (DirExists(installed) || IsInstalledTestDir(exe_dir)) {
      roots.push_back(installed);
    }
  }
  return roots;
}

// Kernels loaded by the diagnostic (see ComputeQueueTest and MemoryAccess).
const char* const kDiagKernels[] = {
    "binary_search_kernels.hsaco",
    "gpuReadWrite_kernels.hsaco",
};

// Assert every diagnostic kernel for gfx_name exists and is an AMDGPU code
// object built for that exact ISA, in every kernel tree that is reachable. Each
// failure names the tree it read, as the two hold different copies.
void ExpectKernelsForTarget(const std::string& gfx_name) {
  const std::vector<std::string> roots = KernelRoots();
  if (roots.empty()) {
    GTEST_SKIP() << "no kernel tree available: neither '" << RDC_KERNELS_HSACO_DIR
                 << "' nor an installed <prefix>/lib/rdc/hsaco beside this binary";
  }

  for (const std::string& base : roots) {
    if (!DirExists(base)) {
      // The installed run above kept this tree: the package should have installed it.
      ADD_FAILURE() << "the installed diagnostic kernels are not there: " << base;
      continue;
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
}

// Regression guard for the reported gap: gfx950 diagnostic kernels must ship.
TEST(DiagKernelPackaging, Gfx950KernelsPresentAndTargeted) { ExpectKernelsForTarget("gfx950"); }

// A previously shipped target, so a wholesale packaging break is also caught
// and the checks are exercised against known-good assets.
TEST(DiagKernelPackaging, Gfx942KernelsPresentAndTargeted) { ExpectKernelsForTarget("gfx942"); }

}  // namespace
