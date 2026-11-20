/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
*/

/*
Regression tests for code-object selection on gfx9 client APUs (gfx902, gfx909, gfx90c).

ROCr reports XNACK as unsupported for these targets (xnack-off mode), so the agent ISA name has
no xnack feature (e.g. amdgcn-amd-amdhsa--gfx90c). Two changes make HIP load the right code
objects there:

  * projects/clr/rocclr/device/device.cpp: the ISA table entries match ROCr (XNACK unsupported),
    and the assertion in amd::Isa::isCompatible() allows an XNACK-capable generic code object
    (gfx9-generic) on such an agent. Before, the agent ISA resolved to an XNACK "any" entry and
    isCompatible() hit an assertion (builds with asserts enabled) for every code object loaded
    on these APUs.
  * projects/clr/hipamd/src/hip_fatbin.cpp (ExtractFatBinaryUsingCOMGR): a code object built with
    an explicit xnack- (--offload-arch=gfx90c:xnack-) is used as the native code object. Before,
    COMGR did not match it against the bare agent ISA name and loading failed.

Fixtures (fixed archs, see CMakeLists.txt), all from fatbin_xnack_off_apu_module.cc:
  fatbin_xnack_off_apu_xnack_off.code          fat binary: gfx902:xnack- gfx909:xnack- gfx90c:xnack-
  fatbin_xnack_off_apu_generic.code            fat binary: gfx9-generic (code object v6)
  fatbin_xnack_off_apu_generic_xnack_on.code   raw ELF: gfx9-generic:xnack+ (code object v6)
  fatbin_xnack_off_apu_generic_xnack_off.code  raw ELF: gfx9-generic:xnack- (code object v6)

The kernel writes 2 if it was compiled with an explicit xnack setting, else 1, so the tests also
check which code object ran.

The two reject fixtures are raw ELF code objects: in a fat binary, COMGR's lookup already skips
them (none of the queried names gfx90c, gfx90c:xnack- and gfx9-generic matches a generic code
object with an explicit xnack setting), so they would never reach amd::Isa::isCompatible(). As raw
ELFs they do; on an agent without XNACK, a generic code object with an explicit xnack+ or xnack-
is incompatible there, as in ROCr's Isa::IsCompatible() (ROCr also refuses to load it).

The clang driver and clang-offload-bundler refuse to put "gfx90c" and "gfx90c:xnack-" into the
same fat binary, so a bare-name-versus-alias preference cannot occur and is not tested.

All cases need a gfx902/gfx909/gfx90c device and skip on any other device.
*/

#include <hip_test_common.hh>
#include <hip/hip_runtime.h>

#include <string>

namespace {

// True on a gfx9 client APU that runs without XNACK.
bool IsXnackOffGfx9Apu() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string arch(prop.gcnArchName);
  const bool apu =
      arch.rfind("gfx902", 0) == 0 || arch.rfind("gfx909", 0) == 0 || arch.rfind("gfx90c", 0) == 0;
  return apu && arch.find("xnack+") == std::string::npos;
}

// Loads `file`, launches xnack_kernel and returns the value it wrote.
int LoadAndRun(const char* file) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, file));
  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "xnack_kernel"));

  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(int)));
  void* args[] = {&d_out};
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  int h_out = 0;
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipModuleUnload(module));
  return h_out;
}

}  // namespace

/*
 * A fat binary that only has xnack- code objects for the gfx9 client APUs loads and its kernel
 * runs (the value 2 shows that the xnack- code object was used).
 */
HIP_TEST_CASE(Unit_FatBin_XnackOffApu_LoadsXnackOffCodeObject) {
  if (!IsXnackOffGfx9Apu()) {
    HIP_SKIP_TEST("Needs a gfx902/gfx909/gfx90c device without XNACK");
    return;
  }
  REQUIRE(LoadAndRun("fatbin_xnack_off_apu_xnack_off.code") == 2);
}

/*
 * A gfx9-generic code object without an xnack setting is compatible with the APU.
 */
HIP_TEST_CASE(Unit_FatBin_XnackOffApu_LoadsGenericCodeObject) {
  if (!IsXnackOffGfx9Apu()) {
    HIP_SKIP_TEST("Needs a gfx902/gfx909/gfx90c device without XNACK");
    return;
  }
  REQUIRE(LoadAndRun("fatbin_xnack_off_apu_generic.code") == 1);
}

/*
 * gfx9-generic code objects with an explicit xnack+ or xnack- are rejected, as in ROCr.
 */
HIP_TEST_CASE(Unit_FatBin_XnackOffApu_RejectsGenericWithExplicitXnack) {
  if (!IsXnackOffGfx9Apu()) {
    HIP_SKIP_TEST("Needs a gfx902/gfx909/gfx90c device without XNACK");
    return;
  }
  const char* file = nullptr;
  SECTION("xnack+") { file = "fatbin_xnack_off_apu_generic_xnack_on.code"; }
  SECTION("xnack-") { file = "fatbin_xnack_off_apu_generic_xnack_off.code"; }
  hipModule_t module = nullptr;
  HIP_CHECK_ERRORS(hipModuleLoad(&module, file), hipErrorInvalidImage, hipErrorNoBinaryForGpu);
  // Clear the expected load error (standard and, on AMD, command error state).
  static_cast<void>(hipGetLastError());
#if HT_AMD
  static_cast<void>(hipExtGetLastError());
#endif
}
