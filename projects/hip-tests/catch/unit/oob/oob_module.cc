#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

// Fixtures are installed flat alongside the test binary; load by basename (cwd-relative).
#define DECL_MODULE_PATH(input_name) constexpr std::string_view input_name = #input_name ".co"

static std::vector<char> ReadFile(std::string_view path) {
  std::ifstream f(std::string(path), std::ios::binary | std::ios::ate);
  REQUIRE(f.good());
  auto size = f.tellg();
  f.seekg(0);
  std::vector<char> buf(static_cast<size_t>(size));
  f.read(buf.data(), size);
  REQUIRE(f.good());
  return buf;
}

// File-backed path: hipModuleLoad(fname) - image_size_ is the exact file size.
HIP_TEST_CASE(OOB_hip_module_load_over) {
  DECL_MODULE_PATH(oob_kernel);
  DECL_MODULE_PATH(elf_huge_shnum);
  DECL_MODULE_PATH(elf_bad_shoff);
  DECL_MODULE_PATH(elf_table_spill);
  DECL_MODULE_PATH(elf_sh_overflow);

  SECTION("valid - sanity") {
    hipModule_t module{};
    HIP_CHECK(hipModuleLoad(&module, oob_kernel.data()));
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("huge shnum") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_huge_shnum.data()), hipErrorInvalidImage);
  }

  SECTION("bad shoff") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_bad_shoff.data()), hipErrorInvalidImage);
  }

  SECTION("table spill") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_table_spill.data()), hipErrorInvalidImage);
  }

  SECTION("sh overflow") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_sh_overflow.data()), hipErrorInvalidImage);
  }
}

// In-memory path: hipModuleLoadData(image) - no length, bound is derived from the
// mapping that contains the (anonymous heap) buffer. These cases reject in
// getElfSize before any device/arch check, so they are arch-independent. The valid
// in-memory load is covered by OOB_hiprtc_roundtrip_loads, which is arch-correct.
HIP_TEST_CASE(OOB_hip_module_load_data_over) {
  DECL_MODULE_PATH(elf_bad_shoff);
  DECL_MODULE_PATH(elf_sh_overflow);

  SECTION("bad shoff in-memory") {
    auto buf = ReadFile(elf_bad_shoff);
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoadData(&module, buf.data()), hipErrorInvalidImage);
  }

  SECTION("sh overflow in-memory") {
    auto buf = ReadFile(elf_sh_overflow);
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoadData(&module, buf.data()), hipErrorInvalidImage);
  }
}

// Runtime-compiled code objects live in an anonymous heap buffer and must still
// load through hipModuleLoadData after the bounds hardening.
HIP_TEST_CASE(OOB_hiprtc_roundtrip_loads) {
  static constexpr char kSource[] = "extern \"C\" __global__ void nop() {}\n";

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::string arch = std::string("--offload-arch=") + props.gcnArchName;
  const char* options[] = {arch.c_str()};

  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, kSource, "nop.cu", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 1, options));

  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  hipModule_t module{};
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleUnload(module));
}
