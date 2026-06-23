#include <hip_test_common.hh>
#include <string_view>

// Fixtures are installed flat alongside the test binary; load by basename (cwd-relative).
#define DECL_MODULE_PATH(input_name) constexpr std::string_view input_name = #input_name ".co"

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
