//===- hotswap_test.cpp - Test HotSwap ISA rewriting API ------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include <cstdio>
#include <cstdlib>

static int tests_passed = 0;
static int tests_failed = 0;

static constexpr const char *kTargetIsa = "amdgcn-amd-amdhsa--gfx1250";

static void check(bool cond, const char *name) {
  if (cond) {
    ++tests_passed;
    printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    fprintf(stderr, "  FAIL: %s\n", name);
  }
}

// A 16-byte fake ELF has no AMDGPU metadata note, so the ISA cannot be derived;
// RetargetCodeObject must fail and leave the original code object untouched.
static void test_RetargetInvalidCodeObjectFallsBack() {
  printf("TEST RetargetInvalidCodeObjectFallsBack...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kTargetIsa, &out_data, &out_size);

  check(rc != 0, "RetargetCodeObject fails when ISA cannot be derived");
  check(out_data == static_cast<const void *>(fake_elf),
        "out_data still points to original input after failure");
  check(out_size == sizeof(fake_elf),
        "out_size still matches original after failure");
}

static void test_RetargetNullOutputPointers() {
  printf("TEST RetargetNullOutputPointers...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  const int rc = rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf),
                                                   kTargetIsa, nullptr, nullptr);
  check(rc != 0, "RetargetCodeObject rejects null output pointers");
}

static void test_RetargetNullInputs() {
  printf("TEST RetargetNullInputs...\n");
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(nullptr, 0, kTargetIsa,
                                                   &out_data, &out_size);
  check(rc != 0, "RetargetCodeObject rejects null elf_data");
}

static void test_RetargetNullTarget() {
  printf("TEST RetargetNullTarget...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf),
                                                   nullptr, &out_data, &out_size);
  check(rc != 0, "RetargetCodeObject rejects null target_isa");
}

int main() {
  test_RetargetInvalidCodeObjectFallsBack();
  test_RetargetNullOutputPointers();
  test_RetargetNullInputs();
  test_RetargetNullTarget();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
