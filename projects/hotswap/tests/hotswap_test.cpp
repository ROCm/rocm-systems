//===- hotswap_test.cpp - Test HotSwap ISA rewriting API ------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char *name) {
  if (cond) {
    ++tests_passed;
    printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    fprintf(stderr, "  FAIL: %s\n", name);
  }
}

static void test_RetargetPassthrough() {
  printf("TEST RetargetPassthrough...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), "amdgcn-amd-amdhsa--gfx1250",
      "amdgcn-amd-amdhsa--gfx1250", &out_data, &out_size);

  check(rc == 0, "RetargetCodeObject succeeds");
  check(out_size == sizeof(fake_elf), "output size matches input");
  if (out_data && out_data != fake_elf) {
    check(memcmp(out_data, fake_elf, sizeof(fake_elf)) == 0,
          "output bytes match input (passthrough)");
    std::free(out_data);
  } else {
    check(out_data != nullptr && out_data != fake_elf,
          "out_data is a new allocation");
  }
}

static void test_RetargetNullOutputPointers() {
  printf("TEST RetargetNullOutputPointers...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), "amdgcn-amd-amdhsa--gfx1250",
      "amdgcn-amd-amdhsa--gfx1250", nullptr, nullptr);
  check(rc != 0, "RetargetCodeObject rejects null output pointers");
}

static void test_RetargetNullInputs() {
  printf("TEST RetargetNullInputs...\n");
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      nullptr, 0, "amdgcn-amd-amdhsa--gfx1250", "amdgcn-amd-amdhsa--gfx1250",
      &out_data, &out_size);
  check(rc != 0, "RetargetCodeObject rejects null elf_data");
}

static void test_RetargetOutputOwnership() {
  // Verifies that on success, out_data is a NEW allocation (not the input).
  // This catches the bug where out_elf was stashed unconditionally —
  // the caller must be able to distinguish "new buffer to free" from
  // "original buffer, don't free".
  printf("TEST RetargetOutputOwnership...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), "amdgcn-amd-amdhsa--gfx1250",
      "amdgcn-amd-amdhsa--gfx1250", &out_data, &out_size);

  check(rc == 0, "RetargetCodeObject succeeds");
  check(out_data != static_cast<const void *>(fake_elf),
        "out_data is NOT the original input (new allocation)");
  check(out_data != nullptr, "out_data is non-null");

  if (out_data && out_data != fake_elf) {
    // Verify we can write to it (proves it's a heap allocation, not const)
    memset(out_data, 0xAB, out_size);
    std::free(out_data);
    check(true, "out_data is writable and freeable");
  }
}

static void test_RetargetFailureLeavesOriginal() {
  // Verifies that on failure, out_data points to the original input
  // (not a dangling pointer or null). This catches use-after-free bugs
  // where the fallback buffer was freed prematurely.
  printf("TEST RetargetFailureLeavesOriginal...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  void *out_data = nullptr;
  size_t out_size = 0;

  // Use an unsupported ISA pair to force failure
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), "amdgcn-amd-amdhsa--gfx_unsupported",
      "amdgcn-amd-amdhsa--gfx_unsupported", &out_data, &out_size);

  check(rc != 0, "RetargetCodeObject fails for unsupported ISA");
  check(out_data == static_cast<const void *>(fake_elf),
        "out_data still points to original input after failure");
  check(out_size == sizeof(fake_elf),
        "out_size still matches original after failure");
}

int main() {
  test_RetargetPassthrough();
  test_RetargetNullOutputPointers();
  test_RetargetNullInputs();
  test_RetargetOutputOwnership();
  test_RetargetFailureLeavesOriginal();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
