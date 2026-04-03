//===- hotswap_test.cpp - Test HotSwap ISA rewriting API ------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include "hotswap_comgr_client.hpp"
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

static void test_ComgrUnavailable() {
  printf("TEST ComgrUnavailable...\n");
  check(!rocr::hotswap::ComgrHotswapAvailable(),
        "ComgrHotswapAvailable returns false when unbound");
}

static void test_RetargetReturnsInputWhenComgrUnavailable() {
  printf("TEST RetargetReturnsInputWhenComgrUnavailable...\n");
  const unsigned char fake_elf[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf),
      "amdgcn-amd-amdhsa--gfx1250", "amdgcn-amd-amdhsa--gfx1250",
      &out_data, &out_size);

  check(rc != 0, "RetargetCodeObject fails without COMGR");
  check(out_data == fake_elf, "out_data points to original input");
  check(out_size == sizeof(fake_elf), "out_size matches original size");
}

static void test_RetargetNullOutputPointers() {
  printf("TEST RetargetNullOutputPointers...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf),
      "amdgcn-amd-amdhsa--gfx1250", "amdgcn-amd-amdhsa--gfx1250",
      nullptr, nullptr);
  check(rc != 0, "RetargetCodeObject rejects null output pointers");
}

int main() {
  test_ComgrUnavailable();
  test_RetargetReturnsInputWhenComgrUnavailable();
  test_RetargetNullOutputPointers();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
