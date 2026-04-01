//===- hotswap_test.cpp - Test HotSwap B0-to-A0 API ----------------------===//
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

static void test_B0A0GrowReturnsInputWhenComgrUnavailable() {
  printf("TEST B0A0GrowReturnsInputWhenComgrUnavailable...\n");
  const unsigned char fake_elf[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  int rc = rocr::hotswap::RetargetCodeObjectB0A0Grow(
      fake_elf, sizeof(fake_elf), &out_data, &out_size);

  check(rc != 0, "RetargetCodeObjectB0A0Grow fails without COMGR");
  check(out_data == fake_elf, "out_data points to original input");
  check(out_size == sizeof(fake_elf), "out_size matches original size");
}

int main() {
  test_ComgrUnavailable();
  test_B0A0GrowReturnsInputWhenComgrUnavailable();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
