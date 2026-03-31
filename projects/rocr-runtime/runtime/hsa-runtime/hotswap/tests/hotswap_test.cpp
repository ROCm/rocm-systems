//===- hotswap_test.cpp - Test HotSwap B0-to-A0 API ----------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../hotswap.hpp"
#include "../hotswap_comgr_client.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
  printf("TEST %s...\n", #name); \
  test_##name(); \
  tests_passed++;

static void test_ComgrUnavailable() {
  // With no COMGR library loaded, the client should report unavailable
  assert(!rocr::hotswap::ComgrHotswapAvailable());
}

static void test_B0A0GrowReturnsInputWhenComgrUnavailable() {
  // When COMGR is not available, RetargetCodeObjectB0A0Grow should
  // fail and leave out_data pointing to the original input
  const unsigned char fake_elf[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  int rc = rocr::hotswap::RetargetCodeObjectB0A0Grow(
      fake_elf, sizeof(fake_elf), &out_data, &out_size);

  // Should fail since COMGR is not available
  assert(rc != 0);
  // out_data should still point to original input (not freed, not allocated)
  assert(out_data == fake_elf);
  assert(out_size == sizeof(fake_elf));
}

int main() {
  TEST(ComgrUnavailable);
  TEST(B0A0GrowReturnsInputWhenComgrUnavailable);

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed;
}
