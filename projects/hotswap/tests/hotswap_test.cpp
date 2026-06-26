//===- hotswap_test.cpp - Test HotSwap ISA rewriting API ------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
// Generated from tests/fixtures/gfx1250_min.hsaco: kGfx1250MinCo[]. Embedding it
// keeps the test GPU-free and free of any runtime file dependency (CI-portable).
#include "gfx1250_min_hsaco.h"
#include <cstdio>
#include <cstdlib>

static int tests_passed = 0;
static int tests_failed = 0;

static constexpr const char *kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";

static void check(bool cond, const char *name) {
  if (cond) {
    ++tests_passed;
    printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    fprintf(stderr, "  FAIL: %s\n", name);
  }
}

// A real gfx1250 code object: COMGR parses its ISA name from the metadata note.
static void test_GetIsaNameRealCodeObject() {
  printf("TEST GetIsaNameRealCodeObject...\n");
  const std::string isa =
      rocr::hotswap::GetCodeObjectIsaName(kGfx1250MinCo, sizeof(kGfx1250MinCo));
  check(isa == kGfx1250Isa, "GetCodeObjectIsaName reads gfx1250 from a real CO");
}

// A 16-byte fake ELF has no metadata note: the ISA name cannot be parsed.
static void test_GetIsaNameInvalidCodeObject() {
  printf("TEST GetIsaNameInvalidCodeObject...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  const std::string isa =
      rocr::hotswap::GetCodeObjectIsaName(fake_elf, sizeof(fake_elf));
  check(isa.empty(), "GetCodeObjectIsaName returns empty for an invalid CO");
}

// A real gfx1250 CO rewrites successfully and yields a fresh output buffer.
static void test_RetargetRealCodeObject() {
  printf("TEST RetargetRealCodeObject...\n");
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      kGfx1250MinCo, sizeof(kGfx1250MinCo), kGfx1250Isa, kGfx1250Isa, &out_data,
      &out_size);

  check(rc == 0, "RetargetCodeObject rewrites a real CO");
  check(out_data != nullptr &&
            out_data != static_cast<const void *>(kGfx1250MinCo),
        "out_data is a fresh allocation");
  check(out_size > 0, "out_size is non-zero");
  if (out_data && out_data != static_cast<const void *>(kGfx1250MinCo)) {
    std::free(out_data);
  }
}

// An un-rewritable (fake) CO fails and leaves the original code object intact.
static void test_RetargetInvalidCodeObjectFallsBack() {
  printf("TEST RetargetInvalidCodeObjectFallsBack...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa, &out_data,
      &out_size);

  check(rc != 0, "RetargetCodeObject fails for an un-rewritable CO");
  check(out_data == static_cast<const void *>(fake_elf),
        "out_data still points to original input after failure");
  check(out_size == sizeof(fake_elf),
        "out_size still matches original after failure");
}

static void test_RetargetNullOutputPointers() {
  printf("TEST RetargetNullOutputPointers...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  const int rc = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa, nullptr, nullptr);
  check(rc != 0, "RetargetCodeObject rejects null output pointers");
}

static void test_RetargetNullInputs() {
  printf("TEST RetargetNullInputs...\n");
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      nullptr, 0, kGfx1250Isa, kGfx1250Isa, &out_data, &out_size);
  check(rc != 0, "RetargetCodeObject rejects null elf_data");
}

static void test_RetargetNullSourceOrTarget() {
  printf("TEST RetargetNullSourceOrTarget...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  void *out_data = nullptr;
  size_t out_size = 0;
  const int rc_src = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), nullptr, kGfx1250Isa, &out_data, &out_size);
  check(rc_src != 0, "RetargetCodeObject rejects null source_isa");
  const int rc_tgt = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, nullptr, &out_data, &out_size);
  check(rc_tgt != 0, "RetargetCodeObject rejects null target_isa");
}

int main() {
  // Line-buffer stdout so the tool's stderr diagnostics interleave next to the
  // test that triggered them (e.g. under ctest -V) instead of clumping.
  setvbuf(stdout, nullptr, _IOLBF, 0);
  printf("Note: the negative tests below intentionally trigger tool diagnostics\n"
         "on stderr (\"COMGR rewrite failed ... rc=2\", \"invalid null ...\").\n"
         "These are EXPECTED; a passing run is the PASS lines + the final count.\n\n");

  test_GetIsaNameRealCodeObject();
  test_GetIsaNameInvalidCodeObject();
  test_RetargetRealCodeObject();
  test_RetargetInvalidCodeObjectFallsBack();
  test_RetargetNullOutputPointers();
  test_RetargetNullInputs();
  test_RetargetNullSourceOrTarget();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
