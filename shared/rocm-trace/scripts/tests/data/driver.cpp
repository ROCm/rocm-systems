// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

// Installs the generated fake probes over stub entries and checks that calls go
// through them unchanged. Exits non-zero on the first failed check. Run it in
// an LTTng session to record the events.

#include "fake_probes.h"

#include <rocm/trace/trace.h>

#include <cstddef>
#include <cstdio>
#include <cstring>

TRACELOGGING_DEFINE_PROVIDER(fake_provider, "fake_provider",
                             (0x1e6b2a3c, 0x5d4f, 0x4a8e, 0x9b, 0x1c, 0x2f,
                              0x3d, 0x4e, 0x5f, 0x6a, 0x7b));

namespace {

int failures = 0;
int no_result_calls = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
  }
}

fakeError_t stub_scalars(int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                         uint32_t, uint64_t, size_t, bool, float, double) {
  return fakeSuccess;
}
fakeError_t stub_alloc(void **ptr, size_t size) {
  if (size == 0)
    return fakeErrorInvalidValue;
  *ptr = reinterpret_cast<void *>(0x1234);
  return fakeSuccess;
}
fakeError_t stub_name(char *out, int len) {
  if (len < 5)
    return fakeErrorInvalidValue;
  std::strcpy(out, "fake");
  return fakeSuccess;
}
fakeError_t stub_load(const char *, fakeStream_t) { return fakeSuccess; }
fakeError_t stub_launch(fakeDim3, fakeDim3, fakeStream_t) {
  return fakeSuccess;
}
fakeError_t stub_open(fakeHandle, void **) { return fakeSuccess; }
fakeError_t stub_raw_fn(void (*)(void *), void *) { return fakeSuccess; }
const char *stub_error_string(fakeError_t e) {
  return e == fakeSuccess ? "no error" : "invalid value";
}
void stub_no_result(int x) { no_result_calls += x; }
fakeDim3 stub_make_dim(uint32_t x) { return fakeDim3{x, 2, 3}; }
fakeError_t stub_get_dim(fakeDim3 *dim) {
  *dim = fakeDim3{7, 8, 9};
  return fakeSuccess;
}
fakeError_t stub_wide(fakeWide, fakeProps *) { return fakeSuccess; }
fakeError_t stub_skipped() { return fakeSuccess; }
void stub_init() {}
fakeError_t stub_count(int, size_t *n) {
  *n = 42;
  return fakeSuccess;
}

FakeTable stub_table() {
  FakeTable t{};
  t.size = sizeof(FakeTable);
  t.fakeScalars_fn = stub_scalars;
  t.fakeAlloc_fn = stub_alloc;
  t.fakeName_fn = stub_name;
  t.fakeLoad_fn = stub_load;
  t.fakeLaunch_fn = stub_launch;
  t.fakeOpen_fn = stub_open;
  t.fakeCallback_fn = nullptr; // a null entry stays null
  t.fakeRawFn_fn = stub_raw_fn;
  t.fakeErrorString_fn = stub_error_string;
  t.fakeNoResult_fn = stub_no_result;
  t.fakeMakeDim_fn = stub_make_dim;
  t.fakeGetDim_fn = stub_get_dim;
  t.fakeWide_fn = stub_wide;
  t.fakeRetired_fn = reinterpret_cast<void *>(0x5678);
  t.fakeSkipped_fn = stub_skipped;
  return t;
}

void call_all(const FakeTable &t, const FakeDeclTable &d) {
  check(t.fakeScalars_fn(-1, -2, -3, -4, 1, 2, 3, 4, 5, true, 1.5f, 2.5) ==
            fakeSuccess,
        "scalars");

  void *ptr = nullptr;
  check(t.fakeAlloc_fn(&ptr, 16) == fakeSuccess &&
            ptr == reinterpret_cast<void *>(0x1234),
        "alloc");
  check(t.fakeAlloc_fn(&ptr, 0) == fakeErrorInvalidValue,
        "alloc failure passes through");

  char buf[16] = {};
  check(t.fakeName_fn(buf, sizeof(buf)) == fakeSuccess &&
            std::strcmp(buf, "fake") == 0,
        "name");
  check(t.fakeLoad_fn("/tmp/kernel.co", reinterpret_cast<fakeStream_t>(0x10)) ==
            fakeSuccess,
        "load");
  check(t.fakeLaunch_fn(fakeDim3{4, 5, 6}, fakeDim3{64, 1, 1}, nullptr) ==
            fakeSuccess,
        "launch");
  check(t.fakeOpen_fn(fakeHandle{}, &ptr) == fakeSuccess, "open");
  check(t.fakeRawFn_fn(nullptr, nullptr) == fakeSuccess,
        "raw function pointer");
  check(std::strcmp(t.fakeErrorString_fn(fakeErrorInvalidValue),
                    "invalid value") == 0,
        "error string");
  t.fakeNoResult_fn(3);
  check(no_result_calls == 3, "no result");
  fakeDim3 made = t.fakeMakeDim_fn(1);
  check(made.x == 1 && made.y == 2 && made.z == 3, "struct result");
  fakeDim3 got{};
  check(t.fakeGetDim_fn(&got) == fakeSuccess && got.x == 7 && got.z == 9,
        "struct output");
  check(t.fakeWide_fn(fakeWideBig, nullptr) == fakeSuccess, "8-byte enum");
  check(t.fakeSkipped_fn() == fakeSuccess, "skipped");

  d.fakeInit_fn();
  size_t n = 0;
  check(d.fakeCount_fn(0, &n) == fakeSuccess && n == 42, "decltype entry");
}

} // namespace

int main() {
  TraceLoggingRegister(fake_provider);

  FakeTable t = stub_table();
  FakeDeclTable d{};
  d.version.minor_id = sizeof(FakeDeclTable);
  d.fakeInit_fn = stub_init;
  d.fakeCount_fn = stub_count;

  rocm_trace_install_FakeTable(&t);
  rocm_trace_install_FakeDeclTable(&d);
  check(t.fakeAlloc_fn != stub_alloc && t.fakeNoResult_fn != stub_no_result,
        "entries replaced");
  check(t.fakeCallback_fn == nullptr, "null entry stays null");
  check(t.fakeSkipped_fn == stub_skipped, "skipped entry untouched");
  check(t.fakeRetired_fn == reinterpret_cast<void *>(0x5678),
        "retired entry untouched");
  check(d.fakeCount_fn != stub_count, "decltype entry replaced");

  // Installing again must not wrap a probe in itself.
  auto const probe = t.fakeAlloc_fn;
  rocm_trace_install_FakeTable(&t);
  check(t.fakeAlloc_fn == probe, "second install is harmless");

  call_all(t, d);

  // A table from an older runtime: only the entries within its size are
  // wrapped.
  FakeTable small = stub_table();
  small.size = offsetof(FakeTable, fakeLoad_fn);
  rocm_trace_install_FakeTable(&small);
  check(small.fakeName_fn != stub_name, "entry within size wrapped");
  check(small.fakeLoad_fn == stub_load, "entry beyond size untouched");

  TraceLoggingUnregister(fake_provider);
  if (failures == 0)
    std::printf("all checks passed\n");
  return failures == 0 ? 0 : 1;
}
