// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

// Installs the generated fake probes over stub entries and checks that calls go
// through them unchanged. Exits non-zero if a check fails. Run it in an LTTng
// session to record the events; test_build.py checks the recorded fields.

#include "fake_probes.h"

#include <rocm/trace/trace.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

TRACELOGGING_DEFINE_PROVIDER(fake_provider, "fake_provider",
                             (0x1e6b2a3c, 0x5d4f, 0x4a8e, 0x9b, 0x1c, 0x2f,
                              0x3d, 0x4e, 0x5f, 0x6a, 0x7b));

int fake_capture_level = ROCM_TRACE_CAPTURE_FULL;

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
  std::memcpy(out, "fake", 5);
  return fakeSuccess;
}
fakeError_t stub_load(const char *, fakeStream_t) { return fakeSuccess; }
fakeError_t stub_get_function(fakeFunction_t *function, const char *) {
  *function = reinterpret_cast<fakeFunction_t>(0x77);
  return fakeSuccess;
}
fakeError_t stub_scratch(char *, size_t) { return fakeSuccess; }
fakeError_t stub_launch(fakeDim3, fakeDim3, fakeStream_t) {
  return fakeSuccess;
}
fakeError_t stub_launch_ex(fakeLaunchConfig config) {
  return config.flags == 3 ? fakeSuccess : fakeErrorInvalidValue;
}
fakeError_t stub_open(fakeHandle, void **) { return fakeSuccess; }
fakeError_t stub_bits(fakeBits, fakeValue) { return fakeSuccess; }
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
fakeError_t stub_wide(fakeWide, fakeFlags, fakeProps *) { return fakeSuccess; }
fakeError_t stub_unnamed(int, const char *) { return fakeSuccess; }
fakeError_t stub_reserved(int, uint64_t, int *exception) {
  *exception = 5;
  return fakeSuccess;
}
fakeError_t stub_throw(int) { throw std::runtime_error("fake failure"); }
fakeError_t stub_skipped() { return fakeSuccess; }
void stub_init() {}
fakeError_t stub_count(int, size_t *n) {
  *n = 42;
  return fakeSuccess;
}
fakeError_t stub_renamed(int value) {
  return value == 1 ? fakeSuccess : fakeErrorInvalidValue;
}

FakeTable stub_table() {
  FakeTable t{};
  t.size = sizeof(FakeTable);
  t.fakeScalars_fn = stub_scalars;
  t.fakeAlloc_fn = stub_alloc;
  t.fakeName_fn = stub_name;
  t.fakeLoad_fn = stub_load;
  t.fakeGetFunction_fn = stub_get_function;
  t.fakeScratch_fn = stub_scratch;
  t.fakeLaunch_fn = stub_launch;
  t.fakeLaunchEx_fn = stub_launch_ex;
  t.fakeOpen_fn = stub_open;
  t.fakeBits_fn = stub_bits;
  t.fakeCallback_fn = nullptr; // a null entry stays null
  t.fakeRawFn_fn = stub_raw_fn;
  t.fakeErrorString_fn = stub_error_string;
  t.fakeNoResult_fn = stub_no_result;
  t.fakeMakeDim_fn = stub_make_dim;
  t.fakeGetDim_fn = stub_get_dim;
  t.fakeWide_fn = stub_wide;
  t.fakeUnnamed_fn = stub_unnamed;
  t.fakeReserved_fn = stub_reserved;
  t.fakeThrow_fn = stub_throw;
  t.fakeRetired_fn = reinterpret_cast<void *>(0x5678);
  t.fakeSkipped_fn = stub_skipped;
  return t;
}

void call_all(const FakeTable &t, const FakeDeclTable &d,
              const FakeRenamedTable &r) {
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
        "output string");

  // 300 characters: the path is capped at 256, the kernel name is not.
  std::string const long_path = "/" + std::string(299, 'p');
  std::string const long_kname = std::string(300, 'k');
  check(t.fakeLoad_fn(long_path.c_str(),
                      reinterpret_cast<fakeStream_t>(0x10)) == fakeSuccess,
        "input string");
  fakeFunction_t function = nullptr;
  check(t.fakeGetFunction_fn(&function, long_kname.c_str()) == fakeSuccess &&
            function == reinterpret_cast<fakeFunction_t>(0x77),
        "kernel name");
  check(t.fakeScratch_fn(buf, sizeof(buf)) == fakeSuccess, "char* input");

  check(t.fakeLaunch_fn(fakeDim3{4, 5, 6}, fakeDim3{64, 1, 1}, nullptr) ==
            fakeSuccess,
        "launch");
  fakeLaunchConfig config{};
  config.grid = fakeDim3{2, 1, 1};
  config.extra[1] = fakeDim3{9, 9, 9};
  config.slots[0] = reinterpret_cast<void *>(0xabc);
  config.weights[1] = 0.5f;
  config.named.b = 12;
  config.flags = 3;
  check(t.fakeLaunchEx_fn(config) == fakeSuccess, "nested struct");
  fakeHandle handle{};
  handle.reserved[0] = 'h';
  check(t.fakeOpen_fn(handle, &ptr) == fakeSuccess, "struct with an array");
  fakeBits bits{};
  bits.u = 0x01020304u;
  fakeValue value{};
  value.kind = 1;
  value.i = 7;
  check(t.fakeBits_fn(bits, value) == fakeSuccess, "unions");
  check(t.fakeRawFn_fn(nullptr, nullptr) == fakeSuccess,
        "raw function pointer");
  check(std::strcmp(t.fakeErrorString_fn(fakeErrorInvalidValue),
                    "invalid value") == 0,
        "string result");
  t.fakeNoResult_fn(3);
  check(no_result_calls == 3, "no result");
  fakeDim3 made = t.fakeMakeDim_fn(1);
  check(made.x == 1 && made.y == 2 && made.z == 3, "struct result");
  fakeDim3 got{};
  check(t.fakeGetDim_fn(&got) == fakeSuccess && got.x == 7 && got.z == 9,
        "struct output");
  check(t.fakeWide_fn(fakeWideBig, fakeFlagHigh, nullptr) == fakeSuccess,
        "enums");
  check(t.fakeUnnamed_fn(1, "unnamed") == fakeSuccess, "unnamed parameters");
  int out = 0;
  check(t.fakeReserved_fn(1, 2, &out) == fakeSuccess && out == 5,
        "reserved names");
  check(t.fakeSkipped_fn() == fakeSuccess, "skipped");

  bool caught = false;
  try {
    t.fakeThrow_fn(1);
  } catch (const std::runtime_error &e) {
    caught = std::strcmp(e.what(), "fake failure") == 0;
  }
  check(caught, "an exception passes through unchanged");

  d.fakeInit_fn();
  size_t n = 0;
  check(d.fakeCount_fn(0, &n) == fakeSuccess && n == 42, "decltype entry");
  check(r.fakeRenamed_fn(1) == fakeSuccess, "renamed parameter");
}

void call_at_lower_capture(const FakeTable &t) {
  void *ptr = nullptr;
  fake_capture_level = ROCM_TRACE_CAPTURE_SCALARS;
  check(t.fakeLoad_fn("/tmp/scalars.co",
                      reinterpret_cast<fakeStream_t>(0x20)) == fakeSuccess,
        "load, scalars");
  check(t.fakeAlloc_fn(&ptr, 32) == fakeSuccess, "alloc, scalars");
  fake_capture_level = ROCM_TRACE_CAPTURE_NONE;
  check(t.fakeAlloc_fn(&ptr, 48) == fakeSuccess, "alloc, none");
  fake_capture_level = ROCM_TRACE_CAPTURE_FULL;
}

} // namespace

int main() {
  TraceLoggingRegister(fake_provider);

  FakeTable t = stub_table();
  FakeDeclTable d{};
  d.version.minor_id = sizeof(FakeDeclTable);
  d.fakeInit_fn = stub_init;
  d.fakeCount_fn = stub_count;
  FakeRenamedTable r{};
  r.size = sizeof(FakeRenamedTable);
  r.fakeRenamed_fn = stub_renamed;

  rocm_trace_install_FakeTable(&t);
  rocm_trace_install_FakeDeclTable(&d);
  rocm_trace_install_FakeRenamedTable(&r);
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

  call_all(t, d, r);
  call_at_lower_capture(t);

  // A table from an older runtime: only the entries within its size are
  // wrapped.
  // (Not named "small": windows.h defines that as a macro.)
  FakeTable older = stub_table();
  older.size = offsetof(FakeTable, fakeLoad_fn);
  rocm_trace_install_FakeTable(&older);
  check(older.fakeName_fn != stub_name, "entry within size wrapped");
  check(older.fakeLoad_fn == stub_load, "entry beyond size untouched");

  TraceLoggingUnregister(fake_provider);
  if (failures == 0)
    std::printf("all checks passed\n");
  return failures == 0 ? 0 : 1;
}
