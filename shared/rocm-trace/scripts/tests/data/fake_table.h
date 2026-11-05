// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

// Dispatch tables with one entry per case the generator handles.
// Self-contained: no system headers, so the tests need no clang resource
// directory.

#ifndef ROCM_TRACE_GEN_FAKE_TABLE_H
#define ROCM_TRACE_GEN_FAKE_TABLE_H

typedef __SIZE_TYPE__ size_t;
typedef __INT8_TYPE__ int8_t;
typedef __INT16_TYPE__ int16_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;

typedef enum fakeError_t {
  fakeSuccess = 0,
  fakeErrorInvalidValue = 1
} fakeError_t;
enum fakeWide : unsigned long long { fakeWideBig = 1ULL << 40 };

typedef struct fakeStream *fakeStream_t;
typedef struct fakeDim3 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} fakeDim3;
typedef struct fakeHandle {
  char reserved[16];
} fakeHandle;
typedef struct fakeProps {
  int a, b, c, d, e, f, g, h, i;
} fakeProps;
typedef void (*fakeCallback_t)(fakeStream_t stream, void *user);

// Numbers of every width, bool, and floating point.
typedef fakeError_t (*t_fakeScalars)(int8_t a, int16_t b, int32_t c, int64_t d,
                                     uint8_t e, uint16_t f, uint32_t g,
                                     uint64_t h, size_t i, bool j, float k,
                                     double l);
// An output pointer, recorded on exit after success.
typedef fakeError_t (*t_fakeAlloc)(void **ptr, size_t size);
// An output string.
typedef fakeError_t (*t_fakeName)(char *name, int len);
// An input string and a handle.
typedef fakeError_t (*t_fakeLoad)(const char *path, fakeStream_t stream);
// By-value structs, recorded one field per member.
typedef fakeError_t (*t_fakeLaunch)(fakeDim3 grid, fakeDim3 block,
                                    fakeStream_t stream);
// A by-value struct with an array member: must be omitted.
typedef fakeError_t (*t_fakeOpen)(fakeHandle handle, void **ptr);
// Function pointers: through a typedef, and spelled out (needs an alias).
typedef fakeError_t (*t_fakeCallback)(fakeStream_t stream, fakeCallback_t cb,
                                      void *user);
typedef fakeError_t (*t_fakeRawFn)(void (*fn)(void *), void *arg);
// A string result and an enum parameter.
typedef const char *(*t_fakeErrorString)(fakeError_t error);
// No result.
typedef void (*t_fakeNoResult)(int x);
// A struct result.
typedef fakeDim3 (*t_fakeMakeDim)(uint32_t x);
// An output struct.
typedef fakeError_t (*t_fakeGetDim)(fakeDim3 *dim);
// An 8-byte enum, and a pointer to a struct too large to record.
typedef fakeError_t (*t_fakeWide)(fakeWide value, fakeProps *props);
// Not traced.
typedef fakeError_t (*t_fakeSkipped)(void);

struct FakeTable {
  size_t size;
  t_fakeScalars fakeScalars_fn;
  t_fakeAlloc fakeAlloc_fn;
  t_fakeName fakeName_fn;
  t_fakeLoad fakeLoad_fn;
  t_fakeLaunch fakeLaunch_fn;
  t_fakeOpen fakeOpen_fn;
  t_fakeCallback fakeCallback_fn;
  t_fakeRawFn fakeRawFn_fn;
  t_fakeErrorString fakeErrorString_fn;
  t_fakeNoResult fakeNoResult_fn;
  t_fakeMakeDim fakeMakeDim_fn;
  t_fakeGetDim fakeGetDim_fn;
  t_fakeWide fakeWide_fn;
  void *fakeRetired_fn;
  t_fakeSkipped fakeSkipped_fn;
};

// HSA style: entries declared with decltype, size inside a version member.
void fakeInit(void);
fakeError_t fakeCount(int device, size_t *count);

struct FakeVersion {
  uint32_t major_id;
  uint32_t minor_id;
};

struct FakeDeclTable {
  FakeVersion version;
  decltype(fakeInit) *fakeInit_fn;
  decltype(fakeCount) *fakeCount_fn;
};

#endif // ROCM_TRACE_GEN_FAKE_TABLE_H
