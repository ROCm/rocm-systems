// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

// Dispatch tables with one entry per case the generator handles. Under Clang
// and GCC, including libclang, it uses no system headers, so the generator
// tests need no clang resource directory.

#ifndef ROCM_TRACE_GEN_FAKE_TABLE_H
#define ROCM_TRACE_GEN_FAKE_TABLE_H

#if defined(__SIZE_TYPE__) && defined(__UINT64_TYPE__)
// Compiler builtins (Clang, GCC), so libclang needs no system headers.
typedef __SIZE_TYPE__ size_t;
typedef __INT8_TYPE__ int8_t;
typedef __INT16_TYPE__ int16_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;
#else
#include <stddef.h>
#include <stdint.h>
#endif

typedef enum fakeError_t {
  fakeSuccess = 0,
  fakeErrorInvalidValue = 1
} fakeError_t;
enum fakeWide : unsigned long long { fakeWideBig = 1ULL << 40 };
enum fakeFlags : unsigned int { fakeFlagHigh = 0x80000000u };

typedef struct fakeStream *fakeStream_t;
typedef struct fakeFunction *fakeFunction_t;
typedef struct fakeDim3 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} fakeDim3;
// A struct holding an array.
typedef struct fakeHandle {
  char reserved[16];
} fakeHandle;
// Nested structs, an array of structs, and an array of pointers.
typedef struct fakeLaunchConfig {
  fakeDim3 grid;
  fakeDim3 block;
  fakeDim3 extra[2];
  void *slots[2];
  float weights[2];
  unsigned int flags;
  struct {
    unsigned char a;
    unsigned short b;
  } named; // a named member of an unnamed struct type
} fakeLaunchConfig;
// A named union, and a struct with an anonymous one.
typedef union fakeBits {
  uint32_t u;
  float f;
} fakeBits;
typedef struct fakeValue {
  int kind;
  union {
    int32_t i;
    double d;
  };
} fakeValue;
// A member with no field mapping.
typedef struct fakeProps {
  int a;
  long double b;
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
// A const char* input, recorded as a capped string, and a handle.
typedef fakeError_t (*t_fakeLoad)(const char *path, fakeStream_t stream);
// A kernel name, recorded whole, and an output handle.
typedef fakeError_t (*t_fakeGetFunction)(fakeFunction_t *function,
                                         const char *kname);
// A non-const char* input: an address, not a string.
typedef fakeError_t (*t_fakeScratch)(char *buffer, size_t size);
// By-value structs, recorded one field per component.
typedef fakeError_t (*t_fakeLaunch)(fakeDim3 grid, fakeDim3 block,
                                    fakeStream_t stream);
typedef fakeError_t (*t_fakeLaunchEx)(fakeLaunchConfig config);
typedef fakeError_t (*t_fakeOpen)(fakeHandle handle, void **ptr);
typedef fakeError_t (*t_fakeBits)(fakeBits bits, fakeValue value);
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
// An 8-byte enum, an unsigned enum, and a pointer to a struct that has no
// field mapping (fine as an address, an error as an output).
typedef fakeError_t (*t_fakeWide)(fakeWide value, fakeFlags flags,
                                  fakeProps *props);
// Unnamed parameters: arg_1, arg_2.
typedef fakeError_t (*t_fakeUnnamed)(int, const char *);
// Names RC-API-1 prefixes with arg_, one of them an output.
typedef fakeError_t (*t_fakeReserved)(int result, uint64_t timestamp,
                                      int *exception);
// Throws in the tests.
typedef fakeError_t (*t_fakeThrow)(int x);
// Not traced.
typedef fakeError_t (*t_fakeSkipped)(void);

struct FakeTable {
  size_t size;
  t_fakeScalars fakeScalars_fn;
  t_fakeAlloc fakeAlloc_fn;
  t_fakeName fakeName_fn;
  t_fakeLoad fakeLoad_fn;
  t_fakeGetFunction fakeGetFunction_fn;
  t_fakeScratch fakeScratch_fn;
  t_fakeLaunch fakeLaunch_fn;
  t_fakeLaunchEx fakeLaunchEx_fn;
  t_fakeOpen fakeOpen_fn;
  t_fakeBits fakeBits_fn;
  t_fakeCallback fakeCallback_fn;
  t_fakeRawFn fakeRawFn_fn;
  t_fakeErrorString fakeErrorString_fn;
  t_fakeNoResult fakeNoResult_fn;
  t_fakeMakeDim fakeMakeDim_fn;
  t_fakeGetDim fakeGetDim_fn;
  t_fakeWide fakeWide_fn;
  t_fakeUnnamed fakeUnnamed_fn;
  t_fakeReserved fakeReserved_fn;
  t_fakeThrow fakeThrow_fn;
  void *fakeRetired_fn;
  t_fakeSkipped fakeSkipped_fn;
};

// HSA style: entries declared with decltype, size inside a version member.
// fakeCount's public declaration names its parameters differently from the
// table typedef style; the public names win (RC-API-1).
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

// A public declaration whose names differ from the table's typedef: the
// public names win (RC-API-1).
typedef fakeError_t (*t_fakeRenamed)(int typedef_name);
extern "C" fakeError_t fakeRenamed(int public_name);

struct FakeRenamedTable {
  size_t size;
  t_fakeRenamed fakeRenamed_fn;
};

#endif // ROCM_TRACE_GEN_FAKE_TABLE_H
