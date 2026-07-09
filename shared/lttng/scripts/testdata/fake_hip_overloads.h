/* Test fixture — mirrors HIP's real hipMallocAsync duplicate-declaration
 * shape, used by test_lttng_curated_codegen.py's live-header resolution
 * tests. Self-contained (no system include resolution needed).
 *
 * fakeMallocAsync: a plain extern declaration plus a later `static
 * inline` convenience overload with an extra `mem_pool` parameter --
 * exactly HIP's real hipMallocAsync(dev_ptr, size, stream) vs.
 * hipMallocAsync(dev_ptr, size, mem_pool, stream) shape. Resolution must
 * prefer the extern declaration when the YAML only references
 * dev_ptr/size/stream.
 *
 * fakeAmbiguousApi: two non-static overloads that both contain the same
 * arg NAMES (just different types for `c`) -- genuinely ambiguous, must
 * fail loudly rather than silently pick one. */
#ifndef FAKE_HIP_OVERLOADS_H_
#define FAKE_HIP_OVERLOADS_H_

typedef unsigned long size_t;

typedef enum { hipSuccess = 0 } hipError_t;

struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t;

struct ihipMemPool_t;
typedef struct ihipMemPool_t* hipMemPool_t;

hipError_t fakeMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
static inline hipError_t fakeMallocAsync(void** dev_ptr, size_t size,
                                         hipMemPool_t mem_pool, hipStream_t stream) {
  return hipSuccess;
}

hipError_t fakeAmbiguousApi(void* a, size_t b, int c);
hipError_t fakeAmbiguousApi(void* a, size_t b, hipStream_t c);

#endif
