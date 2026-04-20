/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/**
 * C wrappers for the rocshmem device API.
 *
 * JIT/bitcode consumers (e.g. PyTorch/Triton) need stable, unmangled symbol names.
 * Each extern "C" function forwards to the corresponding rocshmem:: API.
 *
 * (temporary until extern "C" refactor)
 * Current coverage:
 * - RMA: put/get/p/g + variants (wave, wg, nbi)
 * - AMO: standard, extended, bitwise
 * - Sync: wait_until variants, test
 * - Signal: put_signal variants
 *
 * Intentionally excluded (internal use only):
 * - Context methods
 * - Backend dispatchers
 * - Template functions
 */

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#define ROCSHMEM_DEVICE_API \
  __device__ __attribute__((visibility("default")))

extern "C" {

ROCSHMEM_DEVICE_API int rocshmem_my_pe() {
  return rocshmem::rocshmem_my_pe();
}

ROCSHMEM_DEVICE_API int rocshmem_n_pes() {
  return rocshmem::rocshmem_n_pes();
}

ROCSHMEM_DEVICE_API void *rocshmem_ptr(const void *dest, int pe) {
  return rocshmem::rocshmem_ptr(dest, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem(void *dest, const void *source,
                                         size_t nelems, int pe) {
  rocshmem::rocshmem_putmem(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_wave(void *dest, const void *source,
                                              size_t nelems, int pe) {
  rocshmem::rocshmem_putmem_wave(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_wg(void *dest, const void *source,
                                            size_t nelems, int pe) {
  rocshmem::rocshmem_putmem_wg(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_nbi(void *dest, const void *source,
                                             size_t nelems, int pe) {
  rocshmem::rocshmem_putmem_nbi(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_nbi_wave(void *dest,
                                                  const void *source,
                                                  size_t nelems, int pe) {
  rocshmem::rocshmem_putmem_nbi_wave(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_nbi_wg(void *dest,
                                                const void *source,
                                                size_t nelems, int pe) {
  rocshmem::rocshmem_putmem_nbi_wg(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem(void *dest, const void *source,
                                         size_t nelems, int pe) {
  rocshmem::rocshmem_getmem(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_wave(void *dest, const void *source,
                                              size_t nelems, int pe) {
  rocshmem::rocshmem_getmem_wave(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_wg(void *dest, const void *source,
                                            size_t nelems, int pe) {
  rocshmem::rocshmem_getmem_wg(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_nbi(void *dest, const void *source,
                                             size_t nelems, int pe) {
  rocshmem::rocshmem_getmem_nbi(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_nbi_wave(void *dest,
                                                  const void *source,
                                                  size_t nelems, int pe) {
  rocshmem::rocshmem_getmem_nbi_wave(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_nbi_wg(void *dest,
                                                const void *source,
                                                size_t nelems, int pe) {
  rocshmem::rocshmem_getmem_nbi_wg(dest, source, nelems, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal(void *dest, const void *source,
                                                size_t nelems,
                                                uint64_t *sig_addr,
                                                uint64_t signal, int sig_op,
                                                int pe) {
  rocshmem::rocshmem_putmem_signal(dest, source, nelems, sig_addr, signal,
                                   sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_wg(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_wg(dest, source, nelems, sig_addr, signal,
                                      sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_wave(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_wave(dest, source, nelems, sig_addr, signal,
                                        sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_nbi(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal,
                                       sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_nbi_wg(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_nbi_wg(dest, source, nelems, sig_addr,
                                          signal, sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_nbi_wave(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_nbi_wave(dest, source, nelems, sig_addr,
                                            signal, sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_barrier_all() {
  rocshmem::rocshmem_barrier_all();
}

ROCSHMEM_DEVICE_API void rocshmem_barrier_all_wg() {
  rocshmem::rocshmem_barrier_all_wg();
}

ROCSHMEM_DEVICE_API void rocshmem_barrier_all_wave() {
  rocshmem::rocshmem_barrier_all_wave();
}

ROCSHMEM_DEVICE_API void rocshmem_barrier() {
  rocshmem::rocshmem_barrier();
}

ROCSHMEM_DEVICE_API void rocshmem_barrier_wave() {
  rocshmem::rocshmem_barrier_wave();
}

ROCSHMEM_DEVICE_API void rocshmem_barrier_wg() {
  rocshmem::rocshmem_barrier_wg();
}

ROCSHMEM_DEVICE_API void rocshmem_sync_all() {
  rocshmem::rocshmem_sync_all();
}

ROCSHMEM_DEVICE_API void rocshmem_sync_all_wg() {
  rocshmem::rocshmem_sync_all_wg();
}

ROCSHMEM_DEVICE_API void rocshmem_sync_all_wave() {
  rocshmem::rocshmem_sync_all_wave();
}

ROCSHMEM_DEVICE_API void rocshmem_fence() {
  rocshmem::rocshmem_fence();
}

ROCSHMEM_DEVICE_API void rocshmem_fence_pe(int pe) {
  rocshmem::rocshmem_fence(pe);
}

ROCSHMEM_DEVICE_API void rocshmem_quiet() {
  rocshmem::rocshmem_quiet();
}

ROCSHMEM_DEVICE_API void rocshmem_pe_quiet(const int *target_pes,
                                           size_t npes) {
  rocshmem::rocshmem_pe_quiet(target_pes, npes);
}

ROCSHMEM_DEVICE_API void rocshmem_threadfence_system() {
  rocshmem::rocshmem_threadfence_system();
}

ROCSHMEM_DEVICE_API void rocshmem_query_thread(int *provided) {
  rocshmem::rocshmem_query_thread(provided);
}

ROCSHMEM_DEVICE_API uint64_t rocshmem_signal_fetch(
    const uint64_t *sig_addr) {
  return rocshmem::rocshmem_signal_fetch(sig_addr);
}

ROCSHMEM_DEVICE_API uint64_t rocshmem_signal_fetch_wg(
    const uint64_t *sig_addr) {
  return rocshmem::rocshmem_signal_fetch_wg(sig_addr);
}

ROCSHMEM_DEVICE_API uint64_t rocshmem_signal_fetch_wave(
    const uint64_t *sig_addr) {
  return rocshmem::rocshmem_signal_fetch_wave(sig_addr);
}

// The explicit instantiation pattern pre-compiles all type variants into bitcode,
// so JIT linkers don't need to instantiate templates at link time.
#define WRAP_RMA(T, TNAME)                                                     \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_put(                             \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_put(dest, source, nelems, pe);                \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_put_nbi(                         \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_put_nbi(dest, source, nelems, pe);            \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_p(T *dest, T value, int pe) {    \
    rocshmem::rocshmem_##TNAME##_p(dest, value, pe);                           \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_get(                             \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_get(dest, source, nelems, pe);                \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_get_nbi(                         \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_get_nbi(dest, source, nelems, pe);            \
  }                                                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_g(const T *source, int pe) {        \
    return rocshmem::rocshmem_##TNAME##_g(source, pe);                         \
  }

WRAP_RMA(float, float)
WRAP_RMA(double, double)
WRAP_RMA(char, char)
WRAP_RMA(signed char, schar)
WRAP_RMA(short, short)
WRAP_RMA(int, int)
WRAP_RMA(long, long)
WRAP_RMA(long long, longlong)
WRAP_RMA(unsigned char, uchar)
WRAP_RMA(unsigned short, ushort)
WRAP_RMA(unsigned int, uint)
WRAP_RMA(unsigned long, ulong)
WRAP_RMA(unsigned long long, ulonglong)

ROCSHMEM_DEVICE_API void rocshmem_int64_p(int64_t *dest, int64_t value,
                                          int pe) {
  rocshmem::rocshmem_int64_p(dest, value, pe);
}

#define WRAP_RMA_SUFFIX(T, TNAME, SUFFIX)                                      \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_put##SUFFIX(                     \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_put##SUFFIX(dest, source, nelems, pe);        \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_put_nbi##SUFFIX(                 \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_put_nbi##SUFFIX(dest, source, nelems, pe);    \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_get##SUFFIX(                     \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_get##SUFFIX(dest, source, nelems, pe);        \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_get_nbi##SUFFIX(                 \
      T *dest, const T *source, size_t nelems, int pe) {                       \
    rocshmem::rocshmem_##TNAME##_get_nbi##SUFFIX(dest, source, nelems, pe);    \
  }

#define WRAP_RMA_EXTENDED(T, TNAME)                                            \
  WRAP_RMA_SUFFIX(T, TNAME, _wave)                                             \
  WRAP_RMA_SUFFIX(T, TNAME, _wg)

// clang-format off
WRAP_RMA_EXTENDED(float, float)
WRAP_RMA_EXTENDED(double, double)
WRAP_RMA_EXTENDED(char, char)
WRAP_RMA_EXTENDED(signed char, schar)
WRAP_RMA_EXTENDED(short, short)
WRAP_RMA_EXTENDED(int, int)
WRAP_RMA_EXTENDED(long, long)
WRAP_RMA_EXTENDED(long long, longlong)
WRAP_RMA_EXTENDED(unsigned char, uchar)
WRAP_RMA_EXTENDED(unsigned short, ushort)
WRAP_RMA_EXTENDED(unsigned int, uint)
WRAP_RMA_EXTENDED(unsigned long, ulong)
WRAP_RMA_EXTENDED(unsigned long long, ulonglong)
// clang-format on

#define WRAP_AMO_STANDARD(T, TNAME)                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_compare_swap(                \
      T *dest, T cond, T value, int pe) {                                      \
    return rocshmem::rocshmem_##TNAME##_atomic_compare_swap(                    \
        dest, cond, value, pe);                                                \
  }                                                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_fetch_inc(                   \
      T *dest, int pe) {                                                       \
    return rocshmem::rocshmem_##TNAME##_atomic_fetch_inc(dest, pe);             \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_atomic_inc(T *dest, int pe) {    \
    rocshmem::rocshmem_##TNAME##_atomic_inc(dest, pe);                         \
  }                                                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_fetch_add(                   \
      T *dest, T value, int pe) {                                              \
    return rocshmem::rocshmem_##TNAME##_atomic_fetch_add(dest, value, pe);      \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_atomic_add(                      \
      T *dest, T value, int pe) {                                              \
    rocshmem::rocshmem_##TNAME##_atomic_add(dest, value, pe);                  \
  }

#define WRAP_AMO_EXTENDED(T, TNAME)                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_fetch(                       \
      T *source, int pe) {                                                     \
    return rocshmem::rocshmem_##TNAME##_atomic_fetch(source, pe);               \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_atomic_set(                      \
      T *dest, T value, int pe) {                                              \
    rocshmem::rocshmem_##TNAME##_atomic_set(dest, value, pe);                  \
  }                                                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_swap(                        \
      T *dest, T value, int pe) {                                              \
    return rocshmem::rocshmem_##TNAME##_atomic_swap(dest, value, pe);           \
  }

#define WRAP_AMO_BITWISE(T, TNAME)                                             \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_fetch_and(                   \
      T *dest, T value, int pe) {                                              \
    return rocshmem::rocshmem_##TNAME##_atomic_fetch_and(dest, value, pe);      \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_atomic_and(                      \
      T *dest, T value, int pe) {                                              \
    rocshmem::rocshmem_##TNAME##_atomic_and(dest, value, pe);                  \
  }                                                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_fetch_or(                    \
      T *dest, T value, int pe) {                                              \
    return rocshmem::rocshmem_##TNAME##_atomic_fetch_or(dest, value, pe);       \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_atomic_or(                       \
      T *dest, T value, int pe) {                                              \
    rocshmem::rocshmem_##TNAME##_atomic_or(dest, value, pe);                   \
  }                                                                            \
  ROCSHMEM_DEVICE_API T rocshmem_##TNAME##_atomic_fetch_xor(                   \
      T *dest, T value, int pe) {                                              \
    return rocshmem::rocshmem_##TNAME##_atomic_fetch_xor(dest, value, pe);      \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_atomic_xor(                      \
      T *dest, T value, int pe) {                                              \
    rocshmem::rocshmem_##TNAME##_atomic_xor(dest, value, pe);                  \
  }

WRAP_AMO_STANDARD(int, int)
WRAP_AMO_STANDARD(long, long)
WRAP_AMO_STANDARD(long long, longlong)
WRAP_AMO_STANDARD(unsigned int, uint)
WRAP_AMO_STANDARD(unsigned long, ulong)
WRAP_AMO_STANDARD(unsigned long long, ulonglong)
WRAP_AMO_STANDARD(int32_t, int32)
WRAP_AMO_STANDARD(int64_t, int64)
WRAP_AMO_STANDARD(uint32_t, uint32)
WRAP_AMO_STANDARD(uint64_t, uint64)
WRAP_AMO_STANDARD(size_t, size)
WRAP_AMO_STANDARD(ptrdiff_t, ptrdiff)

WRAP_AMO_EXTENDED(float, float)
WRAP_AMO_EXTENDED(double, double)
WRAP_AMO_EXTENDED(int, int)
WRAP_AMO_EXTENDED(long, long)
WRAP_AMO_EXTENDED(long long, longlong)
WRAP_AMO_EXTENDED(unsigned int, uint)
WRAP_AMO_EXTENDED(unsigned long, ulong)
WRAP_AMO_EXTENDED(unsigned long long, ulonglong)
WRAP_AMO_EXTENDED(int32_t, int32)
WRAP_AMO_EXTENDED(int64_t, int64)
WRAP_AMO_EXTENDED(uint32_t, uint32)
WRAP_AMO_EXTENDED(uint64_t, uint64)
WRAP_AMO_EXTENDED(size_t, size)
WRAP_AMO_EXTENDED(ptrdiff_t, ptrdiff)

WRAP_AMO_BITWISE(unsigned int, uint)
WRAP_AMO_BITWISE(unsigned long, ulong)
WRAP_AMO_BITWISE(unsigned long long, ulonglong)
WRAP_AMO_BITWISE(int32_t, int32)
WRAP_AMO_BITWISE(int64_t, int64)
WRAP_AMO_BITWISE(uint32_t, uint32)
WRAP_AMO_BITWISE(uint64_t, uint64)

#define WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, SUFFIX)                               \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_put_signal##SUFFIX(              \
      T *dest, const T *source, size_t nelems, uint64_t *sig_addr,             \
      uint64_t signal, int sig_op, int pe) {                                   \
    rocshmem::rocshmem_##TNAME##_put_signal##SUFFIX(                           \
        dest, source, nelems, sig_addr, signal, sig_op, pe);                   \
  }

#define WRAP_PUT_SIGNAL(T, TNAME)                                              \
  WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, )                                           \
  WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, _wg)                                        \
  WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, _wave)                                      \
  WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, _nbi)                                       \
  WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, _nbi_wg)                                    \
  WRAP_PUT_SIGNAL_SUFFIX(T, TNAME, _nbi_wave)

WRAP_PUT_SIGNAL(float, float)
WRAP_PUT_SIGNAL(double, double)
WRAP_PUT_SIGNAL(char, char)
WRAP_PUT_SIGNAL(signed char, schar)
WRAP_PUT_SIGNAL(short, short)
WRAP_PUT_SIGNAL(int, int)
WRAP_PUT_SIGNAL(long, long)
WRAP_PUT_SIGNAL(long long, longlong)
WRAP_PUT_SIGNAL(unsigned char, uchar)
WRAP_PUT_SIGNAL(unsigned short, ushort)
WRAP_PUT_SIGNAL(unsigned int, uint)
WRAP_PUT_SIGNAL(unsigned long, ulong)
WRAP_PUT_SIGNAL(unsigned long long, ulonglong)

#define WRAP_WAIT(T, TNAME)                                                    \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_wait_until(                      \
      T *ivars, int cmp, T val) {                                              \
    rocshmem::rocshmem_##TNAME##_wait_until(ivars, cmp, val);                  \
  }                                                                            \
  ROCSHMEM_DEVICE_API size_t rocshmem_##TNAME##_wait_until_any(                \
      T *ivars, size_t nelems, const int *status, int cmp, T val) {            \
    return rocshmem::rocshmem_##TNAME##_wait_until_any(                        \
        ivars, nelems, status, cmp, val);                                      \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_wait_until_all(                  \
      T *ivars, size_t nelems, const int *status, int cmp, T val) {            \
    rocshmem::rocshmem_##TNAME##_wait_until_all(                               \
        ivars, nelems, status, cmp, val);                                      \
  }                                                                            \
  ROCSHMEM_DEVICE_API size_t rocshmem_##TNAME##_wait_until_some(               \
      T *ivars, size_t nelems, size_t *indices, const int *status,             \
      int cmp, T val) {                                                        \
    return rocshmem::rocshmem_##TNAME##_wait_until_some(                       \
        ivars, nelems, indices, status, cmp, val);                             \
  }                                                                            \
  ROCSHMEM_DEVICE_API size_t rocshmem_##TNAME##_wait_until_any_vector(         \
      T *ivars, size_t nelems, const int *status, int cmp, T *vals) {          \
    return rocshmem::rocshmem_##TNAME##_wait_until_any_vector(                 \
        ivars, nelems, status, cmp, vals);                                     \
  }                                                                            \
  ROCSHMEM_DEVICE_API void rocshmem_##TNAME##_wait_until_all_vector(           \
      T *ivars, size_t nelems, const int *status, int cmp, T *vals) {          \
    rocshmem::rocshmem_##TNAME##_wait_until_all_vector(                        \
        ivars, nelems, status, cmp, vals);                                     \
  }                                                                            \
  ROCSHMEM_DEVICE_API size_t rocshmem_##TNAME##_wait_until_some_vector(        \
      T *ivars, size_t nelems, size_t *indices, const int *status,             \
      int cmp, T *vals) {                                                      \
    return rocshmem::rocshmem_##TNAME##_wait_until_some_vector(                \
        ivars, nelems, indices, status, cmp, vals);                            \
  }                                                                            \
  ROCSHMEM_DEVICE_API int rocshmem_##TNAME##_test(                             \
      T *ivars, int cmp, T val) {                                              \
    return rocshmem::rocshmem_##TNAME##_test(ivars, cmp, val);                 \
  }

WRAP_WAIT(float, float)
WRAP_WAIT(double, double)
WRAP_WAIT(char, char)
WRAP_WAIT(signed char, schar)
WRAP_WAIT(short, short)
WRAP_WAIT(int, int)
WRAP_WAIT(long, long)
WRAP_WAIT(long long, longlong)
WRAP_WAIT(unsigned char, uchar)
WRAP_WAIT(unsigned short, ushort)
WRAP_WAIT(unsigned int, uint)
WRAP_WAIT(unsigned long, ulong)
WRAP_WAIT(unsigned long long, ulonglong)
WRAP_WAIT(uint64_t, uint64)

}
