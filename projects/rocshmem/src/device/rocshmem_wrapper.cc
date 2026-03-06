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
 * JIT/bitcode consumers (e.g. Triton) need stable, unmangled symbol names.
 * Each extern "C" function forwards to the corresponding rocshmem:: API.
 * The forwarding call serves as a compile-time check: if parameter types
 * here diverge from the API, the build fails.
 *
 * Signatures must exactly match the declarations in include/rocshmem/.
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

ROCSHMEM_DEVICE_API void rocshmem_int_p(int *dest, int value, int pe) {
  rocshmem::rocshmem_int_p(dest, value, pe);
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

ROCSHMEM_DEVICE_API void rocshmem_ulong_put_signal(
    unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_ulong_put_signal(dest, source, nelems, sig_addr, signal,
                                      sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_ulong_wait_until(unsigned long *ivars,
                                                   int cmp,
                                                   unsigned long val) {
  rocshmem::rocshmem_ulong_wait_until(ivars, cmp, val);
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

}
