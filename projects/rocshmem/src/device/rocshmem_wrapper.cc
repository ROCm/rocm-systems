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
 * C wrapper for the device API so JIT/bitcode linking can resolve symbols.
 * The implementation lives in C++ (namespace rocshmem); we expose extern "C"
 * names and visibility("default") so llvm-link and consumers (e.g. Triton)
 * see stable symbols. Each function here forwards to the rocshmem:: impl.
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
                                         size_t nbytes, int pe) {
  rocshmem::rocshmem_putmem(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_wave(void *dest, const void *source,
                                              size_t nbytes, int pe) {
  rocshmem::rocshmem_putmem_wave(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_wg(void *dest, const void *source,
                                            size_t nbytes, int pe) {
  rocshmem::rocshmem_putmem_wg(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_nbi(void *dest, const void *source,
                                             size_t nbytes, int pe) {
  rocshmem::rocshmem_putmem_nbi(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_nbi_wave(void *dest,
                                                  const void *source,
                                                  size_t nbytes, int pe) {
  rocshmem::rocshmem_putmem_nbi_wave(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_nbi_wg(void *dest,
                                                const void *source,
                                                size_t nbytes, int pe) {
  rocshmem::rocshmem_putmem_nbi_wg(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem(void *dest, const void *source,
                                         size_t nbytes, int pe) {
  rocshmem::rocshmem_getmem(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_wave(void *dest, const void *source,
                                              size_t nbytes, int pe) {
  rocshmem::rocshmem_getmem_wave(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_wg(void *dest, const void *source,
                                            size_t nbytes, int pe) {
  rocshmem::rocshmem_getmem_wg(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_nbi(void *dest, const void *source,
                                             size_t nbytes, int pe) {
  rocshmem::rocshmem_getmem_nbi(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_nbi_wave(void *dest,
                                                  const void *source,
                                                  size_t nbytes, int pe) {
  rocshmem::rocshmem_getmem_nbi_wave(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_getmem_nbi_wg(void *dest,
                                                const void *source,
                                                size_t nbytes, int pe) {
  rocshmem::rocshmem_getmem_nbi_wg(dest, source, nbytes, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal(void *dest, const void *source,
                                                size_t nbytes,
                                                uint64_t *sig_addr,
                                                uint64_t signal, int sig_op,
                                                int pe) {
  rocshmem::rocshmem_putmem_signal(dest, source, nbytes, sig_addr, signal,
                                   sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_wg(
    void *dest, const void *source, size_t nbytes, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_wg(dest, source, nbytes, sig_addr, signal,
                                      sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_wave(
    void *dest, const void *source, size_t nbytes, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_wave(dest, source, nbytes, sig_addr, signal,
                                        sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_nbi(
    void *dest, const void *source, size_t nbytes, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_nbi(dest, source, nbytes, sig_addr, signal,
                                       sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_nbi_wg(
    void *dest, const void *source, size_t nbytes, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_nbi_wg(dest, source, nbytes, sig_addr,
                                          signal, sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_putmem_signal_nbi_wave(
    void *dest, const void *source, size_t nbytes, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  rocshmem::rocshmem_putmem_signal_nbi_wave(dest, source, nbytes, sig_addr,
                                            signal, sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_ulong_put_signal(
    void *dest, const void *source, size_t nelems, void *sig_addr,
    uint64_t signal, int32_t sig_op, int32_t pe) {
  rocshmem::rocshmem_ulong_put_signal(
      static_cast<uint64_t *>(dest), static_cast<const uint64_t *>(source),
      nelems, static_cast<uint64_t *>(sig_addr), signal, sig_op, pe);
}

ROCSHMEM_DEVICE_API void rocshmem_ulong_wait_until(void *sig_addr, int cmp,
                                                   uint64_t cmp_val) {
  rocshmem::rocshmem_ulong_wait_until(static_cast<uint64_t *>(sig_addr), cmp,
                                      cmp_val);
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

ROCSHMEM_DEVICE_API void rocshmem_fence() {
  rocshmem::rocshmem_fence();
}

}
