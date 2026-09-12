###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

import os

types = [
    ("float", "float"),
    ("double", "double"),
    ("char", "char"),
    ("signed char", "schar"),
    ("short", "short"),
    ("int", "int"),
    ("long", "long"),
    ("long long", "longlong"),
    ("unsigned char", "uchar"),
    ("unsigned short", "ushort"),
    ("unsigned int", "uint"),
    ("unsigned long", "ulong"),
    ("unsigned long long", "ulonglong"),
    ("__half", "half"),
    ("__hip_bfloat16", "bfloat16"),
    ("int8_t", "int8"),
    ("int16_t", "int16"),
    ("uint8_t", "uint8"),
    ("uint16_t", "uint16")
]


def put_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_put(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_put(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_put(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_put({T} *dest,\n"
        f"    const {T} *source, size_t nelems, int pe);\n\n"
    )


def generate_put_api():
    expanded_code = """
/**
 * @name SHMEM_PUT
 * @brief Writes contiguous data of \\p nelems elements from \\p source on the
 * calling PE to \\p dest at \\p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \\p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += put_api(type_, tname_)

    expanded_code += """
/**
 * @brief Writes contiguous data of \\p nelems bytes from \\p source on the
 * calling PE to \\p dest at \\p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \\p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem(rocshmem_ctx_t ctx,
                                                    void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem(void *dest, const void *source,
                                                size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \\p nelems bytes from \\p source on the
 * calling PE to \\p dest at \\p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \\p source). The caller must
 * call into __host__ rocshmem_quiet() if remote completion is required.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest,
                                   const void *source, size_t nelems, int pe);

__host__ void rocshmem_putmem(void *dest, const void *source, size_t nelems,
                               int pe);

"""

    return expanded_code


def get_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_get(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_get(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_get(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_get({T} *dest,\n"
        f"    const {T} *source, size_t nelems, int pe);\n\n"
    )


def generate_get_api():
    expanded_code = """
/**
 * @name SHMEM_GET
 * @brief Reads contiguous data of \\p nelems elements from \\p source on \\p pe
 * to \\p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \\p dest).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += get_api(type_, tname_)
    expanded_code += """
/**
 * @brief Reads contiguous data of \\p nelems bytes from \\p source on \\p pe
 * to \\p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \\p dest).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_getmem(rocshmem_ctx_t ctx,
                                                    void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_getmem(void *dest, const void *source,
                                                size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \\p nelems bytes from \\p source on \\p pe
 * to \\p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \\p dest).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_getmem(rocshmem_ctx_t ctx, void *dest,
                                   const void *source, size_t nelems, int pe);

__host__ void rocshmem_getmem(void *dest, const void *source, size_t nelems,
                               int pe);

"""

    return expanded_code


def p_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_p(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value,\n"
        f"    int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_p(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_p(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value,\n"
        f"    int pe);\n"
        f"__host__ void rocshmem_{TNAME}_p(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_p_api():
    expanded_code = """
/**
 * @name SHMEM_P
 * @brief Writes a single value to \\p dest at \\p pe PE to \\p dst at \\p pe.
 * The caller must call into rocshmem_quiet() if remote completion is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] value  Value to write to dest at \\p pe.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += p_api(type_, tname_)
    expanded_code += """__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_p(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_p(
    int64_t *dest, int64_t value, int pe);
"""
    return expanded_code


def g_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_g(\n"
        f"    rocshmem_ctx_t ctx, const {T} *source, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_g(\n"
        f"    const {T} *source, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_g(\n"
        f"    rocshmem_ctx_t ctx, const {T} *source, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_g(\n"
        f"    const {T} *source, int pe);\n\n"
    )


def generate_g_api():
    expanded_code = """
/**
 * @name SHMEM_G
 * @brief reads and returns single value from \\p source at \\p pe.
 * The calling work-group/thread will block until the operation completes.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] source Source address. Must be an address on the symmetric
 *                   heap.
 * @param[in] pe     PE of the remote process.
 *
 * @return the value read from remote \\p source at \\p pe.
 */\n"""
    for type_, tname_ in types:
        expanded_code += g_api(type_, tname_)

    return expanded_code


def put_nbi_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_put_nbi(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_put_nbi(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_put_nbi(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_put_nbi(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n\n"
    )    


def generate_put_nbi_api():
    expanded_code = """
/**
 * @name SHMEM_PUT_NBI
 * @brief Writes contiguous data of \\p nelems elements from \\p source on the
 * calling PE to \\p dest on \\p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += put_nbi_api(type_, tname_)
    expanded_code += """
/**
 * @brief Writes contiguous data of \\p nelems bytes from \\p source on the
 * calling PE to \\p dest on \\p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx,
                                                        void *dest,
                                                        const void *source,
                                                        size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_nbi(void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \\p nelems bytes from \\p source on the
 * calling PE to \\p dest on \\p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * __host__ rocshmem_quiet() if completion notification is required.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                       const void *source, size_t nelems,
                                       int pe);

__host__ void rocshmem_putmem_nbi(void *dest, const void *source,
                                   size_t nelems, int pe);

"""

    return expanded_code


def get_nbi_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_get_nbi(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_get_nbi(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_get_nbi(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_get_nbi({T} *dest,\n"
        f"    const {T} *source, size_t nelems, int pe);\n\n"
    )


def generate_get_nbi_api():
    expanded_code = """
/**
 * @name SHMEM_GET_NBI
 * @brief Reads contiguous data of \\p nelems elements from \\p source on \\p pe
 * to \\p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += get_nbi_api(type_, tname_)
    expanded_code += """
/**
 * @brief Reads contiguous data of \\p nelems bytes from \\p source on \\p pe
 * to \\p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx,
                                                        void *dest,
                                                        const void *source,
                                                        size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_getmem_nbi(void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \\p nelems bytes from \\p source on \\p pe
 * to \\p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * __host__ rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                       const void *source, size_t nelems,
                                       int pe);

__host__ void rocshmem_getmem_nbi(void *dest, const void *source,
                                   size_t nelems, int pe);
"""

    return expanded_code

def add_misc_api():
    return """
/**
 * @brief kernel for performing a getmem RMA operation.
 * Caller enqueues the kernel on given stream
 *
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_getmem_kernel(void *dest,
                                                      const void *source,
                                                      size_t nelems, int pe);

/**
 * @brief kernel for performing a putmem RMA operation.
 * Caller enqueues the kernel on given stream
 *
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_putmem_kernel(void *dest,
                                                      const void *source,
                                                      size_t nelems, int pe);

/**
 * @brief kernel for performing a quiet operation.
 * Caller enqueues the kernel on given stream
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_quiet_kernel();
"""

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def generate_RMA_header(output_dir, copyright):
    expanded_code = copyright

    expanded_code += """
#ifndef LIBRARY_INCLUDE_ROCSHMEM_RMA_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_RMA_HPP

namespace rocshmem {
"""

    expanded_code += (
        generate_put_api() +
        generate_p_api() +
        generate_get_api() +
        generate_g_api() +
        generate_put_nbi_api() +
        generate_get_nbi_api() +
        add_misc_api()
    )

    expanded_code += """
}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_RMA_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_RMA.hpp'
    )

    write_to_file(output_file, expanded_code)
