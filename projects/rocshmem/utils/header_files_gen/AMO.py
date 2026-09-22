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
    ("int", "int"),
    ("long", "long"),
    ("long long", "longlong"),
    ("unsigned int", "uint"),
    ("unsigned long", "ulong"),
    ("unsigned long long", "ulonglong"),
    ("int32_t", "int32"),
    ("int64_t", "int64"),
    ("uint32_t", "uint32"),
    ("uint64_t", "uint64"),
    ("size_t", "size"),
    ("ptrdiff_t", "ptrdiff"),
]


float_types = [
    ("float", "float"),
    ("double", "double"),
]

bitwise_types = types[3:10]


def atomic_fetch_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_fetch(\n"
        f"    rocshmem_ctx_t ctx, {T} *source, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_fetch(\n"
        f"    {T} *source, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_fetch(\n"
        f"    rocshmem_ctx_t ctx, {T} *source, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_fetch(\n"
        f"    {T} *source, int pe);\n\n"
    )


def generate_atomic_fetch_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_FETCH
 * @brief Atomically return the value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The value of \p dest.
 */\n"""

    for type_, tname_ in float_types:
        expanded_code += atomic_fetch_api(type_, tname_)

    for type_, tname_ in types:
        expanded_code += atomic_fetch_api(type_, tname_)

    return expanded_code


def atomic_set_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_atomic_set(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_atomic_set(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_atomic_set(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_atomic_set(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_set_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_SET
 * @brief Atomically set the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */\n"""

    for type_, tname_ in float_types:
        expanded_code += atomic_set_api(type_, tname_)

    for type_, tname_ in types:
        expanded_code += atomic_set_api(type_, tname_)

    return expanded_code


def atomic_compare_swap_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_compare_swap(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} cond, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_compare_swap(\n"
        f"    {T} *dest, {T} cond, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_compare_swap(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} cond, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_compare_swap(\n"
        f"    {T} *dest, {T} cond, {T} value, int pe);\n\n"
    )


def generate_atomic_compare_swap_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_COMPARE_SWAP
 * @brief Atomically compares if the value in \p dest with \p cond is equal
 * then put \p val in \p dest. The operation returns the older value of \p dest
 * to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] cond    The value to be compare with.
 * @param[in] val     The value to be atomically swapped.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest.
 */\n"""
    for type_, tname_ in types:
        expanded_code += atomic_compare_swap_api(type_, tname_)

    return expanded_code


def atomic_swap_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_swap(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_swap(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_swap(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_swap(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_swap_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_SWAP
 * @brief Atomically swap the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */\n"""

    for type_, tname_ in float_types:
        expanded_code += atomic_swap_api(type_, tname_)

    for type_, tname_ in types:
        expanded_code += atomic_swap_api(type_, tname_)

    return expanded_code


def atomic_fetch_inc_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_fetch_inc(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_fetch_inc(\n"
        f"    {T} *dest, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_fetch_inc(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_fetch_inc(\n"
        f"    {T} *dest, int pe);\n\n"
    )


def generate_atomic_fetch_inc_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_FETCH_INC
 * @brief Atomically add 1 to \p dest on \p pe. The operation
 * returns the older value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest before it was incremented by 1.
 */\n"""
    for type_, tname_ in types:
        expanded_code += atomic_fetch_inc_api(type_, tname_)

    return expanded_code


def atomic_inc_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_atomic_inc(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_atomic_inc(\n"
        f"    {T} *dest, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_atomic_inc(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_atomic_inc(\n"
        f"    {T} *dest, int pe);\n\n"
    )


def generate_atomic_inc_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_INC
 * @brief Atomically add 1 to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += atomic_inc_api(type_, tname_)

    return expanded_code


def atomic_fetch_add_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_fetch_add(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_fetch_add(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_fetch_add(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_fetch_add(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_fetch_add_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_FETCH_ADD
 * @brief Atomically add the value \p val to \p dest on \p pe. The operation
 * returns the older value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest before the \p val was added.
 */\n"""
    for type_, tname_ in types:
        expanded_code += atomic_fetch_add_api(type_, tname_)

    return expanded_code


def atomic_add_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_atomic_add(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_atomic_add(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_atomic_add(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_atomic_add(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_add_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_ADD
 * @brief Atomically add the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += atomic_add_api(type_, tname_)

    return expanded_code


def atomic_fetch_and_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_fetch_and(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_fetch_and(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_fetch_and(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_fetch_and(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_fetch_and_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_FETCH_AND
 * @brief Atomically bitwise-and the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */\n"""
    for type_, tname_ in bitwise_types:
        expanded_code += atomic_fetch_and_api(type_, tname_)

    return expanded_code


def atomic_and_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_atomic_and(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_atomic_and(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_atomic_and(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_atomic_and(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_and_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_AND
 * @brief Atomically bitwise-and the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */\n"""
    for type_, tname_ in bitwise_types:
        expanded_code += atomic_and_api(type_, tname_)

    return expanded_code


def atomic_fetch_or_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_fetch_or(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_fetch_or(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_fetch_or(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_fetch_or(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_fetch_or_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_FETCH_OR
 * @brief Atomically bitwise-or the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */\n"""
    for type_, tname_ in bitwise_types:
        expanded_code += atomic_fetch_or_api(type_, tname_)

    return expanded_code


def atomic_or_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_atomic_or(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_atomic_or(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_atomic_or(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_atomic_or(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_or_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_OR
 * @brief Atomically bitwise-or the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */\n"""
    for type_, tname_ in bitwise_types:
        expanded_code += atomic_or_api(type_, tname_)

    return expanded_code


def atomic_fetch_xor_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE {T} rocshmem_ctx_{TNAME}_atomic_fetch_xor(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE {T} rocshmem_{TNAME}_atomic_fetch_xor(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_ctx_{TNAME}_atomic_fetch_xor(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ {T} rocshmem_{TNAME}_atomic_fetch_xor(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_fetch_xor_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_FETCH_XOR
 * @brief Atomically bitwise-xor the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */\n"""
    for type_, tname_ in bitwise_types:
        expanded_code += atomic_fetch_xor_api(type_, tname_)

    return expanded_code


def atomic_xor_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_atomic_xor(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_atomic_xor(\n"
        f"    {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_atomic_xor(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, {T} value, int pe);\n"
        f"__host__ void rocshmem_{TNAME}_atomic_xor(\n"
        f"    {T} *dest, {T} value, int pe);\n\n"
    )


def generate_atomic_xor_api():
    expanded_code = """
/**
 * @name SHMEM_ATOMIC_XOR
 * @brief Atomically bitwise-xor the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */\n"""
    for type_, tname_ in bitwise_types:
        expanded_code += atomic_xor_api(type_, tname_)

    return expanded_code

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def generate_AMO_header(output_dir, copyright):
    expanded_code = copyright

    expanded_code += """
#ifndef LIBRARY_INCLUDE_ROCSHMEM_AMO_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_AMO_HPP

namespace rocshmem {
"""

    expanded_code += (
        generate_atomic_fetch_api() +
        generate_atomic_set_api() +
        generate_atomic_compare_swap_api() +
        generate_atomic_swap_api() +
        generate_atomic_fetch_inc_api() +
        generate_atomic_inc_api() +
        generate_atomic_fetch_add_api() +
        generate_atomic_add_api() +
        generate_atomic_fetch_and_api() +
        generate_atomic_and_api() +
        generate_atomic_fetch_or_api() +
        generate_atomic_or_api() +
        generate_atomic_fetch_xor_api() +
        generate_atomic_xor_api()
    )

    expanded_code += """
}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_AMO_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_AMO.hpp'
    )

    write_to_file(output_file, expanded_code)
