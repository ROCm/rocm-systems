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
]


def alltoall_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_alltoall_wg(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems);\n\n"
    )


def generate_alltoall_api():
    expanded_code = """
/**
 * @name SHMEM_ALLTOALL
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += alltoall_api(type_, tname_)

    return expanded_code


def broadcast_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_broadcast_wg(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems, int pe_root);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_broadcast(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    int nelems, int pe_root, int pe_start, int log_pe_stride,\n"
        f"    int pe_size, long *p_sync);\n"
        f"__host__ void rocshmem_ctx_{TNAME}_broadcast(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems, int pe_root);\n\n"
    )


def generate_broadcast_api():
    expanded_code = """
/**
 * @name SHMEM_BROADCAST
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcase completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelement     Size of the buffer to participate in the broadcast.
 * @param[in] PE_root      Zero-based ordinal of the PE, with respect to the
                           active set, from which the data is copied
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pSync        Temporary sync buffer provided to ROCSHMEM. Must
                           be of size at least ROCSHMEM_REDUCE_SYNC_SIZE.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += broadcast_api(type_, tname_)

    return expanded_code


def fcollect_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_fcollect_wg(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems);\n\n"
    )


def generate_fcollect_api():
    expanded_code = """
/**
 * @name SHMEM_FCOLLECT
 * @brief Concatenates blocks of data from multiple PEs to an array in every
 * PE participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks in source array.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += fcollect_api(type_, tname_)

    return expanded_code


def reduction_api(T, TNAME, Op_API):
    return (
        f"__device__ ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_{Op_API}_reduce_wg(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest, const {T} *source,\n"
        f"    int nreduce);\n"
        f"__host__ int rocshmem_ctx_{TNAME}_{Op_API}_reduce(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest, const {T} *source,\n"
        f"    int nreduce);\n\n"
    )


def arith_reduction_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reduction_api(T, TNAME, op) for op in operations])

def bitwise_reduction_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reduction_api(T, TNAME, op) for op in operations])


def generate_reduction_api():
    expanded_code = """
/**
 * @name SHMEM_REDUCTIONS
 * @brief Perform an allreduce between PEs in the active set. The caller
 * is blocked until the reduction completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nreduce      Size of the buffer to participate in the reduction.
 *
 * @return int (Zero on successful local completion. Nonzero otherwise.)
 */\n"""

    int_types = [
        ("short", "short"),
        ("int", "int"),
        ("long", "long"),
        ("long long", "longlong")
    ]

    float_types = [
        ("float", "float"),
        ("double", "double")
    ]

    for type_, tname_ in int_types:
        expanded_code += arith_reduction_api(type_, tname_)
        expanded_code += bitwise_reduction_api(type_, tname_)

    for type_, tname_ in float_types:
        expanded_code += arith_reduction_api(type_, tname_)

    return expanded_code


def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def generate_COLL_header(output_dir, copyright):
    expanded_code = copyright

    expanded_code += """
#ifndef LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP

namespace rocshmem {
"""

    expanded_code += (
        generate_alltoall_api() +
        generate_broadcast_api() +
        generate_fcollect_api() +
        generate_reduction_api()
    )

    expanded_code += """
}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_COLL.hpp'
    )

    write_to_file(output_file, expanded_code)
