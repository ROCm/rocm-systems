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
    ("__hip_bfloat16", "bfloat16")
]


def alltoall_ctx_wg_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_alltoall_wg(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems);\n\n"
    )

def alltoall_wg_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_alltoall_wg(\n"
        f"    rocshmem_team_t team, {T} *dest, const {T} *source, int nelems);\n\n"
    )

def generate_alltoall_wg_api():
    expanded_code = """
/**
 * @name SHMEM_ALLTOALL
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += alltoall_ctx_wg_api(type_, tname_)
    for type_, tname_ in types:
        expanded_code += alltoall_wg_api(type_, tname_)

    expanded_code += '''/**
 * @name ROCSHMEM_ALLTOALLMEM_WG
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.    
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return int; zero on success, non-zero otherwise
 */
__device__ void rocshmem_ctx_alltoallmem_wg(rocshmem_ctx_t ctx,
    rocshmem_team_t team, void *dest, const void *source, int nelems);
'''

    return expanded_code

def alltoall_wave_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_alltoall_wave(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems);\n\n"
    )

def generate_alltoall_wave_api():
    expanded_code = """
/**
 * @name ROCSHMEM_CTX_ALLTOALL_WAVE
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a wave-level collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return int: zero on success, non-zero otherwise
 */\n"""
    for type_, tname_ in types:
        expanded_code += alltoall_wave_api(type_, tname_)
    expanded_code += """/**
 * @name ROCSHMEM_ALLTOALLMEM_WAVE
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a wave collective.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.    
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return int; zero on success, non-zero otherwise
 */
__device__ int rocshmem_ctx_alltoallmem_wave(rocshmem_ctx_t ctx,
    rocshmem_team_t team, void *dest, const void *source, int nelems);
"""

    return expanded_code

def alltoallv_wg_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_alltoallv_wg(rocshmem_team_t team,\n"
        f"                                                             {T} *dest, const size_t dest_nelems[],\n"
        f"                                                             const size_t dest_displs[],\n"
        f"                                                             {T} *source, const size_t source_nelems[],\n"
        f"                                                             const size_t source_displs[]);\n"
    )

def generate_alltoallv_wg_api():
    expanded_code = """
/**
 * @name SHMEM_ALLTOALLV
 * @brief PE i sends source_nelems[j] of data from source + source_displs[j] to PE j.
 * At the same time, PE i receives dest_nelems[j] of data from PE j to be placed at dest + dest_displs[j].
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team          The team participating in the collective.
 * @param[in] dest          Destination address. Must be an address on the symmetric heap.
 * @param[in] dest_nelems   Array containing number of elements to receive from each participating PE
 * @param[in] dest_displs   Array of offsets into dest buffer for each participating PE
 * @param[in] source        Source address. Must be an address on the symmetric heap.
 * @param[in] source_nelems Array containing number of elements to send from each participating PE
 * @param[in] source_displs Array of offsets into source buffer for each participating PE
 *
 * @return void
 */
\n"""
    _types = [
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
    for type_, tname_ in _types:
        expanded_code += alltoallv_wg_api(type_, tname_)

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
 * is blocked until the broadcast completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Size of the buffer to participate in the broadcast.
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

    expanded_code += """/**
 * @name ROCSHMEM_CTX_BROADCASTMEM_WG
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcast completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Size of buffer to participate in the broadcast.
 * @param[in] PE_root      Root PE (relative to team) from which to broadcast.
 * 
 *
 * @return void
 */
__device__ void rocshmem_ctx_broadcastmem_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
              void *dest, const void *source, int nelems, int PE_root);
"""

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
 * @param[in] ctx          The context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of data blocks in source array.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += fcollect_api(type_, tname_)

    expanded_code += """/**
 * @name ROCSHMEM_CTX_FCOLLECTMEM_WG
 * @brief Concatenates @p nelems bytes from each PE's @p source into every PE's
 * @p dest buffer.
 * Must be called as a work-group collective.
 *
 * @param[in] ctx          The context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of bytes contributed by each PE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_fcollectmem_wg(rocshmem_ctx_t ctx,
    rocshmem_team_t team, void *dest, const void *source, int nelems);
"""

    return expanded_code


def fcollect_wave_api(T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_fcollect_wave(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest,\n"
        f"    const {T} *source, int nelems);\n\n"
    )


def generate_fcollect_wave_api():
    expanded_code = """
/**
 * @name ROCSHMEM_CTX_FCOLLECT_WAVE
 * @brief Concatenates blocks of data from multiple PEs to an array in every
 * PE participating in the collective routine.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] ctx          The context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of data elements contributed by each PE.
 *
 * @return int (Zero on successful local completion. Nonzero otherwise.)
 */\n"""
    for type_, tname_ in types:
        expanded_code += fcollect_wave_api(type_, tname_)

    expanded_code += """/**
 * @name ROCSHMEM_CTX_FCOLLECTMEM_WAVE
 * @brief Concatenates @p nelems bytes from each PE's @p source into every PE's
 * @p dest buffer.
 * Must be called as a wave-level collective.
 *
 * @param[in] ctx          The context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of bytes contributed by each PE.
 *
 * @return int (Zero on successful local completion. Nonzero otherwise.)
 */
__device__ ATTR_NO_INLINE int rocshmem_ctx_fcollectmem_wave(rocshmem_ctx_t ctx,
    rocshmem_team_t team, void *dest, const void *source, int nelems);\n
"""

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

def reduction_wave_api(T, TNAME, Op_API):
    return (
        f"__device__ ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_{Op_API}_reduce_wave(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest, const {T} *source,\n"
        f"    int nreduce);\n"
    )

def arith_reduction_wave_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reduction_wave_api(T, TNAME, op) for op in operations])

def bitwise_reduction_wave_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reduction_wave_api(T, TNAME, op) for op in operations])

def generate_reduction_wave_api():
    expanded_code = """/**
 * @name ROCSHMEM_CTX_REDUCE_WAVE
 * @brief Perform an allreduce between PEs in the active set. The caller
 * is blocked until the reduction completes.
 *
 * This function must be called as a wave-level collective.
 *
 * @param[in] ctx          The context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
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
        expanded_code += arith_reduction_wave_api(type_, tname_)
        expanded_code += bitwise_reduction_wave_api(type_, tname_)

    for type_, tname_ in float_types:
        expanded_code += arith_reduction_wave_api(type_, tname_)

    return expanded_code

def arith_reduction_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reduction_api(T, TNAME, op) for op in operations])

def bitwise_reduction_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reduction_api(T, TNAME, op) for op in operations])


def reducescatter_api(T, TNAME, Op_API):
    return (
        f"__device__ ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_{Op_API}_reduce_scatter_wg(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest, const {T} *source,\n"
        f"    int nreduce);\n"
    )

def arith_reducescatter_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reducescatter_api(T, TNAME, op) for op in operations])

def bitwise_reducescatter_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reducescatter_api(T, TNAME, op) for op in operations])


def generate_reducescatter_wg_api():
    expanded_code = """
/**
 * @name SHMEM_REDUCE_SCATTER
 * @brief Perform a reduce-scatter between PEs in the team. Each PE contributes
 * nreduce * n_pes elements from source; after reduction across all PEs,
 * PE i receives the nreduce elements corresponding to block i.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address (nreduce elements). Must be an
 *                         address on the symmetric heap.
 * @param[in] source       Source address (nreduce * n_pes elements). Must be
 *                         an address on the symmetric heap.
 * @param[in] nreduce      Number of elements each PE receives.
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
        expanded_code += arith_reducescatter_api(type_, tname_)
        expanded_code += bitwise_reducescatter_api(type_, tname_)

    for type_, tname_ in float_types:
        expanded_code += arith_reducescatter_api(type_, tname_)

    return expanded_code

def reducescatter_wave_api(T, TNAME, Op_API):
    return (
        f"__device__ ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_{Op_API}_reduce_scatter_wave(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest, const {T} *source,\n"
        f"    int nreduce);\n"
    )

def arith_reducescatter_wave_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reducescatter_wave_api(T, TNAME, op) for op in operations])

def bitwise_reducescatter_wave_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reducescatter_wave_api(T, TNAME, op) for op in operations])


def generate_reducescatter_wave_api():
    expanded_code = """
/**
 * @name SHMEM_REDUCE_SCATTER_WAVE device-side (wave-level)
 * @brief Wave-level reduce-scatter: PE i receives the element-wise reduction
 * of source[i*nreduce..(i+1)*nreduce-1] across all PEs in the team.
 * Only the wave (wavefront) participates. Returns ROCSHMEM_SUCCESS on success.
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
        expanded_code += arith_reducescatter_wave_api(type_, tname_)
        expanded_code += bitwise_reducescatter_wave_api(type_, tname_)

    for type_, tname_ in float_types:
        expanded_code += arith_reducescatter_wave_api(type_, tname_)

    return expanded_code


def reducescatter_host_api(T, TNAME, Op_API):
    return (
        f"__host__ int rocshmem_ctx_{TNAME}_{Op_API}_reduce_scatter(\n"
        f"    rocshmem_ctx_t ctx, rocshmem_team_t team, {T} *dest, const {T} *source, int nreduce);\n"
    )

def arith_reducescatter_host_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reducescatter_host_api(T, TNAME, op) for op in operations])

def bitwise_reducescatter_host_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reducescatter_host_api(T, TNAME, op) for op in operations])


def generate_reducescatter_host_api():
    expanded_code = """
/**
 * @name SHMEM_REDUCE_SCATTER host-side
 * @brief Host-side reduce-scatter: PE i receives the element-wise reduction
 * of source[i*nreduce..(i+1)*nreduce-1] across all PEs.
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
        expanded_code += arith_reducescatter_host_api(type_, tname_)
        expanded_code += bitwise_reducescatter_host_api(type_, tname_)

    for type_, tname_ in float_types:
        expanded_code += arith_reducescatter_host_api(type_, tname_)

    return expanded_code



def reduce_on_stream_api(T, TNAME, Op_API):
    return (
        f"ATTR_NO_INLINE int rocshmem_ctx_{TNAME}_{Op_API}_reduce_on_stream(\n"
        f"  rocshmem_ctx_t ctx, rocshmem_team_t team,\n"
        f"  {T} *dest, const {T} *source, int nreduce, hipStream_t stream);\n\n"
    )

def arith_reduce_on_stream_api(T, TNAME):
    operations = ["sum", "min", "max", "prod"]
    return "".join([reduce_on_stream_api(T, TNAME, op) for op in operations])

def bitwise_reduce_on_stream_api(T, TNAME):
    operations = ["or", "and", "xor"]
    return "".join([reduce_on_stream_api(T, TNAME, op) for op in operations])


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
 *                         heap.
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


def generate_reduce_on_stream_api():
    expanded_code = """/**
 * @name ROCSHMEM_REDUCE_ON_STREAM
 * @brief Performs a reduction across all PEs in a team on the specified HIP
 *        stream.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nreduce      Size of the buffer to participate in the reduction.
 * @param[in] stream       HIP stream on which the reduction is issued.
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
        expanded_code += arith_reduce_on_stream_api(type_, tname_)
        expanded_code += bitwise_reduce_on_stream_api(type_, tname_)

    for type_, tname_ in float_types:
        expanded_code += arith_reduce_on_stream_api(type_, tname_)

    return expanded_code


def broadcast_wave_api(T, TNAME):
    return (
        f"__device__ int rocshmem_ctx_{TNAME}_broadcast_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,\n"
        f"              {T} *dest, const {T} *source, int nelems, int PE_root);\n\n"
    )

def generate_broadcast_wave_api():
    expanded_code = """
/**
 * @name ROCSHMEM_CTX_TYPE_BROADCAST_WAVE
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcast completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Number of elements to participate in the broadcast.
 * @param[in] PE_root      Root PE (relative to team) from which to broadcast.
 * 
 *
 * @return int; zero when sucessful, non-zero otherwise
 */\n"""

    for type_, tname_ in types:
        expanded_code += broadcast_wave_api(type_, tname_)

    expanded_code += """/**
 * @name ROCSHMEM_CTX_BROADCASTMEM_WAVE
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcast completes.
 *
 * This function must be called as a wave collective.
 *
 * @param[in] ctx          The ROCSHMEM context associated with this operation.
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
 *                         heap.
 * @param[in] nelems       Size of buffer to participate in the broadcast.
 * @param[in] PE_root      Root PE (relative to team) from which to broadcast.
 * 
 *
 * @return int; zero when successful, non-zero otherwise
 */
__device__ int rocshmem_ctx_broadcastmem_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
              void *dest, const void *source, int nelems, int PE_root);
"""

    return expanded_code

def add_misc():
    return '''
/**
 * @brief kernel for performing a barrier synchronization.
 * Caller enqueues the kernel on given stream
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_barrier_all_kernel();

/**
 * @brief kernel for performing a team-scoped barrier synchronization.
 * Caller enqueues the kernel on given stream.
 *
 * @param[in] team  The team participating in the barrier.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_barrier_kernel(rocshmem_team_t team);

/**
 * @brief kernel for performing a sync_all operation.
 * Caller enqueues the kernel on given stream
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_sync_all_kernel();

/**
 * @brief kernel for performing a team-scoped sync operation.
 * Caller enqueues the kernel on given stream.
 *
 * @param[in] team  The team participating in the sync.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_team_sync_kernel(rocshmem_team_t team);

/**
 * @brief kernel for performing an alltoall collective operation.
 * Caller enqueues the kernel on given stream
 *
 * @param[in] team    The team participating in the collective.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] size    Number of bytes to transfer per pair of PEs.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_alltoallmem_kernel(rocshmem_team_t team,
                                                           void *dest,
                                                           const void *source,
                                                           size_t size);

/**
 * @brief kernel for performing a reduce on stream operation.
 *
 * @param[in] team     The team participating in the collective.
 * @param[in] dest     Destination address. Must be an address on the symmetric
 *                     heap.
 * @param[in] source   Source address. Must be an address on the symmetric heap.
 * @param[in] nreduce  Number of elements to reduce.
 *
 * @return void
 */
template <typename T, ROCSHMEM_OP Op>
__global__ ATTR_NO_INLINE void rocshmem_reduce_on_stream_kernel(rocshmem_team_t team,
                                                                T *dest,
                                                                const T *source,
                                                                int nreduce);

/**
 * @brief kernel for performing a broadcast collective operation.
 * Caller enqueues the kernel on given stream
 *
 * @param[in] team    The team participating in the collective.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Number of bytes to broadcast.
 * @param[in] pe_root Root PE (relative to team) from which to broadcast.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_broadcastmem_kernel(
    rocshmem_team_t team, void *dest, const void *source, size_t nelems,
    int pe_root);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_barrier_all();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a wave-front collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_barrier_all_wave();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_barrier_all_wg();

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @param[in] handle GPU side handle.
 *
 * @param[in] team The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_ctx_barrier(rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @param[in] team The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_ctx_barrier_wave(rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @param[in] team The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_ctx_barrier_wg(rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief perform a collective barrier between all PEs in the team world.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @return void
 */
__device__ void rocshmem_barrier();

/**
 * @brief perform a collective barrier between all PEs in the team world.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a wave-front collective.
 *
 * @return void
 */
__device__ void rocshmem_barrier_wave();

/**
 * @brief perform a collective barrier between all PEs in the team world.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @return void
 */
__device__ void rocshmem_barrier_wg();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_sync_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a wave-front collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_sync_all_wave();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_sync_all_wg();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

'''

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
        generate_alltoall_wg_api() +
        generate_alltoall_wave_api() +
        generate_alltoallv_wg_api() +
        generate_broadcast_api() +
        generate_broadcast_wave_api() +
        generate_fcollect_api() +
        generate_fcollect_wave_api() +
        generate_reducescatter_wg_api() +
        generate_reducescatter_wave_api() +
        generate_reducescatter_host_api() +
        generate_reduction_api() + 
        generate_reduction_wave_api() +
        add_misc() +
        generate_reduce_on_stream_api()
    )

    expanded_code += """}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_COLL.hpp'
    )

    write_to_file(output_file, expanded_code)
