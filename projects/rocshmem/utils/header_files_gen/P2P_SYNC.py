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


def wait_until_api(T, TNAME):
    return (
        f"__device__ void rocshmem_{TNAME}_wait_until(\n"
        f"    {T} *ivars, int cmp, {T} val);\n"
        f"__device__ size_t rocshmem_{TNAME}_wait_until_any(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__device__ void rocshmem_{TNAME}_wait_until_all(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__device__ size_t rocshmem_{TNAME}_wait_until_some(\n"
        f"    {T} *ivars, size_t nelems, size_t* indices, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__device__ size_t rocshmem_{TNAME}_wait_until_any_vector(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__device__ void rocshmem_{TNAME}_wait_until_all_vector(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__device__ size_t rocshmem_{TNAME}_wait_until_some_vector(\n"
        f"    {T} *ivars, size_t nelems, size_t* indices, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__host__ void rocshmem_{TNAME}_wait_until(\n"
        f"    {T} *ivars, int cmp, {T} val);\n"
        f"__host__ size_t rocshmem_{TNAME}_wait_until_any(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__host__ void rocshmem_{TNAME}_wait_until_all(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__host__ size_t rocshmem_{TNAME}_wait_until_some(\n"
        f"    {T} *ivars, size_t nelems, size_t* indices, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__host__ size_t rocshmem_{TNAME}_wait_until_any_vector(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__host__ void rocshmem_{TNAME}_wait_until_all_vector(\n"
        f"    {T} *ivars, size_t nelems, const int* status,\n"
        f"    int cmp, {T} val);\n"
        f"__host__ size_t rocshmem_{TNAME}_wait_until_some_vector(\n"
        f"    {T} *ivars, size_t nelems, size_t* indices, const int* status,\n"
        f"    int cmp, {T} val);\n\n"
    )


def generate_wait_until_api():
    expanded_code = """
/**
 * @name SHMEM_WAIT_UNTIL
 * @brief Block the caller until the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ivars Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return void
 */\n"""
    for type_, tname_ in types:
        expanded_code += wait_until_api(type_, tname_)

    return expanded_code


def test_api(T, TNAME):
    return (
        f"__device__ int rocshmem_{TNAME}_test(\n"
        f"    {T} *ivars, int cmp, {T} val);\n"
        f"__host__ int rocshmem_{TNAME}_test(\n"
        f"    {T} *ivars, int cmp, {T} val);\n\n"
    )


def generate_test_api():
    expanded_code = """
/**
 * @name SHMEM_TEST
 * @brief test if the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ivars Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return 1 if the evaluation is true else 0
 */\n"""
    for type_, tname_ in types:
        expanded_code += test_api(type_, tname_)

    return expanded_code


def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def generate_P2P_SYNC_header(output_dir, copyright):
    expanded_code = copyright

    expanded_code += """
#ifndef LIBRARY_INCLUDE_ROCSHMEM_P2P_SYNC_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_P2P_SYNC_HPP

namespace rocshmem {
"""

    expanded_code += (
        generate_wait_until_api() +
        generate_test_api()
    )

    expanded_code += """
}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_P2P_SYNC_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_P2P_SYNC.hpp'
    )

    write_to_file(output_file, expanded_code)
