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


def putmem_signal_dec(SUFFIX):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_putmem_signal{SUFFIX}(\n"
        f"    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,\n"
        f"    uint64_t signal, int sig_op, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal{SUFFIX}(\n"
        f"    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,\n"
        f"    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);\n\n"
    )


def put_signal_typed_dec(T, TNAME, SUFFIX):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_put_signal{SUFFIX}(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source, size_t nelems,\n"
        f"    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_put_signal{SUFFIX}(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, uint64_t *sig_addr,\n"
        f"    uint64_t signal, int sig_op, int pe);\n\n"
    )


def put_signal_dec(SUFFIX):
    return "".join([put_signal_typed_dec(T, TNAME, SUFFIX) for T, TNAME in types])


def signaling_api_dec(SUFFIX):
    return (putmem_signal_dec(SUFFIX) + put_signal_dec(SUFFIX))


def generate_signal_api():

    suffixes = ["", "_wg", "_wave", "_nbi", "_nbi_wg", "_nbi_wave"]

    return "".join([signaling_api_dec(suffix) for suffix in suffixes])


def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def generate_SIG_OP_header(output_dir, copyright):
    expanded_code = copyright

    expanded_code += """
#ifndef LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP

namespace rocshmem {
"""

    expanded_code += generate_signal_api()

    expanded_code += """
}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_SIG_OP.hpp'
    )

    write_to_file(output_file, expanded_code)
