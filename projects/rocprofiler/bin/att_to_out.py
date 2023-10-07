################################################################################
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

import numpy as np
import sys

BYTE_MAP = [str(k) for k in range(10)] + ["a", "b", "c", "d", "e", "f"]


def map8(c):
    return BYTE_MAP[(c // 16) % 16] + BYTE_MAP[c % 16]


def map16(c):
    return map8(c >> 8) + map8(c)


in_filename = sys.argv[1]
out_filename = in_filename.split(".att")[0] + ".out"

in_bytes = np.fromfile(in_filename, dtype=np.uint16)
offset = 4 if in_bytes[0] >= 0xC000 else 0
out_bytes = [map16(c) + "\n" for c in in_bytes[offset:]]

with open(out_filename, "w") as f:
    [f.write(b) for b in out_bytes]
