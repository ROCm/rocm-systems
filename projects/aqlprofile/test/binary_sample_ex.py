# MIT License
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import struct

def sample_ex(source_str, length):
    file = open(source_str, "rb")
    file_out = open("samples.txt", "w")

    count = 0
    sample = file.read(2)
    while sample:
        if length != -1:
            if count >= length:
                break
        #print(sample)
        value = struct.unpack('H', sample)[0]
        t = "{:04x}".format(value)
        #print(t)
        line = t + "\n"
        file_out.write(line)
        sample = file.read(2)
        count += 2

    file.close()
    file_out.close()
