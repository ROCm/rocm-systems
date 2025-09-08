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

def sample_ex(source_str):
    spm_dump = source_str

    file_in = open(spm_dump, "r")

    #extract valid sample data size
    line = file_in.readline()
    line.strip()

    size = int(line, 16)

    line = file_in.readline()
    line.strip()

    size += (int(line, 16) << 16)

    print("valid sampl buffer size: %d"%(size))

    file_out = open("samples.txt", "w")

    line_no = 2

    non_zero_samples = 0

    while (size > 0):
        line = file_in.readline()
        line.strip()
        if line_no >= 16:
            file_out.write(line)
            size -= 2
            if int(line, 16) != 0:
                non_zero_samples += 1
                #print("%d: sample: %s"%(line_no, line))
        line_no += 1


    print ("non_zero_samples: %d"%(non_zero_samples))

    file_in.close()
    file_out.close()
