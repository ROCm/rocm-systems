#!/usr/bin/env python3

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

import re
import sys

if __name__ == "__main__":

    with open(sys.argv[1], "r") as file:
        s = file.read()

        s = re.sub("\,\s*\n", ", ", s)
        s = re.sub(r"[\'|\"]", "", s)
        s = re.sub(r"from.*", "", s)
        s = re.sub(
            r".*SELECT\s*(.*)\,(.*),(.*),(.*),(.*)\s*",
            "          \g<1>:\n            avg: \g<2>\n            min: \g<3>\n            max: \g<4>\n            unit: \g<5>\n            tips:\n",
            s,
        )

        print(s)
