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

        # s = re.sub('SELECT', 'SELECT\n', s)  # only for 100

        s = re.sub(r"(?m)^[(union)|(from)|(where)|(SELECT)].*\n?", "", s)
        s = re.sub('AS "', "AS_", s)
        s = re.sub('As "', "AS_", s)
        s = re.sub("AS '", "AS_", s)
        s = re.sub("As '", "AS_", s)
        s = re.sub('"pct"', "unit: pct", s)

        s = re.sub(r"[\'|\"|,]", "", s)

        # s = re.sub(r'(?m)^(AVG).*\n?', '  avg: \g<0>', s)
        # s = re.sub(r'(?m)^(MIN).*\n?', '  min: \g<0>', s)
        # s = re.sub(r'(?m)^(MAX).*\n?', '  max: \g<0>', s)

        s = re.sub(r"(.+)(?=AS_Avg)", r"  avg: \g<1>", s)
        s = re.sub(r"AS_Avg", "", s)
        s = re.sub(r"(.+)(?=AS_Mean)", r"  avg: \g<1>", s)
        s = re.sub(r"AS_Mean", "", s)
        s = re.sub(r"(.+)(?=AS_Min)", r"  min: \g<1>", s)
        s = re.sub(r"AS_Min", "", s)
        s = re.sub(r"(.+)(?=AS_Max)", r"  max: \g<1>", s)
        s = re.sub(r"AS_Max", "", s)
        s = re.sub(r"(.+)(?=AS_Peak)", r"  peak: \g<1>", s)
        s = re.sub(r"AS_Peak", "", s)
        s = re.sub(r"(.+)(?=AS_Avg)", r"  avg: \g<1>", s)
        s = re.sub(r"AS_Avg", "", s)
        s = re.sub(r"(.+)(?=AS_Mean)", r"  avg: \g<1>", s)
        s = re.sub(r"AS_Mean", "", s)
        s = re.sub(r"AS_Metrics", ":", s)
        s = re.sub(r"(.+)(?=AS_Unit)", r"  unit: \g<1>", s)

        s = re.sub(r"AS_Unit", "\n  tips:", s)

        s = re.sub(r"\n", "\n      ", s)

        # not fully safe
        s = re.sub(r"(.*)(\#\S+)", '\g<1> "\g<2>"', s)

        print(s)
