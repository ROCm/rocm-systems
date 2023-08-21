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

import os, sys, re


# gen_params() takes a text file like the output of rocminfo cmd and parses it into a map {key,value}
# where key is the param and value is the value of this param
# for example: Threadmodel : "posix"
# it also processes encompasing sections to generate a full param name such as (section names separated by '_'):
#     "Agent2_PoolInfo_ISAInfo_ISA1_WorkgroupMaxSizeperDimension_x": "1024(0x400)",
def gen_params(txtfile):
    fields = {}
    counter = 0
    parent_field = ""
    nbr_indent = 0
    nbr_indent_prev = 0
    check_for_dims = False
    with open(txtfile) as fp:
        for line in fp:
            me = re.match(r"\*\*\* Done \*\*\*", line)  # Marks the end of cmd
            if me:
                parent_field = ""
                nbr_indent = 0
                nbr_indent_prev = 0
                check_for_dims = False
                continue
            mv = re.match(
                r"HCC clang version\s+(.*)", line
            )  # outlier: only line with a version number and no ':', special case
            if mv:
                key = "HCCclangversion"
                val = mv.group(1)
                counter = counter + 1
                fields[(counter, key)] = val
                continue
            # Variable 'check_for_dims' is True for text like this:
            # Workgroup Max Size per Dimension:
            #     x                        1024(0x400)
            #     y                        1024(0x400)
            #     z                        1024(0x400)
            if check_for_dims == True:
                mc = re.match(r"\s*([x|y|z])\s+(.*)", line)
                if mc:
                    key_sav = mc.group(1)
                    if parent_field != "":
                        key = parent_field + "." + mc.group(1)
                    else:
                        key = mc.group(1)
                    val = re.sub(r"\s+", "", mc.group(2))
                    counter = counter + 1
                    fields[(counter, key)] = val
                    if key_sav == "z":
                        check_for_dims = False
            nbr_indent_prev = nbr_indent
            mi = re.search(r"^(\s+)\w+.*", line)
            md = re.search(":", line)
            if mi:
                nbr_indent = int(len(mi.group(1)) / 2)  # indentation cnt
            else:
                if not md:
                    tmp = re.sub(r"\s+", "", line)
                    if tmp.isalnum():
                        parent_field = tmp

            if nbr_indent < nbr_indent_prev:
                go_back_parent = nbr_indent_prev - nbr_indent
                for i in range(go_back_parent):  # decrease as many levels up as needed
                    pos = parent_field.rfind(".")
                    if pos != -1:
                        parent_field = parent_field[:pos]
            # Process lines such as :
            # Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
            # Size:                    131897644(0x7dc992c) KB
            for lin in line.split(";"):
                lin = re.sub(r"\s+", "", lin)
                m = re.match(r"(.*):(.*)", lin)
                if m:
                    key, val = m.group(1), m.group(2)
                    if parent_field != "":
                        key = parent_field + "." + key
                    if val == "":
                        mk = re.match(r".*Dimension", key)
                        if mk:  # expect x,y,z on next 3 lines
                            check_for_dims = True
                        parent_field = key
                    else:
                        counter = counter + 1
                        fields[(counter, key)] = val
                else:
                    if nbr_indent != nbr_indent_prev and not check_for_dims:
                        parent_field = parent_field + "." + lin.replace(":", "")

    return fields
