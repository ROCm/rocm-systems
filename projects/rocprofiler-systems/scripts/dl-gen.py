#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import glob

"""
This script reads in function prototypes can generates the implementation pieces
needed to dlsym the function in librocprof-sys.

Example input file:

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names);
    void OnUnload();

generates:

    ##### declaration:

        bool OnLoad(HsaApiTable*, uint64_t, uint64_t, const char* const*) ROCPROFSYS_PUBLIC_API;
        void OnUnload() ROCPROFSYS_PUBLIC_API;

    ##### dlsym:

        ROCPROFSYS_DLSYM(OnLoad_f, m_omnihandle, "OnLoad");
        ROCPROFSYS_DLSYM(OnUnload_f, m_omnihandle, "OnUnload");

    ##### member variables:

        bool (*OnLoad_f)(HsaApiTable*, uint64_t, uint64_t, const char* const*) = nullptr;
        void (*OnUnload_f)() = nullptr;

    ##### callers:

        bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count, const char* const* failed_tool_names)
        {
            return ROCPROFSYS_DL_INVOKE(get_indirect().OnLoad_f, table, runtime_version, failed_tool_count, failed_tool_names);
        }

        void OnUnload()
        {
            return ROCPROFSYS_DL_INVOKE(get_indirect().OnUnload_f);
        }
"""


class function:
    def __init__(self, _f):
        self.return_type = _f.split(" ", 1)[0]
        _f = "".join(_f.split(" ", 1)[1:])
        self.func_name = _f.split("(", 1)[0]
        _f = "".join(_f.split("(", 1)[1:]).rstrip(")")
        self.params = [x.strip() for x in _f.split(",")]
        self.param_types = []
        self.param_names = []
        for itr in self.params:
            _fields = itr.split(" ")
            _len = len(_fields)
            self.param_types.append(" ".join(_fields[0 : (_len - 1)]))
            self.param_names.append(_fields[-1])

    def valid(self):
        return len(self.func_name) > 0

    def member_variables(self):
        return "    {} (*{}_f)({}) = nullptr;".format(
            self.return_type, self.func_name, ", ".join(self.param_types)
        )

    def function_decl(self):
        return "    {} {}({}) ROCPROFSYS_PUBLIC_API;".format(
            self.return_type, self.func_name, ", ".join(self.param_types)
        )

    def dlsym_function(self):
        return '    ROCPROFSYS_DLSYM({0}_f, m_omnihandle, "{0}");'.format(self.func_name)

    def call_dlsym_function(self):
        _param_names = ", ".join(self.param_names)
        if _param_names and _param_names != ", ":
            _param_names = f", {_param_names}"
        return "    {} {}({})\n    {}\n        return ROCPROFSYS_DL_INVOKE(get_indirect().{}_f{});\n    {}".format(
            self.return_type,
            self.func_name,
            ", ".join(self.params),
            "{",
            self.func_name,
            _param_names,
            "}",
        )


def run(fname):
    with open(fname, "r") as f:
        _str = ""
        for itr in f.read():
            _str += itr.replace("\n", " ")

        while "  " in _str:
            _str = _str.replace("  ", " ")
        data = [x.strip(" ") for x in _str.split(";")]

    funcs = []
    for itr in data:
        f = function(itr)
        if f.valid():
            funcs.append(f)

    return funcs


if __name__ == "__main__":
    funcs = []
    for inp in sys.argv[1:]:
        if os.path.exists(inp):
            funcs += run(inp)
        else:
            for itr in glob.glob(f"{inp}*"):
                if os.path.exists(itr):
                    funcs += run(itr)
                else:
                    printf(f"No file matched {itr}")

    if funcs:
        print(f"\n##### declaration:\n")
        for itr in funcs:
            print("{}".format(itr.function_decl()))

        print(f"\n##### dlsym:\n")
        for itr in funcs:
            print("{}".format(itr.dlsym_function()))

        print(f"\n##### member variables:\n")
        for itr in funcs:
            print("{}".format(itr.member_variables()))

        print(f"\n##### callers:")
        for itr in funcs:
            print("")
            print("{}".format(itr.call_dlsym_function()))

        print("")
