##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import re
import subprocess
from pathlib import Path

import pandas as pd

from utils.logger import console_debug, console_error, console_log

cache = dict()


# Note: shortener is now dependent on a rocprof install with llvm
def kernel_name_shortener(df, level):
    def shorten_file(df, level):
        global cache

        column_name = ""
        if "Kernel_Name" in df:
            column_name = "Kernel_Name"
        if "Name" in df:
            column_name = "Name"

        if column_name == "Kernel_Name" or column_name == "Name":
            # loop through all indices
            for index in df.index:
                original_name = df.loc[index, column_name]
                if original_name in cache:
                    continue

                cmd = [cpp_filt, original_name]

                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                demangled_name, e = proc.communicate()
                demangled_name = str(demangled_name, "UTF-8").strip()

                # cache miss, add the shortened name to the dictionary
                new_name = ""
                matches = ""

                names_and_args = re.compile(
                    r"(?P<name>[( )A-Za-z0-9_]+)([ ,*<>()]+)(::)?"
                )

                # works for name Kokkos::namespace::init_lock_array_kernel_threadid(int) [clone .kd]
                if names_and_args.search(demangled_name):
                    matches = names_and_args.findall(demangled_name)
                else:
                    # Works for first case  '__amd_rocclr_fillBuffer.kd'
                    cache[original_name] = new_name
                    if new_name == None or new_name == "":
                        cache[original_name] = demangled_name
                    continue

                current_level = 0
                for name in matches:
                    ##can cause errors if a function name or argument is equal to 'clone'
                    if name[0] == "clone":
                        continue
                    if len(name) == 3:
                        if name[2] == "::":
                            continue

                    if current_level < level:
                        new_name += name[0]
                    # closing '>' is to be taken account by the while loop
                    if name[1].count(">") == 0:
                        if current_level < level:
                            if not (
                                current_level == level - 1 and name[1].count("<") > 0
                            ):
                                new_name += name[1]
                        current_level += name[1].count("<")

                    curr_index = 0
                    # cases include '>'  '> >, ' have to go in depth here to not lose account of commas and current level
                    while name[1].count(">") > 0 and curr_index < len(name[1]):
                        if current_level < level:
                            new_name += name[1][curr_index:]
                            current_level -= name[1][curr_index:].count(">")
                            curr_index = len(name[1])
                        elif name[1][curr_index] == (">"):
                            current_level -= 1
                        curr_index += 1

                cache[original_name] = new_name
                if new_name == None or new_name == "":
                    cache[original_name] = demangled_name

            df[column_name] = df[column_name].map(cache)

        return df

    # Only shorten if valid shortening level
    if level < 5:
        cpp_filt = str(Path("/usr").joinpath("bin", "c++filt"))
        if not Path(cpp_filt).is_file():
            console_error(
                "Could not resolve c++filt in expected directory: %s" % cpp_filt
            )

        try:
            modified_df = shorten_file(df, level)
            console_log("profiling", "Kernel_Name shortening complete.")
            return modified_df
        except pd.errors.EmptyDataError:
            console_debug("profiling", "Skipping shortening on empty csv")
