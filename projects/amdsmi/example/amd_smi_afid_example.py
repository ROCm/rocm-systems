# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from amdsmi import *
import os

amdsmi_init()

def amdsmi_get_afids_from_cper():
    directory_path = "/tmp/cper_dump/"
    print(f"Searching for cper file in {directory_path}")
    with os.scandir(directory_path) as cper_files:
        for cper_file in cper_files:
            if cper_file.is_file():  # Check if the entry is a file (not a subdirectory)
                if ".bin" in cper_file.path:
                    print(f"Found {cper_file.path}")
                    with open(cper_file.path, "rb") as file:
                        raw = file.read()
                        afids, num_afids = amdsmi_interface.amdsmi_get_afids_from_cper(raw)
                        print(f"afids: {afids}")

amdsmi_get_afids_from_cper()

"""
Sample output:

sudo python3 afid.py
Searching for cper file in /tmp/cper_dump/
Found /tmp/cper_dump/cper_entry_0.bin
afids: [17]
Found /tmp/cper_dump/cper_entry_1.bin
afids: [17]
"""