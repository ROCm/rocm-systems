# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
