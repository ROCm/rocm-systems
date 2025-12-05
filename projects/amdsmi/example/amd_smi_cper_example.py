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

def get_severity_mask(severity):
    severity_mask = 0
    if severity == "all":
        # Set bits for NON_FATAL_UNCORRECTED (0), FATAL (1), and NON_FATAL_CORRECTED (2)
        severity_mask |= ((1 << 0) | (1 << 1) | (1 << 2))
    elif severity == "fatal":
        # Set bit corresponding to AMDSMI_CPER_SEV_FATAL (which is 1)
        severity_mask |= (1 << 1)
    elif severity in ("nonfatal", "nonfatal-uncorrected"):
        # Set bit corresponding to AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED (which is 0)
        severity_mask |= (1 << 0)
    elif severity in ("nonfatal-corrected", "corrected"):
        # Set bit corresponding to AMDSMI_CPER_SEV_NON_FATAL_CORRECTED (which is 2)
        severity_mask |= (1 << 2)
    return severity_mask

def gpuid(device):
    for gpu_index, device_handle in enumerate(amdsmi_interface.amdsmi_get_processor_handles()):
        if device.value == device_handle.value:
            return gpu_index

def dump_cper_entry(entry, cper_data, key):
    try:
        os.mkdir("/tmp/cper_dump", mode=0o777, dir_fd=None)
    except FileExistsError:
        pass
    cper_file = f"/tmp/cper_dump/cper_entry_{key}.bin"
    with open(cper_file, "wb") as file:
        size = cper_data[key]["size"]
        data = cper_data[key]["bytes"]
        data = bytes(x % 256 for x in data[:size])
        file.write(data)
        print(f"   Wrote cper data to file: {cper_file}")
    json_file = f"/tmp/cper_dump/cper_entry_{key}.json"
    with open(json_file, "wt") as file:
        file.write(str(entry))

def get_gpu_cper_entries():
    try:
        devices = amdsmi_interface.amdsmi_get_processor_handles()
        buffer_size = 1024*100
        initial_cursor = 0
        severity = "all"
        for device in devices:
            while True:
                entries, new_cursor, cper_data, status_code = amdsmi_get_gpu_cper_entries(
                    device, get_severity_mask(severity), buffer_size, initial_cursor)
                gpu_id = gpuid(device)
                print("#############################################################################")
                print(f"cper entries for severity: '{severity}', gpu #{gpu_id}, cursor: {initial_cursor}-{new_cursor - 1}")
                for key, entry in entries.items():
                    print("----------------")
                    print("Entry", initial_cursor + key)
                    print("   Error Severity:", entry.get("error_severity", "Unknown"))
                    print("   Notify Type:", entry.get("notify_type", "Unknown"))
                    print("   Timestamp:", entry.get("timestamp", ""))
                    print(f"   Cper entry metadata: {entry}")
                    dump_cper_entry(entry, cper_data, key)
                if initial_cursor == new_cursor:
                    break
                initial_cursor = new_cursor
            break
    except AmdSmiException as e:
        print(e)

get_gpu_cper_entries()

"""
Sample output:

cper entries for severity: 'all', gpu #0, cursor: 0-3
----------------
Entry 0
   Error Severity: non_fatal_corrected
   Notify Type: CMC
   Timestamp: 2025/09/07 00:14:22
   Cper entry metadata: {'error_severity': 'non_fatal_corrected', 'notify_type': 'CMC', 'timestamp': '2025/09/07 00:14:22', 'signature': b'CPER', 'revision': 256, 'signature_end': '0xffffffff', 'sec_cnt': 1, 'record_length': 472, 'platform_id': b'0x1002:0x74A2', 'creator_id': b'amdgpu', 'record_id': b'5:1', 'flags': 0, 'persistence_info': 0}
   Wrote cper data to file: /tmp/cper_dump/cper_entry_0.bin
----------------
Entry 1
   Error Severity: non_fatal_corrected
   Notify Type: CMC
   Timestamp: 2025/09/07 00:14:26
   Cper entry metadata: {'error_severity': 'non_fatal_corrected', 'notify_type': 'CMC', 'timestamp': '2025/09/07 00:14:26', 'signature': b'CPER', 'revision': 256, 'signature_end': '0xffffffff', 'sec_cnt': 1, 'record_length': 472, 'platform_id': b'0x1002:0x74A2', 'creator_id': b'amdgpu', 'record_id': b'5:2', 'flags': 0, 'persistence_info': 0}
   Wrote cper data to file: /tmp/cper_dump/cper_entry_1.bin
----------------
Entry 2
   Error Severity: non_fatal_corrected
   Notify Type: CMC
   Timestamp: 2025/09/08 06:12:11
   Cper entry metadata: {'error_severity': 'non_fatal_corrected', 'notify_type': 'CMC', 'timestamp': '2025/09/08 06:12:11', 'signature': b'CPER', 'revision': 256, 'signature_end': '0xffffffff', 'sec_cnt': 1, 'record_length': 472, 'platform_id': b'0x1002:0x74A2', 'creator_id': b'amdgpu', 'record_id': b'5:3', 'flags': 0, 'persistence_info': 0}
   Wrote cper data to file: /tmp/cper_dump/cper_entry_2.bin
----------------
Entry 3
   Error Severity: non_fatal_corrected
   Notify Type: CMC
   Timestamp: 2025/09/08 06:13:59
   Cper entry metadata: {'error_severity': 'non_fatal_corrected', 'notify_type': 'CMC', 'timestamp': '2025/09/08 06:13:59', 'signature': b'CPER', 'revision': 256, 'signature_end': '0xffffffff', 'sec_cnt': 1, 'record_length': 472, 'platform_id': b'0x1002:0x74A2', 'creator_id': b'amdgpu', 'record_id': b'5:4', 'flags': 0, 'persistence_info': 0}
   Wrote cper data to file: /tmp/cper_dump/cper_entry_3.bin
#############################################################################
cper entries for severity: 'all', gpu #0, cursor: 4-3
"""