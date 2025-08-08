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

import sys, os, re
from sqlitedb import SQLiteDB

pinned = ["hipMallocHost", "hipHostMalloc", "hipHostAlloc"]
ondevice = [
    "hipMalloc",
    "hipMallocPitch",
    "hipMallocArray",
    "hipMalloc3DArray",
    "hsa_amd_memory_pool_allocate",
]

mm_table_descr = [
    [
        "BeginNs",
        "EndNs",
        "pid",
        "tid",
        "Name",
        "Direction",
        "SrcType",
        "DstType",
        "Size",
        "Async",
    ],
    {
        "BeginNs": "INTEGER",
        "EndNs": "INTEGER",
        "pid": "INTEGER",
        "tid": "INTEGER",
        "Name": "TEXT",
        "Direction": "TEXT",
        "SrcType": "TEXT",
        "DstType": "TEXT",
        "Size": "INTEGER",
        "Async": "TEXT",
    },
]


def fatal(msg):
    sys.stderr.write(sys.argv[0] + ": " + msg + "\n")
    sys.exit(1)


DELIM = ","


# Mem copy manager class
class MemManager:
    def __init__(self, db, indir):
        self.db = db
        self.allocations = {}
        self.hsa_agent_types = {}
        self.memcopies = {}
        self.filename = ""
        self.fd = 0
        self.parse_hsa_handles(indir + "/" + "hsa_handles.txt")

    def __del__(self):
        if self.fd != 0:
            self.fd.close()

    # Parsing the mapping of HSA agent and memory pool handles
    def parse_hsa_handles(self, infile):
        if os.path.exists(infile):
            inp = open(infile, "r")
            cpu_agent_ptrn = re.compile(r"(0x[0-9a-fA-F]+) agent cpu")
            gpu_agent_ptrn = re.compile(r"(0x[0-9a-fA-F]+) agent gpu")
            for line in inp.readlines():
                m_cpu = cpu_agent_ptrn.match(line)
                if m_cpu:
                    self.hsa_agent_types[str(int(m_cpu.group(1), 16))] = 0  # "cpu"
                m_gpu = gpu_agent_ptrn.match(line)
                if m_gpu:
                    self.hsa_agent_types[str(int(m_gpu.group(1), 16))] = 1  # "gpu"
            inp.close()

    # register alloc and memcpy API calls
    # ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'args', 'Index', 'Data'],
    def register_api(self, rec_vals):
        res = ""
        record_name = rec_vals[4]  # 'Name'
        record_args = rec_vals[5]  # 'args'
        malloc_ptrn = re.compile(r"hip.*Malloc|hsa_amd_memory_pool_allocate")
        mcopy_ptrn = re.compile(r"hipMemcpy|hsa_amd_memory_async_copy")

        if malloc_ptrn.match(record_name):
            self.add_allocation(record_name, record_args)
        elif mcopy_ptrn.match(record_name):
            res = self.add_memcpy(rec_vals)

        return res

    # register memcpy asynchronous copy
    # ['BeginNs', 'EndNs', 'Name', 'pid', 'tid', 'Index', ...
    def register_copy(self, rec_vals):
        data = ""
        event = rec_vals[2]  # 'Name'
        procid = rec_vals[3]  # 'pid'
        recordid = rec_vals[5]  # 'Index'
        # query syncronous memcopy API record
        key = (recordid, procid, 0)
        if key in self.memcopies:
            data = self.memcopies[key]

        # query asyncronous memcopy API record
        key = (recordid, procid, 1)
        if key in self.memcopies:
            if data != "":
                fatal("register_copy: corrupted record sync/async")
            async_copy_start_time = rec_vals[0]
            async_copy_end_time = rec_vals[1]

            tid = rec_vals[4]
            copy_line_header = (
                str(async_copy_start_time)
                + DELIM
                + str(async_copy_end_time)
                + DELIM
                + str(procid)
                + DELIM
                + str(tid)
            )
            copy_line_footer = "Async=" + str(1)
            data = copy_line_header + self.memcopies[key] + copy_line_footer
            self.memcopies[key] = data

        return data

    # register memcpy asynchronous activity
    # rec_vals: ['BeginNs', 'EndNs', 'dev-id', 'queue-id', 'Name', 'pid', 'tid', 'Index', 'Data', ...
    def register_activity(self, rec_vals):
        data = ""
        procid = rec_vals[5]  # 'pid'
        recordid = rec_vals[7]  # 'Index'

        # query syncronous memcopy API record
        key = (recordid, procid, 0)
        if key in self.memcopies:
            data = self.memcopies[key]

        # query asyncronous memcopy API record
        key = (recordid, procid, 1)
        if key in self.memcopies:
            if data != "":
                fatal("register_activity: corrupted record sync/async")

            async_copy_start_time = rec_vals[0]
            async_copy_end_time = rec_vals[1]

            tid = rec_vals[6]
            copy_line_header = (
                str(async_copy_start_time)
                + DELIM
                + str(async_copy_end_time)
                + DELIM
                + str(procid)
                + DELIM
                + str(tid)
            )
            copy_line_footer = "Async=" + str(1)
            data = copy_line_header + self.memcopies[key] + copy_line_footer
            self.memcopies[key] = data

        return data

    # add allocation to map
    def add_allocation(self, event, args):
        choice = 0
        if event == "hipMallocPitch":
            malloc_args_ptrn = re.compile(r"\(ptr\((.*)\) width\((.*)\) height\((.*)\)\)")
            choice = 1
        elif event == "hipMallocArray":
            malloc_args_ptrn = re.compile(
                r"\(array\((.*)\) width\((.*)\) height\((.*)\)\)"
            )
            choice = 1
        elif event == "hipMalloc3DArray":
            malloc_args_ptrn = re.compile(
                r"\(array\((.*)\) width\((.*)\) height\((.*)\) depth\((.*)\)\)"
            )
            choice = 2
        elif event == "hsa_amd_memory_pool_allocate":
            # ({handle=25291264}, 40, 0, 0x7ffc4c7bf1b0)
            malloc_args_ptrn = re.compile(
                r"\({handle=\d+}, (\d+), \d+, (0x[0-9a-fA-F]+)\)"
            )
            choice = 4
        else:
            # (ptr(0x7f3407000000) size(800000000) flags(0))
            malloc_args_ptrn = re.compile(r"\(ptr\((.*)\) size\((.*)\) .*\)")
            choice = 3
        m = malloc_args_ptrn.match(args)
        if m:
            if choice == 4:
                ptr = int(m.group(2), 16)
                size = int(m.group(1))
            elif choice == 3:
                ptr = int(m.group(1), 16)
                size = int(m.group(2))
            elif choice == 1:
                ptr = int(m.group(1), 16)
                size = int(m.group(2)) * int(m.group(3))
            else:
                ptr = int(m.group(1), 16)
                size = int(m.group(2)) * int(m.group(3)) * int(m.group(4))
            self.allocations[ptr] = (size, event)

    # get type of ptr
    def get_ptr_type(self, ptr):
        addr = int(ptr, 16)
        addr_type = "unknown"
        found = 0
        for base, (size, event) in self.allocations.items():
            if addr >= base and addr < base + size:
                found = 1
                break
        if not found:
            addr_type = "pageable"
        elif event in pinned:
            addr_type = "pinned"
        elif event in ondevice:
            addr_type = "device"
        elif ptr in self.hsa_agent_types:
            if self.hsa_agent_types[ptr] == 0:
                addr_type = "pinned"
            elif self.hsa_agent_types[ptr] == 1:
                addr_type = "device"
            else:
                fatal("internal error: ptr(" + ptr + ") cannot be identified")
        else:
            fatal("internal error: ptr(" + ptr + ") cannot be identified")
        return addr_type

    # add memcpy to map
    def add_memcpy(self, recvals):
        recordid = recvals[6]  # same as corrid
        event = recvals[4]
        start_time = recvals[0]  # sync time stamp
        end_time = recvals[1]  # sync time stamp
        args = recvals[5]
        procid = int(recvals[2])  # used to query async entries
        pid = recvals[2]
        tid = recvals[3]

        # hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
        hip_memcpy_ptrn = re.compile(
            r"\(\s*dst\((.*)\) src\((.*)\) sizeBytes\((\d+)\).*\)"
        )
        # hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
        #                   size_t height, hipMemcpyKind kind);
        hip_memcpy_ptrn2 = re.compile(
            r"\(\s*dst\((.*)\) .* src\((.*)\) .* width\((\d+)\) height\((\d+)\).*\)"
        )
        # hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
        #                        size_t count, hipMemcpyKind kind);
        hip_memcpy_ptrn3 = re.compile(
            r"\(\s*dst\((.*)\) .* src\((.*)\) count\((\d+)\).*\)"
        )
        # hipMemcpyToSymbol(const void* symbolName, const void* src, size_t sizeBytes,
        #    size_t offset = 0, hipMemcpyKind kind)
        hip_memcpy_ptrn4 = re.compile(
            r"\(\s*symbol\((.*)\) src\((.*)\) sizeBytes\((\d+)\).*\)"
        )
        # memcopy with kind argument
        hip_memcpy_ptrn_kind = re.compile(r".* kind\((\d+)\)\s*.*")
        # hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
        #                          hsa_agent_t src_agent, size_t size,
        #                          uint32_t num_dep_signals,
        #                          const hsa_signal_t* dep_signals,
        #                          hsa_signal_t completion_signal);
        # "(0x7f8ab6600000, 27064880, 0x7f8b16000000, 27059968, 800000000, 0, 0, 140240759809536) = 0"
        # hsa_memcpy_ptrn_prev used to support format transition and will be cleaned up later.
        hsa_memcpy_ptrn_prev = re.compile(
            r"\((0x[0-9a-fA-F]+), (\d+), (0x[0-9a-fA-F]+), (\d+), (\d+), .*\) = \d"
        )
        # "(0x7fd83bc00000, {handle=16124864}, 0x7fd89b600000, {handle=16119808}, 800000000, 0, 0, {handle=140573877724672}) = 0"
        hsa_memcpy_ptrn = re.compile(
            r"\((0x[0-9a-fA-F]+), {handle=(\d+)}, (0x[0-9a-fA-F]+), {handle=(\d+)}, (\d+), .*\) = \d"
        )
        #    "(0x7f9125cfe7b0, 0x7f9125cfe784, 0x7f9125cfe790, 0x7f9125cfe784, 0x7f9125cfe778, {handle=94324038652880}, 1, 0, 0, {handle=140261380710784}) = 0"
        #    dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, completion_signal
        hsa_memcpy_ptrn2 = re.compile(
            r"\((0x[0-9a-fA-F]+), 0x[0-9a-fA-F]+, (0x[0-9a-fA-F]+), 0x[0-9a-fA-F]+, 0x[0-9a-fA-F]+, {z=(\d+), y=(\d+), x=(\d+)}, {handle=(\d+)}, .*\) = \d"
        )
        # aysnc memcopy
        async_event_ptrn = re.compile(r"Async|async")
        m_basic_hip = hip_memcpy_ptrn.match(args)
        m_basic_hsa3 = hip_memcpy_ptrn4.match(args)
        m_basic_hsa_prev = hsa_memcpy_ptrn_prev.match(args)
        m_basic_hsa = hsa_memcpy_ptrn.match(args)
        m_basic_hsa2 = hsa_memcpy_ptrn2.match(args)
        is_hip = True if not (m_basic_hsa_prev or m_basic_hsa or m_basic_hsa2) else False
        m_2d = hip_memcpy_ptrn2.match(args)
        m_array = hip_memcpy_ptrn3.match(args)
        is_async = 1 if async_event_ptrn.search(event) else 0
        copy_line = ""
        size = 0
        dstptr_type = "unknown"
        srcptr_type = "unknown"
        direction = "unknown"
        kind_switcher = {
            "0": "HtoH",
            "1": "HtoD",
            "2": "DtoH",
            "3": "DtoD",
            "4": "auto",
        }

        condition_matched = False
        if m_basic_hip:
            dstptr = m_basic_hip.group(1)
            dstptr_type = self.get_ptr_type(dstptr)
            srcptr = m_basic_hip.group(2)
            srcptr_type = self.get_ptr_type(srcptr)
            size = int(m_basic_hip.group(3))
            condition_matched = True

        if m_basic_hsa_prev:
            dstptr = m_basic_hsa_prev.group(1)
            dst_agent_ptr = m_basic_hsa_prev.group(2)
            dstptr_type = self.get_ptr_type(dst_agent_ptr)
            srcptr = m_basic_hsa_prev.group(3)
            src_agent_ptr = m_basic_hsa_prev.group(4)
            srcptr_type = self.get_ptr_type(src_agent_ptr)
            size = int(m_basic_hsa_prev.group(5))
            condition_matched = True

        if m_basic_hsa:
            dstptr = m_basic_hsa.group(1)
            dst_agent_ptr = m_basic_hsa.group(2)
            dstptr_type = self.get_ptr_type(dst_agent_ptr)
            srcptr = m_basic_hsa.group(3)
            src_agent_ptr = m_basic_hsa.group(4)
            srcptr_type = self.get_ptr_type(src_agent_ptr)
            size = int(m_basic_hsa.group(5))
            condition_matched = True

        if m_basic_hsa2:
            dstptr = m_basic_hsa2.group(1)
            dst_agent_ptr = m_basic_hsa2.group(6)
            dstptr_type = self.get_ptr_type(dst_agent_ptr)
            srcptr = m_basic_hsa2.group(2)
            src_agent_ptr = m_basic_hsa2.group(6)
            srcptr_type = self.get_ptr_type(src_agent_ptr)
            z = int(m_basic_hsa2.group(3))
            y = int(m_basic_hsa2.group(4))
            x = int(m_basic_hsa2.group(5))
            size = x * y * z
            condition_matched = True

        if m_basic_hsa3:
            dstptr = m_basic_hsa3.group(1)
            dstptr_type = self.get_ptr_type(dstptr)
            srcptr = m_basic_hsa3.group(2)
            srcptr_type = self.get_ptr_type(srcptr)
            size = int(m_basic_hsa3.group(3))
            condition_matched = True

        if m_array:
            dstptr = m_array.group(1)
            dstptr_type = self.get_ptr_type(dstptr)
            srcptr = m_array.group(2)
            srcptr_type = self.get_ptr_type(srcptr)
            size = m_array.group(3)
            condition_matched = True

        if m_2d:
            dstptr = m_2d.group(1)
            dstptr_type = self.get_ptr_type(dstptr)
            srcptr = m_2d.group(2)
            srcptr_type = self.get_ptr_type(srcptr)
            size = int(m_2d.group(3)) * int(m_2d.group(4))
            condition_matched = True

        if not condition_matched:
            fatal('Memcpy args "' + args + '" cannot be identified')

        if not is_async:
            start_time = recvals[0]  # sync time stamp
            end_time = recvals[1]  # sync time stamp
            duration = int(end_time) - int(start_time)

        evt_switcher = {
            "hipMemcpyDtoD": "DtoD",
            "hipMemcpyDtoDAsync": "DtoD",
            "hipMemcpyDtoH": "DtoH",
            "hipMemcpyDtoHAsync": "DtoH",
            "hipMemcpyHtoD": "HtoD",
            "hipMemcpyHtoDAsync": "HtoD",
        }

        if is_hip:
            m = hip_memcpy_ptrn_kind.match(args)
            if m:
                direction = kind_switcher.get(m.group(1), "unknown")
            else:
                direction = evt_switcher.get(event, "unknown")
        else:
            if (
                dst_agent_ptr in self.hsa_agent_types
                and src_agent_ptr in self.hsa_agent_types
            ):
                if self.hsa_agent_types[src_agent_ptr] == 1:
                    direction = "D"
                elif self.hsa_agent_types[src_agent_ptr] == 0:
                    direction = "H"
                if direction != "unknown":
                    direction += "to"
                if self.hsa_agent_types[dst_agent_ptr] == 1:
                    direction += "D"
                elif self.hsa_agent_types[dst_agent_ptr] == 0:
                    direction += "H"

        copy_line_header = ""
        copy_line_footer = ""
        copy_line_header = (
            str(start_time) + DELIM + str(end_time) + DELIM + str(pid) + DELIM + str(tid)
        )
        copy_line_footer = "Async=" + str(is_async)

        copy_line = (
            copy_line_header
            + DELIM
            + event
            + DELIM
            + "Direction="
            + direction
            + DELIM
            + "SrcType="
            + srcptr_type
            + DELIM
            + "DstType="
            + dstptr_type
            + DELIM
            + "Size="
            + str(size)
            + DELIM
            + copy_line_footer
        )

        self.memcopies[(recordid, procid, is_async)] = copy_line
        return copy_line

    def dump_data(self, table_name, file_name):
        # To create memcopy info table in DB
        print("File '" + file_name + "' is generating")
        table_handle = self.db.add_table(table_name, mm_table_descr)

        fld_ptrn = re.compile(r"(.*)=(.*)")
        for key, record in self.memcopies.items():
            rec_vals_array = []
            for rec in record.split(DELIM):
                fld_ptrnm = fld_ptrn.match(rec)
                if fld_ptrnm:
                    rec_vals_array.append(fld_ptrnm.group(2))
                else:
                    rec_vals_array.append(rec)
            self.db.insert_entry(table_handle, rec_vals_array)

        # To dump the memcopy info table as CSV
        self.db.dump_csv(table_name, file_name)
