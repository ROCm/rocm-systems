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

import os
import sys
import re
import subprocess
import bisect
from sqlitedb import SQLiteDB
from mem_manager import MemManager
import dform

mcopy_data_enabled = 0

EXT_PID = 0
COPY_PID = 1
HIP_PID = 2
HSA_PID = 3
OPS_PID = 5
GPU_BASE_PID = 6
NONE_PID = -1

max_gpu_id = -1
START_NS = 0

hsa_activity_found = 0

# dependencies dictionary
dep_dict = {}
kern_dep_list = []
last_hip_api_map = {}
hip_streams = []
from_ids = {}

# stream ID map
stream_counter = 0
stream_id_map = {}


def get_stream_index(stream_id):
    global stream_counter
    stream_ind = 0
    if stream_id.lower() != "nil":
        if not stream_id in stream_id_map:
            stream_counter += 1
            stream_ind = stream_counter
            stream_id_map[stream_id] = stream_ind
        else:
            stream_ind = stream_id_map[stream_id]
    return stream_ind


# patching activity records
def activity_record_patching(
    db, ops_table_name, kernel_found, kernel_name, stream_found, stream_ind, select_expr
):
    if kernel_found != 0:
        db.change_rec_fld(ops_table_name, 'Name = "' + kernel_name + '"', select_expr)
    if stream_found != 0:
        db.change_rec_fld(ops_table_name, "tid = " + str(stream_ind), select_expr)


# global vars
table_descr = [["Index", "KernelName"], {"Index": "INTEGER", "KernelName": "TEXT"}]
var_list = table_descr[0]
var_table = {}
#############################################################


def fatal(msg):
    sys.stderr.write(sys.argv[0] + ": " + msg + "\n")
    sys.exit(1)


dbglog_count = 0


def dbglog(msg):
    global dbglog_count
    dbglog_count += 1
    sys.stderr.write(sys.argv[0] + ": " + msg + "\n")
    fatal("error")


#############################################################

# Dumping sysinfo
sysinfo_begin = 1


def metadata_gen(sysinfo_file, sysinfo_cmd):
    global sysinfo_begin
    if not re.search(r"\.txt$", sysinfo_file):
        raise Exception('wrong output file type: "' + sysinfo_file + '"')
    if sysinfo_begin == 1:
        sysinfo_begin = 0
        with open(sysinfo_file, mode="w") as fd:
            fd.write("")
    with open(sysinfo_file, mode="a") as fd:
        fd.write("CMD: " + sysinfo_cmd + "\n")
    status = subprocess.call(
        sysinfo_cmd + " >> " + sysinfo_file, stderr=subprocess.STDOUT, shell=True
    )
    if status != 0:
        raise Exception('Could not run command: "' + sysinfo_cmd + '"')


# parse results method
def parse_res(infile):
    global max_gpu_id
    if not os.path.isfile(infile):
        return
    inp = open(infile, "r", errors="replace")

    beg_pattern = re.compile(r'^dispatch\[(\d*)\], (.*) kernel-name\("([^"]*)"\)')
    prop_pattern = re.compile(r"([\w-]+)\((\w+)\)")
    ts_pattern = re.compile(r", time\((\d*),(\d*),(\d*),(\d*)\)")
    # var pattern below matches a variable name and a variable value from a one
    # line text in the format of for example "WRITE_SIZE (0.2500000000)" or
    # "GRBM_GUI_ACTIVE (27867)" or "TA_TA_BUSY[0]"
    var_pattern = re.compile(r"^\s*([a-zA-Z0-9_]+(?:\[\d+\])?)\s+\((\d+(?:\.\d+)?)\)")
    pid_pattern = re.compile(r"pid\((\d*)\)")

    dispatch_number = 0
    var_table_pid = 0
    for line in inp.readlines():
        record = line[:-1]

        m = pid_pattern.search(record)
        if m and not os.getenv("ROCP_MERGE_PIDS"):
            var_table_pid = int(m.group(1))

        m = var_pattern.match(record)
        if m:
            if not (var_table_pid, dispatch_number) in var_table:
                fatal("Error: dispatch number not found '" + str(dispatch_number) + "'")
            var = m.group(1)
            val = m.group(2)
            var_table[(var_table_pid, dispatch_number)][var] = val
            if not var in var_list:
                var_list.append(var)

        m = beg_pattern.match(record)
        if m:
            dispatch_number = m.group(1)
            if not (var_table_pid, dispatch_number) in var_table:
                var_table[(var_table_pid, dispatch_number)] = {
                    "Index": dispatch_number,
                    "KernelName": '"' + m.group(3) + '"',
                }

                gpu_id = 0
                queue_id = 0
                disp_pid = 0
                disp_tid = 0

                kernel_properties = m.group(2)
                for prop in kernel_properties.split(", "):
                    m = prop_pattern.match(prop)
                    if m:
                        var = m.group(1)
                        val = m.group(2)
                        var_table[(var_table_pid, dispatch_number)][var] = val
                        if not var in var_list:
                            var_list.append(var)
                        if var == "gpu-id":
                            gpu_id = int(val)
                            if gpu_id > max_gpu_id:
                                max_gpu_id = gpu_id
                        if var == "queue-id":
                            queue_id = int(val)
                        if var == "pid":
                            disp_pid = int(val)
                        if var == "tid":
                            disp_tid = int(val)
                    else:
                        fatal(
                            'wrong kernel property "'
                            + prop
                            + '" in "'
                            + kernel_properties
                            + '"'
                        )
                m = ts_pattern.search(record)
                if m:
                    var_table[(var_table_pid, dispatch_number)]["DispatchNs"] = m.group(1)
                    var_table[(var_table_pid, dispatch_number)]["BeginNs"] = m.group(2)
                    var_table[(var_table_pid, dispatch_number)]["EndNs"] = m.group(3)
                    var_table[(var_table_pid, dispatch_number)]["CompleteNs"] = m.group(4)

                    # filling dependenciws
                    from_ns = int(m.group(1))
                    to_ns = int(m.group(2))
                    from_us = int((from_ns - START_NS) / 1000)
                    to_us = int((to_ns - START_NS) / 1000)

                    kern_dep_list.append((from_ns, disp_pid, disp_tid))

                    gpu_pid = GPU_BASE_PID + int(gpu_id)
                    if not disp_pid in dep_dict:
                        dep_dict[disp_pid] = {}
                    dep_proc = dep_dict[disp_pid]
                    if not gpu_pid in dep_proc:
                        dep_proc[gpu_pid] = {
                            "pid": HSA_PID,
                            "from": [],
                            "to": {},
                            "id": [],
                        }
                    dep_str = dep_proc[gpu_pid]
                    to_id = len(dep_str["from"])
                    dep_str["from"].append((from_us, disp_tid, disp_tid))
                    dep_str["to"][to_id] = to_us
                    ##

    inp.close()


#############################################################


# Comparator to sort a dictionary of tuples. This comparator will convert
# the second element of tuple to an int and return the new tuple. Then
# the dictionary can use the default comparison i.e sort by first element,
# then sort by second element.
def tuple_comparator(tupleElem):
    return tupleElem[0], int(tupleElem[1])


# merge results table
def merge_table():
    global var_list
    keys = sorted(var_table.keys(), key=tuple_comparator)

    fields = set(var_table[keys[0]])
    if "DispatchNs" in fields:
        var_list.append("DispatchNs")
        var_list.append("BeginNs")
        var_list.append("EndNs")
        var_list.append("CompleteNs")
    var_list = [x for x in var_list if x in fields]


#############################################################


# dump CSV results
def dump_csv(file_name):
    global var_list
    keys = sorted(var_table.keys(), key=tuple_comparator)

    with open(file_name, mode="w") as fd:
        fd.write(",".join(var_list) + "\n")
        for pid, ind in keys:
            entry = var_table[(pid, ind)]
            dispatch_number = entry["Index"]
            if ind != dispatch_number:
                fatal("Dispatch #" + ind + " index mismatch (" + dispatch_number + ")\n")
            val_list = [entry[var] if (var in entry) else 'None' for var in var_list]
            fd.write(",".join(val_list) + "\n")

    print("File '" + file_name + "' is generating")


#############################################################


# fill kernels DB
def fill_kernel_db(table_name, db):
    global var_list
    keys = sorted(var_table.keys(), key=tuple_comparator)

    for var in set(var_list).difference(set(table_descr[1])):
        table_descr[1][var] = "INTEGER"
    table_descr[0] = var_list

    table_handle = db.add_table(table_name, table_descr)

    for pid, ind in keys:
        entry = var_table[(pid, ind)]
        dispatch_number = entry["Index"]
        if ind != dispatch_number:
            fatal("Dispatch #" + ind + " index mismatch (" + dispatch_number + ")\n")
        val_list = [entry[var] for var in var_list]
        db.insert_entry(table_handle, val_list)


#############################################################

# Fill Ext DB
ext_table_descr = [
    ["BeginNs", "EndNs", "pid", "tid", "Name", "Index", "__section", "__lane"],
    {
        "BeginNs": "INTEGER",
        "EndNs": "INTEGER",
        "pid": "INTEGER",
        "tid": "INTEGER",
        "Name": "TEXT",
        "Index": "INTEGER",
        "__section": "INTEGER",
        "__lane": "INTEGER",
    },
]


def fill_ext_db(table_name, db, indir, trace_name, api_pid):
    global range_data

    file_name = indir + "/" + trace_name + "_trace.txt"
    # tms pid:tid cid:rid:'.....'
    ptrn_val = re.compile(r'(\d+) (\d+):(\d+) (\d+):(\d+):"(.*)"$')

    range_data = {}
    range_stack = {}
    range_map = {}

    if not os.path.isfile(file_name):
        return 0

    record_id = 0
    table_handle = db.add_table(table_name, ext_table_descr)
    with open(file_name, mode="r", errors="replace") as fd:
        for line in fd.readlines():
            record = line[:-1]
            m = ptrn_val.match(record)
            if m:
                tms = int(m.group(1))
                pid = m.group(2)
                tid = int(m.group(3))
                cid = int(m.group(4))
                rid = int(m.group(5))
                msg = m.group(6)

                rec_vals = []
                if not tid in range_data:
                    range_data[tid] = {}

                if cid != 2:
                    rec_vals.append(tms)
                    rec_vals.append(tms + 1)
                    rec_vals.append(pid)
                    rec_vals.append(tid)
                    rec_vals.append(msg)
                    rec_vals.append(record_id)
                    rec_vals.append(api_pid)    # __section
                    rec_vals.append(tid)        # __lane

                if cid == 1:
                    if not pid in range_stack:
                        range_stack[pid] = {}
                    pid_stack = range_stack[pid]
                    if not tid in pid_stack:
                        pid_stack[tid] = []
                    rec_stack = pid_stack[tid]
                    rec_stack.append(rec_vals)
                    continue

                if cid == 2:
                    if not pid in range_stack:
                        fatal("ROCTX range begin not found, pid(" + pid + ")")
                    pid_stack = range_stack[pid]
                    if not tid in pid_stack:
                        fatal("ROCTX range begin not found, tid(" + tid + ")")
                    rec_stack = pid_stack[tid]
                    rec_vals = rec_stack.pop()
                    rec_vals[1] = tms
                    # record the range's start/stop timestamps, its parent (ranges can be nested), and its message.
                    range_start = rec_vals[0]
                    range_stop = tms
                    range_parent = rec_stack[-1][0] if len(rec_stack) != 0 else 0
                    range_msg = rec_vals[4]
                    range_data[tid][range_start] = (range_stop, range_parent, range_msg)

                # range start
                if cid == 3:
                    range_map[rid] = (tms, msg)
                    continue

                # range stop
                if cid == 4:
                    if rid in range_map:
                        # querying start timestamp if rid exists
                        (tms, msg) = range_map[rid]
                        del range_map[rid]
                    else:
                        fatal("range id(" + str(rid) + ") is not found")
                    rec_vals[0] = tms   # begin timestamp
                    rec_vals[4] = msg   # range message
                    rec_vals[7] = 0     # 0 lane for ranges

                db.insert_entry(table_handle, rec_vals)
                record_id += 1

    return 1


#############################################################
# arguments manipulation routines
def get_field(args, field):
    if args == None:
        return (None, 0)
    ptrn1_field = re.compile(r"^.* " + field + r"\(")
    ptrn2_field = re.compile(r"\) .*$")
    ptrn3_field = re.compile(r"\)\)$")
    (field_name, n) = ptrn1_field.subn("", args, count=1)
    if n != 0:
        (field_name, n) = ptrn2_field.subn("", field_name, count=1)
        if n == 0:
            (field_name, n) = ptrn3_field.subn("", field_name, count=1)
    return (field_name, n)


def set_field(args, field, val):
    return re.subn(
        field + r"\(\w+\)([ \)])", field + "(" + str(val) + ")\\1", args, count=1
    )


hsa_patch_data = {}
ops_patch_data = {}

# Fill API DB
api_table_descr = [
    [
        "BeginNs",
        "EndNs",
        "pid",
        "tid",
        "Name",
        "args",
        "Index",
        "Data",
        "__section",
        "__lane",
    ],
    {
        "BeginNs": "INTEGER",
        "EndNs": "INTEGER",
        "pid": "INTEGER",
        "tid": "INTEGER",
        "Name": "TEXT",
        "args": "TEXT",
        "Index": "INTEGER",
        "Data": "TEXT",
        "__section": "INTEGER",
        "__lane": "INTEGER",
    },
]


# Filling API records DB table
# table_name - created DB table name
# db - DB handle
# indir - input directory
# api_name - traced API name
# api_pid - assigned JSON PID
# dep_pid - PID of dependet domain
# dep_list - list of dependet dospatch events
# dep_filtr - registered dependencies by record ID
def fill_api_db(
    table_name, db, indir, api_name, api_pid, dep_pid, dep_list, dep_filtr, expl_id
):
    global hsa_activity_found
    global memory_manager

    range_start_times = {}
    copy_csv = ""

    # Matches normal API records.
    ptrn_api_record = re.compile(r"(\d+):(\d+) (\d+):(\d+) ([^\(]+)(\(.*)$")
    # Matches records with a function name of "unknown" and no parameters.
    # Capture groups 1-4 should match the same information as in ptrn_api_record.
    # Used to avoid modifying ptrn_api_record regex.
    ptrn_api_record_unknown = re.compile(r"(\d+):(\d+) (\d+):(\d+) (unknown).*$")

    hip_mcopy_ptrn = re.compile(r"hipMemcpy|hipMemset")
    hip_wait_event_ptrn = re.compile(r"WaitEvent")
    hip_sync_event_ptrn = re.compile(r"hipStreamSynchronize")
    hip_sync_dev_event_ptrn = re.compile(r"hipDeviceSynchronize")
    hip_graph_ptrn = re.compile(r"hipGraphLaunch")
    wait_event_ptrn = re.compile(r"WaitEvent|hipStreamSynchronize|hipDeviceSynchronize")
    hip_stream_wait_write_ptrn = re.compile(
        r"hipStreamWaitValue64|hipStreamWriteValue64|hipStreamWaitValue32|hipStreamWriteValue32"
    )
    prop_pattern = re.compile(r"([\w-]+)\((\w+)\)")
    beg_pattern = re.compile(r'^dispatch\[(\d*)\], (.*) kernel-name\("([^"]*)"\)')
    hip_strm_cr_event_ptrn = re.compile(r"hipStreamCreate")
    hsa_mcopy_ptrn = re.compile(r"hsa_amd_memory_async_copy")
    ptrn_fixformat = re.compile(r"(\d+:\d+ \d+:\d+ \w+)\(\s*(.*)\)$")
    ptrn_fixkernel = re.compile(r"\s+kernel=(.*)$")
    ptrn_multi_kernel = re.compile(r"(.*):(\d+)$")
    ptrn_corr_id = re.compile(r"\ :(\d*)$")

    file_name = indir + "/" + api_name + "_api_trace.txt"
    if not os.path.isfile(file_name):
        return 0

    hsa_copy_file_name = indir + "/" + "async_copy_trace.txt"
    hsa_copy_file_name_present = 1 if os.path.isfile(file_name) else 0
    hsa_copy_deps = 1 if (api_pid == HSA_PID and hsa_copy_file_name_present == 1) else 0
    print("hsa_copy_deps: " + str(hsa_copy_deps))

    # parsing an input trace file and creating a DB table
    record_id_dict = {}
    table_handle = db.add_table(table_name, api_table_descr)
    with open(file_name, mode="r", errors="replace") as fd:
        file_lines = fd.readlines()
        total_lines = len(file_lines)
        line_index = 0
        for line in file_lines:
            if (line_index == total_lines - 1) or (line_index % 100 == 0):
                sys.stdout.write(
                    "\rscan "
                    + api_name
                    + " API data "
                    + str(line_index)
                    + ":"
                    + str(total_lines)
                    + " " * 100
                )
            line_index += 1

            record = line[:-1]

            corr_id = 0
            m = ptrn_corr_id.search(record)
            if m:
                corr_id = int(m.group(1))
                record = ptrn_corr_id.sub("", record)

            kernel_arg = ""
            m = ptrn_fixkernel.search(record)
            if m:
                kernel_arg = "kernel(" + m.group(1) + ") "
                record = ptrn_fixkernel.sub("", record)

            mfixformat = ptrn_fixformat.match(record)
            if mfixformat:  # replace '=' in args with parentheses
                reformated_args = (
                    kernel_arg
                    + mfixformat.group(2)
                    .replace("=", "(")
                    .replace(",", ")")
                    .replace("\\", "\\\\")
                    .replace('"', '\\"')
                    + ")"
                )
                record = mfixformat.group(1) + "( " + reformated_args + ")"

            m = ptrn_api_record.match(record)
            if not m:
                m = ptrn_api_record_unknown.match(record)
                if not m:
                    fatal(api_name + " bad record: '" + record + "'")

            rec_vals = []
            rec_len = len(api_table_descr[0]) - 3
            for ind in range(1, rec_len):
                try:
                    rec_vals.append(m.group(ind))
                except IndexError:
                    rec_vals.append(None)
            proc_id = int(rec_vals[2])
            thread_id = int(rec_vals[3])
            record_name = rec_vals[4]
            # record_args is optional, and may be None if an unknown record is found.
            record_args = rec_vals[5]

            # incrementing per-process record id/correlation id
            if not proc_id in record_id_dict:
                record_id_dict[proc_id] = 0
            record_id_dict[proc_id] += 1
            record_id = record_id_dict[proc_id]

            # setting correlationid to record id if correlation id is not defined
            if corr_id == 0:
                corr_id = record_id

            rec_vals.append(corr_id)
            # extracting/converting stream id
            (stream_id, stream_found) = get_field(record_args, "stream")
            if stream_found:
                stream_id = get_stream_index(stream_id)
                (rec_vals[5], found) = set_field(record_args, "stream", stream_id)
                if found == 0:
                    fatal(
                        'set_field() failed for "stream", args: "' + record_args + '"'
                    )
            else:
                (stream_id, stream_found) = get_field(record_args, "hStream")
                if stream_found:
                    stream_id = get_stream_index(stream_id)
                    (rec_vals[5], found) = set_field(
                        record_args, "hStream", stream_id
                    )
                    if found == 0:
                        fatal(
                            'set_field() failed for "stream", args: "'
                            + record_args
                            + '"'
                        )
                else:
                    stream_id = 0

            if hip_strm_cr_event_ptrn.match(record_name):
                hip_streams.append(stream_id)

            if hip_sync_event_ptrn.match(record_name):
                if (proc_id, stream_id) in last_hip_api_map:
                    (last_hip_api_corr_id, last_hip_api_from_pid) = last_hip_api_map[
                        (proc_id, stream_id)
                    ][-1]
                    sync_api_beg_us = int((int(rec_vals[0]) - START_NS) / 1000)
                    if not proc_id in dep_dict:
                        dep_dict[proc_id] = {}
                    if HIP_PID not in dep_dict[proc_id]:
                        dep_dict[proc_id][HIP_PID] = {
                            "pid": last_hip_api_from_pid,
                            "from": [],
                            "to": {},
                            "id": [],
                        }
                    dep_dict[proc_id][HIP_PID]["from"].append(
                        (-1, stream_id, thread_id)
                    )
                    dep_dict[proc_id][HIP_PID]["id"].append(last_hip_api_corr_id)
                    dep_dict[proc_id][HIP_PID]["to"][
                        last_hip_api_corr_id
                    ] = sync_api_beg_us
                    from_ids[(last_hip_api_corr_id, proc_id)] = (
                        len(dep_dict[proc_id][HIP_PID]["from"]) - 1
                    )

            m = beg_pattern.match(record)
            gpu_id = 0
            if m:
                kernel_properties = m.group(2)
                for prop in kernel_properties.split(", "):
                    m = prop_pattern.match(prop)
                    if m:
                        val = m.group(2)
                        var = m.group(1)
                        if var == "gpu-id":
                            gpu_id = int(val)

            if hsa_mcopy_ptrn.match(record_name) or hip_mcopy_ptrn.match(record_name):
                ops_section_id = COPY_PID
            else:
                ops_section_id = GPU_BASE_PID + int(gpu_id)

            if (proc_id, stream_id) not in last_hip_api_map:
                last_hip_api_map[(proc_id, stream_id)] = []
            last_hip_api_map[(proc_id, stream_id)].append((corr_id, ops_section_id))

            # asyncronous opeartion API found
            op_found = 0
            mcopy_found = 0

            # extract kernel name string
            (kernel_str, kernel_found) = get_field(record_args, "kernel")
            if kernel_found == 0:
                kernel_str = ""
            else:
                op_found = 1

            if hip_mcopy_ptrn.match(record_name):
                mcopy_found = 1
                op_found = 1

            # HIP Graph API
            if hip_graph_ptrn.search(record_name):
                op_found = 1

            # HIP WaitEvent API
            if wait_event_ptrn.search(record_name):
                op_found = 1

            if hip_stream_wait_write_ptrn.search(record_name):
                op_found = 1

            # HSA memcopy API
            if hsa_mcopy_ptrn.match(record_name):
                mcopy_found = 1
                op_found = 1

                stream_id = thread_id
                hsa_patch_data[(corr_id, proc_id)] = thread_id

            if op_found:
                roctx_msg = ""

                if not thread_id in range_start_times:
                    range_start_times[thread_id] = (
                        sorted(range_data[thread_id].keys())
                        if thread_id in range_data
                        else []
                    )
                start_times = range_start_times[thread_id]

                index = bisect.bisect_right(start_times, int(rec_vals[0]))
                if index > 0:
                    # We found the range that is closest to this operation. Iterate the
                    # range stack this range is part of until we find a range that entirely
                    # contains the operation.
                    range_start = start_times[index - 1]
                    while range_start != 0:
                        (range_end, range_start, msg) = range_data[thread_id][
                            range_start
                        ]
                        if int(rec_vals[1]) < range_end:
                            # This range contains the operation.
                            roctx_msg = msg
                            break

                ops_patch_data[(corr_id, proc_id)] = (
                    thread_id,
                    stream_id,
                    kernel_str,
                    roctx_msg,
                )

            if op_found:
                op_found = 0
                beg_ns = int(rec_vals[0])
                end_ns = int(rec_vals[1])
                dur_us = int((end_ns - beg_ns) / 1000)
                from_us = int((beg_ns - START_NS) / 1000) + dur_us / 2
                if api_pid == HIP_PID or hsa_copy_deps == 1:
                    if not proc_id in dep_dict:
                        dep_dict[proc_id] = {}
                    dep_proc = dep_dict[proc_id]
                    if not dep_pid in dep_proc:
                        if api_pid == "HIP_PID":
                            dep_proc[dep_pid] = {"pid": api_pid, "from": [], "id": []}
                        else:
                            dep_proc[dep_pid] = {
                                "pid": api_pid,
                                "from": [],
                                "id": [],
                                "to": {},
                            }
                    dep_str = dep_proc[dep_pid]
                    dep_str["from"].append((from_us, stream_id, thread_id))
                    if expl_id:
                        dep_str["id"].append(corr_id)

            # memcopy registering
            api_data = (
                memory_manager.register_api(rec_vals) if mcopy_data_enabled else ""
            )
            rec_vals.append(api_data)

            # setting section and lane
            rec_vals.append(api_pid)    # __section
            rec_vals.append(thread_id)  # __lane

            # inserting an API record to DB
            db.insert_entry(table_handle, rec_vals)

    # inserting of dispatch events correlated to the dependent dispatches
    for from_ns, proc_id, thread_id in dep_list:
        if not proc_id in record_id_dict:
            record_id_dict[proc_id] = 0
        record_id_dict[proc_id] += 1
        corr_id = record_id_dict[proc_id]
        db.insert_entry(
            table_handle,
            [
                from_ns,
                from_ns,
                proc_id,
                thread_id,
                "hsa_dispatch",
                "",
                corr_id,
                "",
                api_pid,
                thread_id,
            ],
        )

    # generating memcopy CSV
    if copy_csv != "":
        file_name = os.environ["PWD"] + "/results_mcopy.csv"
        with open(file_name, mode="w") as fd:
            print("File '" + file_name + "' is generating")
            fd.write(copy_csv)

    return 1


#############################################################

# fill COPY DB
copy_table_descr = [
    ["BeginNs", "EndNs", "Name", "pid", "tid", "Index", "Data", "__section", "__lane"],
    {
        "Index": "INTEGER",
        "Name": "TEXT",
        "args": "TEXT",
        "BeginNs": "INTEGER",
        "EndNs": "INTEGER",
        "pid": "INTEGER",
        "tid": "INTEGER",
        "Data": "TEXT",
        "__section": "INTEGER",
        "__lane": "INTEGER",
    },
]


def fill_copy_db(table_name, db, indir):
    sect_id = COPY_PID
    file_name = indir + "/" + "async_copy_trace.txt"
    ptrn_val = re.compile(r"^(\d+):(\d+) (async-copy):(\d+):(\d+)$")

    if not os.path.isfile(file_name):
        return 0

    table_handle = db.add_table(table_name, copy_table_descr)
    with open(file_name, mode="r", errors="replace") as fd:
        for line in fd.readlines():
            record = line[:-1]
            m = ptrn_val.match(record)
            if not m:
                fatal("bad async-copy entry '" + record + "'")
            else:
                rec_vals = []
                for ind in range(1, 4):
                    rec_vals.append(m.group(ind))
                corr_id = int(m.group(4))
                proc_id = int(m.group(5))

                # querying tid value
                if (corr_id, proc_id) in hsa_patch_data:
                    thread_id = hsa_patch_data[(corr_id, proc_id)]
                else:
                    thread_id = -1

                # completing record
                rec_vals.append(proc_id)    # tid
                rec_vals.append(thread_id)  # tid
                rec_vals.append(corr_id)    # Index

                # registering memcopy information
                activity_data = (
                    memory_manager.register_copy(rec_vals) if mcopy_data_enabled else ""
                )
                rec_vals.append(activity_data)

                # appending straem ID and section ID
                rec_vals.append(COPY_PID)   # __section
                rec_vals.append(thread_id)  # __lane

                # inserting DB activity entry
                db.insert_entry(table_handle, rec_vals)

                # filling dependencies
                to_ns = int(rec_vals[0])
                to_us = int((to_ns - START_NS) / 1000)

                if thread_id != -1:
                    # if not proc_id in dep_dict: dep_dict[proc_id] = {}
                    dep_proc = dep_dict[proc_id]
                    # if not pid in dep_proc: dep_proc[pid] = { 'pid': HSA_PID, 'from': [], 'to': {}, 'id': [] }
                    dep_str = dep_proc[sect_id]
                    dep_str["to"][corr_id] = to_us
                    dep_str["id"].append(corr_id)

    return 1


#############################################################

# fill HCC ops DB
ops_table_descr = [
    [
        "BeginNs",
        "EndNs",
        "dev-id",
        "queue-id",
        "Name",
        "pid",
        "tid",
        "roctx-range",
        "stream-id",
        "Index",
        "Data",
        "__section",
        "__lane",
    ],
    {
        "Index": "INTEGER",
        "Name": "TEXT",
        "args": "TEXT",
        "BeginNs": "INTEGER",
        "EndNs": "INTEGER",
        "dev-id": "INTEGER",
        "queue-id": "INTEGER",
        "pid": "INTEGER",
        "tid": "INTEGER",
        "roctx-range": "TEXT",
        "Data": "TEXT",
        "stream-id": "INTEGER",
        "__section": "INTEGER",
        "__lane": "INTEGER",
    },
]


def fill_ops_db(kernel_table_name, mcopy_table_name, db, indir):
    global max_gpu_id
    file_name = indir + "/" + "hcc_ops_trace.txt"
    ptrn_val = re.compile(r"(\d+):(\d+) (\d+):(\d+) (.*)$")
    ptrn_id = re.compile(r"^([^:]+):(\d+):(\d+)$")
    ptrn_mcopy = re.compile(r"(Memcpy|Copy|Fill)")
    ptrn_barrier = re.compile(r"Marker")

    if not os.path.isfile(file_name):
        return {}

    filtr = {}

    kernel_table_handle = db.add_table(kernel_table_name, ops_table_descr)
    mcopy_table_handle = db.add_table(mcopy_table_name, ops_table_descr)
    with open(file_name, mode="r", errors="replace") as fd:
        file_lines = fd.readlines()
        total_lines = len(file_lines)
        line_index = 0
        for line in file_lines:
            if (line_index == total_lines - 1) or (line_index % 100 == 0):
                sys.stdout.write(
                    "\rscan ops data "
                    + str(line_index)
                    + ":"
                    + str(total_lines)
                    + " " * 100
                )
            line_index += 1

            record = line[:-1]
            m = ptrn_val.match(record)
            if m:
                # parsing trace record
                rec_vals = []
                for ind in range(1, 6):
                    rec_vals.append(m.group(ind))
                label = rec_vals[4]  # record name
                m = ptrn_id.match(label)
                if not m:
                    fatal("bad hcc ops entry '" + record + "'")
                name = m.group(1)
                corr_id = int(m.group(2))
                proc_id = int(m.group(3))

                # checking name for memcopy pattern
                is_barrier = 0
                if ptrn_mcopy.search(name):
                    rec_table_name = mcopy_table_name
                    table_handle = mcopy_table_handle
                    sect_id = COPY_PID
                else:
                    rec_table_name = kernel_table_name
                    table_handle = kernel_table_handle

                    gpu_id = int(rec_vals[2])
                    if gpu_id > max_gpu_id:
                        max_gpu_id = gpu_id

                    if ptrn_barrier.search(name):
                        name = '"<barrier packet>"'
                        is_barrier = 1
                        sect_id = GPU_BASE_PID + int(gpu_id) + 512
                    else:
                        sect_id = GPU_BASE_PID + int(gpu_id)

                thread_id = 0
                stream_id = 0
                roctx_range = ""
                if (corr_id, proc_id) in ops_patch_data:
                    (thread_id, stream_id, name_patch, roctx_range) = ops_patch_data[
                        (corr_id, proc_id)
                    ]
                    if name_patch != "":
                        name = name_patch
                    if roctx_range == "":
                        roctx_range = name
                else:
                    if is_barrier:
                        continue
                    else:
                        if "ROCP_CTRL_RATE" in os.environ:
                            continue
                        else:
                            fatal(
                                "hcc ops data not found: '"
                                + record
                                + "', "
                                + str(corr_id)
                                + ", "
                                + str(proc_id)
                            )

                # activity record
                rec_vals[4] = name              # Name
                rec_vals.append(proc_id)        # pid
                rec_vals.append(thread_id)      # tid
                rec_vals.append(roctx_range)    # roctx-range
                rec_vals.append(stream_id)      # StreamId
                rec_vals.append(corr_id)        # Index

                # registering memcopy information
                activity_data = (
                    memory_manager.register_activity(rec_vals)
                    if mcopy_data_enabled
                    else ""
                )
                rec_vals.append(activity_data)

                # activity record data for stream ID and sction ID
                rec_vals.append(sect_id)    # __section
                rec_vals.append(stream_id)  # __lane

                # inserting DB activity entry
                db.insert_entry(table_handle, rec_vals)

                # registering a dependency filtr
                filtr[(corr_id, proc_id)] = rec_table_name

                # filling a dependencies
                to_ns = int(rec_vals[0])
                to_us = int((to_ns - START_NS) / 1000)

                end_ns = int(rec_vals[1])
                dur_us = int((end_ns - to_ns) / 1000)

                if (corr_id, proc_id) in from_ids:
                    depid = from_ids[(corr_id, proc_id)]
                    from_val = dep_dict[proc_id][HIP_PID]["from"][depid]
                    print("from_val" + str(from_val))
                    from_val_new = (to_us + dur_us, from_val[1], from_val[2])
                    dep_dict[proc_id][HIP_PID]["from"][depid] = from_val_new

                if not proc_id in dep_dict:
                    dep_dict[proc_id] = {}
                dep_proc = dep_dict[proc_id]
                if not sect_id in dep_proc:
                    dep_proc[sect_id] = {"bsp": OPS_PID, "to": {}}
                dep_str = dep_proc[sect_id]
                dep_str["to"][corr_id] = to_us

            else:
                fatal("hcc ops bad record: '" + record + "'")

    return filtr


#############################################################
# main
if len(sys.argv) < 2:
    fatal("Usage: " + sys.argv[0] + " <output CSV file> <input result files list>")

outfile = sys.argv[1]
infiles = sys.argv[2:]
indir = re.sub(r"\/[^\/]*$", r"", infiles[0])
inext = re.sub(r"\s+$", r"", infiles[0])
inext = re.sub(r"^.*(\.[^\.]+)$", r"\1", inext)

dbfile = ""
csvfile = ""

if "ROCP_JSON_REBASE" in os.environ and os.environ["ROCP_JSON_REBASE"] == 0:
    begin_ts_file = indir + "/begin_ts_file.txt"
    if os.path.isfile(begin_ts_file):
        with open(begin_ts_file, mode="r", errors="replace") as fd:
            ind = 0
            for line in fd.readlines():
                val = int(line)
                if ind == 0 or val < START_NS:
                    START_NS = val
                ind += 1
        print("START timestamp found (" + str(START_NS) + "ns)")

if re.search(r"\.csv$", outfile):
    csvfile = outfile
elif re.search(r"\.db$", outfile):
    dbfile = outfile
    csvfile = re.sub(r"\.db$", ".csv", outfile)
else:
    fatal("Bad output file '" + outfile + "'")

if inext == ".txt":
    for f in infiles:
        parse_res(f)
    if len(var_table) != 0:
        merge_table()

if dbfile == "":
    dump_csv(csvfile)
else:
    statfile = re.sub(r"\.csv$", ".stats.csv", csvfile)
    jsonfile = re.sub(r"\.csv$", ".json", csvfile)

    hsa_statfile = re.sub(r"\.stats\.csv$", r".hsa_stats.csv", statfile)
    hip_statfile = re.sub(r"\.stats\.csv$", r".hip_stats.csv", statfile)
    ops_statfile = statfile
    copy_statfile = re.sub(r"\.stats\.csv$", r".copy_stats.csv", statfile)
    memcopy_info_file = re.sub(r"\.stats\.csv$", r".memcopy_info.csv", statfile)
    sysinfo_file = re.sub(r"\.stats\.csv$", r".sysinfo.txt", statfile)
    metadata_gen(sysinfo_file, "rocminfo")

    with open(dbfile, mode="w") as fd:
        fd.truncate()
    db = SQLiteDB(dbfile)
    memory_manager = MemManager(db, indir)

    ext_trace_found = fill_ext_db("rocTX", db, indir, "roctx", EXT_PID)

    hsa_trace_found = fill_api_db(
        "HSA", db, indir, "hsa", HSA_PID, COPY_PID, kern_dep_list, {}, 0
    )
    hsa_activity_found = fill_copy_db("COPY", db, indir)

    hip_trace_found = fill_api_db("HIP", db, indir, "hip", HIP_PID, OPS_PID, [], {}, 1)
    ops_filtr = fill_ops_db("OPS", "COPY", db, indir)

    fill_kernel_db("KERN", db)

    any_trace_found = ext_trace_found | hsa_trace_found | hip_trace_found
    copy_trace_found = 0
    if hsa_activity_found or len(ops_filtr):
        copy_trace_found = 1

    db.open_json(jsonfile)

    if ext_trace_found:
        db.label_json(EXT_PID, "Markers and Ranges", jsonfile)

    if hip_trace_found:
        db.label_json(HIP_PID, "CPU HIP API", jsonfile)

    if hsa_trace_found:
        db.label_json(HSA_PID, "CPU HSA API", jsonfile)

    db.label_json(COPY_PID, "COPY", jsonfile)

    if max_gpu_id >= 0:
        for ind in range(0, int(max_gpu_id) + 1):
            db.label_json(int(ind) + int(GPU_BASE_PID), "GPU" + str(ind), jsonfile)
            db.label_json(int(ind) + int(GPU_BASE_PID) + 512 , "GPU Barriers" + str(ind), jsonfile)

    if ext_trace_found:
        dform.gen_ext_json_trace(db, "rocTX", START_NS, jsonfile)

    if len(var_table) != 0:
        dform.post_process_data(db, "KERN", csvfile)
        dform.gen_table_bins(db, "KERN", statfile, "KernelName", "DurationNs")
        if "BeginNs" in var_list:
            dform.gen_kernel_json_trace(db, "KERN", GPU_BASE_PID, START_NS, jsonfile)

    if hsa_trace_found:
        dform.post_process_data(db, "HSA")
        dform.gen_table_bins(db, "HSA", hsa_statfile, "Name", "DurationNs")
        dform.gen_api_json_trace(db, "HSA", START_NS, jsonfile)

    if copy_trace_found:
        dform.post_process_data(db, "COPY")
        dform.gen_table_bins(db, "COPY", copy_statfile, "Name", "DurationNs")
        dform.gen_api_json_trace(db, "COPY", START_NS, jsonfile)

    if hip_trace_found:
        dform.post_process_data(db, "HIP")
        dform.gen_table_bins(db, "HIP", hip_statfile, "Name", "DurationNs")
        dform.gen_api_json_trace(db, "HIP", START_NS, jsonfile)

    if ops_filtr:
        dform.post_process_data(db, "OPS")
        dform.gen_table_bins(db, "OPS", ops_statfile, "Name", "DurationNs")
        dform.gen_ops_json_trace(db, "OPS", GPU_BASE_PID, START_NS, jsonfile)

    if any_trace_found:
        dep_id = 0
        for proc_id, dep_proc in dep_dict.items():
            for to_pid, dep_str in dep_proc.items():
                if "bsp" in dep_str:
                    bspid = dep_str["bsp"]
                    base_str = dep_proc[bspid]
                    for v in ("pid", "from", "id"):
                        dep_str[v] = base_str[v]
                    base_str["inv"] = 1

            for to_pid, dep_str in dep_proc.items():
                if "inv" in dep_str:
                    continue
                if not "to" in dep_str:
                    continue

                from_pid = dep_str["pid"]
                from_us_list = dep_str["from"]
                to_us_dict = dep_str["to"]
                corr_id_list = dep_str["id"]

                db.flow_json(
                    dep_id,
                    from_pid,
                    from_us_list,
                    to_pid,
                    to_us_dict,
                    corr_id_list,
                    jsonfile,
                )
                dep_id += len(from_us_list)

    db.metadata_json(jsonfile, sysinfo_file)
    db.close_json(jsonfile)

    if mcopy_data_enabled:
        memory_manager.dump_data("MM", memcopy_info_file)

    db.close()

sys.exit(0)
#############################################################
