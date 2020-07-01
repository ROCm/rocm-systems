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

import os, sys, re, subprocess
from sqlitedb import SQLiteDB
from mem_manager import MemManager
import dform

EXT_PID = 0
COPY_PID = 1
HIP_PID = 2
HSA_PID = 3
KFD_PID = 4
OPS_PID = 5
GPU_BASE_PID = 6
NONE_PID = -1

max_gpu_id = -1
START_US = 0

hsa_activity_found = 0

# dependencies dictionary
dep_dict = {}
kern_dep_list = []

# stream ID map
stream_counter = 0
stream_id_map = {}
def get_stream_index(stream_id):
  global stream_counter
  stream_ind = 0
  if stream_id.lower() != 'nil':
    if not stream_id in stream_id_map:
      stream_counter += 1
      stream_ind = stream_counter
      stream_id_map[stream_id] = stream_ind
    else:
      stream_ind = stream_id_map[stream_id]
  return stream_ind

# patching activity records
def activity_record_patching(db, ops_table_name, kernel_found, kernel_name, stream_found, stream_ind, select_expr):
  if kernel_found != 0:
    db.change_rec_fld(ops_table_name, 'Name = "' + kernel_name + '"', select_expr)
  if stream_found != 0:
    db.change_rec_fld(ops_table_name, 'tid = ' + str(stream_ind), select_expr)

# global vars
table_descr = [
  ['Index', 'KernelName'],
  {'Index': 'INTEGER', 'KernelName': 'TEXT'}
]
var_list = table_descr[0]
var_table = {}
#############################################################

def fatal(msg):
  sys.stderr.write(sys.argv[0] + ": " + msg + "\n");
  sys.exit(1)

dbglog_count = 0
def dbglog(msg):
  global dbglog_count
  dbglog_count += 1
  sys.stderr.write(sys.argv[0] + ": " + msg + "\n");
  fatal("error")
#############################################################

# Dumping sysinfo
sysinfo_begin = 1
def metadata_gen(sysinfo_file, sysinfo_cmd):
  global sysinfo_begin
  if not re.search(r'\.txt$', sysinfo_file):
    raise Exception('wrong output file type: "' + sysinfo_file + '"' )
  if sysinfo_begin == 1:
    sysinfo_begin = 0
    with open(sysinfo_file, mode='w') as fd: fd.write('')
  with open(sysinfo_file, mode='a') as fd: fd.write('CMD: ' + sysinfo_cmd + '\n')
  status = subprocess.call(sysinfo_cmd + ' >> ' + sysinfo_file,
                           stderr=subprocess.STDOUT,
                           shell=True)
  if status != 0:
    raise Exception('Could not run command: "' + sysinfo_cmd + '"')

# parse results method
def parse_res(infile):
  global max_gpu_id
  if not os.path.isfile(infile): return
  inp = open(infile, 'r')

  beg_pattern = re.compile("^dispatch\[(\d*)\], (.*) kernel-name\(\"([^\"]*)\"\)")
  prop_pattern = re.compile("([\w-]+)\((\w+)\)");
  ts_pattern = re.compile(", time\((\d*),(\d*),(\d*),(\d*)\)")
  var_pattern = re.compile("^\s*([^\s]*)\s+\((\d*)\)")

  dispatch_number = 0
  for line in inp.readlines():
    record = line[:-1]

    m = var_pattern.match(record)
    if m:
      if not dispatch_number in var_table: fatal("Error: dispatch number not found '" + str(dispatch_number) + "'")
      var = m.group(1)
      val = m.group(2)
      var_table[dispatch_number][var] = val
      if not var in var_list: var_list.append(var)

    m = beg_pattern.match(record)
    if m:
      dispatch_number = m.group(1)
      if not dispatch_number in var_table:
        var_table[dispatch_number] = {
          'Index': dispatch_number,
          'KernelName': "\"" + m.group(3) + "\""
        }

        gpu_id = 0
        queue_id = 0
        disp_pid = 0
        disp_tid = 0

        kernel_properties = m.group(2)
        for prop in kernel_properties.split(', '):
          m = prop_pattern.match(prop)
          if m:
            var = m.group(1)
            val = m.group(2)
            var_table[dispatch_number][var] = val
            if not var in var_list: var_list.append(var);
            if var == 'gpu-id':
              gpu_id = int(val)
              if (gpu_id > max_gpu_id): max_gpu_id = gpu_id
            if var == 'queue-id': queue_id = int(val)
            if var == 'pid': disp_pid = int(val)
            if var == 'tid': disp_tid = int(val)
          else: fatal('wrong kernel property "' + prop + '" in "'+ kernel_properties + '"')
        m = ts_pattern.search(record)
        if m:
          var_table[dispatch_number]['DispatchNs'] = m.group(1)
          var_table[dispatch_number]['BeginNs'] = m.group(2)
          var_table[dispatch_number]['EndNs'] = m.group(3)
          var_table[dispatch_number]['CompleteNs'] = m.group(4)

          ## filling dependenciws
          from_ns = m.group(1)
          from_us = int(from_ns) / 1000
          to_us = int(m.group(2)) / 1000

          kern_dep_list.append((from_ns, disp_pid, disp_tid))

          gpu_pid = GPU_BASE_PID + int(gpu_id)
          if not disp_pid in dep_dict: dep_dict[disp_pid] = {}
          dep_proc = dep_dict[disp_pid]
          if not gpu_pid in dep_proc: dep_proc[gpu_pid] = { 'pid': HSA_PID, 'from': [], 'to': {}, 'id': [] }
          dep_str = dep_proc[gpu_pid]
          to_id = len(dep_str['from'])
          dep_str['from'].append((from_us, disp_tid, queue_id))
          dep_str['to'][to_id] = to_us
          ##

  inp.close()
#############################################################

# merge results table
def merge_table():
  global var_list
  keys = sorted(var_table.keys(), key=int)

  fields = set(var_table[keys[0]])
  if 'DispatchNs' in fields:
    var_list.append('DispatchNs')
    var_list.append('BeginNs')
    var_list.append('EndNs')
    var_list.append('CompleteNs')
  var_list = [x for x in var_list if x in fields]
#############################################################

# dump CSV results
def dump_csv(file_name):
  global var_list
  keys = sorted(var_table.keys(), key=int)

  with open(file_name, mode='w') as fd:
    fd.write(','.join(var_list) + '\n');
    for ind in keys:
      entry = var_table[ind]
      dispatch_number = entry['Index']
      if ind != dispatch_number: fatal("Dispatch #" + ind + " index mismatch (" + dispatch_number + ")\n")
      val_list = [entry[var] for var in var_list]
      fd.write(','.join(val_list) + '\n');

  print("File '" + file_name + "' is generating")
#############################################################

# fill kernels DB
def fill_kernel_db(table_name, db):
  global var_list
  keys = sorted(var_table.keys(), key=int)

  for var in set(var_list).difference(set(table_descr[1])):
    table_descr[1][var] = 'INTEGER'
  table_descr[0] = var_list;

  table_handle = db.add_table(table_name, table_descr)

  for ind in keys:
    entry = var_table[ind]
    dispatch_number = entry['Index']
    if ind != dispatch_number: fatal("Dispatch #" + ind + " index mismatch (" + dispatch_number + ")\n")
    val_list = [entry[var] for var in var_list]
    db.insert_entry(table_handle, val_list)
#############################################################

# Fill Ext DB
ext_table_descr = [
  ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'Index'],
  {'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER', 'Name':'TEXT', 'Index':'INTEGER'}
]
def fill_ext_db(table_name, db, indir, trace_name, api_pid):
  file_name = indir + '/' + trace_name + '_trace.txt'
  ptrn_val = re.compile(r'(\d+) (\d+):(\d+) (\d+):(.*)$')

  if not os.path.isfile(file_name): return 0

  range_stack = {}

  record_id = 0
  table_handle = db.add_table(table_name, ext_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn_val.match(record)
      if m:
        tms = int(m.group(1))
        pid = m.group(2)
        tid = m.group(3)
        cid = int(m.group(4))
        msg = m.group(5)

        rec_vals = []

        if cid != 2:
          rec_vals.append(tms)
          rec_vals.append(tms + 1)
          rec_vals.append(api_pid)
          rec_vals.append(tid)
          rec_vals.append(msg)
          rec_vals.append(record_id)

        if cid == 1:
          if not pid in range_stack: range_stack[pid] = {}
          pid_stack = range_stack[pid]
          if not tid in pid_stack: pid_stack[tid] = []
          rec_stack = pid_stack[tid]
          rec_stack.append(rec_vals)
          continue

        if cid == 2:
          if not pid in range_stack: fatal("ROCTX range begin not found, pid(" + pid + ")");
          pid_stack = range_stack[pid]
          if not tid in pid_stack: fatal("ROCTX range begin not found, tid(" + tid + ")");
          rec_stack = pid_stack[tid]
          rec_vals = rec_stack.pop()
          rec_vals[1] = tms

        db.insert_entry(table_handle, rec_vals)
        record_id += 1

  return 1
#############################################################
# arguments manipulation routines
def get_field(args, field):
  ptrn1_field = re.compile(r'^.*' + field + '\(');
  ptrn2_field = re.compile(r'\) .*$');
  ptrn3_field = re.compile(r'\)\)$');
  (field_name, n) = ptrn1_field.subn('', args, count=1);
  if n != 0:
    (field_name, n) = ptrn2_field.subn('', field_name, count=1)
    if n == 0:
      (field_name, n) = ptrn3_field.subn('', field_name, count=1)
  return (field_name, n)

def set_field(args, field, val):
  return re.subn(field + '\(\w+\)([ \)])', field + '(' + str(val) + ')\\1', args, count=1)

# Fill API DB
api_table_descr = [
  ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'args', 'Index', 'Data'],
  {'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER', 'Name':'TEXT', 'args':'TEXT', 'Index':'INTEGER', 'Data':'TEXT'}
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
def fill_api_db(table_name, db, indir, api_name, api_pid, dep_pid, dep_list, dep_filtr, expl_id):
  global hsa_activity_found
  global memory_manager
  copy_raws = []
  if (hsa_activity_found): copy_raws = db.table_get_raws('COPY')
  copy_csv = ''
  copy_index = 0

  file_name = indir + '/' + api_name + '_api_trace.txt'
  ptrn_val = re.compile(r'(\d+):(\d+) (\d+):(\d+) ([^\(]+)(\(.*)$')
  ptrn_ac = re.compile(r'hsa_amd_memory_async_copy')
  ptrn1_kernel = re.compile(r'^.*kernel\(')
  ptrn2_kernel = re.compile(r'\)\) .*$')
  ptrn_fixformat = re.compile(r'(\d+:\d+ \d+:\d+ \w+)\(\s*(.*)\)$')
  ptrn_fixkernel = re.compile(r'\s+kernel=(.*)$')
  ptrn_multi_kernel = re.compile(r'(.*):(\d+)$')

  if not os.path.isfile(file_name): return 0

  dep_tid_list = []
  dep_from_us_list = []
  dep_id_list = []

  # parsing an input trace file and creating a DB table
  record_id_dict = {}
  table_handle = db.add_table(table_name, api_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]

      kernel_arg = ''
      m = ptrn_fixkernel.search(record)
      if m:
        kernel_arg = 'kernel(' + m.group(1) + ') '
        record = ptrn_fixkernel.sub('', record)

      mfixformat = ptrn_fixformat.match(record)
      if mfixformat: #replace '=' in args with parentheses
        reformated_args = kernel_arg + mfixformat.group(2).replace('=','(').replace(',',')')+')'
        record = mfixformat.group(1) + '(' + reformated_args + ')'

      m = ptrn_val.match(record)
      if m:
        rec_vals = []
        rec_len = len(api_table_descr[0]) - 1
        for ind in range(1, rec_len):
          rec_vals.append(m.group(ind))
        proc_id = int(rec_vals[2])
        thrd_id = int(rec_vals[3])
        record_name = rec_vals[4]
        record_args = rec_vals[5]

        # incrementing per-process record id/correlation id
        if not proc_id in record_id_dict: record_id_dict[proc_id] = 0
        corr_id = record_id_dict[proc_id]
        record_id_dict[proc_id] += 1
        rec_vals.append(corr_id)

        # extracting/converting stream id
        (stream_id, stream_found) = get_field(record_args, 'stream')
        if stream_found != 0:
          stream_id = get_stream_index(stream_id)
          (rec_vals[5], found) = set_field(record_args, 'stream', stream_id)
          if found == 0: fatal('set_field() failed for "stream", args: "' + record_args + '"')
        else: stream_id = 0

        # dependencies filling
        if ptrn_ac.search(record_name) or (corr_id, proc_id) in dep_filtr:
          beg_ns = int(rec_vals[0])
          end_ns = int(rec_vals[1])
          from_us = (beg_ns / 1000) + ((end_ns - beg_ns) / 1000)

          if not proc_id in dep_dict: dep_dict[proc_id] = {}
          dep_proc = dep_dict[proc_id]
          found = 1 if dep_pid in dep_proc else 0
          if found == 0 and dep_pid == OPS_PID:
            dep_proc[dep_pid] = { 'pid': api_pid, 'from': [], 'id': [] }
            found = 1
          if found == 1:
            dep_str = dep_proc[dep_pid]
            dep_str['from'].append((from_us, thrd_id, stream_id))
            if expl_id: dep_str['id'].append(corr_id)

          # memcopy data
          if len(copy_raws) != 0:
            copy_data = list(copy_raws[copy_index])
            args_str = rec_vals[5]
            args_str = re.sub(r'\(', r'', args_str)
            args_str = re.sub(r'\).*$', r'', args_str)
            copy_line = str(copy_data[0]) + ', ' + str(copy_data[1]) + ', ' + record_name + ', ' + args_str
            copy_csv += str(copy_index) + ', ' + copy_line + '\n'
            copy_index += 1

        # patching activity properties: kernel name, stream-id
        if (corr_id, proc_id) in dep_filtr:
          ops_table_name = dep_filtr[(corr_id, proc_id)]

          select_expr = '"Index" = ' + str(corr_id) + ' AND "proc-id" = ' + str(proc_id)
          record_args = rec_vals[rec_len - 2]

          # extract kernel name string
          (kernel_str, kernel_found) = get_field(record_args, 'kernel')
          is_kernel_list = 1 if kernel_found != 0 and kernel_str[-1] == ';' else 0

          if is_kernel_list != 0:
            for kernel_item in kernel_str[:-1].split(';'):
              m = ptrn_multi_kernel.match(kernel_item)
              if m:
                kernel_name = m.group(1)
                dev_id = m.group(2)
                select_expr += ' AND "dev-id" = ' + dev_id
                activity_record_patching(db, ops_table_name, 1, kernel_name, stream_found, stream_id, select_expr)
              else:
                fatal('Bad multi-kernel format: "' + kernel_item + '" in "' + kernel_str + '"')
          else:
            activity_record_patching(db, ops_table_name, kernel_found, kernel_str, stream_found, stream_id, select_expr)

        api_data = memory_manager.register_api(rec_vals) if len(dep_filtr) else ''
        rec_vals.append(api_data)

        rec_vals[2] = api_pid

        db.insert_entry(table_handle, rec_vals)
      else: fatal(api_name + " bad record: '" + record + "'")

  # inserting of dispatch events correlated to the dependent dispatches
  for (from_ns, proc_id, thrd_id) in dep_list:
    if not proc_id in record_id_dict: record_id_dict[proc_id] = 0
    corr_id = record_id_dict[proc_id]
    record_id_dict[proc_id] += 1
    db.insert_entry(table_handle, [from_ns, from_ns, api_pid, thrd_id, 'hsa_dispatch', '', corr_id, ''])

  # generating memcopy CSV
  if copy_csv != '':
    file_name = os.environ['PWD'] + '/results_mcopy.csv'
    with open(file_name, mode='w') as fd:
      print("File '" + file_name + "' is generating")
      fd.write(copy_csv)

  return 1
#############################################################

# fill COPY DB
copy_table_descr = [
  ['BeginNs', 'EndNs', 'Name', 'pid', 'tid', 'Index', 'proc-id'],
  {'Index':'INTEGER', 'proc-id':'INTEGER', 'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER'}
]
def fill_copy_db(table_name, db, indir):
  pid = COPY_PID
  file_name = indir + '/' + 'async_copy_trace.txt'
  ptrn_val = re.compile(r'(\d+):(\d+) (.*)$')
  ptrn_id = re.compile(r'^async-copy:(\d+):(\d+)$')

  if not os.path.isfile(file_name): return 0

  table_handle = db.add_table(table_name, copy_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn_val.match(record)
      if m:
        rec_vals = []
        for ind in range(1,4): rec_vals.append(m.group(ind))
        rec_vals.append(COPY_PID)
        rec_vals.append(0)

        m = ptrn_id.match(rec_vals[2])
        if not m: fatal("bad async-copy entry '" + record + "'")
        corr_id = int(m.group(1))
        proc_id = int(m.group(2))
        rec_vals.append(corr_id)
        rec_vals.append(proc_id)

        db.insert_entry(table_handle, rec_vals)

        # filling dependencies
        if not proc_id in dep_dict: dep_dict[proc_id] = {}
        dep_proc = dep_dict[proc_id]
        if not pid in dep_proc: dep_proc[pid] = { 'pid': HSA_PID, 'from': [], 'to': {}, 'id': [] }
        dep_str = dep_proc[pid]
        dep_str['to'][corr_id] = int(rec_vals[0]) / 1000

      else: fatal("async-copy bad record: '" + record + "'")

  return 1
#############################################################

# fill HCC ops DB
ops_table_descr = [
  ['BeginNs', 'EndNs', 'dev-id', 'queue-id', 'Name', 'pid', 'tid', 'Index', 'proc-id', 'Data'],
  {'Index':'INTEGER', 'proc-id':'INTEGER', 'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'dev-id':'INTEGER', 'queue-id':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER', 'Data':'TEXT'}
]
def fill_ops_db(kernel_table_name, mcopy_table_name, db, indir):
  global max_gpu_id
  file_name = indir + '/' + 'hcc_ops_trace.txt'
  ptrn_val = re.compile(r'(\d+):(\d+) (\d+):(\d+) (.*)$')
  ptrn_id = re.compile(r'^([^:]+):(\d+):(\d+)$')
  ptrn_mcopy = re.compile(r'(Memcpy|Copy|Fill)')
  ptrn_barrier = re.compile(r'Marker')

  if not os.path.isfile(file_name): return {}

  filtr = {}

  kernel_table_handle = db.add_table(kernel_table_name, ops_table_descr)
  mcopy_table_handle = db.add_table(mcopy_table_name, ops_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn_val.match(record)
      if m:
        # parsing trace record
        rec_vals = []
        for ind in range(1,6): rec_vals.append(m.group(ind))
        label = rec_vals[4] # record name
        m = ptrn_id.match(label)
        if not m: fatal("bad hcc ops entry '" + record + "'")
        name = m.group(1)
        corr_id = int(m.group(2)) - 1
        proc_id = int(m.group(3))

        # checking name for memcopy pattern
        if ptrn_mcopy.search(name):
          rec_table_name = mcopy_table_name
          table_handle = mcopy_table_handle
          pid = COPY_PID;
        else:
          rec_table_name = kernel_table_name
          table_handle = kernel_table_handle

          gpu_id = int(rec_vals[2]);
          if (gpu_id > max_gpu_id): max_gpu_id = gpu_id
          pid = GPU_BASE_PID + int(gpu_id)

          if ptrn_barrier.search(name):
            name = '"<barrier packet>"'

        # insert DB record
        rec_vals[4] = name                       # Name
        rec_vals.append(pid)                     # pid
        rec_vals.append(0)                       # tid
        rec_vals.append(corr_id)                 # Index
        rec_vals.append(proc_id)                 # proc-id
        rec_vals.append('')                      # Data
        db.insert_entry(table_handle, rec_vals)

        # registering a dependency filtr
        filtr[(corr_id, proc_id)] = rec_table_name

        # filling a dependencies
        if not proc_id in dep_dict: dep_dict[proc_id] = {}
        dep_proc = dep_dict[proc_id]
        if not pid in dep_proc: dep_proc[pid] = { 'bsp': OPS_PID, 'to': {} }
        dep_str = dep_proc[pid]
        dep_str['to'][corr_id] = int(rec_vals[0]) / 1000

      else:
        fatal("hcc ops bad record: '" + record + "'")

  return filtr
#############################################################
# main
if (len(sys.argv) < 2): fatal("Usage: " + sys.argv[0] + " <output CSV file> <input result files list>")

outfile = sys.argv[1]
infiles = sys.argv[2:]
indir = re.sub(r'\/[^\/]*$', r'', infiles[0])
inext = re.sub(r'\s+$', r'', infiles[0])
inext = re.sub(r'^.*(\.[^\.]+)$', r'\1', inext)

dbfile = ''
csvfile = ''

begin_ts_file = indir + '/begin_ts_file.txt'
if os.path.isfile(begin_ts_file):
  with open(begin_ts_file, mode='r') as fd:
    ind = 0
    for line in fd.readlines():
      val = int(line) / 1000
      if ind == 0 or val < START_US: START_US = val
      ind += 1
  print('START timestamp found (' + str(START_US) + 'us)')

if re.search(r'\.csv$', outfile):
  csvfile = outfile
elif re.search(r'\.db$', outfile):
  dbfile = outfile
  csvfile = re.sub(r'\.db$', '.csv', outfile)
else:
  fatal("Bad output file '" + outfile + "'")

if inext == '.txt':
  for f in infiles: parse_res(f)
  if len(var_table) != 0: merge_table()

if dbfile == '':
  dump_csv(csvfile)
else:
  statfile = re.sub(r'\.csv$', '.stats.csv', csvfile)
  jsonfile = re.sub(r'\.csv$', '.json', csvfile)

  hsa_statfile = re.sub(r'\.stats\.csv$', r'.hsa_stats.csv', statfile)
  hip_statfile = re.sub(r'\.stats\.csv$', r'.hip_stats.csv', statfile)
  kfd_statfile = re.sub(r'\.stats\.csv$', r'.kfd_stats.csv', statfile)
  ops_statfile = statfile
  copy_statfile = re.sub(r'\.stats\.csv$', r'.copy_stats.csv', statfile)
  sysinfo_file = re.sub(r'\.stats\.csv$', r'.sysinfo.txt', statfile)
  metadata_gen(sysinfo_file, 'rocminfo')

  with open(dbfile, mode='w') as fd: fd.truncate()
  db = SQLiteDB(dbfile)
  memory_manager = MemManager(db)

  ext_trace_found = fill_ext_db('rocTX', db, indir, 'roctx', EXT_PID)

  kfd_trace_found = fill_api_db('KFD', db, indir, 'kfd', KFD_PID, NONE_PID, [], {}, 0)

  hsa_activity_found = fill_copy_db('COPY', db, indir)
  hsa_trace_found = fill_api_db('HSA', db, indir, 'hsa', HSA_PID, COPY_PID, kern_dep_list, {}, 0)

  ops_filtr = fill_ops_db('OPS', 'COPY', db, indir)
  hip_trace_found = fill_api_db('HIP', db, indir, 'hip', HIP_PID, OPS_PID, [], ops_filtr, 1)

  fill_kernel_db('A', db)

  any_trace_found = ext_trace_found | kfd_trace_found | hsa_trace_found | hip_trace_found
  copy_trace_found = 0
  if hsa_activity_found or len(ops_filtr): copy_trace_found = 1

  if any_trace_found:
    db.open_json(jsonfile)

  if ext_trace_found:
    db.label_json(EXT_PID, "Markers and Ranges", jsonfile)

  if hip_trace_found:
    db.label_json(HIP_PID, "CPU HIP API", jsonfile)

  if hsa_trace_found:
    db.label_json(HSA_PID, "CPU HSA API", jsonfile)

  if kfd_trace_found:
    db.label_json(KFD_PID, "CPU KFD API", jsonfile)

  db.label_json(COPY_PID, "COPY", jsonfile)

  if any_trace_found and max_gpu_id >= 0:
    for ind in range(0, int(max_gpu_id) + 1):
      db.label_json(int(ind) + int(GPU_BASE_PID), "GPU" + str(ind), jsonfile)

  if ext_trace_found:
    dform.gen_ext_json_trace(db, 'rocTX', START_US, jsonfile)

  if len(var_table) != 0:
    dform.post_process_data(db, 'A', csvfile)
    dform.gen_table_bins(db, 'A', statfile, 'KernelName', 'DurationNs')
    if hsa_trace_found and 'BeginNs' in var_list:
      dform.gen_kernel_json_trace(db, 'A', GPU_BASE_PID, START_US, jsonfile)

  if hsa_trace_found:
    dform.post_process_data(db, 'HSA')
    dform.gen_table_bins(db, 'HSA', hsa_statfile, 'Name', 'DurationNs')
    dform.gen_api_json_trace(db, 'HSA', START_US, jsonfile)

  if copy_trace_found:
    dform.post_process_data(db, 'COPY')
    dform.gen_table_bins(db, 'COPY', copy_statfile, 'Name', 'DurationNs')
    dform.gen_api_json_trace(db, 'COPY', START_US, jsonfile)

  if hip_trace_found:
    dform.post_process_data(db, 'HIP')
    dform.gen_table_bins(db, 'HIP', hip_statfile, 'Name', 'DurationNs')
    dform.gen_api_json_trace(db, 'HIP', START_US, jsonfile)

  if ops_filtr:
    dform.post_process_data(db, 'OPS')
    dform.gen_table_bins(db, 'OPS', ops_statfile, 'Name', 'DurationNs')
    dform.gen_ops_json_trace(db, 'OPS', GPU_BASE_PID, START_US, jsonfile)

  if kfd_trace_found:
    dform.post_process_data(db, 'KFD')
    dform.gen_table_bins(db, 'KFD', kfd_statfile, 'Name', 'DurationNs')
    dform.gen_api_json_trace(db, 'KFD', START_US, jsonfile)

  if any_trace_found:
    dep_id = 0
    for (proc_id, dep_proc) in dep_dict.items():
      for (to_pid, dep_str) in dep_proc.items():
        if 'bsp' in dep_str:
          bspid = dep_str['bsp']
          base_str = dep_proc[bspid]
          for v in ('pid', 'from', 'id'):
            dep_str[v] = base_str[v]
          base_str['inv'] = 1

      for (to_pid, dep_str) in dep_proc.items():
        if 'inv' in dep_str: continue
        if not 'to' in dep_str: continue

        from_pid = dep_str['pid']
        from_us_list = dep_str['from']
        to_us_dict = dep_str['to']
        corr_id_list = dep_str['id']

        db.flow_json(dep_id, from_pid, from_us_list, to_pid, to_us_dict, corr_id_list, START_US, jsonfile)
        dep_id += len(from_us_list)

  if any_trace_found:
    db.metadata_json(jsonfile, sysinfo_file)
    db.close_json(jsonfile);

  memory_manager.dump_data()

  db.close()

sys.exit(0)
#############################################################
