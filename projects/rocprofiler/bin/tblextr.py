#!/usr/bin/python

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
from sqlitedb import SQLiteDB
import dform

# Parsing results in the format:
#dispatch[0], queue_index(0), kernel_name("SimpleConvolution"), time(1048928000311041,1048928006154674,1048928006168274,1048928006170503):
#  GRBM_GUI_ACTIVE (74332)
#  SQ_WAVES (4096)
#  SQ_INSTS_VMEM_RD (36864)

COPY_PID = 0
OPS_PID = 1
HSA_PID = 2
HIP_PID = 3
GPU_BASE_PID = 4
max_gpu_id = -1
START_US = 0

# dependencies dictionary
dep_dict = {}
kern_dep_list = []

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
#############################################################

# parse results method
def parse_res(infile):
  global max_gpu_id
  if not os.path.isfile(infile): fatal("Error: input file '" + infile + "' not found")
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
      var_table[dispatch_number][m.group(1)] = m.group(2)
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
            if var == 'tid': disp_tid = val
          else: fatal('wrong kernel property "' + prop + '" in "'+ kernel_properties + '"')
        m = ts_pattern.search(record)
        if m:
          var_table[dispatch_number]['DispatchNs'] = m.group(1)
          var_table[dispatch_number]['BeginNs'] = m.group(2)
          var_table[dispatch_number]['EndNs'] = m.group(3)
          var_table[dispatch_number]['CompleteNs'] = m.group(4)

          gpu_pid = GPU_BASE_PID + int(gpu_id)
          if not gpu_pid in dep_dict: dep_dict[gpu_pid] = {}
          dep_str = dep_dict[gpu_pid]
          if not 'tid' in dep_str: dep_str['tid'] = []
          if not 'from' in dep_str: dep_str['from'] = []
          if not 'to' in dep_str: dep_str['to'] = {}
          to_id = len(dep_str['tid'])
          from_us = int(m.group(1)) / 1000
          to_us = int(m.group(2)) / 1000
          dep_str['to'][to_id] = to_us
          dep_str['from'].append(from_us)
          dep_str['tid'].append(disp_tid)
          dep_str['pid'] = HSA_PID
          kern_dep_list.append((disp_tid, m.group(1)))

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

# fill HSA DB
hsa_table_descr = [
  ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'args', 'Index'],
  {'Index':'INTEGER', 'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER'}
]
def fill_api_db(table_name, db, indir, api_name, api_pid, dep_pid, dep_list, dep_filtr, expl_id):
  file_name = indir + '/' + api_name + '_api_trace.txt'
  ptrn_val = re.compile(r'(\d+):(\d+) (\d+):(\d+) ([^\(]+)(\(.*)$')
  ptrn_ac = re.compile(r'hsa_amd_memory_async_copy')

  if not os.path.isfile(file_name): return 0

  dep_tid_list = []
  dep_from_us_list = []
  dep_id_list = []

  global START_US
  with open(file_name, mode='r') as fd:
    line = fd.readline()
    record = line[:-1]
    m = ptrn_val.match(record)
    if m: START_US = int(m.group(1)) / 1000
    START_US = 0

  record_id = 0
  table_handle = db.add_table(table_name, hsa_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn_val.match(record)
      if m:
        rec_vals = []
        for ind in range(1,7):
          rec_vals.append(m.group(ind))
        rec_vals[2] = api_pid
        rec_vals.append(record_id)
        db.insert_entry(table_handle, rec_vals)
        if ptrn_ac.search(rec_vals[4]) or record_id in dep_filtr:
          beg_ns = int(rec_vals[0])
          end_ns = int(rec_vals[1])
          from_us = (beg_ns / 1000) + ((end_ns - beg_ns) / 1000)
          dep_from_us_list.append(from_us)
          dep_tid_list.append(int(rec_vals[3]))
          dep_id_list.append(record_id) 
        record_id += 1
      else: fatal("hsa bad record")

  for (tid, from_ns) in dep_list:
    db.insert_entry(table_handle, [from_ns, from_ns, api_pid, tid, 'hsa_dispatch', '', record_id])
    record_id += 1

  if not dep_pid in dep_dict: dep_dict[dep_pid] = {}
  dep_dict[dep_pid]['pid'] = api_pid
  dep_dict[dep_pid]['tid'] = dep_tid_list
  dep_dict[dep_pid]['from'] = dep_from_us_list
  if expl_id: dep_dict[dep_pid]['id'] = dep_id_list

  return 1
#############################################################

# fill COPY DB
copy_table_descr = [
  ['BeginNs', 'EndNs', 'Name', 'pid', 'tid', 'Index'],
  {'Index':'INTEGER', 'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER'}
]
def fill_copy_db(table_name, db, indir):
  file_name = indir + '/' + 'async_copy_trace.txt'
  ptrn_val = re.compile(r'(\d+):(\d+) (.*)$')
  ptrn_id = re.compile(r'^async-copy(\d+)$')

  if not COPY_PID in dep_dict: dep_dict[COPY_PID] = {}
  dep_to_us_dict = {}

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
        if m: dep_to_us_dict[int(m.group(1))] = int(rec_vals[0]) / 1000
        else: fatal("bad async-copy entry")
        rec_vals.append(m.group(1))
        db.insert_entry(table_handle, rec_vals)
      else: fatal("async-copy bad record")

  dep_dict[COPY_PID]['to'] = dep_to_us_dict
#############################################################

# fill HCC ops DB
ops_table_descr = [
  ['BeginNs', 'EndNs', 'dev-id', 'queue-id', 'Name', 'pid', 'tid', 'Index'],
  {'Index':'INTEGER', 'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'dev-id':'INTEGER', 'queue-id':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER'}
]
def fill_ops_db(table_name, db, indir):
  global max_gpu_id
  file_name = indir + '/' + 'hcc_ops_trace.txt'
  ptrn_val = re.compile(r'(\d+):(\d+) (\d+):(\d+) (.*)$')
  ptrn_id = re.compile(r'^[^:]+:(\d+)$')

  if not os.path.isfile(file_name): return {}

  filtr = {}

  record_id = 0
  table_handle = db.add_table(table_name, ops_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn_val.match(record)
      if m:
        rec_vals = []
        for ind in range(1,6): rec_vals.append(m.group(ind))
        gpu_id = int(rec_vals[2]);
        if (gpu_id > max_gpu_id): max_gpu_id = gpu_id
        gpu_pid = GPU_BASE_PID + int(gpu_id)
        rec_vals.append(gpu_pid)
        rec_vals.append(0)
        m = ptrn_id.match(rec_vals[4])
        if not m: fatal("bad hcc ops entry '" + record + "'")
        corr_id = int(m.group(1)) - 1
        rec_vals.append(corr_id)
        db.insert_entry(table_handle, rec_vals)
        filtr[corr_id] = 1

        if not gpu_pid in dep_dict:
          dep_dict[gpu_pid] = {}
          dep_dict[gpu_pid]['to'] = {}
        dep_dict[gpu_pid]['to'][corr_id] = int(rec_vals[0]) / 1000
        dep_dict[gpu_pid]['bsp'] = OPS_PID
      else: fatal("async-copy bad record")

  return filtr
#############################################################
# main
if (len(sys.argv) < 3): fatal("Usage: " + sys.argv[0] + " <output CSV file> <input result files list>")

outfile = sys.argv[1]
infiles = sys.argv[2:]
indir = re.sub(r'\/[^\/]*$', r'', infiles[0])

dbfile = ''
csvfile = ''

if re.search(r'\.csv$', outfile):
  csvfile = outfile
elif re.search(r'\.db$', outfile):
  dbfile = outfile
  csvfile = re.sub(r'\.db$', '.csv', outfile)
else:
  fatal("Bad output file '" + outfile + "'")

for f in infiles: parse_res(f)
if len(var_table) == 0: sys.exit(1)
merge_table()

if dbfile == '':
  dump_csv(csvfile)
else:
  statfile = re.sub(r'\.csv$', '.stats.csv', csvfile)
  jsonfile = re.sub(r'\.csv$', '.json', csvfile)

  with open(dbfile, mode='w') as fd: fd.truncate()
  db = SQLiteDB(dbfile)

  hsa_trace_found = fill_api_db('HSA', db, indir, 'hsa', HSA_PID, COPY_PID, kern_dep_list, {}, 0)
  if hsa_trace_found:
    fill_copy_db('COPY', db, indir) 

  ops_filtr = fill_ops_db('OPS', db, indir)
  hip_trace_found = fill_api_db('HIP', db, indir, 'hip', HIP_PID, OPS_PID, [], ops_filtr, 1)

  fill_kernel_db('A', db)

  any_trace_found = hsa_trace_found | hip_trace_found
  if any_trace_found:
    db.open_json(jsonfile)

  if hsa_trace_found:
    db.label_json(HSA_PID, "CPU HSA API", jsonfile)
    db.label_json(COPY_PID, "COPY", jsonfile)

  if hip_trace_found:
    db.label_json(HIP_PID, "CPU HIP API", jsonfile)

  if any_trace_found and max_gpu_id >= 0:
    for ind in range(0, int(max_gpu_id) + 1):
      db.label_json(int(ind) + int(GPU_BASE_PID), "GPU" + str(ind), jsonfile)

  dform.post_process_data(db, 'A', csvfile)
  dform.gen_table_bins(db, 'A', statfile, 'KernelName', 'DurationNs')
  if hsa_trace_found and 'BeginNs' in var_list:
    dform.gen_kernel_json_trace(db, 'A', GPU_BASE_PID, START_US, jsonfile)

  if hsa_trace_found:
    statfile = re.sub(r'stats', r'hsa_stats', statfile)
    dform.post_process_data(db, 'HSA')
    dform.gen_table_bins(db, 'HSA', statfile, 'Name', 'DurationNs')
    dform.gen_api_json_trace(db, 'HSA', START_US, jsonfile)
  
    dform.post_process_data(db, 'COPY')
    dform.gen_api_json_trace(db, 'COPY', START_US, jsonfile)

  if hip_trace_found:
    statfile = re.sub(r'stats', r'hip_stats', statfile)
    dform.post_process_data(db, 'HIP')
    dform.gen_table_bins(db, 'HIP', statfile, 'Name', 'DurationNs')
    dform.gen_api_json_trace(db, 'HIP', START_US, jsonfile)
  
    dform.post_process_data(db, 'OPS')
    dform.gen_ops_json_trace(db, 'OPS', GPU_BASE_PID, START_US, jsonfile)

  if any_trace_found:
    for (to_pid, dep_str) in dep_dict.items():
      if 'bsp' in dep_str:
        bspid = dep_str['bsp']
        base_str = dep_dict[bspid]
        for v in ('pid', 'tid', 'from', 'id'):
          dep_str[v] = base_str[v]
        base_str['inv'] = 1

    dep_id = 0
    for (to_pid, dep_str) in dep_dict.items():
      if 'inv' in dep_str: continue
      from_pid = dep_str['pid']
      tid_list = dep_str['tid']
      from_us_list = dep_str['from']
      to_us_dict = dep_str['to']
      corr_id_list = []
      if 'id' in dep_str: corr_id_list = dep_str['id']
      db.flow_json(dep_id, from_pid, tid_list, from_us_list, to_pid, to_us_dict, corr_id_list, START_US, jsonfile)
      dep_id += len(tid_list)

  if any_trace_found:
    db.close_json(jsonfile);
  db.close()

sys.exit(0)
#############################################################
