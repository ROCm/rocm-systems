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
HSA_PID = 1
GPU_BASE_PID = 2
max_gpu_id = 0

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
        kernel_properties = m.group(2)
        for prop in kernel_properties.split(', '):
          m = prop_pattern.match(prop)
          if m:
            var = m.group(1)
            val = m.group(2)
            var_table[dispatch_number][var] = val
            if not var in var_list: var_list.append(var);
            if re.search(r'gpu-id', var):
              if (val > max_gpu_id): max_gpu_id = val
          else: fatal('wrong kernel property "' + prop + '" in "'+ kernel_properties + '"')
        m = ts_pattern.search(record)
        if m:
          var_table[dispatch_number]['DispatchNs'] = m.group(1)
          var_table[dispatch_number]['BeginNs'] = m.group(2)
          var_table[dispatch_number]['EndNs'] = m.group(3)
          var_table[dispatch_number]['CompleteNs'] = m.group(4)

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
  ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'args'],
  {'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER'}
]
def fill_hsa_db(table_name, db, indir):
  file_name = indir + '/' + 'hsa_api_trace.txt'
  ptrn = re.compile(r'(\d+):(\d+) (\d+):(\d+) ([^\(]+)(\(.*)$')

  table_handle = db.add_table(table_name, hsa_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn.match(record)
      if m:
        rec_vals = []
        for ind in range(1,7):
          if hsa_table_descr[0][ind - 1] == 'pid':
            rec_vals.append(HSA_PID)
          else:
            rec_vals.append(m.group(ind))
        db.insert_entry(table_handle, rec_vals)
#############################################################

# fill COPY DB
copy_table_descr = [
  ['BeginNs', 'EndNs', 'Name', 'pid', 'tid'],
  {'Name':'TEXT', 'args':'TEXT', 'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER'}
]
def fill_copy_db(table_name, db, indir):
  file_name = indir + '/' + 'async_copy_trace.txt'
  ptrn = re.compile(r'(\d+):(\d+) (.*)$')

  table_handle = db.add_table(table_name, copy_table_descr)
  with open(file_name, mode='r') as fd:
    for line in fd.readlines():
      record = line[:-1]
      m = ptrn.match(record)
      if m:
        rec_vals = []
        for ind in range(1,4): rec_vals.append(m.group(ind))
        rec_vals.append(COPY_PID)
        rec_vals.append(0)
        db.insert_entry(table_handle, rec_vals)
#############################################################
# main
if (len(sys.argv) < 3): fatal("Usage: " + sys.argv[0] + " <output CSV file> <input result files list>")

outfile = sys.argv[1]
infiles = sys.argv[2:]
indir = re.sub(r'\/[^\/]*$', r'', infiles[0])
print "indir: '" + indir + "'"

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

  fill_hsa_db('HSA', db, indir) 
  fill_copy_db('COPY', db, indir) 
  fill_kernel_db('A', db)

  db.open_json(jsonfile);
  db.label_json(HSA_PID, "CPU", jsonfile)
  db.label_json(COPY_PID, "COPY", jsonfile)
  for ind in range(0, int(max_gpu_id) + 1): db.label_json(int(ind) + int(GPU_BASE_PID), "GPU" + str(ind), jsonfile)

  if 'BeginNs' in var_list:
    dform.post_process_data(db, 'A', csvfile)
    dform.gen_table_bins(db, 'A', statfile, 'KernelName', 'DurationNs')
    dform.gen_kernel_json_trace(db, 'A', jsonfile)
  else:
    db.dump_csv('A', csvfile)

  statfile = re.sub(r'stats', r'hsa_stats', statfile)
  dform.post_process_data(db, 'HSA')
  dform.gen_table_bins(db, 'HSA', statfile, 'Name', 'DurationNs')
  dform.gen_api_json_trace(db, 'HSA', jsonfile)

  dform.post_process_data(db, 'COPY')
  dform.gen_api_json_trace(db, 'COPY', jsonfile)

  db.close_json(jsonfile);
  db.close()

sys.exit(0)
#############################################################
