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

#!/usr/bin/python
import os, sys, re

# Parsing results in the format:
#dispatch[0], queue_index(0), kernel_name("SimpleConvolution"), time(1048928000311041,1048928006154674,1048928006168274,1048928006170503):
#  GRBM_GUI_ACTIVE (74332)
#  SQ_WAVES (4096)
#  SQ_INSTS_VMEM_RD (36864)

# global vars
var_list = ['Index', 'KernelName', 'DispatchNs', 'BeginNs', 'EndNs', 'CompleteNs']
var_table = {}
#############################################################

def fatal(msg):
  sys.stderr.write(sys.argv[0] + ": " + msg + "\n");
  sys.exit(1)
#############################################################

# parse results method
def parse_res(infile):
  if not os.path.isfile(infile): fatal("Error: input file '" + infile + "' not found")
  inp = open(infile, 'r')

  beg_pattern = re.compile("^dispatch\[(\d*)\], queue_index\(\d*\), kernel_name\(\"([^\"]*)\"\)")
  ts_pattern = re.compile(", time\((\d*),(\d*),(\d*),(\d*)\)")
  var_pattern = re.compile("^\s*([^\s]*)\s+\((\d*)\)")

  dispatch_number = 0
  for line in inp.readlines():
    record = line[:-1]

    m = var_pattern.match(record)
    if m:
      if not dispatch_number in var_table: fatal("Error: dispatch number not unique '" + str(dispatch_number) + "'")
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
          'KernelName': "\"" + m.group(2) + "\""
        }
        m = ts_pattern.search(record)
        if m:
          var_table[dispatch_number]['DispatchNs'] = m.group(1)
          var_table[dispatch_number]['BeginNs'] = m.group(2)
          var_table[dispatch_number]['EndNs'] = m.group(3)
          var_table[dispatch_number]['CompleteNs'] = m.group(4)

  inp.close()
#############################################################

# print results table method
def print_tbl(outfile):
  global var_list
  if len(var_table) == 0: return 1

  out = open(outfile, 'w')

  keys = var_table.keys()
  keys.sort(key=int)

  entry = var_table[keys[0]]
  list1 = []
  for var in var_list:
    if var in entry:
      list1.append(var)
  var_list = list1

  for var in var_list: out.write(var + ',')
  out.write("\n")

  for ind in keys:
    entry = var_table[ind]
    dispatch_number = entry['Index']
    if ind != dispatch_number: fatal("Dispatch #" + ind + " index mismatch (" + dispatch_number + ")\n")
    for var in var_list: out.write(entry[var] + ',')
    out.write("\n")

  out.close()
  return 0
#############################################################

# main
if (len(sys.argv) < 3): fatal("Usage: " + sys.argv[0] + " <output CSV file> <input result files list>")

outfile = sys.argv[1]
infiles = sys.argv[2:]
for f in infiles :
  parse_res(f)
ret = print_tbl(outfile)
sys.exit(ret)
#############################################################
