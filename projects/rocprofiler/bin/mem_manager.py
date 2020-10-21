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

pinned = ['hipMallocHost', 'hipHostMalloc', 'hipHostAlloc']
ondevice = ['hipMalloc', 'hipMallocPitch', 'hipMallocArray', 'hipMalloc3DArray']

mm_table_descr = [
  ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'Direction', 'SrcType', 'DstType', 'Size', 'BW', 'Async'],
  {'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER', 'Name':'TEXT', 'Direction':'TEXT', 'SrcType':'TEXT', 'DstType':'TEXT', 'Size':'INTEGER', 'BW':'TEXT', 'Async':'TEXT'}
]

def fatal(msg):
  sys.stderr.write(sys.argv[0] + ": " + msg + "\n");
  sys.exit(1)

DELIM = ','

# Mem copy manager class
class MemManager:

  def __init__(self, db):
    self.db = db
    self.allocations = {}
    self.memcopies = {}
    self.filename = ''
    self.fd = 0

  def __del__(self):
    if self.fd != 0: self.fd.close()

  # register alloc and memcpy API calls
  # ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'args', 'Index', 'Data'],
  def register_api(self, rec_vals):
    res = ''
    record_name = rec_vals[4]  # 'Name'
    record_args = rec_vals[5]  # 'args'
    malloc_ptrn = re.compile(r'hip.*Malloc')
    mcopy_ptrn = re.compile(r'hipMemcpy')

    if malloc_ptrn.match(record_name):
      self.add_allocation(record_name, record_args)
    elif mcopy_ptrn.match(record_name):
      res = self.add_memcpy(rec_vals)

    return res

  # register memcpy asynchronous activity
  # rec_vals: ['BeginNs', 'EndNs', 'dev-id', 'queue-id', 'Name', 'pid', 'tid', 'Index', 'proc-id', 'Data'],
  def register_activity(self, rec_vals):
    data = ''
    event = rec_vals[4]     # 'Name'
    recordid = rec_vals[7]  # 'Index'
    procid = rec_vals[8]    # 'proc-id'
    size_ptrn = re.compile(DELIM + 'Size=(\d+)' + DELIM)

    # query syncronous memcopy API record
    key = (recordid, procid, 0)
    if key in self.memcopies:
      data = self.memcopies[key]

    # query asyncronous memcopy API record
    key = (recordid, procid, 1)
    if key in self.memcopies:
      if data != '': fatal('register_activity: corrupted record sync/async')

      async_copy_start_time = rec_vals[0]
      async_copy_end_time = rec_vals[1]

      duration = int(async_copy_end_time) - int(async_copy_start_time)
      size = 0
      m = size_ptrn.search(self.memcopies[key])
      if m:
        size = m.group(1)
      bandwidth = round(float(size) * 1000 / duration, 2)

      tid = rec_vals[6]
      copy_line_header = str(async_copy_start_time) + DELIM + str(async_copy_end_time) + DELIM + str(procid) + DELIM + str(tid)
      copy_line_footer = 'BW=' + str(bandwidth) + DELIM + 'Async=' + str(1)
      data = copy_line_header + self.memcopies[key] + copy_line_footer
      self.memcopies[key] = data

    return data

  # add allocation to map
  def add_allocation(self, event, args):
    choice = 0
    if event == "hipMallocPitch":
      malloc_args_ptrn = re.compile(r'\(ptr\((.*)\) width\((.*)\) height\((.*)\)\)')
      choice = 1
    elif event == "hipMallocArray":
      malloc_args_ptrn = re.compile(r'\(array\((.*)\) width\((.*)\) height\((.*)\)\)')
      choice = 1
    elif event == "hipMalloc3DArray":
      malloc_args_ptrn = re.compile(r'\(array\((.*)\) width\((.*)\) height\((.*)\) depth\((.*)\)\)')
      choice = 2
    else:
      #(ptr(0x7f3407000000) size(800000000) flags(0))
      malloc_args_ptrn = re.compile(r'\(ptr\((.*)\) size\((.*)\) .*\)')
      choice = 3
    m = malloc_args_ptrn.match(args)
    if m:
      ptr = int(m.group(1), 16)
      if choice == 3:
        size = int(m.group(2))
      elif choice == 1:
        size = int(m.group(2)) * int(m.group(3))
      else:
        size = int(m.group(2)) * int(m.group(3)) * int(m.group(4))
      self.allocations[ptr] = (size, event)

  #get type of ptr
  def get_ptr_type(self, ptr):
    addr = int(ptr, 16)
    addr_type = 'unknown'
    found = 0
    for base, (size, event) in self.allocations.items():
      if addr >= base and addr < base + size:
        found = 1
        break
    if not found:
      addr_type = 'pageable'
    elif event in pinned:
      addr_type = 'pinned'
    elif event in ondevice:
      addr_type = 'device'
    else:
      fatal('internal error: ptr(' + ptr + ') cannot be identified')
    return addr_type

  # add memcpy to map
  def add_memcpy(self, recvals):
    recordid = recvals[6]  #same as corrid
    event = recvals[4]
    start_time = recvals[0] # sync time stamp
    end_time = recvals[1] # sync time stamp
    args = recvals[5]
    procid = int(recvals[2]) # used to query async entries
    pid = recvals[2]
    tid = recvals[3]

    select_expr = '"Index" = ' + str(recordid) + ' AND "proc-id" = ' + str(procid)

    # hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
    hipMemcpy_ptrn = re.compile(r'\(\s*dst\((.*)\) src\((.*)\) sizeBytes\((\d+)\).*\)')
    # hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
    #                   size_t height, hipMemcpyKind kind);
    hipMemcpy_ptrn2 = re.compile(r'\(\s*dst\((.*)\) .* src\((.*)\) .* width\((\d+)\) height\((\d+)\).*\)')
    # hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
    #                        size_t count, hipMemcpyKind kind);
    hipMemcpy_ptrn3 = re.compile(r'\(\s*dst\((.*)\) .* src\((.*)\) count\((\d+)\).*\)')
    # memcopy with kind argument
    hipMemcpy_ptrn_kind = re.compile(r'.* kind\((\d+)\)\s*.*')
    # aysnc memcopy
    async_event_ptrn = re.compile(r'Async|async')

    m_basic = hipMemcpy_ptrn.match(args)
    m_2d = hipMemcpy_ptrn2.match(args)
    m_array = hipMemcpy_ptrn3.match(args)

    is_async = 1 if async_event_ptrn.search(event) else 0
    async_copy_start_time = -1
    async_copy_end_time = -1
    tid = -1

    copy_line = ''
    size = 0
    dstptr_type = 'unknown'
    srcptr_type = 'unknown'
    direction = 'unknown'
    bandwidth = 0
    duration = 0

    switcher = {
      '0': "HtoH",
      '1': "HtoD",
      '2': "DtoH",
      '3': "DtoD",
      '4': "auto",
    }

    condition_matched = False
    if m_basic:
      dstptr = m_basic.group(1)
      dstptr_type = self.get_ptr_type(dstptr)
      srcptr = m_basic.group(2)
      srcptr_type = self.get_ptr_type(srcptr)
      size = int(m_basic.group(3))
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
      size = m_2d.group(3)*m_2d.group(4)
      condition_matched = True

    if not condition_matched: fatal('Memcpy args \"' + args + '\" cannot be identified')

    if not is_async:
      start_time = recvals[0] # sync time stamp
      end_time = recvals[1] # sync time stamp
      duration = (int(end_time) - int(start_time))
      bandwidth = round(float(size) * 1000 / duration, 2)

    m = hipMemcpy_ptrn_kind.match(args)
    if m:
      direction = switcher.get(m.group(1), "unknown")

    copy_line_header = ''
    copy_line_footer = ''
    if not is_async:
        copy_line_header = str(start_time) + DELIM + str(end_time) + DELIM + pid + DELIM + tid
        copy_line_footer = "BW=" + str(bandwidth) + DELIM + 'Async=' + str(is_async)

    copy_line = copy_line_header + DELIM + event + DELIM + 'Direction=' + direction + DELIM + 'SrcType=' + srcptr_type + DELIM + 'DstType=' + dstptr_type + DELIM + "Size=" + str(size) + DELIM + copy_line_footer

    self.memcopies[(recordid, procid, is_async)] = copy_line
    return copy_line;

  def dump_data(self, table_name, file_name):
    # To create memcopy info table in DB
    print("File '" + file_name + "' is generating")
    table_handle = self.db.add_table(table_name, mm_table_descr)

    fld_ptrn = re.compile(r'(.*)=(.*)')
    for (key, record) in self.memcopies.items():
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
