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
ondevice = ['hipMallocPitch', 'hipMallocArray', 'hipMalloc3DArray']

mm_table_descr = [
  ['BeginNs', 'EndNs', 'pid', 'tid', 'Name', 'Direction', 'SrcType', 'DstType', 'Size', 'BW', 'Async'],
  {'BeginNs':'INTEGER', 'EndNs':'INTEGER', 'pid':'INTEGER', 'tid':'INTEGER', 'Name':'TEXT', 'Direction':'TEXT', 'SrcType':'TEXT', 'DstType':'TEXT', 'Size':'INTEGER', 'BW':'TEXT', 'Async':'TEXT'}
]

def fatal(msg):
  sys.stderr.write(sys.argv[0] + ": " + msg + "\n");
  sys.exit(1)

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

  # register allo and memcpy API calls
  def register_api(self, rec_vals):
    res = ''
    malloc_ptrn = re.compile(r'hip.*Malloc')
    record_name = rec_vals[4]
    record_args = rec_vals[5]
    mallocm = malloc_ptrn.match(record_name)
    if mallocm:
      self.add_allocation(record_name, record_args)
    else:
      hip_ptrn_mcopy = re.compile(r'hipMemcpy')
      memcpym = hip_ptrn_mcopy.match(record_name)
      if memcpym:
        res = self.add_memcpy(rec_vals)

    return res

  # add allocation to map
  def add_allocation(self, event, args):
    choice = 0
    if event == "hipMallocPitch":
      hipMalloc_ptrn = re.compile(r'\(ptr\((.*)\) width\((.*)\) height\((.*)\)\)')
      choice = 1
    elif event == "hipMallocArray":
      hipMalloc_ptrn = re.compile(r'\(array\((.*)\) width\((.*)\) height\((.*)\)\)')
      choice = 1
    elif event == "hipMalloc3DArray":
      hipMalloc_ptrn = re.compile(r'\(array\((.*)\) width\((.*)\) height\((.*)\) depth\((.*)\)\)')
      choice = 2
    else:
      hipMalloc_ptrn = re.compile(r'\(ptr\((.*)\) size\((.*)\)\)')
      choice = 3
    mhipMalloc = hipMalloc_ptrn.match(args)
    if mhipMalloc:
      ptr = mhipMalloc.group(1)
      if choice == 3:
        size = mhipMalloc.group(2)
      elif choice == 1:
        size = mhipMalloc.group(2) * mhipMalloc.group(3)
      else:
        size = mhipMalloc.group(2) * mhipMalloc.group(3) * mhipMalloc.group(4)
      self.allocations[ptr]=(size, event)
    return

  #get type of ptr
  def get_ptr_type(self, ptr):
    ptr_type = 'unknown'
    found = 0
    for base in self.allocations.keys():
      (size, event) = self.allocations[base]
      size = re.sub('\).*$', '', size)
      size = '0x' + size
      #print("ptr(" + str(ptr) + ") base(" + base + ") size='" + size + "'")
      if int(ptr, 16) >= int(base, 16) and int(ptr, 16) < int(base, 16) + int(size, 16):
        found = 1
        break
    if not found:
      ptr_type = 'pageable'
    elif event in pinned:
      ptr_type = 'pinned'
    elif event in ondevice:
      ptr_type = 'device'
    return ptr_type

  # add memcpy to map
  def add_memcpy(self, recvals):
    recordid = recvals[6]  #same as corrid
    event = recvals[4]
    start_time = recvals[0] # sync time stamp
    end_time = recvals[1] # sync time stamp
    args = recvals[5]
    procid = recvals[2] # used to query async entries
    pid = recvals[2]
    tid = recvals[3]
    separator = ','

    select_expr = '"Index" = ' + str(recordid) + ' AND "proc-id" = ' + str(procid)
    async_copy_records = self.db.table_get_record('COPY', select_expr)  #List of async copy record fields
    async_copy_start_time = async_copy_records[0]
    async_copy_end_time = async_copy_records[1]

    #hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
    hipMemcpy_ptrn = re.compile(r'\(dst\((.*)\) src\((.*)\) sizeBytes\((\d+)\).*\)')
    #hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
    #                   size_t height, hipMemcpyKind kind);
    hipMemcpy_ptrn2 = re.compile(r'\(dst\((.*)\) .* src\((.*)\) .* width\((\d+)\) height\((\d+)\).*\)')
    #hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
    #                        size_t count, hipMemcpyKind kind);
    hipMemcpy_ptrn3 = re.compile(r'\(dst\((.*)\) .* src\((.*)\) count\((\d+)\).*\)')
    mhip = hipMemcpy_ptrn.match(args)
    mhip2 = hipMemcpy_ptrn2.match(args)
    mhip3 = hipMemcpy_ptrn3.match(args)

    if mhip or mhip2 or mhip3:
      size = 0
      dstptr_type = 'unknown'
      srcptr_type = 'unknown'
      if mhip:
        dstptr = mhip.group(1)
        dstptr_type = self.get_ptr_type(dstptr)
        srcptr = mhip.group(2)
        srcptr_type = self.get_ptr_type(srcptr)
        size = int(mhip.group(3), 16)
      if mhip3:
        dstptr = mhip3.group(1)
        dstptr_type = self.get_ptr_type(dstptr)
        srcptr = mhip3.group(2)
        srcptr_type = self.get_ptr_type(srcptr)
        size = int(mhip3.group(3), 16)
      if mhip2:
        dstptr = mhip2.group(1)
        dstptr_type = self.get_ptr_type(dstptr)
        srcptr = mhip2.group(2)
        srcptr_type = self.get_ptr_type(srcptr)
        size = int(mhip2.group(3)*mhip2.group(4), 16)
      bandwidth = float(size) / (int(end_time) - int(start_time))
      copy_line = str(start_time) + ',' + str(end_time) + ',' + pid + ',' + tid + ',' + event
      hipMemcpy_ptrn_kind = re.compile(r'.* kind\((\d+)\)\s*.*')
      mkind = hipMemcpy_ptrn_kind.match(args)
      if mkind:
        if mkind.group(1) == '0': copy_line += separator + "Direction=hipMemcpyHostToHost"
        if mkind.group(1) == '1': copy_line += separator + "Direction=hipMemcpyHostToDevice"
        if mkind.group(1) == '2': copy_line += separator + "Direction=hipMemcpyDeviceToHost"
        if mkind.group(1) == '3': copy_line += separator + "Direction=hipMemcpyDeviceToDevice"
        if mkind.group(1) == '4': copy_line += separator + "Direction=runtime copy-kind"
      else:
        copy_line += ", -"
      copy_line += separator + 'SrcType=' + srcptr_type + separator + 'DstType=' + dstptr_type + separator + "Size=" + str(round(size*1e-6, 2)) + separator + "BW=" + str(round(bandwidth,2))
      async_event_ptrn = re.compile(r'hipMemcpy.*Async')
      masync = async_event_ptrn.match(event)
      if masync:
        copy_line += separator + 'Async=yes'
      else:
        copy_line += separator + 'Async=no'

    self.memcopies[recordid] = copy_line
    return copy_line;

  def dump_data(self):
    # To create “MM” table in DB on the finish:
    table_name = "MM"
    file_name = os.environ['PWD'] + '/results_memcopy_info.csv'
    print("File '" + file_name + "' is generating")
    table_handle = self.db.add_table(table_name, mm_table_descr)

    fld_ptrn = re.compile(r'(.*)=(.*)')
    for (key, record) in self.memcopies.items():
      rec_vals_array = []
      for rec in record.split(','):
        fld_ptrnm = fld_ptrn.match(rec)
        if fld_ptrnm:
          rec_vals_array.append(fld_ptrnm.group(2))
        else:
          rec_vals_array.append(rec)
      self.db.insert_entry(table_handle, rec_vals_array)
    # To dump the MM table as CSV by:
    self.db.dump_csv(table_name, file_name)

