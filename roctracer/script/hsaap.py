#!/usr/bin/env python3

################################################################################
# Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
################################################################################

from __future__ import print_function
import os, sys, re

H_OUT='hsa_prof_str.h'
CPP_OUT='hsa_prof_str.inline.h'
API_TABLES_H = 'hsa_api_trace.h'
API_HEADERS_H = (
  ('CoreApi', 'hsa.h'),
  ('AmdExt', 'hsa_ext_amd.h'),
  ('ImageExt', 'hsa_ext_image.h'),
  ('AmdExt', API_TABLES_H),
)

LICENSE = \
'/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.\n' + \
'\n' + \
' Permission is hereby granted, free of charge, to any person obtaining a copy\n' + \
' of this software and associated documentation files (the "Software"), to deal\n' + \
' in the Software without restriction, including without limitation the rights\n' + \
' to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n' + \
' copies of the Software, and to permit persons to whom the Software is\n' + \
' furnished to do so, subject to the following conditions:\n' + \
'\n' + \
' The above copyright notice and this permission notice shall be included in\n' + \
' all copies or substantial portions of the Software.\n' + \
'\n' + \
' THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n' + \
' IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n' + \
' FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\n' + \
' AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n' + \
' LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n' + \
' OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n' + \
' THE SOFTWARE. */\n'

#############################################################
# Error handler
def fatal(module, msg):
  print (module + ' Error: "' + msg + '"', file = sys.stderr)
  sys.exit(1)

# Get next text block
def NextBlock(pos, record):
  if len(record) == 0: return pos

  space_pattern = re.compile(r'(\s+)')
  word_pattern = re.compile(r'([\w\*]+)')
  if record[pos] != '(':
    m = space_pattern.match(record, pos)
    if not m:
      m = word_pattern.match(record, pos)
    if m:
      return pos + len(m.group(1))
    else:
      fatal('NextBlock', "bad record '" + record + "' pos(" + str(pos) + ")")
  else:
    count = 0
    for index in range(pos, len(record)):
      if record[index] == '(':
        count = count + 1
      elif record[index] == ')':
        count = count - 1
        if count == 0:
          index = index + 1
          break
    if count != 0:
      fatal('NextBlock', "count is not zero (" + str(count) + ")")
    if record[index - 1] != ')':
      fatal('NextBlock', "last char is not ')' '" + record[index - 1] + "'")
    return index

#############################################################
# API table parser class
class API_TableParser:
  def fatal(self, msg):
    fatal('API_TableParser', msg)

  def __init__(self, header, name):
    self.name = name

    if not os.path.isfile(header):
      self.fatal("file '" + header + "' not found")

    self.inp = open(header, 'r')

    self.beg_pattern = re.compile('^\s*struct\s+' + name + 'Table\s*{\s*$')
    self.end_pattern = re.compile('^\s*};\s*$')
    self.array = []
    self.parse()

  # normalizing a line
  def norm_line(self, line):
    return re.sub(r'^\s+', r' ', line[:-1])

  # check for start record
  def is_start(self, record):
    return self.beg_pattern.match(record)

  # check for end record
  def is_end(self, record):
    return self.end_pattern.match(record)

  # check for declaration entry record
  def is_entry(self, record):
    return re.match(r'^\s*decltype\(([^\)]*)\)', record)

  # parse method
  def parse(self):
    active = 0
    for line in self.inp.readlines():
      record = self.norm_line(line)
      if self.is_start(record): active = 1
      if active != 0:
        if self.is_end(record): return
        m = self.is_entry(record)
        if m:
          self.array.append(m.group(1))

#############################################################
# API declaration parser class
class API_DeclParser:
  def fatal(self, msg):
    fatal('API_DeclParser', msg)

  def __init__(self, header, array, data):
    if not os.path.isfile(header):
      self.fatal("file '" + header + "' not found")

    self.inp = open(header, 'r')

    self.end_pattern = re.compile('\);\s*$')
    self.data = data
    for call in array:
      if call in data:
        self.fatal(call + ' is already found')
      self.parse(call)

  # api record filter
  def api_filter(self, record):
    record = re.sub(r'\sHSA_API\s', r' ', record)
    record = re.sub(r'\sHSA_DEPRECATED\s', r' ', record)
    return record

  # check for start record
  def is_start(self, call, record):
    return re.search('\s' + call + '\s*\(', record)

  # check for API method record
  def is_api(self, call, record):
    record = self.api_filter(record)
    return re.match('\s+\S+\s+' + call + '\s*\(', record)

  # check for end record
  def is_end(self, record):
    return self.end_pattern.search(record)

  # parse method args
  def get_args(self, record):
    struct = {'ret': '', 'args': '', 'astr': {}, 'alst': [], 'tlst': []}
    record = re.sub(r'^\s+', r'', record)
    record = re.sub(r'\s*(\*+)\s*', r'\1 ', record)
    rind = NextBlock(0, record)
    struct['ret'] = record[0:rind]
    pos = record.find('(')
    end = NextBlock(pos, record);
    args = record[pos:end]
    args = re.sub(r'^\(\s*', r'', args)
    args = re.sub(r'\s*\)$', r'', args)
    args = re.sub(r'\s*,\s*', r',', args)
    struct['args'] = re.sub(r',', r', ', args)
    if len(args) == 0: return struct

    pos = 0
    args = args + ','
    while pos < len(args):
      ind1 = NextBlock(pos, args) # type
      ind2 = NextBlock(ind1, args) # space
      if args[ind2] != '(':
        while ind2 < len(args):
          end = NextBlock(ind2, args)
          if args[end] == ',': break
          else: ind2 = end
        name = args[ind2:end]
      else:
        ind3 = NextBlock(ind2, args) # field
        m = re.match(r'\(\s*\*\s*(\S+)\s*\)', args[ind2:ind3])
        if not m:
          self.fatal("bad block3 '" + args + "' : '" + args[ind2:ind3] + "'")
        name = m.group(1)
        end = NextBlock(ind3, args) # the rest
      item = args[pos:end]
      struct['astr'][name] = item
      struct['alst'].append(name)
      struct['tlst'].append(item)
      if args[end] != ',':
        self.fatal("no comma '" + args + "'")
      pos = end + 1

    return struct

  # parse given api
  def parse(self, call):
    record = ''
    active = 0
    found = 0
    api_name = ''
    prev_line = ''

    self.inp.seek(0)
    for line in self.inp.readlines():
      record += ' ' + line[:-1]
      record = re.sub(r'^\s*', r' ', record)

      if active == 0:
        if self.is_start(call, record):
          active = 1
          m = self.is_api(call, record)
          if not m:
            record = ' ' + prev_line + ' ' + record
            m = self.is_api(call, record)
            if not m:
              self.fatal("bad api '" + line + "'")

      if active == 1:
        if self.is_end(record):
          self.data[call] = self.get_args(record)
          active = 0
          found = 0

      if active == 0: record = ''
      prev_line = line

#############################################################
# API description parser class
class API_DescrParser:
  def fatal(self, msg):
    fatal('API_DescrParser', msg)

  def __init__(self, out_h_file, hsa_dir, api_table_h, api_headers, license):
    out_macro = re.sub(r'[\/\.]', r'_', out_h_file.upper()) + '_'

    self.h_content = ''
    self.cpp_content = ''
    self.api_names = []
    self.api_calls = {}
    self.api_rettypes = set()
    self.api_id = {}

    api_data = {}
    api_list = []
    ns_calls = []

    for i in range(0, len(api_headers)):
      (name, header) = api_headers[i]

      if i < len(api_headers) - 1:
        api = API_TableParser(hsa_dir + api_table_h, name)
        api_list = api.array
        self.api_names.append(name)
        self.api_calls[name] = api_list
      else:
        api_list = ns_calls
        ns_calls = []

      for call in api_list:
        if call in api_data:
          self.fatal("call '"  + call + "' is already found")

      API_DeclParser(hsa_dir + header, api_list, api_data)

      for call in api_list:
        if not call in api_data:
          # Not-supported functions
          ns_calls.append(call)
        else:
          # API ID map
          self.api_id[call] = 'HSA_API_ID_' + call
          # Return types
          self.api_rettypes.add(api_data[call]['ret'])

    self.api_rettypes.discard('void')
    self.api_data = api_data
    self.ns_calls = ns_calls

    self.h_content += "/* Generated by " + os.path.basename(__file__) + " */\n" + license + "\n\n"

    self.h_content += "/* HSA API tracing primitives\n"
    for (name, header) in api_headers:
      self.h_content += " '" + name + "', header '" + header + "', " + str(len(self.api_calls[name])) + ' funcs\n'
    for call in self.ns_calls:
      self.h_content += ' ' + call + ' was not parsed\n'
    self.h_content += " */\n"
    self.h_content += '\n'
    self.h_content += '#ifndef ' + out_macro + '\n'
    self.h_content += '#define ' + out_macro + '\n'

    self.h_content += self.add_section('API ID enumeration', '  ', self.gen_id_enum)

    self.h_content += '/* Declarations of APIs intended for use only by tools. */\n'
    self.h_content += 'typedef void (*hsa_amd_queue_intercept_packet_writer)(const void*, uint64_t);\n'
    self.h_content += 'typedef void (*hsa_amd_queue_intercept_handler)(const void*, uint64_t, uint64_t, void*,\n'
    self.h_content += '                                                hsa_amd_queue_intercept_packet_writer);\n'
    self.h_content += 'typedef void (*hsa_amd_runtime_queue_notifier)(const hsa_queue_t*, hsa_agent_t, void*);\n'

    self.h_content += self.add_section('API arg structure', '    ', self.gen_arg_struct)
    self.h_content += self.add_section('API output stream', '    ', self.gen_out_stream)
    self.h_content += '#endif /* ' + out_macro + ' */\n'

    self.cpp_content += "/* Generated by " + os.path.basename(__file__) + " */\n" + license + "\n\n"

    self.cpp_content += '#include <hsa/hsa_api_trace.h>\n'
    self.cpp_content += '#include <atomic>\n'
    self.cpp_content += 'namespace roctracer::hsa_support::detail {\n'

    self.cpp_content += 'static CoreApiTable CoreApi_saved_before_cb;\n'
    self.cpp_content += 'static AmdExtTable AmdExt_saved_before_cb;\n'
    self.cpp_content += 'static ImageExtTable ImageExt_saved_before_cb;\n\n'

    self.cpp_content += self.add_section('API callback functions', '', self.gen_callbacks)
    self.cpp_content += self.add_section('API intercepting code', '', self.gen_intercept)
    self.cpp_content += self.add_section('API get_name function', '    ', self.gen_get_name)
    self.cpp_content += self.add_section('API get_code function', '  ', self.gen_get_code)
    self.cpp_content += '\n};\n'

  # add code section
  def add_section(self, title, gap, fun):
    content = ''
    n = 0
    content +=  '\n/* section: ' + title + ' */\n\n'
    content += fun(-1, '-', '-', {})
    for index in range(len(self.api_names)):
      last = (index == len(self.api_names) - 1)
      name = self.api_names[index]
      if n != 0:
        if gap == '': content += fun(n, name, '-', {})
        content += '\n'
      content += gap + '/* block: ' + name + ' API */\n'
      for call in self.api_calls[name]:
        content += fun(n, name, call, self.api_data[call])
        n += 1
    content += fun(n, '-', '-', {})
    return content

  # generate API ID enumeration
  def gen_id_enum(self, n, name, call, data):
    content = ''
    if n == -1:
      content += 'enum hsa_api_id_t {\n'
      return content
    if call != '-':
      content += '  ' + self.api_id[call] + ' = ' + str(n) + ',\n'
    else:
      content += '\n'
      content += '  HSA_API_ID_DISPATCH = ' + str(n) + ',\n'
      content += '  HSA_API_ID_NUMBER = ' + str(n + 1) + ',\n'
      content += '};\n'
    return content

  # generate API args structure
  def gen_arg_struct(self, n, name, call, struct):
    content = ''
    if n == -1:
      content += 'struct hsa_api_data_t {\n'
      content += '  uint64_t correlation_id;\n'
      content += '  uint32_t phase;\n'
      content += '  union {\n'
      for ret_type in self.api_rettypes:
        content += '    ' + ret_type + ' ' + ret_type + '_retval;\n'
      content += '  };\n'
      content += '  union {\n'
      return content
    if call != '-':
      content +=   '    struct {\n'
      for (var, item) in struct['astr'].items():
        content += '      ' + item + ';\n'
        if call == "hsa_amd_memory_async_copy_rect" and item == "const hsa_dim3_t* range":
          content += '      hsa_dim3_t range__val;\n'
      content +=   '    } ' + call + ';\n'
    else:
      content += '  } args;\n'
      content += '  uint64_t *phase_data;\n'
      content += '};\n'
    return content

  # generate API callbacks
  def gen_callbacks(self, n, name, call, struct):
    content = ''
    if n == -1:
      content += '/* section: Static declarations */\n'
      content += '\n'
    if call != '-':
      call_id = self.api_id[call];
      ret_type = struct['ret']
      content += 'static ' + ret_type + ' ' + call + '_callback(' + struct['args'] + ') {\n'

      content += '  hsa_trace_data_t trace_data;\n'
      content += '  bool enabled{false};\n'
      content += '\n'
      content += '  if (auto function = report_activity.load(std::memory_order_relaxed); function &&\n'
      content += '      (enabled =\n'
      content += '           function(ACTIVITY_DOMAIN_HSA_API, ' + call_id + ', &trace_data) == 0)) {\n'
      content += '    if (trace_data.phase_enter != nullptr) {\n'

      for var in struct['alst']:
        item = struct['astr'][var];
        if re.search(r'char\* ', item):
          # FIXME: we should not strdup the char* arguments here, as the callback will not outlive the scope of this function. Instead, we
          # should generate a helper function to capture the content of the arguments similar to hipApiArgsInit for HIP. We also need a
          # helper to free the memory that is allocated to capture the content.
          content += '      trace_data.api_data.args.' + call + '.' + var + ' = ' + '(' + var + ' != NULL) ? strdup(' + var + ')' + ' : NULL;\n'
        else:
          content += '      trace_data.api_data.args.' + call + '.' + var + ' = ' + var + ';\n'
          if call == 'hsa_amd_memory_async_copy_rect' and var == 'range':
            content += '      trace_data.api_data.args.' + call + '.' + var + '__val = ' + '*(' + var + ');\n'

      content += '      trace_data.phase_enter(' + call_id + ', &trace_data);\n'
      content += '    }\n'
      content += '  }\n'
      content += '\n'

      if ret_type != 'void':
        content +=  '  trace_data.api_data.' + ret_type + '_retval = '
      content += '  ' + name + '_saved_before_cb.' + call + '_fn(' + ', '.join(struct['alst']) + ');\n'

      content += '\n'
      content += '  if (enabled && trace_data.phase_exit != nullptr)\n'
      content += '    trace_data.phase_exit(' + call_id + ', &trace_data);\n'

      if ret_type != 'void':
        content += '  return trace_data.api_data.' + ret_type + '_retval;\n'
      content += '}\n'

    return content

  # generate API intercepting code
  def gen_intercept(self, n, name, call, struct):
    content = ''
    if n > 0 and call == '-':
      content += '};\n'
    if n == 0 or (call == '-' and name != '-'):
      content += 'static void Install' + name + 'Wrappers(' + name + 'Table* table) {\n'
      content += '  ' + name + '_saved_before_cb = *table;\n'
    if call != '-':
      if call != 'hsa_shut_down':
        content += '  table->' + call + '_fn = ' + call + '_callback;\n'
      else:
        content += '  { void* p = (void*)' + call + '_callback; (void)p; }\n'
    return content

  # generate API name function
  def gen_get_name(self, n, name, call, struct):
    content = ''
    if n == -1:
      content += 'static const char* GetApiName(uint32_t id) {\n'
      content += '  switch (id) {\n'
      return content
    if call != '-':
      content += '    case ' + self.api_id[call] + ': return "' + call + '";\n'
    else:
      content += '  }\n'
      content += '  return "unknown";\n'
      content += '}\n'
    return content

  # generate API code function
  def gen_get_code(self, n, name, call, struct):
    content = ''
    if n == -1:
      content += 'static uint32_t GetApiCode(const char* str) {\n'
      return content
    if call != '-':
      content += '  if (strcmp("' + call + '", str) == 0) return ' + self.api_id[call] + ';\n'
    else:
      content += '  return HSA_API_ID_NUMBER;\n'
      content += '}\n'
    return content

  # generate stream operator
  def gen_out_stream(self, n, name, call, struct):
    content = ''
    if n == -1:
      content += '#ifdef __cplusplus\n'
      content += '#include "hsa_ostream_ops.h"\n'
      content += 'typedef std::pair<uint32_t, hsa_api_data_t> hsa_api_data_pair_t;\n'
      content += 'inline std::ostream& operator<< (std::ostream& out, const hsa_api_data_pair_t& data_pair) {\n'
      content += '  const uint32_t cid = data_pair.first;\n'
      content += '  const hsa_api_data_t& api_data = data_pair.second;\n'
      content += '  switch(cid) {\n'
      return content
    if call != '-':
      content += '    case ' + self.api_id[call] + ': {\n'
      content += '      out << "' + call + '(";\n'
      arg_list = struct['alst']
      if len(arg_list) != 0:
        for ind in range(len(arg_list)):
          arg_var = arg_list[ind]
          arg_val = 'api_data.args.' + call + '.' + arg_var
          if re.search(r'char\* ', struct['astr'][arg_var]):
            content += '      out << "0x" << std::hex << (uint64_t)' + arg_val
          else:
            content += '      out << ' + arg_val
            if call == "hsa_amd_memory_async_copy_rect" and arg_var == "range":
              content += ' << ", ";\n'
              content += '      out << ' + arg_val + '__val'
          '''
          arg_item = struct['tlst'][ind]
          if re.search(r'\(\* ', arg_item): arg_pref = ''
          elif re.search(r'void\* ', arg_item): arg_pref = ''
          elif re.search(r'\*\* ', arg_item): arg_pref = '**'
          elif re.search(r'\* ', arg_item): arg_pref = '*'
          else: arg_pref = ''
          if arg_pref != '':
            content += '      if (' + arg_val + ') out << ' + arg_pref + '(' + arg_val + '); else out << ' + arg_val
          else:
            content += '      out << ' + arg_val
          '''
          if ind < len(arg_list) - 1: content += ' << ", ";\n'
          else: content += ';\n'
      if struct['ret'] != 'void':
        content += '      out << ") = " << api_data.' + struct['ret'] + '_retval;\n'
      else:
        content += '      out << ") = void";\n'
      content += '      break;\n'
      content += '    }\n'
    else:
      content += '    default:\n'
      content += '      out << "ERROR: unknown API";\n'
      content += '      abort();\n'
      content += '  }\n'
      content += '  return out;\n'
      content += '}\n'
      content += '#endif\n'
    return content

#############################################################
# main
# Usage
if len(sys.argv) != 3:
  print ("Usage:", sys.argv[0], " <OUT prefix> <HSA runtime include path>", file=sys.stderr)
  sys.exit(1)
else:
  PREFIX = sys.argv[1] + '/'
  HSA_DIR = sys.argv[2] + '/'

descr = API_DescrParser(H_OUT, HSA_DIR, API_TABLES_H, API_HEADERS_H, LICENSE)

out_file = PREFIX + H_OUT
print ('Generating "' + out_file + '"')
f = open(out_file, 'w')
f.write(descr.h_content[:-1])
f.close()

out_file = PREFIX + CPP_OUT
print ('Generating "' + out_file + '"')
f = open(out_file, 'w')
f.write(descr.cpp_content[:-1])
f.close()
#############################################################
