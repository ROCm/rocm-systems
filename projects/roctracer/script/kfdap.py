#!/usr/bin/python
from __future__ import print_function
import os, sys, re

OUT_H = 'inc/kfd_prof_str.h'
OUT_CPP = 'src/kfd_wrapper.cpp'
API_HEADERS_H = (
  ('HSAKMTAPI', 'hsakmt.h'),
)

LICENSE = \
'/*\n' + \
'Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.\n' + \
'\n' + \
'Permission is hereby granted, free of charge, to any person obtaining a copy\n' + \
'of this software and associated documentation files (the "Software"), to deal\n' + \
'in the Software without restriction, including without limitation the rights\n' + \
'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n' + \
'copies of the Software, and to permit persons to whom the Software is\n' + \
'furnished to do so, subject to the following conditions:\n' + \
'\n' + \
'The above copyright notice and this permission notice shall be included in\n' + \
'all copies or substantial portions of the Software.\n' + \
'\n' + \
'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n' + \
'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n' + \
'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\n' + \
'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n' + \
'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n' + \
'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n' + \
'THE SOFTWARE.\n' + \
'*/\n'

#############################################################
# Error handler
def fatal(module, msg):
  print (module + ' Error: "' + msg + '"', file = sys.stderr)
  sys.exit(1)

# Get next text block
def NextBlock(pos, record):
  if len(record) == 0: return pos

  space_pattern = re.compile(r'(\s+)')
  word_pattern = re.compile(r'([\w\*]+\[*\]*)')
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
        count = count-1
        if count == 0:
          index = index + 1
          break
    if count != 0:
      fatal('NextBlock', "count is not zero (" + str(count) + ")")
    if record[index-1] != ')':
      fatal('NextBlock', "last char is not ')' '" + record[index-1] + "'")
    return index

#############################################################
# API table parser class
class API_TableParser:
  def fatal(self, msg):
    fatal('API_TableParser', msg)

  def __init__(self, header, name, full_fct):
    self.name = name
    self.full_fct = full_fct

    if not os.path.isfile(header):
      self.fatal("file '" + header + "' not found")

    self.inp = open(header, 'r')

    self.beg_pattern = re.compile(name)
    self.end_pattern = re.compile('.*\)\s*;\s*$');
    self.array = []
    self.parse()

  # normalizing a line
  def norm_line(self, line):
    return re.sub(r'^\s+', r' ', line)

  def fix_comment_line(self, line):
    return re.sub(r'\/\/.*', r'', line)

  def remove_ret_line(self, line):
    return re.sub(r'\n', r'', line)

  # check for start record
  def is_start(self, record):
    return self.beg_pattern.match(record)

  # check for end record
  def is_end(self, record):
    return self.end_pattern.match(record)

  # check for declaration entry record
  def is_entry(self, record):
    return re.match(r'^\s*HSAKMTAPI\s*(.*)\s*\((.*)\)', record)

  # parse method
  def parse(self):
    active = 0
    record = "";
    cumulate = 0;
    self.full_fct = {}
    rettype = ""
    prev_line = ""
    for line in self.inp.readlines():
      line = self.norm_line(line)
      line = self.fix_comment_line(line)

      if cumulate == 1: record += " " + line;
      else: record = line;
      if self.is_start(line): rettype = prev_line.strip(); cumulate = 1; prev_line = line; continue;
      if self.is_end(line): record = self.remove_ret_line(record); cumulate = 0; active = 1;
      else: prev_line = line; continue;
      if active != 0:
        m = self.is_entry(record)
        if m:
          mycall_full = rettype  + " " + m.group(1) + ' (' + m.group(2) + ')'
          mycall = m.group(1)
          self.full_fct[mycall] = mycall_full
          self.array.append(mycall)
          rettype = "";
      prev_line = line

#############################################################
# API declaration parser clas
class API_DeclParser:
  def fatal(self, msg):
    fatal('API_DeclParser', msg)

  def __init__(self, header, array, data, full_fct):
    if not os.path.isfile(header):
      self.fatal("file '" + header + "' not found")

    self.inp = open(header, 'r')

    self.end_pattern = re.compile('\)\s*;\s*$')
    self.data = data
    for call in array:
      if call in data:
        self.fatal(call + ' is already found')
      self.parse(call,full_fct)

  # check for start record
  def is_start(self, call, record):
    return re.search('\s*' + call + '\s*\(', record)

  # check for API method record
  def is_api(self, call, record):
    return re.match('\s*' + call + '\s*\(', record)


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
    if args == "void":
      return struct

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
  def parse(self, call, full_fct):
    if call in full_fct:
      self.data[call] = self.get_args(full_fct[call])
    else:
      self.data[call] = self.get_args(call)

#############################################################
# API description parser class
class API_DescrParser:
  def fatal(self, msg):
    fatal('API_DescrParser', msg)

  def __init__(self, out_file, kfd_dir, api_headers, license):
    out_macro = re.sub(r'[\/\.]', r'_', out_file.upper()) + '_'

    self.content_h = ''
    self.content_cpp = ''

    self.api_names = []
    self.api_calls = {}
    self.api_rettypes = set()
    self.api_id = {}

    api_data = {}
    full_fct = {}
    api_list = []
    ns_calls = []

    (name, header) = api_headers[0]
    api = API_TableParser(kfd_dir + header, name, full_fct)
    full_fct = api.full_fct
    api_list = api.array
    self.api_names.append(name)
    self.api_calls[name] = api_list

    for call in api_list:
      if call in api_data:
        self.fatal("call '"  + call + "' is already found")

    API_DeclParser(kfd_dir + header, api_list, api_data, full_fct)

    for call in api_list:
      if not call in api_data:
        # Not-supported functions
        ns_calls.append(call)
      else:
        # API ID map
        self.api_id[call] = 'KFD_API_ID_' + call
        # Return types
        self.api_rettypes.add(api_data[call]['ret'])

    self.api_data = api_data
    self.ns_calls = ns_calls

    self.content_h += "// automatically generated\n\n" + license + '\n'

    self.content_h += "/////////////////////////////////////////////////////////////////////////////\n"
    for call in self.ns_calls:
      self.content_h += '// ' + call + ' was not parsed\n'
    self.content_h += '\n'
    self.content_h += '#ifndef ' + out_macro + '\n'
    self.content_h += '#define ' + out_macro + '\n'

    self.content_h += '\n'

    self.content_h += '#include <dlfcn.h>\n'
    self.content_h += '#include <string.h>\n'
    self.content_h += '#include <roctracer_kfd.h>\n'
    self.content_h += '#include <hsakmt.h>\n'

    self.content_h += '#define PUBLIC_API __attribute__((visibility(\"default\")))\n'

    self.add_section('API ID enumeration', '  ', self.gen_id_enum)
    self.add_section('API arg structure', '    ', self.gen_arg_struct)

    self.content_h += '\n'
    self.content_h += '#if PROF_API_IMPL\n'
    self.content_h += '#include <roctracer_cb_table.h>\n'
    self.content_h += 'namespace roctracer {\n'
    self.content_h += 'namespace kfd_support {\n'

    self.add_section('API get_name function', '    ', self.gen_get_name)
    self.add_section('API get_code function', '  ', self.gen_get_code)

    self.add_section('API intercepting code', '', self.gen_intercept_decl)
    self.add_section('API intercepting code', '', self.gen_intercept)
    self.add_section('API callback functions', '', self.gen_callbacks)

    self.content_h += '\n};};\n'
    self.content_h += '#endif // PROF_API_IMPL\n'

    self.content_cpp += "// automatically generated\n\n" + license + '\n'
    self.content_cpp += "/////////////////////////////////////////////////////////////////////////////\n\n"
    self.content_cpp += '#define PROF_API_IMPL 1\n'
    self.content_cpp += '#include \"kfd_prof_str.h\"\n'

    self.add_section('API output stream', '    ', self.gen_out_stream)
    self.add_section_cpp('API callback fcts', '    ', self.gen_public_api)
    self.content_h += '#endif // ' + out_macro + '_'
    self.content_cpp += '}\n'
    self.content_cpp += '\n'

  # add code section
  def add_section_cpp(self, title, gap, fun):
    n = 0
    self.content_cpp += '\n// section: ' + title + '\n\n'
    fun(-1, '-', '-', {})
    for index in range(len(self.api_names)):
      last = (index == len(self.api_names)-1)
      name = self.api_names[index]

      if n != 0:
        if gap == '': fun(n, name, '-', {})
        self.content_cpp += '\n'
      self.content_cpp += gap + '// block: ' + name + ' API\n'
      for call in self.api_calls[name]:
        fun(n, name, call, self.api_data[call])
        n += 1
    fun(n, '-', '-', {})

  def add_section(self, title, gap, fun):
    n = 0
    self.content_h += '\n// section: ' + title + '\n\n'
    fun(-1, '-', '-', {})
    for index in range(len(self.api_names)):
      last = (index == len(self.api_names)-1)
      name = self.api_names[index]

      if n != 0:
        if gap == '': fun(n, name, '-', {})
        self.content_h += '\n'
      self.content_h += gap + '// block: ' + name + ' API\n'
      for call in self.api_calls[name]:
        fun(n, name, call, self.api_data[call])
        n += 1
    fun(n, '-', '-', {})

  # check if it's an array decl
  def is_arr(self, record):
    return re.match(r'\s*(.*)\s+(.*)\[\]\s*', record)

  # generate API ID enumeration
  def gen_id_enum(self, n, name, call, data):
    if n == -1:
      self.content_h += 'enum kfd_api_id_t {\n'
      return
    if call != '-':
      self.content_h += '  ' + self.api_id[call] + ' = ' + str(n) + ',\n'
    else:
      self.content_h += '\n'
      self.content_h += '  KFD_API_ID_NUMBER = ' + str(n) + ',\n'
      self.content_h += '  KFD_API_ID_ANY = ' + str(n + 1) + ',\n'
      self.content_h += '};\n'

  # generate API args structure
  def gen_arg_struct(self, n, name, call, struct):
    if n == -1:
      self.content_h += 'typedef struct kfd_api_data_s {\n'
      self.content_h += '  uint64_t correlation_id;\n'
      self.content_h += '  uint32_t phase;\n'
      if len(self.api_rettypes) != 0:
        self.content_h += '  union {\n'
        for ret_type in self.api_rettypes:
          if ret_type != 'void':
            self.content_h += '    ' + ret_type + ' ' + ret_type + '_retval;\n'
        self.content_h += '  };\n'
      self.content_h += '  union {\n'
      return
    if call != '-':
      self.content_h += '    struct {\n'
      for (var, item) in struct['astr'].items():
        m = self.is_arr(item)
        if m:
          self.content_h += '      ' + m.group(1)  + '* ' +  m.group(2) + ';\n'
        else:
          self.content_h += '      ' + item + ';\n'
      self.content_h += '    } ' +  call + ';\n'
    else:
      self.content_h += '  } args;\n'
      self.content_h += '} kfd_api_data_t;\n'

  # generate API callbacks
  def gen_callbacks(self, n, name, call, struct):
    if n == -1:
      self.content_h += 'typedef CbTable<KFD_API_ID_NUMBER> cb_table_t;\n'
      self.content_h += 'cb_table_t cb_table;\n'
      self.content_h += '\n'
    if call != '-':
      call_id = self.api_id[call];
      ret_type = struct['ret']
      self.content_h += ret_type + ' ' + call + '_callback(' + struct['args'] + ') {\n'  # 'static '  +
      self.content_h += '  if (' + name + '_table == NULL) intercept_KFDApiTable();\n'
      self.content_h += '  kfd_api_data_t api_data{};\n'
      for var in struct['alst']:
        self.content_h += '  api_data.args.' + call + '.' + var.replace("[]","") + ' = ' + var.replace("[]","") + ';\n'
      self.content_h += '  activity_rtapi_callback_t api_callback_fun = NULL;\n'
      self.content_h += '  void* api_callback_arg = NULL;\n'
      self.content_h += '  cb_table.get(' + call_id + ', &api_callback_fun, &api_callback_arg);\n'
      self.content_h += '  api_data.phase = 0;\n'
      self.content_h += '  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_KFD_API, ' + call_id + ', &api_data, api_callback_arg);\n'
      if ret_type != 'void':
        self.content_h += '  ' + ret_type + ' ret = '
      tmp_str = '  ' + name + '_table->' + call + '_fn(' + ', '.join(struct['alst']) + ');\n'
      self.content_h += tmp_str.replace("[]","")
      if ret_type != 'void':
        self.content_h += '  api_data.' + ret_type + '_retval = ret;\n'
      self.content_h += '  api_data.phase = 1;\n'
      self.content_h += '  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_KFD_API, ' + call_id + ', &api_data, api_callback_arg);\n'
      if ret_type != 'void':
        self.content_h += '  return ret;\n'
      self.content_h += '}\n'

  # Generates API intercepting table struct definition
  def gen_intercept_decl(self, n, name, call, struct):
    if n > 0 and call == '-':
      self.content_h += '} HSAKMTAPI_table_t;\n' #was HSAKMTAPI_table_t
    if n == 0 or (call == '-' and name != '-'):
      self.content_h += 'typedef struct {\n'
    if call != '-':
      self.content_h += '  decltype(' + call + ')* ' + call + '_fn;\n'

  # generate API intercepting code
  def gen_intercept(self, n, name, call, struct):
    if n > 0 and call == '-':
      self.content_h += '};\n'
    if n == 0 or (call == '-' and name != '-'):
      self.content_h += name + '_table_t* ' + name + '_table = NULL;\n'
      self.content_h += 'void intercept_' + 'KFDApiTable' + '(void) {\n'
      self.content_h += '  ' + name + '_table = new ' + name + '_table_t{}' + ';\n'

    if call != '-':
      self.content_h += '  typedef decltype(' + name + '_table_t::' + call + '_fn) ' + call + '_t;\n'
      self.content_h += '  ' + name + '_table->' + call + '_fn = (' + call + '_t)' + 'dlsym(RTLD_NEXT,\"'  + call + '\");\n'

  # generate API name function
  def gen_get_name(self, n, name, call, struct):
    if n == -1:
      self.content_h += 'const char* GetApiName(const uint32_t& id) {\n' #static
      self.content_h += '  switch (id) {\n'
      return
    if call != '-':
      self.content_h += '    case ' + self.api_id[call] + ': return "' + call + '";\n'
    else:
      self.content_h += '  }\n'
      self.content_h += '  return "unknown";\n'
      self.content_h +=  '}\n'

  # generate API code function
  def gen_get_code(self, n, name, call, struct):
    if n == -1:
      self.content_h += 'uint32_t GetApiCode(const char* str) {\n' # static
      return
    if call != '-':
      self.content_h += '  if (strcmp("' + call + '", str) == 0) return ' + self.api_id[call] + ';\n'
    else:
      self.content_h += '  return KFD_API_ID_NUMBER;\n'
      self.content_h += '}\n'

  # generate stream operator
  def gen_out_stream(self, n, name, call, struct):
    if n == -1:
      self.content_h += '#ifdef __cplusplus\n'
      self.content_h += 'typedef std::pair<uint32_t, kfd_api_data_t> kfd_api_data_pair_t;\n'
      self.content_h += 'inline std::ostream& operator<< (std::ostream& out, const kfd_api_data_pair_t& data_pair) {\n'
      self.content_h += '  const uint32_t cid = data_pair.first;\n'
      self.content_h += '  const kfd_api_data_t& api_data = data_pair.second;\n'
      self.content_h += '  switch(cid) {\n'
      return
    if call != '-':
      self.content_h += '    case ' + self.api_id[call] + ': {\n'
      self.content_h += '      out << "' + call + '(";\n'
      arg_list = struct['alst']
      if len(arg_list) != 0:
        for ind in range(len(arg_list)):
          arg_var = arg_list[ind]
          arg_val = 'api_data.args.' + call + '.' + arg_var
          if re.search(r'MemFlags',arg_var):
            continue
          self.content_h += '      typedef decltype(' + arg_val.replace("[]","") + ') arg_val_type_t' + str(ind) + ';\n'
          self.content_h += '      roctracer::kfd_support::output_streamer<arg_val_type_t' + str(ind) + '>::put(out, ' + arg_val.replace("[]","") + ')'
          if ind < len(arg_list)-1: self.content_h += ' << ", ";\n'
          else: self.content_h += ';\n'
      if struct['ret'] != 'void':
        self.content_h += '      out << ") = " << api_data.' + struct['ret'] + '_retval;\n'
      else:
        self.content_h += '      out << ") = void";\n'
      self.content_h += '      break;\n'
      self.content_h += '    }\n'
    else:
      self.content_h += '    default:\n'
      self.content_h += '      out << "ERROR: unknown API";\n'
      self.content_h += '      abort();\n'
      self.content_h += '  }\n'
      self.content_h += '  return out;\n'
      self.content_h += '}\n'
      self.content_h += '#endif\n'
      self.content_cpp += 'inline std::ostream& operator<< (std::ostream& out, const HsaMemFlags& v) { out << "HsaMemFlags"; return out; }\n'

  # generate PUBLIC_API for all API fcts
  def gen_public_api(self, n, name, call, struct):
    if n == -1:
      self.content_cpp += 'extern "C" {\n'
      self.content_cpp += 'PUBLIC_API bool RegisterApiCallback(uint32_t op, void* callback, void* user_data) {\n';
      self.content_cpp += '    roctracer::kfd_support::cb_table.set(op, reinterpret_cast<activity_rtapi_callback_t>(callback), user_data);\n';
      self.content_cpp += '    return true;\n';
      self.content_cpp += '}\n';
      self.content_cpp += 'PUBLIC_API bool RemoveApiCallback(uint32_t op) {\n'
      self.content_cpp += '    roctracer::kfd_support::cb_table.set(op, NULL, NULL);\n';
      self.content_cpp += '    return true;\n';
      self.content_cpp += '}\n\n';

    if call != '-' and call != 'hsaKmtCloseKFD' and call != 'hsaKmtOpenKFD':
      self.content_cpp += 'PUBLIC_API ' + struct['ret'] + " " + call + '(' + struct['args'] + ') { return roctracer::kfd_support::' + call + '_callback('
      for i in range(0,len(struct['alst'])):
        if i == (len(struct['alst'])-1):
          self.content_cpp += struct['alst'][i].replace("[]","")
        else:
          self.content_cpp += struct['alst'][i].replace("[]","") + ', '
      self.content_cpp +=  ');} \n'

#############################################################
# main
# Usage
if len(sys.argv) != 3:
  print ("Usage:", sys.argv[0], " <OUT prefix> <KFD include path>", file = sys.stderr)
  sys.exit(1)
else:
  PREFIX = sys.argv[1] + '/'
  KFD_DIR = sys.argv[2] + '/'

descr = API_DescrParser(OUT_H, KFD_DIR, API_HEADERS_H, LICENSE)

out_file = PREFIX + OUT_H
print ('Generating "' + out_file + '"')
f = open(out_file, 'w')
f.write(descr.content_h[:-1])
f.close()

out_file = PREFIX + OUT_CPP
print ('Generating "' + out_file + '"')
f = open(out_file, 'w')
f.write(descr.content_cpp[:-1])
f.close()

#############################################################
