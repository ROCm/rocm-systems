#!/usr/bin/python
import os, sys, re

OUT='inc/hsa_prof_str.h'
API_TABLES_H = 'hsa_api_trace.h'
API_HEADERS_H = ( 
  ('CoreApiTable', 'hsa.h'),
  ('AmdExtTable', 'hsa_ext_amd.h'),
  ('ImageExtTable', 'hsa_ext_image.h'),
  ('AmdExtTable', API_TABLES_H),
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
  print >>sys.stderr, module + ' Error: "' + msg + '"'
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

    self.beg_pattern = re.compile('^\s*struct\s+' + name + '\s*{\s*$')
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

  def __init__(self, out_file, hsa_dir, api_table_h, api_headers, license):
    out_macro = re.sub(r'[\/\.]', r'_', out_file.upper()) + '_'

    self.content = ''
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

    self.content += "// automatically generated\n\n" + license + '\n'
    
    self.content += "/////////////////////////////////////////////////////////////////////////////\n"
    self.content += "//\n"
    self.content += "// HSA API tracing primitives\n"
    self.content += "//\n"
    for (name, header) in api_headers:
      self.content += "// '" + name + "', header '" + header + "', " + str(len(self.api_calls[name])) + ' funcs\n'
    for call in self.ns_calls:
      self.content += '// ' + call + ' was not parsed\n'
    self.content += "//\n"
    self.content += "/////////////////////////////////////////////////////////////////////////////\n"
    self.content += '\n'
    self.content += '#ifndef ' + out_macro + '\n'
    self.content += '#define ' + out_macro + '\n'

    self.add_section('API ID enumeration', '  ', self.gen_id_enum)
    self.add_section('API arg structure', '    ', self.gen_arg_struct)

    self.content += '\n'
    self.content += '#if PROF_API_IMPL\n'
    self.content += 'namespace roctracer {\n'
    self.content += 'namespace hsa_support {\n'
    self.add_section('API callback functions', '', self.gen_callbacks)
    self.add_section('API intercepting code', '', self.gen_intercept)
    self.add_section('API get_name function', '    ', self.gen_get_name)
    self.add_section('API get_code function', '  ', self.gen_get_code)
    self.content += '\n};};\n'
    self.content += '#endif // PROF_API_IMPL\n'

    self.add_section('API output stream', '    ', self.gen_out_stream)
    self.content += '\n'

    self.content += '#endif // ' + out_macro

  # add code section
  def add_section(self, title, gap, fun):
    n = 0
    self.content +=  '\n// section: ' + title + '\n\n'
    fun(-1, '-', '-', {})
    for index in range(len(self.api_names)):
      last = (index == len(self.api_names) - 1)
      name = self.api_names[index]
      if n != 0:
        if gap == '': fun(n, name, '-', {})
        self.content += '\n'
      self.content += gap + '// block: ' + name + ' API\n'
      for call in self.api_calls[name]:
        fun(n, name, call, self.api_data[call])
        n += 1
    fun(n, '-', '-', {})

  # generate API ID enumeration
  def gen_id_enum(self, n, name, call, data):
    if n == -1:
      self.content += 'enum hsa_api_id_t {\n'
      return
    if call != '-':
      self.content += '  ' + self.api_id[call] + ' = ' + str(n) + ',\n'
    else:
      self.content += '\n'
      self.content += '  HSA_API_ID_NUMBER = ' + str(n) + ',\n'
      self.content += '  HSA_API_ID_ANY = ' + str(n + 1) + ',\n'
      self.content += '};\n'
    
  # generate API args structure
  def gen_arg_struct(self, n, name, call, struct):
    if n == -1:
      self.content += 'struct hsa_api_data_t {\n'
      self.content += '  uint64_t correlation_id;\n'
      self.content += '  uint32_t phase;\n'
      self.content += '  union {\n'
      for ret_type in self.api_rettypes:
        self.content += '    ' + ret_type + ' ' + ret_type + '_retval;\n'
      self.content += '  };\n'
      self.content += '  union {\n'
      return
    if call != '-':
      self.content +=   '    struct {\n'
      for (var, item) in struct['astr'].items():
        self.content += '      ' + item + ';\n'
      self.content +=   '    } ' + call + ';\n'
    else:
      self.content += '  } args;\n'
      self.content += '};\n'
    
  # generate API callbacks
  def gen_callbacks(self, n, name, call, struct):
    if n == -1:
      self.content += 'typedef CbTable<HSA_API_ID_NUMBER> cb_table_t;\n'
      self.content += 'extern cb_table_t cb_table;\n'
      self.content += '\n'
    if call != '-':
      call_id = self.api_id[call];
      ret_type = struct['ret']
      self.content += 'static ' + ret_type + ' ' + call + '_callback(' + struct['args'] + ') {\n'
      self.content += '  hsa_api_data_t api_data{};\n'
      for var in struct['alst']:
        item = struct['astr'][var];
        if re.search(r'char\* ', item):
          self.content += '  api_data.args.' + call + '.' + var + ' = ' + '(' + var + ' != NULL) ? strdup(' + var + ')' + ' : NULL;\n'
        else:
          self.content += '  api_data.args.' + call + '.' + var + ' = ' + var + ';\n'
      self.content += '  activity_rtapi_callback_t api_callback_fun = NULL;\n'
      self.content += '  void* api_callback_arg = NULL;\n'
      self.content += '  cb_table.get(' + call_id + ', &api_callback_fun, &api_callback_arg);\n'
      self.content += '  api_data.phase = 0;\n'
      self.content += '  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, ' + call_id + ', &api_data, api_callback_arg);\n'
      if ret_type != 'void':
        self.content += '  ' + ret_type + ' ret ='
      self.content += '  ' + name + '_saved.' + call + '_fn(' + ', '.join(struct['alst']) + ');\n'
      if ret_type != 'void':
        self.content += '  api_data.' + ret_type + '_retval = ret;\n'
      self.content += '  api_data.phase = 1;\n'
      self.content += '  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, ' + call_id + ', &api_data, api_callback_arg);\n'
      if ret_type != 'void':
        self.content += '  return ret;\n'
      self.content += '}\n'

  # generate API intercepting code
  def gen_intercept(self, n, name, call, struct):
    if n > 0 and call == '-':
      self.content += '};\n'
    if n == 0 or (call == '-' and name != '-'):
      self.content += 'static void intercept_' + name + '(' + name + '* table) {\n'
      self.content += '  ' + name + '_saved = *table;\n'
    if call != '-':
      if call != 'hsa_shut_down':
        self.content += '  table->' + call + '_fn = ' + call + '_callback;\n'
      else:
        self.content += '  { void* p = (void*)' + call + '_callback; (void)p; }\n'

  # generate API name function
  def gen_get_name(self, n, name, call, struct):
    if n == -1:
      self.content += 'static const char* GetApiName(const uint32_t& id) {\n'
      self.content += '  switch (id) {\n'
      return
    if call != '-':
      self.content += '    case ' + self.api_id[call] + ': return "' + call + '";\n'
    else:
      self.content += '  }\n'
      self.content += '  return "unknown";\n'
      self.content += '}\n'

  # generate API code function
  def gen_get_code(self, n, name, call, struct):
    if n == -1:
      self.content += 'static uint32_t GetApiCode(const char* str) {\n'
      return
    if call != '-':
      self.content += '  if (strcmp("' + call + '", str) == 0) return ' + self.api_id[call] + ';\n'
    else:
      self.content += '  return HSA_API_ID_NUMBER;\n'
      self.content += '}\n'

  # generate stream operator
  def gen_out_stream(self, n, name, call, struct):
    if n == -1:
      self.content += 'typedef std::pair<uint32_t, hsa_api_data_t> hsa_api_data_pair_t;\n'
      self.content += 'inline std::ostream& operator<< (std::ostream& out, const hsa_api_data_pair_t& data_pair) {\n'
      self.content += '  const uint32_t cid = data_pair.first;\n'
      self.content += '  const hsa_api_data_t& api_data = data_pair.second;\n'
      self.content += '  switch(cid) {\n'
      return
    if call != '-':
      self.content += '    case ' + self.api_id[call] + ': {\n'
      self.content += '      out << "' + call + '(";\n'
      arg_list = struct['alst']
      if len(arg_list) != 0:
        for ind in range(len(arg_list)):
          arg_var = arg_list[ind]
          arg_val = 'api_data.args.' + call + '.' + arg_var
          self.content += '      typedef decltype(' + arg_val + ') arg_val_type_t' + str(ind) + ';\n'
          self.content += '      roctracer::hsa_support::output_streamer<arg_val_type_t' + str(ind) + '>::put(out, ' + arg_val + ')'
          '''
          arg_item = struct['tlst'][ind]
          if re.search(r'\(\* ', arg_item): arg_pref = ''
          elif re.search(r'void\* ', arg_item): arg_pref = ''
          elif re.search(r'\*\* ', arg_item): arg_pref = '**'
          elif re.search(r'\* ', arg_item): arg_pref = '*'
          else: arg_pref = ''
          if arg_pref != '':
            self.content += '      if (' + arg_val + ') out << ' + arg_pref + '(' + arg_val + '); else out << ' + arg_val
          else:
            self.content += '      out << ' + arg_val
          '''
          if ind < len(arg_list) - 1: self.content += ' << ", ";\n'
          else: self.content += ';\n'
      if struct['ret'] != 'void':
        self.content += '      out << ") = " << api_data.' + struct['ret'] + '_retval;\n'
      else:
        self.content += '      out << ") = void";\n'
      self.content += '      break;\n'
      self.content += '    }\n'
    else:
      self.content += '    default:\n'
      self.content += '      out << "ERROR: unknown API";\n'
      self.content += '      abort();\n'
      self.content += '  }\n'
      self.content += '  return out;\n'
      self.content += '}\n'

#############################################################
# main
# Usage
if len(sys.argv) != 3:
  print >>sys.stderr, "Usage:", sys.argv[0], " <rocTracer root> <HSA runtime include path>"
  sys.exit(1)
else:
  ROOT = sys.argv[1] + '/'
  HSA_DIR = sys.argv[2] + '/'

descr = API_DescrParser(OUT, HSA_DIR, API_TABLES_H, API_HEADERS_H, LICENSE)

out_file = ROOT + OUT
print 'Generating "' + out_file + '"'
f = open(out_file, 'w')
f.write(descr.content[:-1])
f.close()
#############################################################
