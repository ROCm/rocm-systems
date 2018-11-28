#!/usr/bin/python
import os, sys, re

HEADER = "hip_cbstr.h"
REC_MAX_LEN = 1024

#############################################################
# Filling API map with API call name and args
def filtr_api_args(args_str):
  args_str = re.sub(r'^\s*', r'', args_str);
  args_str = re.sub(r'\s*$', r'', args_str);
  args_str = re.sub(r'\s*,\s*', r',', args_str);
  args_str = re.sub(r'\s+', r' ', args_str);
  args_str = re.sub(r'void \*', r'void* ', args_str);
  args_str = re.sub(r'(enum|struct) ', '', args_str);
  return args_str

def list_api_args(args_str):
  args_list = []
  for arg_pair in args_str.split(','):
    arg_pair = re.sub(r'\s+=\s+\S+$','', arg_pair);
    m = re.match("^(.*)\s(\S+)$", arg_pair);
    if m:
      arg_type = m.group(1)
      arg_name = m.group(2)
#      m = re.match("^(.*_t)\s(.*)$", arg_type)
#      if m:
#        arg_type = m.group(1)
#        arg_name = m.group(2)
      args_list.append((arg_type, arg_name))
  return args_list;

def filtr_api_types(args_str):
  args_str = filtr_api_args(args_str)
  args_list = list_api_args(args_str)
  types_str = ''
  for arg_tuple in args_list:
    types_str += arg_tuple[0] + ', '
  return types_str

def norm_api_types(types_str):
  types_str = re.sub(r'uint32_t,', r'unsigned int,', types_str)
  types_str = re.sub(r'unsigned,', r'unsigned int,', types_str)
  return types_str

def filtr_api_opts(args_str):
  args_str = filtr_api_args(args_str)
  args_list = list_api_args(args_str)
  opts_list = []
  for arg_tuple in args_list:
    opts_list.append(arg_tuple[1])
  return opts_list

def fill_api_map(out, api_name, args_str):
  args_str = filtr_api_args(args_str)
  out[api_name + '.a'] = args_str
  out[api_name] = list_api_args(args_str)

def patch_args(api_opts, eta_opts, content):
  api_opts_list = api_opts.split(',');
  eta_opts_list = eta_opts.split(',');
  length = len(api_opts_list)
  for index in range(0, length):
    content = re.sub(' ' + api_opts_list[index], ' ' + eta_opts_list[index], content)
  return content
#############################################################
# Parsing API header
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
def parse_api(inp_file, out):
  beg_pattern = re.compile("^hipError_t");
  api_pattern = re.compile("^hipError_t\s+([^\(]+)\(([^\)]*)\)");
  end_pattern = re.compile("Texture");

  inp = open(inp_file, 'r')

  found = 0
  record = ""
  line_num = -1 

  for line in inp.readlines():
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      print "Error: bad record \"" + record + "\"\nfile '" + inp_file + ", line (" + str(line_num) + ")"
      break;

    if beg_pattern.match(record): found = 1

    if found != 0:
      record = re.sub("\s__dparm\([^\)]*\)", '', record);
      m = api_pattern.match(record)
      if m:
        found = 0
        if end_pattern.search(record): break
        out[m.group(1)] = filtr_api_args(m.group(2))
      else: continue

    record = ""

  inp.close()
#############################################################
# Patching API implementation
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
#    HIP_INIT_CB(hipSetupArgument, arg, size, offset);
def patch_content(inp_file, api_map, out):
  beg_pattern = re.compile("^hipError_t");
  api_pattern = re.compile("^hipError_t\s+([^\(]+)\(([^\)]*)\)\s*{");
  target_pattern = re.compile("^(\s*HIP_INIT[^\(]*)(_API\()(.*)\);\s*$");

  inp = open(inp_file, 'r')

  api_name = ""
  api_valid = 0
  api_valid_always = 1

  content = ''
  sub_content = ''
  record = ''
  line_num = -1 
  found = 0

  for line in inp.readlines():
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      print "Error: bad record \"" + record + "\"\nfile '" + inp_file + ", line (" + str(line_num) + ")"
      break;

    if beg_pattern.match(record): found = 1

    if found != 0:
      record = re.sub("\s__dparm\([^\)]*\)", '', record);
      m = api_pattern.match(record)
      if m:
        found = 0
        api_name = m.group(1);
        if api_name in api_map:
          api_args = filtr_api_args(m.group(2))
          eta_args = api_map[api_name]
          api_types = filtr_api_types(api_args)
          eta_types = filtr_api_types(eta_args)
          if api_types != eta_types:
            api_types = norm_api_types(api_types)
            eta_types = norm_api_types(eta_types)
          if api_types == eta_types:
            if api_name in out:
              print "Error: API redefined \"" + api_name + "\", record \"" + record + "\"\nfile '" + inp_file + ", line (" + str(line_num) + ")"
              sys.exit(1)
            api_valid = 1
            out[api_name] = filtr_api_opts(api_args)
          elif not api_name in out:
            api_diff = '\t\t' + inp_file + " line(" + str(line_num) + ")\n\t\tapi: " + api_types + "\n\t\teta: " + eta_types
            print "\t" + api_name + ':\n' + api_diff + '\n'

        content += sub_content
        sub_content = ''
      else:
        sub_content += line
        continue

    if (api_valid_always == 1) || (api_valid == 1):
      m = target_pattern.match(line)
      if m:
        api_valid = 0
        if not re.search("_CB_API\(", line):
          print (api_name);
          api_label = api_name
          if m.group(3) != "": api_label += ', '
          line = m.group(1) + '_CB' + m.group(2) + api_label + m.group(3) + ");\n"

    content += line
    record = ""

  inp.close()

  if len(out) != 0:
    return content
  else:
    return '' 

# srcs path walk
def patch_src(api_map, src_path, src_patt, out):
  pattern = re.compile(src_patt)
  src_path = re.sub(r'\s', '', src_path)
  for src_dir in src_path.split(':'):
    print "Patching " + src_dir + " for '" + src_patt + "'"
    for root, dirs, files in os.walk(src_dir):
      for fnm in files:
        if pattern.search(fnm):
          file = root + '/' + fnm
          print "\t" + file
          content = patch_content(file, api_map, out);
          if content != '':
            f = open(file, 'w')
            f.write(content)
            f.close()
#############################################################
# main
# Usage
if (len(sys.argv) < 2):
  print >>sys.stderr, "Usage:", sys.argv[0], " <input HIP API .h file> [patched srcs path]"
  sys.exit(1)

# API header file given as an argument
api_hfile = sys.argv[1]
if not os.path.isfile(api_hfile):
  print >>sys.stderr, "Error: input file '" + api_hfile + "' not found"
  sys.exit(1)

# API declaration map
api_map = {}
# API options map
opts_map = {}

# Parsing API header
parse_api(api_hfile, api_map)

# Patching API implementation sources
# Sources path is given as an argument
if len(sys.argv) == 3:
  src_path = sys.argv[2]
  src_patt = "\.cpp$"
  patch_src(api_map, src_path, src_patt, opts_map)

# Converting api map to map of lists
for name in api_map.keys():
  args_str = api_map[name];

# Printing not found APIs
if len(opts_map) != 0:
  for name in api_map.keys():
    args_str = api_map[name];
    api_map[name] = list_api_args(args_str)
    if not name in opts_map:
      print "Not found: " + name
#############################################################
# Generating the header
#api_map['hipLaunchKernel'] = [
#  ('void*', 'kernel'),
#  ('hipStream_t', 'stream')
#]
#api_map['hipKernel'] = [
#  ('const char*', 'name'),
#  ('uint64_t', 'start'),
#  ('uint64_t', 'end')
#]

f = open(HEADER, 'w')
f.write('// automatically generated sources\n')
f.write('#ifndef _HIP_CBSTR_H\n');
f.write('#define _HIP_CBSTR_H\n');
f.write('#include <sstream>\n');
f.write('#include <string>\n');

# Generating the callbacks function type
f.write('\n// HIP API callbacks function type\n\
struct hip_cb_data_t;\n\
struct hip_act_record_t;\n\
typedef void (*hip_cb_fun_t)(uint32_t domain, uint32_t cid, const void* data, void* arg);\n\
typedef void (*hip_cb_act_t)(uint32_t cid, hip_act_record_t** record, const void* data, void* arg);\n\
typedef void (*hip_cb_async_t)(uint32_t op_id, void* record, void* arg);\n\
')

# Generating the callbacks ID enumaration
f.write('\n// HIP API callbacks ID enumaration\n')
f.write('enum hip_cb_id_t {\n')
cb_id = 0
for name in api_map.keys():
  f.write('  HIP_API_ID_' + name + ' = ' + str(cb_id) + ',\n')
  cb_id += 1
f.write('  HIP_API_ID_NUMBER = ' + str(cb_id) + ',\n')
f.write('};\n')

# Generating the callbacks ID enumaration
f.write('\n// Return HIP API string\n')
f.write('static const char* hip_api_name(const uint32_t& id) {\n')
f.write('  switch(id) {\n')
for name in api_map.keys():
  f.write('    case HIP_API_ID_' + name + ': return "' +  name + '";\n')
f.write('  };\n')
f.write('  return "unknown";\n')
f.write('};\n')

# Generating the callbacks data structure
f.write('\n// HIP API callbacks data structure\n')
f.write(
'struct hip_cb_data_t {\n' +
'  uint64_t correlation_id;\n' +
'  uint32_t phase;\n' +
'  union {\n'
)
for name, args in api_map.items():
  if len(args) != 0:
    f.write('    struct {\n')
    for arg_tuple in args:
      f.write('      ' + arg_tuple[0] + ' ' + arg_tuple[1] + ';\n')
    f.write('    } ' + name + ';\n')
f.write(
'  } args;\n' +
'};\n'
)

# Generating the callbacks args data filling macros
f.write('\n// HIP API callbacks args data filling macros\n')
for name, args in api_map.items():
  f.write('#define INIT_' + name + '_CB_ARGS_DATA(cb_data) { \\\n')
  opts_list = []
  if name in opts_map:
    opts_list = opts_map[name]
  for ind in range(0, len(args)):
    arg_tuple = args[ind]
    arg_type = arg_tuple[0]
    fld_name = arg_tuple[1]
    arg_name = arg_tuple[1]
    if len(opts_list) != 0:
      arg_name = opts_list[ind]
    f.write('  cb_data.args.' + name + '.' + fld_name + ' = (' + arg_type + ')' + arg_name + '; \\\n')
  f.write('};\n')
f.write('#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)\n')

# Generating the method for the API string, name and parameters
f.write('\n')
f.write('#if 0\n')
f.write('// HIP API string method, method name and parameters\n')
f.write('const char* hipApiString(hip_cb_id_t id, const hip_cb_data_t* data) {\n')
f.write('  std::ostringstream oss;\n')
f.write('  switch (id) {\n')
for name, args in api_map.items():
  f.write('    case HIP_API_ID_' + name + ':\n')
  f.write('      oss << "' + name + '("')
  for ind in range(0, len(args)):
    arg_tuple = args[ind]
    arg_name = arg_tuple[1]
    if ind != 0: f.write(' << ","')
    f.write('\n          << " ' + arg_name  + '=" << data->args.' + name + '.' + arg_name)
  f.write('\n          << ")";\n')
  f.write('    break;\n')
f.write('    default: oss << "unknown";\n')
f.write('  };\n')
f.write('  return strdup(oss.str().c_str());\n')
f.write('};\n')
f.write('#endif\n')

# Generating the activity record type
f.write('\
\n\
// HIP API activity record type\n\
// Base record type\n\
struct hip_act_record_t {\n\
  uint32_t domain;                                                    // activity domain id\n\
  uint32_t op_id;                                                     // operation id, dispatch/copy/barrier\n\
  uint32_t activity_kind;                                             // activity kind\n\
  uint64_t correlation_id;                                            // activity correlation ID\n\
  uint64_t begin_ns;                                                  // host begin timestamp, nano-seconds\n\
  uint64_t end_ns;                                                    // host end timestamp, nano-seconds\n\
};\n\
// Async record type\n\
struct hip_async_record_t : hip_act_record_t {\n\
  int device_id;\n\
  uint64_t stream_id;\n\
};\n\
// Dispatch record type\n\
struct hip_dispatch_record_t : hip_async_record_t {};\n\
// Barrier record type\n\
struct hip_barrier_record_t : hip_async_record_t {};\n\
// Memcpy record type\n\
struct hip_copy_record_t : hip_async_record_t {\n\
  size_t bytes;\n\
};\n\
// Generic async operation record\n\
typedef hip_copy_record_t hip_ops_record_t;\n\
')

# # Generating the callbacks table
# f.write('\n// HIP API callbacks table\n')
# f.write('\
# struct hip_cb_table_t {\n\
#   struct { hip_cb_fun_t act; hip_cb_fun_t fun; void* arg; } arr[HIP_API_ID_NUMBER];\n\
# };\n\
# #define HIP_CALLBACKS_TABLE hip_cb_table_t HIP_API_callbacks_table{};\n\
# ')
# f.write('\
# inline bool HIP_SET_ACTIVITY(uint32_t id, hip_cb_fun_t fun, void* arg = NULL) {\n\
#   (void)arg;\n\
#   extern hip_cb_table_t HIP_API_callbacks_table;\n\
#   if (id < HIP_API_ID_NUMBER) {\n\
#     HIP_API_callbacks_table.arr[id].act = fun;\n\
#     return true;\n\
#   }\n\
#   return false;\n\
# }\n')
# f.write('\
# inline bool HIP_SET_CALLBACK(uint32_t id, hip_cb_fun_t fun, void* arg) {\n\
#   extern hip_cb_table_t HIP_API_callbacks_table; \n\
#   if (id < HIP_API_ID_NUMBER) {\n\
#     HIP_API_callbacks_table.arr[id].fun = fun;\n\
#     HIP_API_callbacks_table.arr[id].arg = arg;\n\
#     return true;\n\
#   }\n\
#   return false;\n\
# }\n')
# 
# # Generating the callback spawning class
# f.write('\n// HIP API callbacks spawning class macro\n\
# #define CB_SPAWNER_OBJECT(cb_id) \\\n\
#   class api_callbacks_spawner_t { \\\n\
#    public: \\\n\
#     api_callbacks_spawner_t(hip_cb_data_t& cb_data) : cb_data_(cb_data) { \\\n\
#       hip_cb_id_t id = HIP_API_ID_##cb_id; \\\n\
#       cb_data_.id = id; \\\n\
#       cb_data_.correlation_id = UINT_MAX; \\\n\
#       cb_data_.name = #cb_id; \\\n\
#       extern const hip_cb_table_t* getApiCallbackTabel(); \\\n\
#       const hip_cb_table_t* cb_table = getApiCallbackTabel(); \\\n\
#       cb_act_ = cb_table->arr[id].act; \\\n\
#       cb_fun_ = cb_table->arr[id].fun; \\\n\
#       cb_arg_ = cb_table->arr[id].arg; \\\n\
#       cb_data_.on_enter = true; \\\n\
#       if (cb_act_ != NULL) cb_act_(&cb_data_, NULL); \\\n\
#       if (cb_fun_ != NULL) cb_fun_(&cb_data_, cb_arg_); \\\n\
#     } \\\n\
#     ~api_callbacks_spawner_t() { \\\n\
#       cb_data_.on_enter = false; \\\n\
#       if (cb_act_ != NULL) cb_act_(&cb_data_, NULL); \\\n\
#       if (cb_fun_ != NULL) cb_fun_(&cb_data_, cb_arg_); \\\n\
#     } \\\n\
#    private: \\\n\
#     hip_cb_data_t& cb_data_; \\\n\
#     hip_cb_fun_t cb_act_; \\\n\
#     hip_cb_fun_t cb_fun_; \\\n\
#     void* cb_arg_; \\\n\
#   }; \\\n\
#   hip_cb_data_t cb_data{}; \\\n\
#   INIT_CB_ARGS_DATA(cb_id, cb_data); \\\n\
#   api_callbacks_spawner_t api_callbacks_spawner(cb_data); \n\
# ')

f.write('#endif  // _HIP_CBSTR\n');

print "Header '" + HEADER + "' is generated"
#############################################################
