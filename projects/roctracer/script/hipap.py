#!/usr/bin/python
import os, sys, re

HEADER = "hip_prof_str.h"
REC_MAX_LEN = 1024

#############################################################
# Normalizing API arguments
def filtr_api_args(args_str):
  args_str = re.sub(r'^\s*', r'', args_str);
  args_str = re.sub(r'\s*$', r'', args_str);
  args_str = re.sub(r'\s*,\s*', r',', args_str);
  args_str = re.sub(r'\s+', r' ', args_str);
  args_str = re.sub(r'void \*', r'void* ', args_str);
  args_str = re.sub(r'(enum|struct) ', '', args_str);
  return args_str

# Normalizing types
def norm_api_types(type_str):
  type_str = re.sub(r'uint32_t', r'unsigned int', type_str)
  type_str = re.sub(r'^unsigned$', r'unsigned int', type_str)
  return type_str

# Creating a list of arguments [(type, name), ...]
def list_api_args(args_str):
  args_str = filtr_api_args(args_str)
  args_list = []
  if args_str != '':
    for arg_pair in args_str.split(','):
      if arg_pair == 'void': continue
      arg_pair = re.sub(r'\s*=\s*\S+$','', arg_pair);
      m = re.match("^(.*)\s(\S+)$", arg_pair);
      if m:
        arg_type = norm_api_types(m.group(1))
        arg_name = m.group(2)
        args_list.append((arg_type, arg_name))
      else:
        print "Bad args: args_str: '" + args_str + "' arg_pair: '" + arg_pair + "'"
        sys.exit(1)
  return args_list;

# Creating arguments string "type0, type1, ..."
def filtr_api_types(args_str):
  args_list = list_api_args(args_str)
  types_str = ''
  for arg_tuple in args_list:
    types_str += arg_tuple[0] + ', '
  return types_str

# Creating options list [name0, name1, ...]
def filtr_api_opts(args_str):
  args_list = list_api_args(args_str)
  opts_list = []
  for arg_tuple in args_list:
    opts_list.append(arg_tuple[1])
  return opts_list
#############################################################
# Parsing API header
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
def parse_api(inp_file, out):
  beg_pattern = re.compile("^hipError_t");
  api_pattern = re.compile("^hipError_t\s+([^\(]+)\(([^\)]*)\)");
  end_pattern = re.compile("Texture");
  hidden_pattern = re.compile(r'__attribute__\(\(visibility\("hidden"\)\)\)')
  nms_open_pattern = re.compile(r'namespace hip_impl {')
  nms_close_pattern = re.compile(r'}')

  inp = open(inp_file, 'r')

  found = 0
  hidden = 0
  nms_level = 0;
  record = ""
  line_num = -1 

  for line in inp.readlines():
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      print "Error: bad record \"" + record + "\"\nfile '" + inp_file + ", line (" + str(line_num) + ")"
      break;

    if beg_pattern.match(record) and (hidden == 0) and (nms_level == 0): found = 1

    if found != 0:
      record = re.sub("\s__dparm\([^\)]*\)", '', record);
      m = api_pattern.match(record)
      if m:
        found = 0
        if end_pattern.search(record): break
        out[m.group(1)] = m.group(2)
      else: continue

    hidden = 0
    if hidden_pattern.match(line): hidden = 1
#    print "> " + str(hidden) + ": " + line

    if nms_open_pattern.match(line): nms_level += 1
    if (nms_level > 0) and nms_close_pattern.match(line): nms_level -= 1
    if nms_level < 0:
      print "Error: nms level < 0"
      sys.exit(1)

    record = ""

  inp.close()
#############################################################
# Patching API implementation
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
#    HIP_INIT_CB(hipSetupArgument, arg, size, offset);
# inp_file - input implementation source file
# api_map - input public API map [<api name>] => <api args>
# out - output map  [<api name>] => <api args>
def patch_content(inp_file, api_map, out):
  # API definition begin pattern
  beg_pattern = re.compile("^(hipError_t|const char\s*\*\s+[_\w]+\()");
  # API definition complete pattern
  api_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(([^\)]*)\)\s*{");
  # API init macro pattern
  init_pattern = re.compile("^\s*HIP_INIT[_\w]*_API\(([^,]+)(,|\))");
  target_pattern = re.compile("^(\s*HIP_INIT[^\(]*)(_API\()(.*)\);\s*$");

  # Open input file
  inp = open(inp_file, 'r')

  # API name
  api_name = ""
  # Valid public API found flag
  api_valid = 0

  # Input file patched content
  content = ''
  # Sub content for found API defiition
  sub_content = ''
  # Current record, accumulating several API definition related lines
  record = ''
  # Current input file line number
  line_num = -1 
  # API beginning found flag
  found = 0

  # Reading input file
  for line in inp.readlines():
    # Accumulating record
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      print "Error: bad record \"" + record + "\"\nfile '" + inp_file + ", line (" + str(line_num) + ")"
      break;

    # Looking for API begin
    if beg_pattern.match(record): found = 1

    # Matching complete API definition
    if found == 1:
      record = re.sub("\s__dparm\([^\)]*\)", '', record);
      m = api_pattern.match(record)
      # Checking if complete API matched
      if m:
        found = 2
        api_name = m.group(2);
        # Checking if API name is in the API map
        if api_name in api_map:
          #print "> " + api_name
          # Getting API arguments
          api_args = m.group(3)
          # Getting etalon arguments from the API map
          eta_args = api_map[api_name]
          if eta_args == '':
            eta_args = api_args
            api_map[api_name] = eta_args
          # Normalizing API arguments
          api_types = filtr_api_types(api_args)
          #print "> " + api_name, ": '" + api_args + "' : '" + api_types + "'"
          # Normalizing etalon arguments
          eta_types = filtr_api_types(eta_args)
          #print "> " + api_name + ": '" + eta_args + "' : '" + eta_types + "'"
          # Comparing API and etalon arguments
          # Normalizing types if not matching
          api_types_n = api_types
          eta_types_n = eta_types
          #if api_types != eta_types:
          #  api_types_n = norm_api_types(api_types)
          #  eta_types_n = norm_api_types(eta_types)
          # Comparing API and etalon arguments
          if api_types_n == eta_types_n:
            # API is already found
            if api_name in out:
              print "Error: API redefined \"" + api_name + "\", record \"" + record + "\"\nfile '" + inp_file + "', line (" + str(line_num) + ")"
              sys.exit(1)
            # Set valid public API found flag
            api_valid = 1
            # Set output API map with API arguments list
            out[api_name] = filtr_api_opts(api_args)
          else:
            # Warning about mismatched API, possible non public overloaded version
            api_diff = '\t\t' + inp_file + " line(" + str(line_num) + ")\n\t\tapi: " + api_types_n + "\n\t\teta: " + eta_types_n
            print "\t" + api_name + ':\n' + api_diff + '\n'

    # API found action
    if found == 2:
      # Looking for INIT macro
      m = init_pattern.match(line)
      if m:
        found = 0
        if api_valid == 1:
          api_valid = 0
          print (api_name);
        else:
          # Registering dummy API for non public API if the name in INIT is not NONE
          dummy_name = m.group(1)
          if (not dummy_name in api_map) and (dummy_name != 'NONE'):
            if dummy_name in out:
              print "Error: API reinit \"" + api_name + "\", record \"" + record + "\"\nfile '" + inp_file + "', line (" + str(line_num) + ")"
              sys.exit(1)
            out[dummy_name] = []
      elif re.search('}', line):
        found = 0
        # Expect INIT macro for valid public API
        if api_valid == 1:
          api_valid = 0
          if api_name in out:
            del out[api_name]
            del api_map[api_name]
            out['.' + api_name] = 1
          else:
            print "Error: API is not in out \"" + api_name + "\", record \"" + record + "\"\nfile '" + inp_file + "', line (" + str(line_num) + ")"
            sys.exit(1)

    if found != 1: record = ""
    content += line

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
  print >>sys.stderr, "  $ hipap.py hip/include/hip/hcc_detail/hip_runtime_api.h hip/src"
  sys.exit(1)

# API header file given as an argument
api_hfile = sys.argv[1]
if not os.path.isfile(api_hfile):
  print >>sys.stderr, "Error: input file '" + api_hfile + "' not found"
  sys.exit(1)

# API declaration map
api_map = {
  'hipHccModuleLaunchKernel': ''
}
# API options map
opts_map = {}
# Private API list
priv_lst = []

# Parsing API header
parse_api(api_hfile, api_map)

# Patching API implementation sources
# Sources path is given as an argument
if len(sys.argv) == 3:
  src_path = sys.argv[2]
  src_patt = "\.cpp$"
  patch_src(api_map, src_path, src_patt, opts_map)

# Checking for non-conformant APIs
for name in opts_map.keys():
  m = re.match(r'\.(\S*)', name)
  if m:
    print "Init missing: " + m.group(1)
    del opts_map[name]

# Converting api map to map of lists
# Printing not found APIs
not_found = 0
if len(opts_map) != 0:
  for name in api_map.keys():
    args_str = api_map[name];
    api_map[name] = list_api_args(args_str)
    if not name in opts_map:
      print "Not found: " + name
      not_found += 1
if not_found != 0:
  print "Error:", not_found, "API calls not found"
  sys.exit(1)
#############################################################

f = open(HEADER, 'w')
f.write('// automatically generated sources\n')
f.write('#ifndef _HIP_PROF_STR_H\n');
f.write('#define _HIP_PROF_STR_H\n');
f.write('#include <sstream>\n');
f.write('#include <string>\n');

# Generating dummy macro for non-public API
f.write('\n// Dummy API primitives\n')
f.write('#define INIT_NONE_CB_ARGS_DATA(cb_data) {};\n')
for name in opts_map:
  if not name in api_map:
    opts_lst = opts_map[name]
    if len(opts_lst) != 0:
      print ("Error: bad dummy API \"" + name + "\", args: ", opts_lst) 
      sys.exit(1)
    f.write('#define INIT_'+ name + '_CB_ARGS_DATA(cb_data) {};\n')
    priv_lst.append(name)

for name in priv_lst:
  print "Private: ", name

# Generating the callbacks ID enumaration
f.write('\n// HIP API callbacks ID enumaration\n')
f.write('enum hip_api_id_t {\n')
cb_id = 0
for name in api_map.keys():
  f.write('  HIP_API_ID_' + name + ' = ' + str(cb_id) + ',\n')
  cb_id += 1
f.write('  HIP_API_ID_NUMBER = ' + str(cb_id) + ',\n')
f.write('  HIP_API_ID_ANY = ' + str(cb_id + 1) + ',\n')
f.write('\n')
f.write('  HIP_API_ID_NONE = HIP_API_ID_NUMBER,\n')
for name in priv_lst:
  f.write('  HIP_API_ID_' + name + ' = HIP_API_ID_NUMBER,\n')
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
'struct hip_api_data_t {\n' +
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
  if name in opts_map:
    opts_list = opts_map[name]
    if len(args) != len(opts_list):
      print "Error: \"" + name + "\" API args and opts mismatch, args: ", args, ", opts: ", opts_list
    for ind in range(0, len(args)):
      arg_tuple = args[ind]
      arg_type = arg_tuple[0]
      fld_name = arg_tuple[1]
      arg_name = opts_list[ind]
      f.write('  cb_data.args.' + name + '.' + fld_name + ' = (' + arg_type + ')' + arg_name + '; \\\n')
  f.write('};\n')
f.write('#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)\n')

# Generating the method for the API string, name and parameters
f.write('\n')
f.write('#if 0\n')
f.write('// HIP API string method, method name and parameters\n')
f.write('const char* hipApiString(hip_api_id_t id, const hip_api_data_t* data) {\n')
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

f.write('#endif  // _HIP_PROF_STR_H\n');

print "Header '" + HEADER + "' is generated"
#############################################################
