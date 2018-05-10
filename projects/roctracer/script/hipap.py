#!/usr/bin/python
import os, sys, re

HEADER = "hip_cbstr.h"
REC_MAX_LEN = 1024

def fill_api_map(out, api_name, args_str):
  args_list = []

  args_str = re.sub(r'^\s*', r'', args_str);
  args_str = re.sub(r'\s*$', r'', args_str);
  args_str = re.sub(r'\s*,\s*', r',', args_str);
  args_str = re.sub(r'\s+', r' ', args_str);

  for arg_pair in args_str.split(','):
    arg_pair = re.sub(r'\s+=\s+\S+$', '', arg_pair);
    m = re.match("^(.*)\s(\S+)$", arg_pair);
    if m: args_list.append((m.group(1), m.group(2)))

  out[api_name] = args_list;

# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
def parse_api(inp, out):
  end_pattern = re.compile("Texture");
  beg_pattern = re.compile("^hipError_t");
  api_pattern = re.compile("^hipError_t\s+([^\(]+)\(([^\)]*)\)");

  found = 0
  record = ""
  line_num = -1 

  for line in inp.readlines():
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      print "Error: bad record \"" + record + "\"\nfile '" + hfile + ", line (" + str(line_num) + ")"
      break;

    if beg_pattern.match(record): found = 1

    if found:
      m = api_pattern.match(record)
      if m:
        found = 0
        if end_pattern.search(record): break
        fill_api_map(out, m.group(1), m.group(2))
      else: continue

    record = ""
      
#############################################################

if (len(sys.argv) != 2):
  print >>sys.stderr, "Usage:", sys.argv[0], " <input HIP API .h file>"
  sys.exit(1)

hfile = sys.argv[1]
if not os.path.isfile(hfile):
  print >>sys.stderr, "Error: input file '" + hfile + "' not found"
  sys.exit(1)

inp = open(hfile, 'r')
api_map = {}
parse_api(inp, api_map)

api_map['hipLaunchKernel'] = [('void*', 'kernel'), ('hipStream_t', 'stream')]

f = open(HEADER, 'w')
f.write('// automatically generated sources\n')
f.write('#ifndef HIP__CBSTR_H__\n');
f.write('#define HIP__CBSTR_H__\n');

# Generating the callbacks function type
f.write('\n// HIP API callbacks function type\n\
struct hip_cb_data_t;\n\
typedef void (*hip_cb_fun_t)(const hip_cb_data_t* data, void* arg);\n\
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

# Generating the callbacks data structure
f.write('\n// HIP API callbacks data structure\n')
f.write(
'struct hip_cb_data_t {\n' +
'  const char* name;\n' +
'  hip_cb_id_t id;\n' +
'  uint32_t correlation_id;\n' +
'  bool on_enter;\n' +
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
  for arg_tuple in args:
    arg_type = arg_tuple[0];
    arg_name = arg_tuple[1];
    f.write('  cb_data.args.' + name + '.' + arg_name + ' = (' + arg_type + ')' + arg_name + '; \\\n');
  f.write('};\n')
f.write('#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)\n')

# Generating the callbacks table
f.write('\n// HIP API callbacks table\n')
f.write('\
struct hip_cb_table_t {\n\
  struct { hip_cb_fun_t act; hip_cb_fun_t fun; void* arg; } arr[HIP_API_ID_NUMBER];\n\
};\n\
#define HIP_CALLBACKS_TABLE hip_cb_table_t HIP_API_callbacks_table{};\n\
')
f.write('\
inline bool HIP_SET_ACTIVITY(uint32_t id, hip_cb_fun_t fun, void* arg = NULL) {\n\
  (void)arg;\n\
  extern hip_cb_table_t HIP_API_callbacks_table;\n\
  if (id < HIP_API_ID_NUMBER) {\n\
    HIP_API_callbacks_table.arr[id].act = fun;\n\
    return true;\n\
  }\n\
  return false;\n\
}\n')
f.write('\
inline bool HIP_SET_CALLBACK(uint32_t id, hip_cb_fun_t fun, void* arg) {\n\
  extern hip_cb_table_t HIP_API_callbacks_table; \n\
  if (id < HIP_API_ID_NUMBER) {\n\
    HIP_API_callbacks_table.arr[id].fun = fun;\n\
    HIP_API_callbacks_table.arr[id].arg = arg;\n\
    return true;\n\
  }\n\
  return false;\n\
}\n')

# Generating the callback spawning class
f.write('\n// HIP API callbacks spawning class macro\n\
#define CB_SPAWNER_OBJECT(cb_id) \\\n\
  class api_callbacks_spawner_t { \\\n\
   public: \\\n\
    api_callbacks_spawner_t(hip_cb_data_t& cb_data) : cb_data_(cb_data) { \\\n\
      hip_cb_id_t id = HIP_API_ID_##cb_id; \\\n\
      cb_data_.id = id; \\\n\
      cb_data_.correlation_id = UINT_MAX; \\\n\
      cb_data_.name = #cb_id; \\\n\
      extern const hip_cb_table_t* getApiCallbackTabel(); \\\n\
      const hip_cb_table_t* cb_table = getApiCallbackTabel(); \\\n\
      cb_act_ = cb_table->arr[id].act; \\\n\
      cb_fun_ = cb_table->arr[id].fun; \\\n\
      cb_arg_ = cb_table->arr[id].arg; \\\n\
      cb_data_.on_enter = true; \\\n\
      if (cb_act_ != NULL) cb_act_(&cb_data_, NULL); \\\n\
      if (cb_fun_ != NULL) cb_fun_(&cb_data_, cb_arg_); \\\n\
    } \\\n\
    ~api_callbacks_spawner_t() { \\\n\
      cb_data_.on_enter = false; \\\n\
      if (cb_act_ != NULL) cb_act_(&cb_data_, NULL); \\\n\
      if (cb_fun_ != NULL) cb_fun_(&cb_data_, cb_arg_); \\\n\
    } \\\n\
   private: \\\n\
    hip_cb_data_t& cb_data_; \\\n\
    hip_cb_fun_t cb_act_; \\\n\
    hip_cb_fun_t cb_fun_; \\\n\
    void* cb_arg_; \\\n\
  }; \\\n\
  hip_cb_data_t cb_data{}; \\\n\
  INIT_CB_ARGS_DATA(cb_id, cb_data); \\\n\
  api_callbacks_spawner_t api_callbacks_spawner(cb_data); \n\
')

f.write('#endif  // HIP__CBSTR__');

print "Header '" + HEADER + "' is generated"
#############################################################
