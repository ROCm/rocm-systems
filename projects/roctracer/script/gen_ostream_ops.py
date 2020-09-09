#!/usr/bin/python

import os, sys, re
import CppHeaderParser
import argparse
import string

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


header = 'template <typename T>\n' + \
'struct output_streamer {\n' + \
'  inline static std::ostream& put(std::ostream& out, const T& v) { return out; }\n' + \
'};\n\n'

header_basic = \
'template <typename T>\n' + \
'  inline static std::ostream& operator<<(std::ostream& out, const T& v) {\n' + \
'     using std::operator<<;\n' + \
'     static bool recursion = false;\n' + \
'     if (recursion == false) { recursion = true; out << v; recursion = false; }\n' + \
'     return out; }\n'

structs_analyzed = {}
global_ops_hip = ''
global_str = ''

# process_struct traverses recursively all structs to extract all fields
def process_struct(file_handle, cppHeader_struct, cppHeader, parent_hier_name, apiname):
# file_handle: handle for output file {api_name}_ostream_ops.h to be generated
# cppHeader_struct: cppHeader struct being processed
# cppHeader: cppHeader object created by CppHeaderParser.CppHeader(...)
# parent_hier_name: parent hierarchical name used for nested structs/enums
# apiname: for example hip, kfd.
    global global_str

    if cppHeader_struct == 'max_align_t': #function pointers not working in cppheaderparser
        return
    if cppHeader_struct not in cppHeader.classes:
        return
    if cppHeader_struct in structs_analyzed:
        return

    structs_analyzed[cppHeader_struct] = 1
    for l in reversed(range(len(cppHeader.classes[cppHeader_struct]["properties"]["public"]))):
        key = 'name'
        name = ""
        if key in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
           if parent_hier_name != '':
             name = parent_hier_name + '.' + cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key]
           else:
             name = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key]
        if name == '':
           continue
        key2 = 'type'
        mtype = ""
        if key2 in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            mtype = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key2]
        if mtype == '':
          continue
        key3 = 'array_size'
        array_size = ""
        if key3 in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            array_size = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key3]
        key4 = 'property_of_class'
        prop = ""
        if  key4 in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            prop = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key4]

        str = ''
        if "union" not in mtype:
            if apiname.lower() == 'hip' or apiname.lower() == 'hsa':
              str += "   roctracer::" + apiname.lower() + "_support::operator<<(out, \"" + name + " = \");\n"
              str += "   roctracer::" + apiname.lower() + "_support::operator<<(out, v."+name+");\n"
              str += "   roctracer::" + apiname.lower() + "_support::operator<<(out, \", \");\n"
            else:
              str += "   roctracer::" + apiname.lower() + "_support::output_streamer<const char*>::put(out, \"" + name + " = \");\n"
              if array_size == "":
                str += "   roctracer::" + apiname.lower() + "_support::output_streamer<" + mtype + ">::put(out, v." + name + ");\n"
              else:
                str += "   roctracer::" + apiname.lower() + "_support::output_streamer<" + mtype + "[" + array_size + "]>::put(out, v." + name + ");\n"
              str += "   roctracer::" + apiname.lower() + "_support::output_streamer<const char*>::put(out, \", \");\n"
            if "void" not in mtype:
                global_str += str
        else:
            if prop != '':
              next_cppHeader_struct = prop + "::"
              process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)
              next_cppHeader_struct = prop + "::" + mtype + " "
              process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)
            next_cppHeader_struct = cppHeader_struct + "::"
            process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)

#  Parses API header file and generates ostream ops files ostream_ops.h
def gen_cppheader(infilepath, outfilepath, structs_depth):
# infilepath: API Header file to be parsed
# outfilepath: Output file where ostream operators are written
    global_ops_hip = ''
    global_ops_hsa = ''
    global global_str
    try:
        cppHeader = CppHeaderParser.CppHeader(infilepath)
    except CppHeaderParser.CppParseError as e:
        print(e)
        sys.exit(1)
    mpath = os.path.dirname(outfilepath)
    if mpath == "":
       mpath = os.getcwd()
    apiname = outfilepath.replace(mpath+"/","")
    apiname = apiname.replace("_ostream_ops.h","")
    apiname = apiname.upper()
    f = open(outfilepath,"w+")
    f.write("// automatically generated\n")
    f.write(LICENSE + '\n')
    header_s = \
      '#ifndef INC_' + apiname + '_OSTREAM_OPS_H_\n' + \
      '#define INC_' + apiname + '_OSTREAM_OPS_H_\n' + \
      '#ifdef __cplusplus\n' + \
      '#include <iostream>\n' + \
      '\n' + \
      '#include "roctracer.h"\n'
    if apiname.lower() == 'hip':
      header_s = header_s + '\n' + \
      '#include "hip/hip_runtime_api.h"\n' + \
      '#include "hip/hcc_detail/hip_vector_types.h"\n\n'

    f.write(header_s)
    f.write('\n')
    f.write('namespace roctracer {\n')
    f.write('namespace ' + apiname.lower() + '_support {\n')
    if structs_depth != -1:
      f.write('static int ' + apiname.upper() + '_depth_max = ' + str(structs_depth) + ';\n')
    f.write('// begin ostream ops for '+ apiname + ' \n')
    if apiname.lower() == "hip" or apiname.lower() == "hsa":
      f.write("// basic ostream ops\n")
      f.write(header_basic)
      f.write("// End of basic ostream ops\n\n")
    else:
      f.write(header)

    for c in cppHeader.classes:
        if "union" in c:
            continue
        if  apiname.lower() == 'hsa':
          if c == 'max_align_t' or c == '__fsid_t': #already defined for hip
            continue
        if len(cppHeader.classes[c]["properties"]["public"])!=0:
          if apiname.lower() == 'hip' or apiname.lower() == 'hsa':
            f.write("inline static std::ostream& operator<<(std::ostream& out, const " + c + "& v)\n")
            f.write("{\n")
            f.write("  roctracer::" + apiname.lower() + "_support::operator<<(out, '{');\n")
            if structs_depth != -1:
              f.write("  " + apiname.upper() + "_depth_max++;\n")
              f.write("  if (" + apiname.upper() + "_depth_max <= " + str(structs_depth) + ") {\n" )
            process_struct(f, c, cppHeader, "", apiname)
            global_str = "\n".join(global_str.split("\n")[0:-2])
            if structs_depth != -1: #reindent
              global_str = string.split(global_str, '\n')
              global_str = ['    ' + string.lstrip(line) for line in global_str]
              global_str = string.join(global_str, '\n')
            f.write(global_str+"\n")
            if structs_depth != -1:
              f.write("  };\n")
              f.write("  " + apiname.upper() + "_depth_max--;\n")
            f.write("  roctracer::" + apiname.lower() + "_support::operator<<(out, '}');\n")
            f.write("  return out;\n")
            f.write("}\n")
            global_str = ''
          else:
            f.write("\ntemplate<>\n")
            f.write("struct output_streamer<" + c + "&> {\n")
            f.write("  inline static std::ostream& put(std::ostream& out, "+c+"& v)\n")
            f.write("{\n")
            f.write("  roctracer::" + apiname.lower() + "_support::output_streamer<char>::put(out, '{');\n")
            if structs_depth != -1:
              f.write(apiname.upper() + "_depth_max++;\n")
              f.write("  if (" + apiname.upper() + "_depth_max <= " + str(structs_depth) + ") {\n" )
            process_struct(f, c, cppHeader, "", apiname)
            global_str = "\n".join(global_str.split("\n")[0:-2])
            if structs_depth != -1: #reindent
              global_str = string.split(global_str, '\n')
              global_str = ['    ' + string.lstrip(line) for line in global_str]
              global_str = string.join(global_str, '\n')
            f.write(global_str+"\n")
            if structs_depth != -1:
              f.write("  };\n")
              f.write("  " + apiname.upper() + "_depth_max--;\n")
            f.write("  roctracer::" + apiname.lower() + "_support::output_streamer<char>::put(out, '}');\n")
            f.write("  return out;\n")
            f.write("}\n")
            f.write("};\n")
            global_str = ''
          if apiname.lower() == 'hip':
            global_ops_hip += "inline static std::ostream& operator<<(std::ostream& out, const " + c + "& v)\n" + "{\n" + "  roctracer::hip_support::operator<<(out, v);\n" + "  return out;\n" + "}\n\n"
          if apiname.lower() == 'hsa':
            global_ops_hsa += "inline static std::ostream& operator<<(std::ostream& out, const " + c + "& v)\n" + "{\n" + "  roctracer::hsa_support::operator<<(out, v);\n" + "  return out;\n" + "}\n\n"

    footer = \
    '// end ostream ops for '+ apiname + ' \n'
    footer += '};};\n\n'
    f.write(footer)
    f.write(global_ops_hip)
    f.write(global_ops_hsa)
    footer = '#endif //__cplusplus\n' + \
             '#endif // INC_' + apiname + '_OSTREAM_OPS_H_\n' + \
             ' \n'
    f.write(footer)
    f.close()
    print('File ' + outfilepath + ' generated')

    return

parser = argparse.ArgumentParser(description='genOstreamOps.py: generates ostream operators for all typedefs in provided input file.')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in', metavar='file', help='Header file to be parsed', required=True)
requiredNamed.add_argument('-out', metavar='file', help='Output file with ostream operators', required=True)
requiredNamed.add_argument('-depth', metavar='N', type=int, help='Depth for nested structs', required=False)

structs_depth = 0
args = vars(parser.parse_args())

if __name__ == '__main__':
    if args['depth'] != None: structs_depth = args['depth']
    gen_cppheader(args['in'], args['out'], structs_depth)
