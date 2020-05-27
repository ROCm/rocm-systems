#!/usr/bin/python

import os, sys, re
import CppHeaderParser
import argparse

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

header = \
'template <typename T>\n' + \
'struct output_streamer {\n' + \
'  inline static std::ostream& put(std::ostream& out, const T& v) { return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<void*> {\n' + \
  'inline static std::ostream& put(std::ostream& out, void* v) { out << std::hex << v; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<const void*> {\n' + \
  'inline static std::ostream& put(std::ostream& out, const void* v) { out << std::hex << v; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<bool> {\n' + \
'  inline static std::ostream& put(std::ostream& out, bool v) { out << std::hex << "<bool " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint8_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint8_t v) { out << std::hex << "<uint8_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint16_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint16_t v) { out << std::hex << "<uint16_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint32_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint32_t v) { out << std::hex << "<uint32_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint64_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint64_t v) { out << std::hex << "<uint64_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<bool*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, bool* v) { out << std::hex << "<bool " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint8_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint8_t* v) { out << std::hex << "<uint8_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint16_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint16_t* v) { out << std::hex << "<uint16_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint32_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint32_t* v) { out << std::hex << "<uint32_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\ntemplate<>\n' + \
'struct output_streamer<uint64_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint64_t* v) { out << std::hex << "<uint64_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\n'

header_hip = \
'template <typename T>\n' + \
'  std::ostream& operator<<(std::ostream& out, const T& v) {\n' + \
'     using std::operator<<;\n' + \
'     static bool recursion = false;\n' + \
'     if (recursion == false) { recursion = true; out << v; recursion = false; }\n' + \
'     return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, void* v) { using std::operator<<; out << std::hex << v; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, const void* v) { using std::operator<<; out << std::hex << v; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, bool v) { using std::operator<<; out << std::hex << "<bool " << "0x" << v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint8_t v) { using std::operator<<; out << std::hex << "<uint8_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint16_t v) { using std::operator<<; out << std::hex << "<uint16_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint32_t v) { using std::operator<<; out << std::hex << "<uint32_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint64_t v) { using std::operator<<; out << std::hex << "<uint64_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, bool* v) {  using std::operator<<; out << std::hex << "<bool " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint8_t* v) { using std::operator<<; out << std::hex << "<uint8_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint16_t* v) { using std::operator<<; out << std::hex << "<uint16_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint32_t* v) { using std::operator<<; out << std::hex << "<uint32_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'std::ostream& operator<<(std::ostream& out, uint64_t* v) { using std::operator<<; out << std::hex << "<uint64_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'\n'

structs_analyzed = {}
global_ops_hip = ''

# process_struct traverses recursively all structs to extract all fields
def process_struct(file_handle, cppHeader_struct, cppHeader, parent_hier_name, apiname):
# file_handle: handle for output file {api_name}_ostream_ops.h to be generated
# cppHeader_struct: cppHeader struct being processed 
# cppHeader: cppHeader object created by CppHeaderParser.CppHeader(...) 
# parent_hier_name: parent hierarchical name used for nested structs/enums
# apiname: for example hip, kfd.

    if cppHeader_struct == 'max_align_t': #function pointers not working in cppheaderparser
        return
    if cppHeader_struct not in cppHeader.classes:
        return
    if cppHeader_struct in structs_analyzed:
        return

    structs_analyzed[cppHeader_struct] = 1;
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

        if "union" not in mtype:
            if apiname.lower() == 'hip':
              str = "   roctracer::hip_support::operator<<(out, v."+name+");\n"
            else:
              if array_size == "":
                str = "   roctracer::" + apiname.lower() + "_support::output_streamer<"+mtype+">::put(out,v."+name+");\n"
              else:
                str = "   roctracer::" + apiname.lower() + "_support::output_streamer<"+mtype+"["+array_size+"]>::put(out,v."+name+");\n"
            if "void" not in mtype:
                file_handle.write(str)
        else:
            if prop != '':
              next_cppHeader_struct = prop + "::"
              process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)
              next_cppHeader_struct = prop + "::" + mtype + " "
              process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)
            next_cppHeader_struct = cppHeader_struct + "::"
            process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)

#  Parses API header file and generates ostream ops files ostream_ops.h and basic_ostream_ops.h
def gen_cppheader(infilepath, outfilepath):
# infilepath: API Header file to be parsed
# outfilepath: Output file where ostream operators are written
    global_ops_hip = ''
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
    f2 = open(mpath + "/basic_ostream_ops.h","w+")
    f.write("// automatically generated\n")
    f2.write("// automatically generated\n")
    f.write(LICENSE + '\n')
    f2.write(LICENSE + '\n')
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
    f.write('// begin ostream ops for '+ apiname + ' \n')
    if apiname.lower() != 'hip':
      f.write('#include "basic_ostream_ops.h"' + '\n')
    else:
      f.write("// HIP basic ostream ops\n")
      f.write(header_hip)
      f.write("// End of HIP basic ostream ops\n\n")
    f2.write(header)
    for c in cppHeader.classes:
        if "union" in c:
            continue
        if len(cppHeader.classes[c]["properties"]["public"])!=0:
          if apiname.lower() == 'hip':
            f.write("std::ostream& operator<<(std::ostream& out, " + c + "& v)\n")
            f.write("{\n")
            global_ops_hip = global_ops_hip + "std::ostream& operator<<(std::ostream& out, const " + c + "& v)\n" + "{\n" + "   roctracer::hip_support::operator<<(out, v);\n" + "   return out;\n" + "}\n\n"
            process_struct(f, c, cppHeader, "", apiname)
            f.write("   return out;\n")
            f.write("}\n")
          else:
            f.write("\ntemplate<>\n")
            f.write("struct output_streamer<" + c + "&> {\n")
            f.write("  inline static std::ostream& put(std::ostream& out, "+c+"& v)\n")
            f.write("{\n")
            process_struct(f, c, cppHeader, "", apiname)
            f.write("   return out;\n")
            f.write("}\n")
            f.write("};\n")

    footer = \
    '// end ostream ops for '+ apiname + ' \n'
    footer += '};};\n\n'
    f.write(footer)
    f.write(global_ops_hip)

    footer = '#endif //__cplusplus\n' + \
             '#endif // INC_' + apiname + '_OSTREAM_OPS_H_\n' + \
             ' \n'
    f.write(footer)
    f.close()
    f2.close()
    print('File ' + outfilepath + ' generated')
    print('File ' + mpath + '/basic_ostream_ops.h generated')

    return

parser = argparse.ArgumentParser(description='genOstreamOps.py: generates ostream operators for all typedefs in provided input file.')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in', metavar='file', help='Header file to be parsed', required=True)
requiredNamed.add_argument('-out', metavar='file', help='Output file with ostream operators', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    gen_cppheader(args['in'], args['out'])
