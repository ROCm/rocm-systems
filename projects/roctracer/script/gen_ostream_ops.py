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

HEADER = \
'template <typename T>\n' + \
'struct output_streamer {\n' + \
'  inline static std::ostream& put(std::ostream& out, const T& v) { return out; }\n' + \
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

structs_done = {}
def process_struct(f,c,cppHeader,nname):

    if c not in cppHeader.classes:
        return
    if c in structs_done:
        return

    structs_done[c] = 1;
    for l in range(len(cppHeader.classes[c]["properties"]["public"])):
        key = 'name'
        name = ""
        if key in cppHeader.classes[c]["properties"]["public"][l]:
           name = cppHeader.classes[c]["properties"]["public"][l][key]
        key2 = 'type'
        mtype = ""
        if key2 in cppHeader.classes[c]["properties"]["public"][l]:
            mtype = cppHeader.classes[c]["properties"]["public"][l][key2]
        key3 = 'array_size'
        array_size = ""
        if key3 in cppHeader.classes[c]["properties"]["public"][l]:
            array_size = cppHeader.classes[c]["properties"]["public"][l][key3]
        key4 = 'property_of_class'
        prop = ""
        if  key4 in cppHeader.classes[c]["properties"]["public"][l]:
            prop = cppHeader.classes[c]["properties"]["public"][l][key4]

        if mtype != "" and "union" not in mtype:
            if array_size == "":
                str = "   roctracer::kfd_support::output_streamer<"+mtype+">::put(out,v."+name+");\n"
            else:
                str = "   roctracer::kfd_support::output_streamer<"+mtype+"["+array_size+"]>::put(out,v."+name+");\n"

            if nname != "" and nname not in str:
                #print("injecting ",nname, "in ", str)
                str = str.replace("v.","v."+nname+".")
            if "void" not in mtype:
                f.write(str)
        else:
            nc = prop+"::"
            process_struct(f,nc,cppHeader,name)
            nc = prop+"::"+mtype+" "
            process_struct(f,nc,cppHeader,name)
            nc = c+"::"
            process_struct(f,nc,cppHeader,name)


def gen_cppheader(infilepath,outfilepath):
    try:
        cppHeader = CppHeaderParser.CppHeader(infilepath)
    except CppHeaderParser.CppParseError as e:
        print(e)
        sys.exit(1)

    f = open(outfilepath,"w+")
    f.write("// automatically generated\n")
    f.write(LICENSE)
    f.write("\n")
    HEADER_S = \
      '#ifndef INC_KFD_OSTREAM_OPS_H_\n' + \
      '#define INC_KFD_OSTREAM_OPS_H_\n' + \
      '#include <iostream>\n' + \
      '\n' + \
      '#include "roctracer.h"\n' + \
      '#include "hsakmt.h"\n'
    f.write(HEADER_S)
    f.write('\n')
    f.write('namespace roctracer {\n')
    f.write('namespace kfd_support {\n')
    f.write('// begin ostream ops for KFD \n')
    f.write(HEADER)
    for c in cppHeader.classes:
        if "union" in c:
            continue
        f.write("\ntemplate<>\n")
        f.write("struct output_streamer<"+c+"&> {\n")
        f.write("  inline static std::ostream& put(std::ostream& out, "+c+"& v)\n")
        f.write("{\n")
        process_struct(f,c,cppHeader,"")
        f.write("    return out;\n")
        f.write("}\n")
        f.write("};\n")

    FOOTER = \
    '// end ostream ops for KFD \n'
    FOOTER += '};};\n' + \
       '\n' + \
       '#endif // INC_KFD_OSTREAM_OPS_H_\n' + \
       ' \n'
    FOOTER2 = '\n\n' + \
        '#endif // INC_BASIC_OSTREAM_OPS_H_\n' + \
        ' \n'
    f.write(FOOTER)

    f.close()
    return

parser = argparse.ArgumentParser(description='genOstreamOps.py: generates ostream operators for all typedefs in provided input file.')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in','--in', help='Header file to be parsed', required=True)
requiredNamed.add_argument('-out','--out', help='Output file with ostream operators', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    gen_cppheader(args['in'],args['out'])

