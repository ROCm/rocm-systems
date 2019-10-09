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
'#ifndef INC_ROCTRACER_KFD_H_\n' + \
'#define INC_ROCTRACER_KFD_H_\n' + \
'#include <iostream>\n' + \
'#include <mutex>\n' + \
'\n' + \
'#include <hsa.h>\n' + \
'\n' + \
'#include "roctracer.h"\n' + \
'#include "hsakmt.h"\n' + \
'\n' + \
'namespace roctracer {\n' + \
'namespace kfd_support {\n' + \
'template <typename T>\n' + \
'struct output_streamer {\n' + \
'  inline static std::ostream& put(std::ostream& out, const T& v) { return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<bool> {\n' + \
'  inline static std::ostream& put(std::ostream& out, bool v) { out << std::hex << "<bool " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint8_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint8_t v) { out << std::hex << "<uint8_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint16_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint16_t v) { out << std::hex << "<uint16_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint32_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint32_t v) { out << std::hex << "<uint32_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint64_t> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint64_t v) { out << std::hex << "<uint64_t " << "0x" << v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\n' + \
'template<>\n' + \
'struct output_streamer<bool*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, bool* v) { out << std::hex << "<bool " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint8_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint8_t* v) { out << std::hex << "<uint8_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint16_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint16_t* v) { out << std::hex << "<uint16_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint32_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint32_t* v) { out << std::hex << "<uint32_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'template<>\n' + \
'struct output_streamer<uint64_t*> {\n' + \
'  inline static std::ostream& put(std::ostream& out, uint64_t* v) { out << std::hex << "<uint64_t " << "0x" << *v << std::dec << ">"; return out; }\n' + \
'};\n' + \
'\n'

FOOTER = \
'// end ostream ops for KFD\n' + \
'};};\n' + \
'\n' + \
'#include <inc/kfd_prof_str.h>\n' + \
'\n' + \
'#endif // INC_ROCTRACER_KFD_H_\n' + \
' \n'

rx_dict = {
    'struct_name': re.compile(r'typedef (?P<struct_name>.*)\n'),
    'field_type': re.compile(r'\s+name\[raw_type\]=(?P<field_type>.*)\n'),
    'field_name': re.compile(r'\s+name\[name\]=(?P<field_name>.*)\n'),
    'array_size_val': re.compile(r'\s+name\[array_size\]=(?P<array_size_val>.*)\n'),
}

def _parse_line(line):

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    return None, None

def parse_file(infilepath,outfilepath):
    f= open(outfilepath,"w+")
    f.write("// automatically generated\n")
    f.write(LICENSE)
    f.write("/////////////////////////////////////////////////////////////////////////////")
    f.write("\n")
    f.write(HEADER)
    f.write("// begin ostream ops for KFD\n")

    with open(infilepath, 'r') as file_object:
        line = file_object.readline()
        flag=0
        tmp_str=""
        while line:
            key, match = _parse_line(line)
            if key == 'struct_name':
                if tmp_str!="":
                    f.write(tmp_str+"\n")
                    tmp_str=""
                if flag == 1:
                    f.write("    return out;\n")
                    f.write("}\n")
                    f.write("};\n")
                flag=0
                struct_name = match.group('struct_name')
                if ("anon" not in struct_name and "union" not in struct_name) or args['debug']: 
                    f.write("template<>\n")
                    f.write("struct output_streamer<"+struct_name+"&> {\n")
                    f.write("  inline static std::ostream& put(std::ostream& out, "+struct_name+"& v)\n")
                    f.write("{\n")
                    flag=1;
            if flag==1 and key == 'field_type':
                field_type = match.group('field_type')
                if field_type == "":
                    field_type="notype"
            if flag==1 and key == 'array_size_val':
                array_size_val = match.group('array_size_val')
                tmp_str=tmp_str.replace(field_type,field_type+"["+array_size_val+"]")
                f.write(tmp_str+"\n")
                tmp_str=""
            if flag==1 and key == 'field_name':
                if tmp_str!="":
                    f.write(tmp_str+"\n")
                tmp_str=""
                field_name = match.group('field_name')
                if field_name == "":
                    field_name="noname"
                if (field_name!="noname" and field_type!="notype") or args['debug'] :
                    tmp_str="    roctracer::kfd_support::output_streamer<"+field_type+">::put(out,v."+field_name+")"+";";
                    tmp_str=tmp_str.replace('<::', '<')
                    #f.write(tmp_str+"\n")
            line = file_object.readline()
    if tmp_str!="":
        f.write(tmp_str+"\n")
        tmp_str=""
    if flag==1:
        f.write("    return out;\n")
        f.write("}\n")
        f.write("};\n")
    f.write(FOOTER)
    f.close()
    print ("File "+outfilepath+" has been generated.")
    return 

def gen_cppheader_lut(infilepath):
    try:
        cppHeader = CppHeaderParser.CppHeader(infilepath)
    except CppHeaderParser.CppParseError as e:
        print(e)
        sys.exit(1)

    f= open("/tmp/cppheader_lut.txt","w+")
    for c in cppHeader.classes:
        f.write("typedef %s\n"%(c))
        for l in range(len(cppHeader.classes[c]["properties"]["public"])):
            for key in cppHeader.classes[c]["properties"]["public"][l].keys():
                f.write("	name[%s]=%s\n"%(key,cppHeader.classes[c]["properties"]["public"][l][key]))
    f.close()
    #print ("File /tmp/cppheader_lut.txt has been generated.")
    return


parser = argparse.ArgumentParser(description='genOstreamOps.py: generates ostream operators for all typedefs in provided input file.')
parser.add_argument('-debug','--debug', help='Debug option for features not supported by CppHeaderParser', action='store_true')
requiredNamed=parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in','--in', help='Header file to be parsed', required=True)
requiredNamed.add_argument('-out','--out', help='Output file with ostream operators', required=True)
args = vars(parser.parse_args())

if __name__ == '__main__':
    gen_cppheader_lut(args['in'])
    parse_file("/tmp/cppheader_lut.txt",args['out'])

