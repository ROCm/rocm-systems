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

import os, sys, re
import CppHeaderParser
import argparse
import string

LICENSE = (
    "/*\n"
    + "Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.\n"
    + "\n"
    + "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    + 'of this software and associated documentation files (the "Software"), to deal\n'
    + "in the Software without restriction, including without limitation the rights\n"
    + "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    + "copies of the Software, and to permit persons to whom the Software is\n"
    + "furnished to do so, subject to the following conditions:\n"
    + "\n"
    + "The above copyright notice and this permission notice shall be included in\n"
    + "all copies or substantial portions of the Software.\n"
    + "\n"
    + 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
    + "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    + "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\n"
    + "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
    + "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
    + "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n"
    + "THE SOFTWARE.\n"
    + "*/\n"
)


header_basic = (
    "namespace detail {\n"
    + "template <typename T>\n"
    + "  inline static std::ostream& operator<<(std::ostream& out, const T& v) {\n"
    + "     using std::operator<<;\n"
    + "     static bool recursion = false;\n"
    + "     if (recursion == false) { recursion = true; out << v; recursion = false; }\n"
    + "     return out;\n  }\n"
    + "\n"
    + "  inline static std::ostream &operator<<(std::ostream &out, const unsigned char &v) {\n"
    + "    out << (unsigned int)v;\n"
    + "    return out;\n  }\n"
    + "\n"
    + "  inline static std::ostream &operator<<(std::ostream &out, const char &v) {\n"
    + "    out << (unsigned char)v;\n"
    + "    return out;\n  }\n"
)

structs_analyzed = {}
global_ops = ""
global_str = ""
output_filename_h = None
apiname = ""


# process_struct traverses recursively all structs to extract all fields
def process_struct(file_handle, cppHeader_struct, cppHeader, parent_hier_name, apiname):
    # file_handle: handle for output file {api_name}_ostream_ops.h to be generated
    # cppHeader_struct: cppHeader struct being processed
    # cppHeader: cppHeader object created by CppHeaderParser.CppHeader(...)
    # parent_hier_name: parent hierarchical name used for nested structs/enums
    # apiname: for example hip.
    global global_str

    if (
        cppHeader_struct == "max_align_t"
    ):  # function pointers not working in cppheaderparser
        return
    if cppHeader_struct not in cppHeader.classes:
        return
    if cppHeader_struct in structs_analyzed:
        return
    structs_analyzed[cppHeader_struct] = 1
    for l in reversed(
        range(len(cppHeader.classes[cppHeader_struct]["properties"]["public"]))
    ):
        key = "name"
        name = ""
        if key in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            if parent_hier_name != "":
                name = (
                    parent_hier_name
                    + "."
                    + cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key]
                )
            else:
                name = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key]
        if name == "":
            continue
        key2 = "type"
        mtype = ""
        if key2 in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            mtype = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key2]
        if mtype == "":
            continue
        key3 = "array_size"
        array_size = ""
        if key3 in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            array_size = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][
                key3
            ]
        key4 = "property_of_class"
        prop = ""
        if key4 in cppHeader.classes[cppHeader_struct]["properties"]["public"][l]:
            prop = cppHeader.classes[cppHeader_struct]["properties"]["public"][l][key4]

        str = ""
        if "union" not in mtype:
            indent = ""
            str += (
                '    if (std::string("'
                + cppHeader_struct
                + "::"
                + name
                + '").find('
                + apiname.upper()
                + "_structs_regex"
                + ") != std::string::npos)   {\n"
            )
            indent = "    "
            str += (
                indent
                + "  roctracer::"
                + apiname.lower()
                + '_support::detail::operator<<(out, "'
                + name
                + '=");\n'
            )
            str += (
                indent
                + "  roctracer::"
                + apiname.lower()
                + "_support::detail::operator<<(out, v."
                + name
                + ");\n"
            )
            str += (
                indent
                + "  roctracer::"
                + apiname.lower()
                + '_support::detail::operator<<(out, ", ");\n'
            )
            str += "    }\n"
            if "void" not in mtype:
                global_str += str
        else:
            if prop != "":
                next_cppHeader_struct = prop + "::"
                process_struct(
                    file_handle, next_cppHeader_struct, cppHeader, name, apiname
                )
                next_cppHeader_struct = prop + "::" + mtype + " "
                process_struct(
                    file_handle, next_cppHeader_struct, cppHeader, name, apiname
                )
            next_cppHeader_struct = cppHeader_struct + "::"
            process_struct(file_handle, next_cppHeader_struct, cppHeader, name, apiname)


#  Parses API header file and generates ostream ops files ostream_ops.h
def gen_cppheader(infilepath, outfilepath, rank):
    # infilepath: API Header file to be parsed
    # outfilepath: Output file where ostream operators are written
    global global_ops
    global output_filename_h
    global apiname
    global global_str
    try:
        cppHeader = CppHeaderParser.CppHeader(infilepath)
    except CppHeaderParser.CppParseError as e:
        print(e)
        sys.exit(1)
    if rank == 0 or rank == 2:
        mpath = os.path.dirname(outfilepath)
        if mpath == "":
            mpath = os.getcwd()
        apiname = outfilepath.replace(mpath + "/", "")
        output_filename_h = open(outfilepath, "w+")
        apiname = apiname.replace("_ostream_ops.h", "")
        apiname = apiname.upper()
        output_filename_h.write("// automatically generated\n")
        output_filename_h.write(LICENSE + "\n")
        header_s = (
            "#ifndef INC_"
            + apiname
            + "_OSTREAM_OPS_H_\n"
            + "#define INC_"
            + apiname
            + "_OSTREAM_OPS_H_\n"
            + "\n"
        )
        if apiname.upper() == 'HIP':
            header_s = (
                header_s
                + "#include <hip/hip_runtime.h>\n"
                + "#include <hip/hip_deprecated.h>\n"
            )
        header_s = (
            header_s
            + '#include "src/core/session/tracer/src/roctracer.h"\n'
            + "\n"
            + "#ifdef __cplusplus\n"
            + "#include <iostream>\n"
            + "#include <string>\n"
        )

        output_filename_h.write(header_s)
        output_filename_h.write("\n")
        output_filename_h.write("namespace roctracer {\n")
        output_filename_h.write("namespace " + apiname.lower() + "_support {\n")
        output_filename_h.write("static int " + apiname.upper() + "_depth_max = 1;\n")
        output_filename_h.write("static int " + apiname.upper() + "_depth_max_cnt = 0;\n")
        output_filename_h.write(
            "static std::string " + apiname.upper() + '_structs_regex = "";\n'
        )
        output_filename_h.write("// begin ostream ops for " + apiname + " \n")
        output_filename_h.write("// basic ostream ops\n")
        output_filename_h.write(header_basic)
        output_filename_h.write("// End of basic ostream ops\n\n")

    for c in cppHeader.classes.copy():
        # Types defined inside of unions are incorrectly prepended with "union " after parsing by CppHeaderParser
        # Remove "union " from the beginning of the full class name to correct the eventual output
        if "union " in c[0:6] and "::union" not in c[-8:]:
            new_name = c[6:]
            cppHeader.classes[new_name] = cppHeader.classes[c]
            del cppHeader.classes[c]

    for c in cppHeader.classes:
        if c[-2] == ":" and c[-1] == ":":
            continue  # ostream operator cannot be overloaded for anonymous struct therefore it is skipped
        if "::union" in c:
            continue
        if c in structs_analyzed:
            continue
        if (
            c == "max_align_t" or c == "__fsid_t"
        ):  # Skipping as it is defined in multiple domains
            continue
        if len(cppHeader.classes[c]["properties"]["public"]) != 0:
            output_filename_h.write(
                "inline static std::ostream& operator<<(std::ostream& out, const "
                + c
                + "& v)\n"
            )
            output_filename_h.write("{\n")
            output_filename_h.write("  std::operator<<(out, '{');\n")
            output_filename_h.write("  " + apiname.upper() + "_depth_max_cnt++;\n")
            output_filename_h.write(
                "  if ("
                + apiname.upper()
                + "_depth_max == -1 || "
                + apiname.upper()
                + "_depth_max_cnt <= "
                + apiname.upper()
                + "_depth_max"
                + ") {\n"
            )
            process_struct(output_filename_h, c, cppHeader, "", apiname)
            global_str = "\n".join(global_str.split("\n")[0:-3])
            if global_str != "":
                global_str += "\n    }\n"
            output_filename_h.write(global_str)
            output_filename_h.write("  };\n")
            output_filename_h.write("  " + apiname.upper() + "_depth_max_cnt--;\n")
            output_filename_h.write("  std::operator<<(out, '}');\n")
            output_filename_h.write("  return out;\n")
            output_filename_h.write("}\n")
            global_str = ""
            global_ops += (
                "inline static std::ostream& operator<<(std::ostream& out, const "
                + c
                + "& v)\n"
                + "{\n"
                + "  roctracer::"
                + apiname.lower()
                + "_support::detail::operator<<(out, v);\n"
                + "  return out;\n"
                + "}\n\n"
            )

    if rank == 1 or rank == 2:
        footer = "// end ostream ops for " + apiname + " \n"
        footer += "};};};\n\n"
        output_filename_h.write(footer)
        output_filename_h.write(global_ops)
        footer = (
            "#endif //__cplusplus\n"
            + "#endif // INC_"
            + apiname
            + "_OSTREAM_OPS_H_\n"
            + " \n"
        )
        output_filename_h.write(footer)
        output_filename_h.write("#include <hip/amd_detail/hip_prof_str.h>")
        output_filename_h.close()
        print("File " + outfilepath + " generated")

    return


parser = argparse.ArgumentParser(
    description="genOstreamOps.py: generates ostream operators for all typedefs in provided input file."
)
requiredNamed = parser.add_argument_group("Required arguments")
requiredNamed.add_argument(
    "-in",
    metavar="fileList",
    help="Comma separated list of header files to be parsed",
    required=True,
)
requiredNamed.add_argument(
    "-out", metavar="file", help="Output file with ostream operators", required=True
)

args = vars(parser.parse_args())

if __name__ == "__main__":
    flist = args["in"].split(",")
    if len(flist) == 1:
        gen_cppheader(flist[0], args["out"], 2)
    else:
        for i in range(len(flist)):
            if i == 0:
                gen_cppheader(flist[i], args["out"], 0)
            elif i == len(flist) - 1:
                gen_cppheader(flist[i], args["out"], 1)
            else:
                gen_cppheader(flist[i], args["out"], -1)
