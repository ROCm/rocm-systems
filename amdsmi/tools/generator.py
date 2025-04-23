# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse
import tempfile
import shutil
import platform
from subprocess import run, PIPE
from ctypeslib.clang2py import main as clangToPy

HEADER = \
"""# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
"""


def parseArgument():
    parser = argparse.ArgumentParser(description="parse input arguments")
    parser.add_argument('-o','--output', type=str, required=True,
                        help='The output file name')
    parser.add_argument('-i','--input', type=str, required=True,
                        help='The input file name')
    parser.add_argument('-l', '--library', type=str, required=True,
                        help='Loading dynamic link libraries')
    parser.add_argument('-e', '--extra-args', type=str, required=False,
                        help='Parse extra arguments to clang')
    args = vars(parser.parse_args())

    return args['output'], args['input'], args['library'], args['extra_args']


def replace_line(full_path_file_name, string_to_replace, new_string):
    """
    Replaces a specific string in a file with a new string.

    Args:
        full_path_file_name (str): The full path of the file to modify.
        string_to_replace (str): The string to be replaced.
        new_string (str): The new string to replace the old string with.

    Returns:
        None
    """
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, 'w') as new_file:
        with open(full_path_file_name, 'r+', encoding='UTF-8') as old_file:
            for line in old_file:
                new_file.write(line.replace(string_to_replace, new_string))

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def write_header(full_path_file_name):
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, 'w') as new_file:
        new_file.write(HEADER)
        with open(full_path_file_name, 'r+', encoding='UTF-8') as old_file:
            for line in old_file:
                new_file.write(line)

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def write_file(full_path_file_name, contents):
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, 'w') as new_file:
        for line in contents:
            new_file.write(f'{line}\n')

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def find_replacement(search_str1, search_str2, line):
    pos1 = line.find(search_str1)
    if pos1 < 0:
        return ''

    if len(search_str2):
        pos2 = line.find(search_str2, pos1)
        if pos2 < 0:
            return ''
    else:
        pos2 = len(line) - 1

    return line[pos1:pos2+1]


def find_line_num(search_str, line):
    pos1 = line.find(search_str)
    if pos1 < 0:
        return 0
    items = line[pos1:].split(':')
    if len(items) < 2:
        return 0

    line_num = items[1].strip()
    if not line_num.isdigit():
        return 0

    line_num = int(line_num)
    return (line_num)


def main():
    open_bracket = '['
    close_bracket = ']'
    open_parenthesis = '('
    close_parenthesis = ')'
    open_curly_brace = '{'
    close_curly_brace = '}'

    output_file, input_file, library, clang_extra_args =  parseArgument()

    # make args string easy to append
    if clang_extra_args is None:
        clang_extra_args = ""
    else:
        clang_extra_args = " " + clang_extra_args

    library_name = os.path.basename(library)

    clang_include_dir = \
        run(["clang", "--print-resource-dir"], stdout=PIPE, stderr=PIPE, encoding="utf-8").stdout.strip()

    os_platform = platform.system()
    if os_platform == "Windows":
        clang_include_dir += "\\include"
        if "Program Files(x86)" in clang_include_dir:
            clang_include_dir = clang_include_dir.replace("Program Files(x86)", "Progra~2")
        elif "Program Files" in clang_include_dir:
            clang_include_dir = clang_include_dir.replace("Program Files", "Progra~1")

        arguments = [input_file, "-o", output_file]
        line_to_replace = "_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')"
        new_line = "_libraries['FIXME_STUB'] = ctypes.CDLL('{}')".format(library_name)
    elif os_platform == "Linux":
        clang_include_dir += "/include"
        arguments = [input_file, "-o", output_file, "-l", library]
        library_path = os.path.join(os.path.dirname(__file__), library)
        line_to_replace = "_libraries['{}'] = ctypes.CDLL('{}')".format(library_name, library_path)
        new_line = f"""from pathlib import Path
# {library_name} can be located in several different places.
# Look for it with below priority:
# 1. ROCM_HOME/ROCM_PATH environment variables
#    - ROCM_HOME/lib
#    - ROCM_PATH/lib (usually set to /opt/rocm/)
# 2. Decided by the linker
#    - LD_LIBRARY_PATH env var
#    - defined path in /etc/ld.so.conf.d/
# 3. Relative to amdsmi_wrapper.py
#    - parent directory
#    - current directory
def find_smi_library():
    err = OSError("Could not load {library_name}")
    possible_locations = []
    # 1.
    rocm_path = os.getenv("ROCM_HOME", os.getenv("ROCM_PATH"))
    if rocm_path:
        possible_locations.append(os.path.join(rocm_path, "lib/{library_name}"))
    # 2.
    possible_locations.append("{library_name}")
    # 3.
    libamd_smi_parent_dir = Path(__file__).resolve().parent / "{library_name}"
    libamd_smi_cwd = Path.cwd() / "{library_name}"
    possible_locations.append(libamd_smi_parent_dir)
    possible_locations.append(libamd_smi_cwd)

    for location in possible_locations:
        try:
            lib = ctypes.CDLL(location)
            return lib, location
        except OSError as e:
            err = e
            continue
    raise err

try:
    _libraries['{library_name}'], location = find_smi_library()
    #print(f"found smi lib in [", location, "]")
except OSError as e:
    print(e)
    print("Unable to find {library_name} library try installing amd-smi-lib from your package manager")"""
    else:
        print("Unknown operating system. It is only supporing Linux and Windows.")
        return

    arguments.append("--clang-args=-I" + clang_include_dir + clang_extra_args)
    clangToPy(arguments)

    replace_line(output_file, line_to_replace, new_line)
    write_header(output_file)

    # Custom handling for <anonymous|unnamed> struct in Linux
    if os_platform == "Linux":
        with open(input_file, 'r') as fin:
            input_file_contents = fin.read()
        input_file_array = input_file_contents.split('\n')

        with open(output_file, 'r') as fin:
            output_file_contents = fin.read()
        output_file_array = output_file_contents.split('\n')

        # Find all unamed occurences in the output_file
        struct_name_dict = {}
        for index, line in enumerate(output_file_array):
            if 'amdsmi.h:' in line:
                # Handling "struct_struct (<anonymous:unnamed> at amdsmi.h:<num>:<num>)"
                if 'anonymous' in line or 'unnamed' in line:
                    search_name = 'unnamed'
                    if 'anonymous' in line:
                        search_name = 'anonymous'

                    # Find the amdsmi.h line number for this instance
                    # Example 1:
                    #    class struct_struct (anonymous at amdsmi.h:370:9)(Structure):
                    #    line_num = 370
                    # Example 2:
                    #    class struct_struct (unnamed at amdsmi.h:782:9)(Structure):
                    #    line_num = 782
                    line_num = find_line_num(search_name, line)
                    if line_num == 0:
                        print(f'Error: {index+1}: Could determine amdsmi.h line number in {line}, skipping replacement')
                        continue

                    # Using in amdsmi.h starting at the line_num to find the structure name
                    # Search the following lines for "open curly bracket" that has a name
                    # Example 1:
                    #    369: typedef union {
                    #    370:     struct {
                    #    371:         uint64_t function_number : 3;
                    #    375:     };
                    #    377: } amdsmi_bdf_t;
                    #    struct_name = amdsmi_bdf_t
                    # Example 2:
                    #    782: struct {
                    #    783:     uint64_t gfx;
                    #    786: } engine_usage;
                    #    struct_name = engine_usage
                    struct_name = ''
                    for i in range(1, 50):
                        input_line = input_file_array[line_num + i]
                        if close_curly_brace in input_line:
                            struct_name = find_replacement(close_curly_brace, ';', input_line)
                            struct_name = struct_name[1:-1].strip()
                            if len(struct_name):
                                struct_name = struct_name.split(open_bracket)[0]
                                break
                    if not len(struct_name):
                        print(f'Error: {index+1}: Could not find struct name using line number {line_num}, skipping replacement')
                        continue

                    # Generate the replacement for this line
                    # Example:
                    #     class struct_struct (unnamed at amdsmi.h:782:9)(Structure):
                    # becomes
                    #     class struct_engine_usage(Structure):
                    str_replace = find_replacement('struct_struct', close_parenthesis, line)
                    if len(str_replace) > 0:
                        str_with = f'struct_{struct_name}'
                    else:
                        # Example
                        #     (unnamed at amdsmi.h:787:9)', 'uint32_t', 'uint64_t', 'uint8_t',
                        # becomes
                        #     'struct_memory_usage', 'uint32_t', 'uint64_t', 'uint8_t',
                        str_replace = find_replacement(f'({search_name}', close_parenthesis, line)
                        if len(str_replace) == 0:
                            print(f'Error: {index+1}: Could not find structure name in {line}, skipping replacement')
                            continue
                        str_with = f"'struct_{struct_name}"

                    # Save the line number and struct_name association for possible additional replacements
                    if line_num not in struct_name_dict:
                        struct_name_dict[line_num] = struct_name

                    # Do the replace
                    new_line = line.replace(str_replace, str_with)

                    # Look for special replacements that has the struct_name
                    # Example
                    #     ('_0', struct_struct (anonymous at amdsmi.h:370:9)),
                    # becomes
                    #     ('struct_amdsmi_bdf_t', struct_amdsmi_bdf_t),
                    if '_0' in new_line:
                        new_line = new_line.replace('_0', f'struct_{struct_name}')

                    # Look for special replacements that has an amdsmi.h:
                    # Example
                    #     amdsmi.h:370:9)', 'uint8_t',
                    # becomes
                    #     'uint8_t,
                    if 'amdsmi.h:' in new_line:
                        str_replace = find_replacement('amdsmi.h:', ',', line)
                        if len(str_replace) > 0:
                            new_line = new_line.replace(str_replace, '')

                    # Save the replaced line into the array
                    output_file_array[index] = new_line

            # Look for special replacements
            new_line = output_file_array[index]

            # Example
            #     union_amdsmi_bdf_t._anonymous_ = ('_0',)
            # becomes
            #
            if '_anonymous_' in new_line:
                new_line = ''
                output_file_array[index] = new_line

            # Example
            #     'struct_pcie_static_', 'struct_struct (anonymous at
            # becomes
            #     'struct_pcie_static_',
            name = ", 'struct_struct"
            if name in new_line:
                str_replace = find_replacement(name, '', new_line)
                if len(str_replace) > 0:
                    new_line = new_line.replace(str_replace, ',')
                    output_file_array[index] = new_line

            # Example
            #     amdsmi_get_utilization_count.argtypes = [amdsmi_processor_handle, struct_amdsmi_utilization_counter_t * 0, uint32_t, ctypes.POINTER(ctypes.c_uint64)]
            # becomes
            #     amdsmi_get_utilization_count.argtypes = [amdsmi_processor_handle, ctypes.POINTER(struct_amdsmi_utilization_counter_t), uint32_t, ctypes.POINTER(ctypes.c_uint64)]
            name = "amdsmi_get_utilization_count.argtypes"
            if name in new_line:
                str_replace = find_replacement(name, '', new_line)
                if len(str_replace) > 0:
                    str_with = 'amdsmi_get_utilization_count.argtypes = [amdsmi_processor_handle, ctypes.POINTER(struct_amdsmi_utilization_counter_t), uint32_t, ctypes.POINTER(ctypes.c_uint64)]'
                    new_line = new_line.replace(str_replace, str_with)
                    output_file_array[index] = new_line

        write_file(output_file, output_file_array)

if __name__ == "__main__":
    main()
