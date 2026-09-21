# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import re
import argparse
import tempfile
import shutil
import platform
import re
from subprocess import run, PIPE
from pathlib import Path
from ctypeslib.clang2py import main as clangToPy

HEADER = """# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
"""


def parseArgument():
    parser = argparse.ArgumentParser(description="parse input arguments")
    parser.add_argument("-o", "--output", type=str, required=True, help="The output file name")
    parser.add_argument("-i", "--input", type=str, required=True, help="The input file name")
    parser.add_argument(
        "-l", "--library", type=str, required=True, help="Loading dynamic link libraries"
    )
    parser.add_argument(
        "-e", "--extra-args", type=str, required=False, help="Parse extra arguments to clang"
    )
    args = vars(parser.parse_args())

    return args["output"], args["input"], args["library"], args["extra_args"]


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
    with os.fdopen(fh, "w") as new_file:
        with open(full_path_file_name, "r+", encoding="UTF-8") as old_file:
            for line in old_file:
                new_file.write(line.replace(string_to_replace, new_string))

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def write_header(full_path_file_name):
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, "w") as new_file:
        new_file.write(HEADER)
        with open(full_path_file_name, "r+", encoding="UTF-8") as old_file:
            for line in old_file:
                new_file.write(line)

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def write_file(full_path_file_name, contents):
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, "w") as new_file:
        for line in contents:
            new_file.write(f"{line}\n")

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def insert_layout_ms(full_path_file_name):
    # Python 3.14 deprecates the implicit ctypes layout when _pack_ is set
    # (becomes an error in 3.19). _layout_ = 'ms' keeps the MSVC-compatible
    # layout that packed structs already used pre-3.14, so the ABI is unchanged.
    fh, abs_path = tempfile.mkstemp()
    standalone = re.compile(r"^(\s*)(\S+)\._pack_ = ")
    inline = re.compile(r"^(\s+)_pack_ = ")
    pack_count = 0
    layout_count = 0
    with os.fdopen(fh, "w", encoding="UTF-8") as new_file:
        with open(full_path_file_name, "r", encoding="UTF-8") as old_file:
            for line in old_file:
                new_file.write(line)
                if "._pack_ = " in line or line.lstrip().startswith("_pack_ = "):
                    pack_count += 1
                m = standalone.match(line)
                if m:
                    layout_count += 1
                    new_file.write(f"{m.group(1)}{m.group(2)}._layout_ = 'ms'\n")
                    continue
                mi = inline.match(line)
                if mi:
                    layout_count += 1
                    new_file.write(f"{mi.group(1)}_layout_ = 'ms'\n")

    if pack_count != layout_count:
        raise RuntimeError(
            f"_layout_ insertion missed a _pack_ style: found {pack_count} "
            f"_pack_ lines but added {layout_count} _layout_ lines"
        )

    shutil.copymode(full_path_file_name, abs_path)
    os.remove(full_path_file_name)
    shutil.move(abs_path, full_path_file_name)


def find_replacement(search_str1, search_str2, line):
    pos1 = line.find(search_str1)
    if pos1 < 0:
        return ""

    if len(search_str2):
        pos2 = line.find(search_str2, pos1)
        if pos2 < 0:
            return ""
    else:
        pos2 = len(line) - 1

    return line[pos1 : pos2 + 1]


def find_line_num(search_str, line):
    pos1 = line.find(search_str)
    if pos1 < 0:
        return 0
    items = line[pos1:].split(":")
    if len(items) < 2:
        return 0

    line_num = items[1].strip()
    if not line_num.isdigit():
        return 0

    line_num = int(line_num)
    return line_num


def main():
    open_bracket = "["
    close_parenthesis = ")"
    close_curly_brace = "}"

    output_file, input_file, library, clang_extra_args = parseArgument()

    # make args string easy to append
    if clang_extra_args is None:
        clang_extra_args = ""
    else:
        clang_extra_args = " " + clang_extra_args

    library_name = os.path.basename(library)

    # SONAME major must match src/CMakeLists.txt SOVERSION, which is itself
    # AMDSMI_LIB_VERSION_MAJOR in include/amd_smi/amdsmi.h. Parse it from the
    # header we are already processing rather than hardcoding it, so a major
    # bump cannot silently desync the loader's system-library SONAME.
    soname_major = ""
    try:
        _m = re.search(
            r"#define\s+AMDSMI_LIB_VERSION_MAJOR\s+(\d+)",
            Path(input_file).read_text(encoding="utf-8"),
        )
        if _m:
            soname_major = _m.group(1)
    except OSError:
        pass
    if not soname_major:
        raise SystemExit(
            "generator.py: could not parse AMDSMI_LIB_VERSION_MAJOR from " + str(input_file)
        )

    clang_include_dir = run(
        ["clang", "--print-resource-dir"], stdout=PIPE, stderr=PIPE, encoding="utf-8"
    ).stdout.strip()

    os_platform = platform.system()
    if os_platform == "Windows":
        clang_include_dir += "\\include"
        if "Program Files(x86)" in clang_include_dir:
            clang_include_dir = clang_include_dir.replace("Program Files(x86)", "Progra~2")
        elif "Program Files" in clang_include_dir:
            clang_include_dir = clang_include_dir.replace("Program Files", "Progra~1")

        arguments = [input_file, "-o", output_file]
        line_to_replace = (
            "_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')"
        )
        new_line = "_libraries['FIXME_STUB'] = ctypes.CDLL('{}')".format(library_name)
    elif os_platform == "Linux":
        clang_include_dir += "/include"
        arguments = [input_file, "-o", output_file, "-l", library]
        library_path = os.path.join(os.path.dirname(__file__), library)
        line_to_replace = "_libraries['{}'] = ctypes.CDLL('{}')".format(library_name, library_path)
        new_line = f"""from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic library loading
# ---------------------------------------------------------------------------
# This wrapper supports three self-contained install paths:
#
#   1. pip wheel
#      The wheel ships ``libamd_smi_python.so`` next to this file.
#
#   2. system package (rpm / deb)
#      The wrapper is placed in the system Python's ``site-packages``;
#      the package also drops an ``ld.so.conf.d`` entry pointing at
#      ``/opt/rocm/lib`` so the dynamic linker resolves the SONAME
#      ``libamd_smi.so.<SOVERSION>`` without further help.
#
#   3. relocatable ROCm tree (TheRock rocm-sdk wheels, portable tarball)
#      Wrapper at ``<root>/share/amd_smi/amdsmi/``, library at
#      ``<root>/lib/<SONAME>`` -- resolved by fixed relative path.
#
# A user installs ONE of these packages. We never combine paths
# from them -- no ROCM_HOME / ROCM_PATH ladders, no walking up to a
# ROCm root, no LD_LIBRARY_PATH probing. ``AMDSMI_LIB_OVERRIDE`` stays
# as a single-purpose escape hatch for ABI tests that need to load an
# alternate .so explicitly.
# ---------------------------------------------------------------------------


# SONAME the system rpm/deb ships and the relocatable tree names. Always the
# system lib, never the wheel-private libamd_smi_python.so, even when the
# wrapper is generated with -l libamd_smi_python.so. Major = src/CMakeLists.txt
# SOVERSION (= AMDSMI_LIB_VERSION_MAJOR in amdsmi.h).
_AMDSMI_LIB_SONAME = "libamd_smi.so.{soname_major}"

# Whether the loader may fall back to the system SONAME after the bundled
# wheel-library check. The committed wrapper and the system rpm/deb keep this
# True. The pip wheel build flips it to False at package time so a wheel never
# loads a system libamd_smi.so -- a wheel missing its bundled
# libamd_smi_python.so fails loudly instead of silently importing an unrelated
# system library (which would risk symbol conflicts inside PyTorch / JAX).
_AMDSMI_ALLOW_SYSTEM_FALLBACK = True


def _load_library():
    \"\"\"Load the AMD SMI shared library.

    Order:
      1. ``AMDSMI_LIB_OVERRIDE`` env var (ABI-test escape hatch).
      2. ``libamd_smi_python.so`` next to this file (pip wheel).
      3. ``<root>/lib/<SONAME>`` relative to this file (relocatable ROCm tree).
      4. SONAME via the dynamic linker (system rpm / deb); skipped when
         _AMDSMI_ALLOW_SYSTEM_FALLBACK is False (pip wheel).
    \"\"\"
    mode = getattr(ctypes, "RTLD_LOCAL", 0)

    override = os.getenv("AMDSMI_LIB_OVERRIDE")
    if override:
        return ctypes.CDLL(override, mode=mode), override

    bundled = Path(__file__).resolve().parent / "libamd_smi_python.so"
    if bundled.exists():
        return ctypes.CDLL(str(bundled), mode=mode), str(bundled)

    if not _AMDSMI_ALLOW_SYSTEM_FALLBACK:
        raise OSError(
            "bundled libamd_smi_python.so is missing from this amdsmi wheel; "
            "refusing to fall back to a system libamd_smi.so"
        )

    # Relocatable ROCm tree: <root>/lib/<SONAME>, one fixed location relative
    # to this file, tried before the bare-SONAME linker lookup (not a search).
    # amdsmi_interface.py resolves librocm-core.so the same way. A
    # present-but-unloadable file (missing deps) must not shadow the system
    # linker lookup, so fall through on OSError.
    here = Path(__file__).resolve()
    if len(here.parents) > 3:
        relocatable = here.parents[3] / "lib" / _AMDSMI_LIB_SONAME
        if relocatable.exists():
            try:
                return ctypes.CDLL(str(relocatable), mode=mode), str(relocatable)
            except OSError:
                pass

    return ctypes.CDLL(_AMDSMI_LIB_SONAME, mode=mode), _AMDSMI_LIB_SONAME


class _MissingLibrary:
    \"\"\"Sentinel installed when the .so could not be loaded.

    Module import stays tolerant so doc / lint / multi-stage container
    builds work without a runtime ROCm install. Any *call* of a wrapped
    C symbol raises OSError with the underlying error.
    \"\"\"

    __slots__ = ("_err",)

    def __init__(self, err=None):
        object.__setattr__(self, "_err", err)

    def _raise(self):
        raise OSError(
            "AMD SMI shared library could not be loaded.\\n"
            f"Underlying error: {{self._err}}\\n"
            "Hint: install amd-smi-lib (rpm/deb) or pip-install the amdsmi wheel."
        )

    def __getattr__(self, name):
        if not name.startswith("amdsmi_"):
            raise AttributeError(name)
        return _MissingLibrary(self._err)

    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        self._raise()


try:
    _libraries['libamd_smi.so'], _loaded_lib_path = _load_library()
except OSError as _load_err:
    _libraries['libamd_smi.so'] = _MissingLibrary(_load_err)
    _loaded_lib_path = None


#Add support for amdsmi_free_name_value_pairs
amdsmi_free_name_value_pairs = _libraries['libamd_smi.so'].amdsmi_free_name_value_pairs
amdsmi_free_name_value_pairs.restype = None
amdsmi_free_name_value_pairs.argtypes = [ctypes.POINTER(None)]"""
    else:
        print("Unknown operating system. It is only supporting Linux and Windows.")
        return

    arguments.append("--clang-args=-I" + clang_include_dir + clang_extra_args)
    clangToPy(arguments)

    replace_line(output_file, line_to_replace, new_line)
    write_header(output_file)

    # Custom handling for <anonymous|unnamed> struct in Linux
    if os_platform == "Linux":
        with open(input_file, "r") as fin:
            input_file_contents = fin.read()
        input_file_array = input_file_contents.split("\n")

        with open(output_file, "r") as fin:
            output_file_contents = fin.read()
        output_file_array = output_file_contents.split("\n")

        # Find all unnamed occurrences in the output_file
        struct_name_dict = {}
        for index, line in enumerate(output_file_array):
            if "amdsmi.h:" in line:
                # Handling "struct_struct (<anonymous:unnamed> at amdsmi.h:<num>:<num>)"
                if "anonymous" in line or "unnamed" in line:
                    search_name = "unnamed"
                    if "anonymous" in line:
                        search_name = "anonymous"

                    # Find the amdsmi.h line number for this instance
                    # Example 1:
                    #    class struct_struct (anonymous at amdsmi.h:370:9)(Structure):
                    #    line_num = 370
                    # Example 2:
                    #    class struct_struct (unnamed at amdsmi.h:782:9)(Structure):
                    #    line_num = 782
                    line_num = find_line_num(search_name, line)
                    if line_num == 0:
                        print(
                            f"Error: {index + 1}: Could determine amdsmi.h line number in {line}, skipping replacement"
                        )
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
                    struct_name = ""
                    for i in range(1, 50):
                        input_line = input_file_array[line_num + i]
                        if close_curly_brace in input_line:
                            struct_name = find_replacement(close_curly_brace, ";", input_line)
                            struct_name = struct_name[1:-1].strip()
                            if len(struct_name):
                                struct_name = struct_name.split(open_bracket)[0]
                                break
                    if not len(struct_name):
                        print(
                            f"Error: {index + 1}: Could not find struct name using line number {line_num}, skipping replacement"
                        )
                        continue

                    # Generate the replacement for this line
                    # Example:
                    #     class struct_struct (unnamed at amdsmi.h:782:9)(Structure):
                    # becomes
                    #     class struct_engine_usage(Structure):
                    str_replace = find_replacement("struct_struct", close_parenthesis, line)
                    if len(str_replace) > 0:
                        str_with = f"struct_{struct_name}"
                    else:
                        # Example
                        #     (unnamed at amdsmi.h:787:9)', 'uint32_t', 'uint64_t', 'uint8_t',
                        # becomes
                        #     'struct_memory_usage', 'uint32_t', 'uint64_t', 'uint8_t',
                        str_replace = find_replacement(f"({search_name}", close_parenthesis, line)
                        if len(str_replace) == 0:
                            print(
                                f"Error: {index + 1}: Could not find structure name in {line}, skipping replacement"
                            )
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
                    if "_0" in new_line:
                        new_line = new_line.replace("_0", f"struct_{struct_name}")

                    # Look for special replacements that has an amdsmi.h:
                    # Example
                    #     amdsmi.h:370:9)', 'uint8_t',
                    # becomes
                    #     'uint8_t,
                    if "amdsmi.h:" in new_line:
                        str_replace = find_replacement("amdsmi.h:", ",", line)
                        if len(str_replace) > 0:
                            new_line = new_line.replace(str_replace, "")

                    # Save the replaced line into the array
                    output_file_array[index] = new_line

            # Look for special replacements
            new_line = output_file_array[index]

            # Example
            #     union_amdsmi_bdf_t._anonymous_ = ('_0',)
            # becomes
            #
            if "_anonymous_" in new_line:
                new_line = ""
                output_file_array[index] = new_line

            # Example
            #     'struct_pcie_static_', 'struct_struct (anonymous at
            # becomes
            #     'struct_pcie_static_',
            name = ", 'struct_struct"
            if name in new_line:
                str_replace = find_replacement(name, "", new_line)
                if len(str_replace) > 0:
                    new_line = new_line.replace(str_replace, ",")
                    output_file_array[index] = new_line

            # Example
            #     amdsmi_get_utilization_count.argtypes = [amdsmi_processor_handle, struct_amdsmi_utilization_counter_t * 0, uint32_t, ctypes.POINTER(ctypes.c_uint64)]
            # becomes
            #     amdsmi_get_utilization_count.argtypes = [amdsmi_processor_handle, ctypes.POINTER(struct_amdsmi_utilization_counter_t), uint32_t, ctypes.POINTER(ctypes.c_uint64)]
            name = "amdsmi_get_utilization_count.argtypes"
            if name in new_line:
                str_replace = find_replacement(name, "", new_line)
                if len(str_replace) > 0:
                    str_with = "amdsmi_get_utilization_count.argtypes = [amdsmi_processor_handle, ctypes.POINTER(struct_amdsmi_utilization_counter_t), uint32_t, ctypes.POINTER(ctypes.c_uint64)]"
                    new_line = new_line.replace(str_replace, str_with)
                    output_file_array[index] = new_line

        # trim last newline - avoids pre-commit hook error
        if output_file_array[-1] == "":
            output_file_array = output_file_array[:-1]

        write_file(output_file, output_file_array)

    insert_layout_ms(output_file)


if __name__ == "__main__":
    main()
