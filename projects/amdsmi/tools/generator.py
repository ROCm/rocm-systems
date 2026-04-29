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
from pathlib import Path
from ctypeslib.clang2py import main as clangToPy

HEADER = """# Copyright (C) Advanced Micro Devices. All rights reserved.
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
import sys
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
# Two install contexts, each shipping its own .so under a DIFFERENT name
# so a process with both installed cannot double-load the same SONAME:
#
#   pip wheel    : <site-packages>/amdsmi/libamd_smi_python.so
#   system pkg   : /opt/rocm/lib64/{library_name}   (or .../lib/...)
#
# Pip context wins if libamd_smi_python.so sits next to this wrapper.
# Otherwise we resolve the system .so via the wrapper's path-derived
# ROCm root, then ROCM_HOME / ROCM_PATH, then the dynamic linker.
# Whichever .so is loaded is stored under _libraries['{library_name}'].
# ---------------------------------------------------------------------------

_libraries = {{}}


# AMDSMI_LOADER_BEGIN -- the block bounded by these two markers must
# stay byte-for-byte identical (post-render) with the f-string template
# in tools/generator.py. tools/check_loader_drift.py enforces this.
def _dir_has_libs(p):
    \"\"\"Return True if *p* contains a lib64/ or lib/ subdir (ROCm root marker).\"\"\"
    return (p / "lib64").is_dir() or (p / "lib").is_dir()


def _pip_context_dir(module_dir):
    \"\"\"Return *module_dir* if it looks like a pip-install layout, else None.

    A pip install ships the SONAME-renamed library next to the wrapper:
        <site-packages>/amdsmi/amdsmi_wrapper.py
        <site-packages>/amdsmi/libamd_smi_python.so
    \"\"\"
    if (module_dir / "libamd_smi_python.so").exists():
        return module_dir
    return None


def _system_context_root(module_dir):
    \"\"\"Resolve the ROCm root from a system-package wrapper layout, or None.

    System layout:
        <rocm_root>/share/amd_smi/amdsmi/amdsmi_wrapper.py
        <rocm_root>/lib{{64}}/{library_name}

    Walks three directories up from *module_dir* and confirms the result
    has a lib64 or lib subdir.
    \"\"\"
    candidate = module_dir.parent.parent.parent
    if _dir_has_libs(candidate):
        return candidate
    return None


def _env_rocm_root():
    \"\"\"Return the first valid ROCM_HOME / ROCM_PATH env var, else None.\"\"\"
    for env_var in ("ROCM_HOME", "ROCM_PATH"):
        env_val = os.getenv(env_var)
        if env_val:
            p = Path(env_val).resolve()
            if _dir_has_libs(p):
                return p
    return None


def _detect_install_context():
    \"\"\"Classify the current install as ``"pip"`` or ``"system"``.

    Returns
    -------
    tuple[str, Path]
        ``("pip",    module_dir)``  - *module_dir* contains the wrapper **and**
        ``libamd_smi_python.so``.
        ``("system", rocm_root)``   - *rocm_root* is the resolved ROCm prefix
        (e.g. ``/opt/rocm``).

    All paths are fully resolved so symlinks like
    ``/opt/rocm -> /opt/rocm-X.Y.Z`` are handled automatically.
    \"\"\"
    module_dir = Path(__file__).resolve().parent  # .../amdsmi/

    pip_dir = _pip_context_dir(module_dir)
    if pip_dir is not None:
        return "pip", pip_dir

    sys_root = _system_context_root(module_dir)
    if sys_root is not None:
        return "system", sys_root

    env_root = _env_rocm_root()
    if env_root is not None:
        return "system", env_root

    # Last resort - default ROCm location.
    return "system", Path("/opt/rocm").resolve()


def _build_candidate_paths():
    \"\"\"Return an ordered list of .so paths to try, best-match first.

    AMDSMI_LIB_OVERRIDE, if set, takes precedence over every other
    candidate -- intended for local development against an in-tree build
    and for ABI-compatibility tests that load a curated alternate .so.
    \"\"\"
    override = os.getenv("AMDSMI_LIB_OVERRIDE")
    if override:
        return [Path(override)]

    context, base = _detect_install_context()
    candidates = []

    if context == "pip":
        # .so is self-contained inside the wheel / site-packages.
        # Do NOT fall back to the system {library_name}: loading both libraries
        # in the same process causes segfaults during static initialisation of
        # std::variant tables on older toolchains (GCC 8 / glibc 2.28).
        candidates.append(base / "libamd_smi_python.so")
        return candidates

    # System package - .so lives under <rocm_root>/lib64 or <rocm_root>/lib.
    # Prefer lib64 (where amd-smi packages actually install on x86_64),
    # but accept lib for distributions that use a unified libdir.
    for libdir in ("lib64", "lib"):
        candidates.append(base / libdir / "{library_name}")

    # Fallbacks - same lib64-then-lib precedence
    for env_var in ("ROCM_HOME", "ROCM_PATH"):
        env_val = os.getenv(env_var)
        if env_val:
            for libdir in ("lib64", "lib"):
                candidates.append(Path(env_val) / libdir / "{library_name}")

    # Let the dynamic linker try LD_LIBRARY_PATH / ld.so.conf.d
    candidates.append("{library_name}")

    return candidates


def _load_library():
    \"\"\"Load the AMD SMI shared library for the detected install context.

    Returns
    -------
    tuple[ctypes.CDLL, str]
        The loaded library handle and the path that was successfully loaded.

    Raises
    ------
    OSError
        If none of the candidate paths could be loaded.
    \"\"\"
    candidates = _build_candidate_paths()
    last_err = None
    mode = getattr(ctypes, "RTLD_LOCAL", 0)
    debug = bool(os.getenv("AMDSMI_DEBUG_LOAD"))

    for candidate in candidates:
        try:
            lib = ctypes.CDLL(str(candidate), mode=mode)
            resolved = str(candidate)
            if debug:
                sys.stderr.write(f"[amdsmi] loaded {{resolved}}\\n")
            return lib, resolved
        except OSError as exc:
            if debug:
                sys.stderr.write(f"[amdsmi] WARNING: {{exc}}\\n")
            last_err = exc

    raise last_err or OSError(
        "Could not load AMD SMI library.  Searched:\\n"
        + "\\n".join(f"  - {{c}}" for c in candidates)
    )


class _MissingLibrary:
    \"\"\"Sentinel installed when the .so could not be loaded.

    Module import stays tolerant so downstream tools (sphinx-autodoc,
    mypy/ruff, api_summary, multi-stage container builds) still work
    without a runtime ROCm install. Any *call* of a wrapped C symbol
    raises OSError listing every path that was attempted.
    \"\"\"

    __slots__ = ("_err", "_searched")

    def __init__(self, err=None, searched=()):
        object.__setattr__(self, "_err", err)
        object.__setattr__(self, "_searched", tuple(searched))

    def _raise(self):
        searched_lines = "\\n".join(f"  - {{c}}" for c in self._searched)
        raise OSError(
            "AMD SMI shared library could not be loaded.\\n"
            f"Underlying error: {{self._err}}\\n"
            "Searched the following paths (in order):\\n"
            f"{{searched_lines}}\\n"
            "Hint: install amd-smi-lib (rpm/deb) or set ROCM_PATH/ROCM_HOME "
            "to your ROCm install root, or add the directory containing "
            "{library_name} to LD_LIBRARY_PATH."
        )

    # Chained `lib.amdsmi_init` access at module load time is tolerated;
    # unrelated attribute names (feature probes) honestly miss.
    def __getattr__(self, name):
        if not name.startswith("amdsmi_"):
            raise AttributeError(name)
        return _MissingLibrary(self._err, self._searched)

    # Tolerate `func.restype = X` / `func.argtypes = [...]` at module load.
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        self._raise()
# AMDSMI_LOADER_END


try:
    _libraries['{library_name}'], _loaded_lib_path = _load_library()
except OSError as _load_err:
    # Defer the failure to first use. Importing the module still succeeds so
    # that doc/lint/CI tools that do not actually call into the .so keep
    # working. The first attribute access on any wrapped function raises
    # with a helpful message. Per-candidate detail is opt-in via
    # AMDSMI_DEBUG_LOAD (already emitted by _load_library above); the line
    # below is just the silent-mode summary.
    if os.getenv("AMDSMI_DEBUG_LOAD"):
        sys.stderr.write(
            "[amdsmi] Module imported in degraded mode; calling any "
            "amdsmi_* function will raise OSError.\\n"
        )
    _libraries['{library_name}'] = _MissingLibrary(
        _load_err, _build_candidate_paths()
    )
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

        write_file(output_file, output_file_array)


if __name__ == "__main__":
    main()
