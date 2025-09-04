import re
import sys
from pathlib import Path
import os
 
start_lineno = -1
comment_lines = []
def parse_api(insert_text):
    match = re.match(r"^(\w+)\((.*)\)$", insert_text.strip())
    if match:
        name, args = match.groups()
        return name, args
    return None, None

def find_pattern(name, args):
    file_path = Path.cwd() / "hip_api_trace.hpp"
    file_path = file_path.resolve()
    inp = open(file_path, 'r+')
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    pattern1 = re.compile(rf'\bend_of_typedefs\b')
    pattern2 = re.compile(rf'\bend_of_dispatch_table\b')
    typedef_idx = None
    dispatch_idx = None

    
    for lineno, line in enumerate(lines, start=1):
        if pattern1.search(line):
           print(f"Function: {name}, Args: {args}")
           api_name = f"typedef hipError_t (*t_{name})({args});\n"
           typedef_idx = lineno - 1
        if pattern2.search(line):
           api_name = f"  t_{name} {name}_fn\n"
           dispatch_idx = lineno - 1
           print(f"Function: {name}, Args: {args}")
        if typedef_idx is not None and dispatch_idx is not None:
            break

    inserts = []
    if typedef_idx is not None:
        inserts.append((typedef_idx, f"typedef hipError_t (*t_{name})({args});\n"))
    if dispatch_idx is not None:
        inserts.append((dispatch_idx, f"  t_{name} {name}_fn\n"))

    # 3) Insert from bottom to top to avoid index shifting
    for idx, text in sorted(inserts, key=lambda t: t[0], reverse=True):
        lines.insert(idx, text)

    # 4) Ensure file ends with newline and write once
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    file_path.write_text("".join(lines), encoding="utf-8")

    file_path.write_text("".join(lines), encoding="utf-8", newline="")

    return -1

def find_pattern2(name, args):
    file_path = Path.cwd().parent.parent.parent / "src" / "hip_api_trace.cpp"
    file_path = file_path.resolve()
    inp = open(file_path, 'r+')
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    new_lines = []
    new_value = None
    first_pattern = re.compile(rf"\bend_of_definitions\b")
    second_pattern = re.compile(rf"\bend_of_dispatch_table\b")
    third_pattern = re.compile(rf"\bend_of_hip_enforce_abi\b")
    def_idx = None
    dispatch_idx = None
    enforce_idx = None
    inserts = []

    new_value = None
    macro_pat = re.compile(r"^\s*#\s*define\s+LAST_HIP_ABI_OFFSET\s+(\d+)")
    for i, line in enumerate(lines):
        m = macro_pat.match(line)
        if m:
            old_value = int(m.group(1))
            new_value = old_value + 1
            lines[i] = f"#define LAST_HIP_ABI_OFFSET {new_value}\n"
            break
    if new_value is None:
        raise ValueError("LAST_HIP_ABI_OFFSET not found")

 
    for lineno, line in enumerate(lines, start=1):
       if first_pattern.search(line):
           def_idx = lineno - 1
           inserts.append((def_idx, f"hipError_t " + name + "(" + args + ");\n"))
       elif second_pattern.search(line):
           dispatch_idx = lineno - 1
           inserts.append((dispatch_idx, f"  ptrDispatchTable->" + name + "_fn = hip::" + name + ";\n"))
       elif third_pattern.search(line):
           enforce_idx = lineno - 1
           inserts.append((enforce_idx, f"HIP_ENFORCE_ABI(HipDispatchTable," +  name + "_fn," +  str(new_value) + ");\n"))

    for idx, text in sorted(inserts, key=lambda t: t[0], reverse=True):
        lines.insert(idx, text)

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    file_path.write_text("".join(lines), encoding="utf-8")

    file_path.write_text("".join(lines), encoding="utf-8", newline="")

    return -1

def find_pattern3(name, args):
    file_path = Path.cwd().parent.parent.parent / "src" / "hip_table_interface.cpp"
    file_path = file_path.resolve()
    inp = open(file_path, 'r+')
    lines = inp.readlines()
    pattern1 = re.compile(rf'\bend_of_table_interface\b')

    for lineno, line in enumerate(lines, start=1):
       if pattern1.search(line):
           api_block = (
               f"hipError_t {name}({args}) {{\n"
               f"  return hip::GetHipDispatchTable()->{name}({args});\n"
               f"}}\n"
           )
           lines.insert(lineno - 1, api_block + "\n")
           break

    file_path.write_text("".join(lines), encoding="utf-8", newline="")
    return -1

def find_pattern4(name, hipversion):
    file_path = Path.cwd().parent.parent.parent / "src" / "hip_hcc.map.in"
    file_path = file_path.resolve()
    inp = open(file_path, 'r+')
    lines = inp.readlines()
    pattern1 = re.compile(rf'\b{hipversion}\b')

    for lineno, line in enumerate(lines, start=1):
       if pattern1.search(line):
           print(f"Function: {line}, Args: {lineno}")
           lines.insert(lineno - 1, "    " + name + "\n")
           break

    file_path.write_text("".join(lines), encoding="utf-8", newline="")
    return -1

def insert_api(filename, lineno, insert_text):
    path = Path(filename).resolve()
    if not path.exists():
       print(f"Error: file not found: {filename}")
       sys.exit(1)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Insert the new API declaration
    lines.insert(lineno - 1, insert_text + "\n")
    print(f"Function: {lineno}, Args: {insert_text}")
    path.write_text("".join(lines), encoding="utf-8")

def insert_end(insert_text):
    file_path = Path.cwd().parent.parent.parent / "src" / "amdhip.def"
    file_path = file_path.resolve()
    inp = open(file_path, 'r+')

    with open(file_path, "a") as f:
        f.write(insert_text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_struct_body.py <cpp_file> <struct_name>")
        sys.exit(1)
 
    hipversion = sys.argv[1]
    api = sys.argv[2]
    api, args = parse_api(api)
    api_name = api

    find_pattern(api_name, args)
    find_pattern2(api_name, args)
    find_pattern3(api_name, args)
    find_pattern4(api_name, hipversion)
    insert_end(api_name)




