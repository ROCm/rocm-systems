#!/usr/bin/env python3
"""Extract compilation flags from build.ninja for the build server.

Parses SPLIT[bc], SPLIT[asm], SPLIT[cobj/hipfb/host] custom commands,
CXX_COMPILER__rccl_Release host compile rules, and the
CXX_SHARED_LIBRARY_LINKER__rccl_Release link command to produce a
configuration file that the rccl-build-server reads at startup.

Usage:
    python3 extract-ninja-flags.py /path/to/rccl/build [output_file]

The output defaults to <build_dir>/build_server_flags.conf.
"""

import subprocess
import sys
from pathlib import Path


def parse_command_line(cmd_str):
    """Split a shell command string into tokens, handling simple quoting."""
    tokens = []
    current = []
    in_quote = None
    for ch in cmd_str:
        if in_quote:
            if ch == in_quote:
                in_quote = None
            else:
                current.append(ch)
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch == ' ' or ch == '\t':
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append(''.join(current))
    return tokens


def extract_flags_from_command(tokens):
    """Given tokenized COMMAND, return (compiler, flags, output, source).

    Expects: cd <dir> && <compiler> <flags...> -o <output> <source>
    """
    start = 0
    if tokens and tokens[0] == 'cd':
        try:
            amp_idx = tokens.index('&&')
            start = amp_idx + 1
        except ValueError:
            pass

    compiler = tokens[start]
    rest = tokens[start + 1:]

    source = rest[-1]
    rest = rest[:-1]

    output = None
    flags = []
    skip_next = False
    for i, tok in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        if tok == '-o' and i + 1 < len(rest):
            output = rest[i + 1]
            skip_next = True
        else:
            flags.append(tok)

    return compiler, flags, output, source


def find_split_commands(ninja_path):
    """Parse build.ninja and return SPLIT[bc] and SPLIT[asm] commands."""
    callee_bc = []
    kernel_bc = []
    asm_cmds = []

    with open(ninja_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if 'DESC = SPLIT[bc]' in line or 'DESC = SPLIT[asm]' in line:
            is_asm = 'SPLIT[asm]' in line
            cmd_line = None
            for j in range(i - 1, max(i - 5, -1), -1):
                if lines[j].strip().startswith('COMMAND ='):
                    cmd_line = lines[j].strip()
                    break

            if cmd_line:
                cmd_str = cmd_line[len('COMMAND = '):]
                tokens = parse_command_line(cmd_str)
                compiler, flags, output, source = extract_flags_from_command(tokens)
                entry = {
                    'compiler': compiler,
                    'flags': flags,
                    'output': output,
                    'source': source,
                }
                if is_asm:
                    asm_cmds.append(entry)
                elif '-DNCCL_FUNC_ONLY' in flags:
                    callee_bc.append(entry)
                else:
                    kernel_bc.append(entry)
        i += 1

    return callee_bc, kernel_bc, asm_cmds


def find_split_post_commands(ninja_path):
    """Parse build.ninja for SPLIT[cobj], SPLIT[hipfb], SPLIT[host] commands."""
    result = {}
    with open(ninja_path) as f:
        lines = f.readlines()

    for marker in ('SPLIT[cobj]', 'SPLIT[hipfb]', 'SPLIT[host]'):
        for i, line in enumerate(lines):
            if f'DESC = {marker}' in line:
                for j in range(i - 1, max(i - 5, -1), -1):
                    if lines[j].strip().startswith('COMMAND ='):
                        cmd_str = lines[j].strip()[len('COMMAND = '):]
                        tokens = parse_command_line(cmd_str)
                        if tokens and tokens[0] == 'cd':
                            try:
                                amp = tokens.index('&&')
                                tokens = tokens[amp + 1:]
                            except ValueError:
                                pass
                        result[marker] = tokens
                        break
                break

    return result


def find_host_compile_rules(ninja_path):
    """Parse CXX_COMPILER__rccl_Release build statements.

    Returns (sources, defines, flags, includes) where sources is a list of
    absolute source file paths, and defines/flags/includes are from the
    first build statement (all host TUs share identical flags in CMake).
    Also returns onerank source separately.
    """
    sources = []
    onerank_source = None
    ref_defines = ref_flags = ref_includes = None

    with open(ninja_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if ': CXX_COMPILER__rccl_Release ' in line:
            # "build <out>: CXX_COMPILER__rccl_Release <source> || ..."
            parts = line.split(': CXX_COMPILER__rccl_Release ', 1)
            if len(parts) == 2:
                rest = parts[1]
                src = rest.split('||')[0].split('|')[0].strip()
                is_onerank = 'onerank.cu.cpp' in src

                # Read the variable block that follows
                defines = flags = includes = ''
                j = i + 1
                while j < len(lines) and lines[j].startswith('  '):
                    stripped = lines[j].strip()
                    if stripped.startswith('DEFINES = '):
                        defines = stripped[len('DEFINES = '):]
                    elif stripped.startswith('FLAGS = '):
                        flags = stripped[len('FLAGS = '):]
                    elif stripped.startswith('INCLUDES = '):
                        includes = stripped[len('INCLUDES = '):]
                    j += 1

                if is_onerank:
                    onerank_source = src
                else:
                    sources.append(src)

                if ref_defines is None:
                    ref_defines = defines
                    ref_flags = flags
                    ref_includes = includes

        i += 1

    return sources, onerank_source, ref_defines or '', ref_flags or '', ref_includes or ''


def find_link_command(ninja_path):
    """Parse CXX_SHARED_LIBRARY_LINKER__rccl_Release for librccl.so link info.

    Returns (link_flags, link_libraries, link_path, object_inputs, soname).
    """
    with open(ninja_path) as f:
        lines = f.readlines()

    link_flags = ''
    link_libraries = ''
    link_path = ''
    soname = ''
    object_inputs = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if ': CXX_SHARED_LIBRARY_LINKER__rccl_Release ' in line:
            parts = line.split(': CXX_SHARED_LIBRARY_LINKER__rccl_Release ', 1)
            if len(parts) == 2:
                inputs_str = parts[1].split('||')[0].split('|')[0].strip()
                object_inputs = inputs_str.split()

            j = i + 1
            while j < len(lines) and lines[j].startswith('  '):
                stripped = lines[j].strip()
                if stripped.startswith('LINK_FLAGS = '):
                    link_flags = stripped[len('LINK_FLAGS = '):]
                elif stripped.startswith('LINK_LIBRARIES = '):
                    link_libraries = stripped[len('LINK_LIBRARIES = '):]
                elif stripped.startswith('LINK_PATH = '):
                    link_path = stripped[len('LINK_PATH = '):]
                elif stripped.startswith('SONAME = '):
                    soname = stripped[len('SONAME = '):]
                j += 1
            break

        i += 1

    return link_flags, link_libraries, link_path, object_inputs, soname


def extract_gpu_arch(flags):
    """Extract the GPU architecture from --offload-arch=XXX."""
    for f in flags:
        if f.startswith('--offload-arch='):
            return f.split('=', 1)[1]
    return None


def extract_mllvm_flags(flags):
    """Extract -mllvm <arg> pairs from a flag list, returning the LLVM args."""
    llvm_flags = []
    i = 0
    while i < len(flags):
        if flags[i] == '-mllvm' and i + 1 < len(flags):
            llvm_flags.append(flags[i + 1])
            i += 2
        else:
            i += 1
    return llvm_flags


def get_resource_dir(compiler):
    """Query the compiler for its resource directory."""
    try:
        result = subprocess.run(
            [compiler, '--print-resource-dir'],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def write_config(path, compiler, gpu_arch, resource_dir, callee_flags,
                 kernel_flags, backend_flags, callee_sources, kernel_sources,
                 host_flags=None, host_sources=None,
                 onerank_source=None, onerank_flags=None,
                 split_cobj_args=None, split_hipfb_args=None,
                 split_host_args=None,
                 link_flags=None, link_libraries=None, link_path=None,
                 link_objects=None, soname=None):
    """Write the build server configuration file."""
    with open(path, 'w') as f:
        f.write("# rccl-build-server flag configuration\n")
        f.write("# Auto-generated from build.ninja by extract-ninja-flags.py\n")
        f.write("# Re-run after any CMake reconfigure.\n\n")

        f.write("[meta]\n")
        f.write(f"compiler={compiler}\n")
        f.write(f"gpu_arch={gpu_arch}\n")
        if resource_dir:
            f.write(f"resource_dir={resource_dir}\n")
        f.write("\n")

        f.write("[callee_flags]\n")
        for flag in callee_flags:
            f.write(f"{flag}\n")
        f.write("\n")

        f.write("[kernel_flags]\n")
        for flag in kernel_flags:
            f.write(f"{flag}\n")
        f.write("\n")

        f.write("[backend_flags]\n")
        for flag in backend_flags:
            f.write(f"{flag}\n")
        f.write("\n")

        f.write("[callee_sources]\n")
        for src in sorted(callee_sources):
            f.write(f"{src}\n")
        f.write("\n")

        f.write("[kernel_sources]\n")
        for src in sorted(kernel_sources):
            f.write(f"{src}\n")
        f.write("\n")

        # --- Host compilation ---
        if host_flags is not None:
            f.write("[host_flags]\n")
            for flag in host_flags:
                f.write(f"{flag}\n")
            f.write("\n")

        if host_sources is not None:
            f.write("[host_sources]\n")
            for src in sorted(host_sources):
                f.write(f"{src}\n")
            f.write("\n")

        if onerank_source:
            f.write("[onerank_source]\n")
            f.write(f"{onerank_source}\n")
            f.write("\n")

        if onerank_flags is not None:
            f.write("[onerank_flags]\n")
            for flag in onerank_flags:
                f.write(f"{flag}\n")
            f.write("\n")

        # --- SPLIT post-device commands ---
        if split_cobj_args is not None:
            f.write("[split_cobj]\n")
            for arg in split_cobj_args:
                f.write(f"{arg}\n")
            f.write("\n")

        if split_hipfb_args is not None:
            f.write("[split_hipfb]\n")
            for arg in split_hipfb_args:
                f.write(f"{arg}\n")
            f.write("\n")

        if split_host_args is not None:
            f.write("[split_host]\n")
            for arg in split_host_args:
                f.write(f"{arg}\n")
            f.write("\n")

        # --- Final link ---
        if link_flags:
            f.write("[link_flags]\n")
            for tok in parse_command_line(link_flags):
                f.write(f"{tok}\n")
            f.write("\n")

        if link_libraries:
            f.write("[link_libraries]\n")
            for tok in parse_command_line(link_libraries):
                f.write(f"{tok}\n")
            f.write("\n")

        if link_path:
            f.write("[link_path]\n")
            for tok in parse_command_line(link_path):
                f.write(f"{tok}\n")
            f.write("\n")

        if link_objects:
            f.write("[link_objects]\n")
            for obj in link_objects:
                f.write(f"{obj}\n")
            f.write("\n")

        if soname:
            f.write("[link_soname]\n")
            f.write(f"{soname}\n")
            f.write("\n")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <build_dir> [output_file]", file=sys.stderr)
        sys.exit(1)

    build_dir = Path(sys.argv[1]).resolve()
    ninja_path = build_dir / 'build.ninja'
    if not ninja_path.exists():
        print(f"Error: {ninja_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else str(build_dir / 'build_server_flags.conf')

    callee_bc, kernel_bc, asm_cmds = find_split_commands(str(ninja_path))

    if not callee_bc:
        print("Error: no callee SPLIT[bc] commands found", file=sys.stderr)
        sys.exit(1)
    if not kernel_bc:
        print("Error: no kernel SPLIT[bc] commands found", file=sys.stderr)
        sys.exit(1)

    ref_callee = callee_bc[0]
    for cmd in callee_bc[1:]:
        if cmd['flags'] != ref_callee['flags']:
            print(f"Warning: callee flags differ between {cmd['source']} "
                  f"and {ref_callee['source']}", file=sys.stderr)

    ref_kernel = kernel_bc[0]
    for cmd in kernel_bc[1:]:
        if cmd['flags'] != ref_kernel['flags']:
            print(f"Warning: kernel flags differ between {cmd['source']} "
                  f"and {ref_kernel['source']}", file=sys.stderr)

    compiler = ref_callee['compiler']
    gpu_arch = extract_gpu_arch(ref_callee['flags'])
    if not gpu_arch:
        print("Error: could not extract GPU arch from flags", file=sys.stderr)
        sys.exit(1)

    # Extract backend (SPLIT[asm]) flags — specifically the -mllvm options
    # that control LLVM backend behaviour.
    backend_flags = []
    if asm_cmds:
        ref_asm = asm_cmds[0]
        for cmd in asm_cmds[1:]:
            if cmd['flags'] != ref_asm['flags']:
                print(f"Warning: asm flags differ between {cmd['source']} "
                      f"and {ref_asm['source']}", file=sys.stderr)
                break
        backend_flags = extract_mllvm_flags(ref_asm['flags'])

    callee_sources = [cmd['source'] for cmd in callee_bc]
    kernel_sources = [cmd['source'] for cmd in kernel_bc]
    resource_dir = get_resource_dir(compiler)

    # --- Parse host compile rules ---
    host_sources, onerank_source, host_defines, host_flags_str, host_includes = \
        find_host_compile_rules(str(ninja_path))

    # Build the host compiler flags list: DEFINES + INCLUDES + FLAGS
    # (mirroring the rule template: $DEFINES $INCLUDES $FLAGS)
    host_flags = []
    if host_defines:
        host_flags.extend(parse_command_line(host_defines))
    if host_includes:
        host_flags.extend(parse_command_line(host_includes))
    if host_flags_str:
        host_flags.extend(parse_command_line(host_flags_str))

    # onerank uses the same flags but needs the full HIP pipeline (no --offload-host-only)
    onerank_flags = list(host_flags)

    # --- Parse SPLIT post-device commands ---
    split_cmds = find_split_post_commands(str(ninja_path))
    split_cobj_args = split_cmds.get('SPLIT[cobj]')
    split_hipfb_args = split_cmds.get('SPLIT[hipfb]')
    split_host_args = split_cmds.get('SPLIT[host]')

    # --- Parse link command ---
    link_flags, link_libraries, link_path, link_objects, soname = \
        find_link_command(str(ninja_path))

    write_config(
        output_path,
        compiler=compiler,
        gpu_arch=gpu_arch,
        resource_dir=resource_dir,
        callee_flags=ref_callee['flags'],
        kernel_flags=ref_kernel['flags'],
        backend_flags=backend_flags,
        callee_sources=callee_sources,
        kernel_sources=kernel_sources,
        host_flags=host_flags,
        host_sources=host_sources,
        onerank_source=onerank_source,
        onerank_flags=onerank_flags,
        split_cobj_args=split_cobj_args,
        split_hipfb_args=split_hipfb_args,
        split_host_args=split_host_args,
        link_flags=link_flags,
        link_libraries=link_libraries,
        link_path=link_path,
        link_objects=link_objects,
        soname=soname,
    )

    n_callee = len(callee_sources)
    n_kernel = len(kernel_sources)
    n_host = len(host_sources)

    print(f"Extracted from {ninja_path}")
    print(f"  Compiler:  {compiler}")
    print(f"  GPU arch:  {gpu_arch}")
    print(f"  Callee:    {n_callee} sources, {len(ref_callee['flags'])} flags")
    print(f"  Kernel:    {n_kernel} sources, {len(ref_kernel['flags'])} flags")
    print(f"  Backend:   {len(backend_flags)} LLVM flags")
    if backend_flags:
        for f in backend_flags:
            print(f"             {f}")
    print(f"  Host:      {n_host} sources, {len(host_flags)} flags")
    if onerank_source:
        print(f"  Onerank:   {onerank_source}")
    for name in ('SPLIT[cobj]', 'SPLIT[hipfb]', 'SPLIT[host]'):
        args = split_cmds.get(name)
        if args:
            print(f"  {name}: {args[0]} ...")
    if link_flags or link_libraries:
        print(f"  Link:      flags={len(parse_command_line(link_flags))} "
              f"libs={len(parse_command_line(link_libraries))} "
              f"objects={len(link_objects)}")
    print(f"  Output:    {output_path}")

    callee_set = set(ref_callee['flags'])
    kernel_set = set(ref_kernel['flags'])
    only_callee = callee_set - kernel_set
    only_kernel = kernel_set - callee_set
    if only_callee or only_kernel:
        print(f"\n  Flag differences (callee vs kernel):")
        for f in sorted(only_callee):
            print(f"    callee only: {f}")
        for f in sorted(only_kernel):
            print(f"    kernel only: {f}")


if __name__ == '__main__':
    main()
