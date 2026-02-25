#!/usr/bin/env python3
"""Extract device compilation flags from build.ninja for the build server.

Parses the SPLIT[bc] and SPLIT[asm] custom commands in build.ninja to produce
a configuration file that the rccl-build-server reads at startup.  This
replaces hardcoded compiler flags with the canonical flags from CMake.

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
                 kernel_flags, backend_flags, callee_sources, kernel_sources):
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
    )

    n_callee = len(callee_sources)
    n_kernel = len(kernel_sources)

    print(f"Extracted from {ninja_path}")
    print(f"  Compiler:  {compiler}")
    print(f"  GPU arch:  {gpu_arch}")
    print(f"  Callee:    {n_callee} sources, {len(ref_callee['flags'])} flags")
    print(f"  Kernel:    {n_kernel} sources, {len(ref_kernel['flags'])} flags")
    print(f"  Backend:   {len(backend_flags)} LLVM flags")
    if backend_flags:
        for f in backend_flags:
            print(f"             {f}")
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
