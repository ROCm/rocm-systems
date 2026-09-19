#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile fresh LLVM code objects and check their flags and execution."""

import argparse
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--rocjitsu', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--runtime-preload', default=os.environ.get('RJ_LLVM_SMOKE_PRELOAD', '')
    )
    parser.add_argument(
        '--target',
        choices=['gfx90a', 'gfx950', 'gfx1250', 'gfx1250-strict', 'gfx12-5-generic'],
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = (
        Path(__file__).parent.parent / 'kernels/llvm_codegen_smoke.ll'
    ).read_text()
    results = []
    env = dict(os.environ, HSA_XNACK='0')

    def run(label, command):
        command = [str(arg) for arg in command]
        print(shlex.join(command), flush=True)
        command_env = env
        if args.runtime_preload and command[0] == str(args.rocjitsu):
            command_env = dict(env)
            command_env['LD_PRELOAD'] = ':'.join(
                part
                for part in [args.runtime_preload, env.get('LD_PRELOAD', '')]
                if part
            )
        try:
            proc = subprocess.run(
                command,
                env=command_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            output, code = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as error:
            output, code = str(error), 124
        (args.output / f'{label}.log').write_text(output)
        results.append({'stage': label, 'command': command, 'returncode': code})
        if code:
            print(output, flush=True)
        return code == 0

    run('llvm-version', [args.llvm_bin / 'llc', '--version'])
    for mode, xnack, sramecc in [
        ('any', None, None),
        ('off-any', 0, None),
        ('on-any', 1, None),
        ('off-on', 0, 1),
        ('on-on', 1, 1),
        ('off-off', 0, 0),
    ]:
        ir = args.output / f'{mode}.ll'
        assembly = ir.with_suffix('.s')
        obj = ir.with_suffix('.o')
        direct_obj = ir.with_suffix('.direct.o')
        code_object = ir.with_suffix('.hsaco')
        module_flags = []
        for name, value in [('xnack', xnack), ('sramecc', sramecc)]:
            if value is not None:
                module_flags.append(
                    f'!{len(module_flags)} = !{{i32 1, !"amdgpu.{name}", i32 {value}}}'
                )
        flags = ''
        if module_flags:
            refs = ', '.join(f'!{i}' for i in range(len(module_flags)))
            flags = (
                f'\n!llvm.module.flags = !{{{refs}}}\n' + '\n'.join(module_flags) + '\n'
            )
        ir.write_text(source + flags)
        # Go through assembly so the new .amdgcn_target handling is exercised,
        # in addition to the module flags used by the code generator.
        if not run(
            f'{mode}-compile',
            [
                args.llvm_bin / 'llc',
                '-mtriple=amdgcn-amd-amdhsa',
                f'-mcpu={args.target}',
                '-O2',
                ir,
                '-o',
                assembly,
            ],
        ):
            continue
        if not run(
            f'{mode}-assemble',
            [
                args.llvm_bin / 'llvm-mc',
                '-triple=amdgcn-amd-amdhsa',
                f'-mcpu={args.target}',
                '-filetype=obj',
                assembly,
                '-o',
                obj,
            ],
        ):
            continue
        if not run(
            f'{mode}-link',
            [args.llvm_bin / 'ld.lld', '-shared', obj, '-o', code_object],
        ):
            continue
        run(
            f'{mode}-headers',
            [args.llvm_bin / 'llvm-readobj', '--file-headers', code_object],
        )
        run(
            f'{mode}-disassemble',
            [args.llvm_bin / 'llvm-objdump', '--disassemble-all', code_object],
        )
        actual = struct.unpack_from('<I', code_object.read_bytes(), 48)[0]
        if run(
            f'{mode}-direct-compile',
            [
                args.llvm_bin / 'llc',
                '-mtriple=amdgcn-amd-amdhsa',
                f'-mcpu={args.target}',
                '-O2',
                '-filetype=obj',
                ir,
                '-o',
                direct_obj,
            ],
        ):
            direct_flags = struct.unpack_from('<I', direct_obj.read_bytes(), 48)[0]
            results.append(
                {
                    'stage': f'{mode}-direct-flags',
                    'actual': hex(direct_flags),
                    'expected': hex(actual),
                    'returncode': 0 if direct_flags == actual else 1,
                }
            )
        machine = {
            'gfx90a': 0x3F,
            'gfx950': 0x4F,
            'gfx1250': 0x49,
            'gfx1250-strict': 0xEB,
            'gfx12-5-generic': 0x5B,
        }[args.target]
        # gfx1250 and strict have XNACK permanently enabled. The target ID
        # omits its mode, but ELF records ON even with an explicit OFF flag.
        if args.target in ('gfx1250', 'gfx1250-strict', 'gfx12-5-generic'):
            expected = machine | (3 << 8)
        else:
            expected = machine | ((1 if xnack is None else xnack + 2) << 8)
        expected |= (1 if sramecc is None else sramecc + 2) << 10
        if args.target == 'gfx12-5-generic':
            expected |= 1 << 24  # Generic ABI version 1.
        matched = actual == expected
        results.append(
            {
                'stage': f'{mode}-flags',
                'actual': hex(actual),
                'expected': hex(expected),
                'returncode': 0 if matched else 1,
            }
        )
        print(f'{mode}: e_flags={actual:#x}, expected={expected:#x}', flush=True)
        fixed_xnack = args.target in ('gfx1250', 'gfx1250-strict', 'gfx12-5-generic')
        # The current HIP/ROCr stack advertises no selectable SRAM-ECC mode
        # for gfx1250. Explicit modes remain unsupported, independently of XNACK.
        incompatible = (
            (sramecc is not None) if fixed_xnack else (sramecc == 0 or xnack == 1)
        )
        action = 'reject' if incompatible else 'execute'
        run(
            f'{mode}-{action}',
            [
                args.rocjitsu,
                '--config',
                args.config,
                '--',
                args.helper,
                action,
                code_object,
            ],
        )

    # Fixed-on targets must keep accepting older producer flags, but must not
    # silently accept an explicitly OFF XNACK contract. These are separate
    # metadata controls; never rewrite the compiler-produced smoke objects.
    original = args.output / 'any.hsaco'
    if (
        args.target in ('gfx1250', 'gfx1250-strict', 'gfx12-5-generic')
        and original.exists()
    ):
        for label, bits, action in [
            ('legacy-xnack-unsupported', 0, 'execute'),
            ('xnack-any-compatible', 0x100, 'execute'),
            ('xnack-off-incompatible', 0x200, 'reject'),
        ]:
            image = bytearray(original.read_bytes())
            flags = struct.unpack_from('<I', image, 48)[0]
            struct.pack_into('<I', image, 48, (flags & ~0x300) | bits)
            control = args.output / f'{label}.hsaco'
            control.write_bytes(image)
            run(
                label,
                [
                    args.rocjitsu,
                    '--config',
                    args.config,
                    '--',
                    args.helper,
                    action,
                    control,
                ],
            )

    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    failed = [r['stage'] for r in results if r['returncode']]
    print('FAIL: ' + ', '.join(failed) if failed else 'PASS: all smoke stages')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
