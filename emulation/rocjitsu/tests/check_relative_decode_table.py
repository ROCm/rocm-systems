# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check linked REL32 table entries against the generated factory rows."""

import argparse
import os
import re
import struct
import subprocess
from bisect import bisect_left
from pathlib import Path


def inspect(tool, *options):
    return subprocess.check_output(
        [tool, *options], text=True, env={**os.environ, 'LC_ALL': 'C'}
    ).splitlines()


def decoder_symbols(nm, image):
    symbols = {}
    lines = inspect(nm, '--defined-only', '-S', '--format=posix', image)
    if not lines:
        raise AssertionError(f'{image}: no symbols available to inspect')
    for line in lines:
        if line.startswith('rj_'):
            name, kind, address, *size = line.split()
            symbols[name] = kind, int(address, 16), int(size[0], 16) if size else 0
    return symbols


def expected_tables(source):
    assembly = source.with_suffix('.S')
    prefix = re.search(
        r'^#define RJ_FACTORY_NAME\(scope, name\) (rj_\w+)_##scope##_##name$',
        assembly.read_text(),
        re.MULTILINE,
    )
    if not prefix:
        raise ValueError(f'{assembly}: missing factory prefix')
    macros = dict(
        re.findall(
            r'^#define (RJ_\w+_ROWS(?:_\d+)?)\s+(.+)$',
            source.read_text().replace('\\\n', ''),
            re.MULTILINE,
        )
    )
    roots = [name for name in macros if name.endswith('_ROWS')]
    if not roots:
        raise ValueError(f'{source}: no factory rows')
    for name in roots:
        rows = macros[name].strip()
        if rows.startswith(name):
            rows = ' '.join(macros[chunk] for chunk in rows.split())
        default = re.search(r'RJ_DECODE_DEFAULT\((\w+),\s*(\w+)\)', rows)
        if not default:
            raise ValueError(f'{name}: missing default')
        default_target = f'{prefix[1]}_{default[1]}_{default[2]}'
        slots = {}
        for slot, scope, factory in re.findall(
            r'RJ_DECODE_(?:CASE|ALIAS)\((\d+),\s*(\w+),\s*(\w+)\)', rows
        ):
            slots[int(slot)] = f'{prefix[1]}_{scope}_{factory}'
        for first, count in re.findall(
            r'RJ_DECODE_GAP\((\d+),\s*(\d+),\s*\w+,\s*\w+\)', rows
        ):
            # Compare the assembled gap with the portable switch's default.
            for slot in range(int(first), int(first) + int(count)):
                slots[slot] = default_target
        table = name.removeprefix('RJ_').removesuffix('_ROWS').lower()
        yield f'{prefix[1]}_table_{table}', [slots[slot] for slot in range(len(slots))]


def dynamic_relocations(image):
    # REL32 tables are only available on little-endian ELF64 x86. Inspect
    # allocated relocation sections, including packed RELR if the linker uses it.
    if image[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', image, 18)[0] != 62:
        raise ValueError('relative tables require little-endian x86-64 ELF')
    section_offset = struct.unpack_from('<Q', image, 40)[0]
    entry_size, entry_count = struct.unpack_from('<HH', image, 58)
    offsets = set()
    for index in range(entry_count):
        header = section_offset + index * entry_size
        _, kind, flags, _, start, size = struct.unpack_from('<IIQQQQ', image, header)
        if not flags & 2:  # SHF_ALLOC
            continue
        if kind in (4, 9):  # SHT_RELA, SHT_REL
            entries = struct.iter_unpack(
                '<QQq' if kind == 4 else '<QQ', image[start : start + size]
            )
            offsets.update(entry[0] for entry in entries)
        elif kind == 19:  # SHT_RELR
            cursor = 0
            for (word,) in struct.iter_unpack('<Q', image[start : start + size]):
                if word & 1:
                    offsets.update(
                        cursor + 8 * bit for bit in range(63) if word & (2 << bit)
                    )
                    cursor += 63 * 8
                else:
                    offsets.add(word)
                    cursor = word + 8
    return sorted(offsets)


def check(args):
    symbols = decoder_symbols(args.nm, args.image)
    marker = decoder_symbols(args.nm, args.mode_from) if args.mode_from else symbols
    relative = 'rj_test_expect_decode_rel32' in marker
    switch = 'rj_test_expect_decode_switch' in marker
    if relative == switch:
        raise AssertionError('missing or ambiguous decoder representation marker')
    actual = {name for name in symbols if re.fullmatch(r'rj_\w+_table_\w+', name)}
    if args.expect_switch and not switch:
        raise AssertionError('portable fixture was compiled without switch dispatch')
    if switch:
        if actual:
            raise AssertionError(f'switch dispatch retained {sorted(actual)}')
        return

    if args.model:
        maps = [args.source_root / args.model / 'decode_tables.inc']
    else:
        maps = sorted(args.source_root.glob('*/decode_tables.inc'))
    expected = dict(table for source in maps for table in expected_tables(source))
    if not expected or actual != expected.keys():
        raise AssertionError(f'unexpected table set: {actual ^ expected.keys()}')
    if args.table_object:
        # A linked image can hide a definition even when its archive reference
        # permitted a shared library to supply the factory.
        hidden = {
            line.split()[-1]
            for line in inspect(args.readelf, '-Ws', args.table_object)
            if re.search(r'\bGLOBAL\s+HIDDEN\s+UND(?:EF)?\s+rj_\w+$', line)
        }
        factories = {factory for slots in expected.values() for factory in slots}
        if missing := factories - hidden:
            raise AssertionError(f'{min(missing)}: expected a hidden undefined factory')
    headers = inspect(args.readelf, '-lWd', args.image)
    if any('TEXTREL' in line for line in headers):
        raise AssertionError('image permits text relocations')
    loads = [line.split() for line in headers if line.lstrip().startswith('LOAD ')]
    image = args.image.read_bytes()
    relocations = dynamic_relocations(image)
    for name, factories in expected.items():
        kind, start, size = symbols[name]
        if kind not in ('R', 'r') or size != len(factories) * 4:
            raise AssertionError(f'{name}: wrong symbol type or size')
        end = start + size
        for fields in loads:
            segment = int(fields[2], 16)
            if segment <= start and end <= segment + int(fields[4], 16):
                flags = ''.join(fields[6:-1])
                if 'R' not in flags or 'W' in flags:
                    raise AssertionError(f'{name}: expected a read-only segment')
                pos = int(fields[1], 16) + start - segment
                break
        else:
            raise AssertionError(f'{name}: outside a file-backed load segment')
        index = bisect_left(relocations, start)
        if index < len(relocations) and relocations[index] < end:
            raise AssertionError(
                f'{name}: dynamic relocation at {relocations[index]:#x}'
            )
        for slot, factory in enumerate(factories):
            target = start + struct.unpack_from('<i', image, pos + slot * 4)[0]
            if factory not in symbols:
                raise AssertionError(f'{name}[{slot}]: missing factory {factory}')
            kind, address, _ = symbols[factory]
            if kind not in ('T', 't', 'W', 'w') or target != address:
                raise AssertionError(
                    f'{name}[{slot}]: target {target:#x}, expected {factory} at {address:#x}'
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nm')
    parser.add_argument('readelf')
    parser.add_argument('source_root', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('model', nargs='?')
    parser.add_argument('--mode-from', type=Path)
    parser.add_argument('--table-object', type=Path)
    parser.add_argument('--expect-switch', action='store_true')
    check(parser.parse_args())
