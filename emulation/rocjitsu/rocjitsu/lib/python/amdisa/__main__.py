# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point: ``python -m amdisa``."""

import argparse
import sys
import xml.etree.ElementTree as elem_tree

from amdisa import (
    Cdna1Profile,
    Cdna2Profile,
    CdnaProfile,
    CodegenConfig,
    CodeGenerator,
    Parser,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
)
from amdisa import xml_schema as xs
from amdisa.cross_isa import CrossIsaAnalyzer
from amdisa.encoding_translator_codegen import generate_encoding_fields, generate_encoding_translators
from amdisa.legalization import LegalizationGenerator
from amdisa.legalization_codegen import emit_all as emit_legalization
from amdisa.semantics import derive_all_semantics

_PROFILES = {
    'cdna': CdnaProfile,
    'cdna1': Cdna1Profile,
    'cdna2': Cdna2Profile,
    'cdna3': CdnaProfile,
    'cdna4': CdnaProfile,
    'rdna1': Rdna1Profile,
    'rdna2': Rdna2Profile,
    'rdna3': Rdna3Profile,
    'rdna3.5': Rdna3_5Profile,
    'rdna4': Rdna4Profile,
}


def _detect_profile(isa_xml: str) -> str:
    """Detect the ISA profile from the XML architecture name.

    Parses only the architecture name element to determine the profile
    without loading the full spec.
    """
    root = elem_tree.parse(isa_xml).getroot()
    isa_node = xs.get_node(root, xs.ISA)
    arch_node = xs.get_node(isa_node, xs.ARCH)
    arch_name_raw = xs.get_node_text(xs.get_node(arch_node, xs.ARCH_NAME))
    parts = arch_name_raw.split()
    family = parts[1].lower()
    version = parts[2]
    key = f'{family}{version}'
    if key in _PROFILES:
        return key
    key_underscore = f'{family}{version.replace(".", "_")}'
    if key_underscore in _PROFILES:
        return key_underscore
    if family == 'cdna':
        return 'cdna'
    if family == 'rdna':
        major = int(version.split('.')[0])
        if major >= 4:
            return 'rdna4'
        if major >= 3:
            return 'rdna3'
        return 'rdna1'
    return 'cdna'


def _run_multi(args) -> None:
    """Multi-ISA mode: parse all XMLs, run CrossIsaAnalyzer, generate shared + per-ISA."""
    specs = []
    for entry in args.multi:
        if ':' not in entry:
            print(f'error: --multi entry must be name:xml_path, got: {entry}', file=sys.stderr)
            sys.exit(1)
        name, xml_path = entry.split(':', 1)
        profile_key = name.replace('.', '_')
        if profile_key not in _PROFILES:
            profile_key = _detect_profile(xml_path)
        profile = _PROFILES[profile_key]()
        spec = Parser(xml_path, profile).parse()
        sem = derive_all_semantics(spec)
        specs.append((name, spec, sem))

    analyzer = CrossIsaAnalyzer()
    plan = analyzer.analyze(specs)

    print(f'Cross-ISA analysis: {plan.total_universal} universal, '
          f'{plan.total_family_shared} family-shared, '
          f'{plan.total_exclusive} exclusive', file=sys.stderr)

    config = CodegenConfig(use_shared=args.use_shared)

    # Generate per-ISA files, accumulating shared execute bodies.
    all_shared_bodies: dict[tuple[str, str], tuple] = {}
    for name, spec, sem in specs:
        code_gen = CodeGenerator(spec, args.output, sem, config=config,
                                 shared_plan=plan)
        if args.gen_all:
            code_gen.gen_all()
        # Merge shared execute bodies (first ISA wins for each key).
        for key, data in code_gen._shared_execute_bodies.items():
            if key not in all_shared_bodies:
                all_shared_bodies[key] = data

    # Write accumulated shared execute templates once (only when requested).
    if args.gen_shared_execute and all_shared_bodies:
        first_spec = specs[0][1]
        first_sem = specs[0][2]
        writer = CodeGenerator(first_spec, args.output, first_sem,
                               config=config, shared_plan=plan)
        writer._shared_execute_bodies = all_shared_bodies
        writer._write_shared_execute_templates()

    # Single unified shared execute header — no per-encoding stubs needed.

    # Legalization table generation (--gen-legalization).
    if args.gen_legalization:
        leg_gen = LegalizationGenerator(specs)
        pairs = args.legalization_pairs
        if pairs:
            pair_list = []
            for p in pairs:
                s, d = p.split('->')
                pair_list.append((s.strip(), d.strip()))
        else:
            pair_list = None
        results = leg_gen.generate_all(pair_list)
        leg_output = args.legalization_output or args.output or '.'
        generated = emit_legalization(leg_output, results)
        for src, dst, entries in results:
            counts = leg_gen.summary(entries)
            print(f'  {src} -> {dst}: {len(entries)} entries '
                  f'({counts["identity"]} identity, {counts["substitute"]} substitute, '
                  f'{counts["lower"]} lower, {counts["expand"]} expand, '
                  f'{counts["illegal"]} illegal)', file=sys.stderr)
        print(f'Generated {len(generated)} files in {leg_output}', file=sys.stderr)

    # Encoding field structs + translator generation (--gen-encoding-translators).
    if args.gen_encoding_translators:
        enc_output = args.encoding_translator_output or args.output or '.'
        generate_encoding_fields(specs, enc_output)
        if args.encoding_pair:
            src_n, dst_n = args.encoding_pair.split('->')
            src_n, dst_n = src_n.strip(), dst_n.strip()
            spec_map = {name: (spec, sem) for name, spec, sem in specs}
            if src_n not in spec_map or dst_n not in spec_map:
                print(f'error: --encoding-pair references unknown ISA: {src_n} or {dst_n}',
                      file=sys.stderr)
                sys.exit(1)
            src_spec, _ = spec_map[src_n]
            dst_spec, _ = spec_map[dst_n]
            generate_encoding_translators(src_spec, dst_spec, src_n, dst_n, enc_output)


def main() -> None:
    """Parse an AMD GPU ISA XML spec and generate C++ sources."""
    arg_parser = argparse.ArgumentParser(
        description="Parse a machine-readable AMD GPU ISA specification and generate C++ sources"
    )
    arg_parser.add_argument(
        "isafile", nargs='?', default=None,
        help="XML file with machine-readable AMD GPU ISA specification"
    )
    arg_parser.add_argument(
        "--multi", nargs='+', metavar='NAME:XML',
        help="Multi-ISA mode: parse all XMLs and generate shared execute() templates. "
             "Each argument is name:xml_path (e.g., cdna1:/path/to/cdna1.xml)."
    )
    arg_parser.add_argument(
        "--profile",
        choices=sorted(_PROFILES),
        default=None,
        help="ISA profile to use (auto-detected from XML if omitted)",
    )
    arg_parser.add_argument(
        "--gen-all", action="store_true", help="Generate C++ for all files"
    )
    arg_parser.add_argument(
        "--gen-decoder", action="store_true", help="Generate C++ decoder files"
    )
    arg_parser.add_argument(
        "--gen-isa", action="store_true", help="Generate C++ ISA type files"
    )
    arg_parser.add_argument(
        "--gen-opr-types", action="store_true", help="Generate C++ operand types"
    )
    arg_parser.add_argument(
        "--gen-encodings", action="store_true", help="Generate C++ encoding types"
    )
    arg_parser.add_argument(
        "--gen-insts", action="store_true", help="Generate C++ instruction types"
    )
    arg_parser.add_argument(
        "--gen-operand",
        action="store_true",
        help="Generate C++ ISA-specific operand class",
    )
    arg_parser.add_argument(
        "--gen-machine-inst-encodings",
        action="store_true",
        help="Generate C++ machine inst encoding types",
    )
    arg_parser.add_argument(
        "-o", "--output", help="Output path for generated C++ files"
    )
    arg_parser.add_argument(
        "--use-shared",
        action="store_true",
        help="Emit using-aliases for structs that match the shared baseline "
             "headers (machine_insts_scalar.h, machine_insts_cdna.h)",
    )
    arg_parser.add_argument(
        "--gen-shared-execute",
        action="store_true",
        help="Generate shared/execute_*.h template headers (requires --multi).",
    )
    arg_parser.add_argument(
        "--gen-legalization",
        action="store_true",
        help="Generate C++ legalization tables for DBT (requires --multi).",
    )
    arg_parser.add_argument(
        "--legalization-output",
        metavar="DIR",
        help="Output directory for legalization tables (defaults to -o value).",
    )
    arg_parser.add_argument(
        "--legalization-pairs",
        nargs='+',
        metavar='SRC->DST',
        help="Restrict legalization to specific pairs (e.g., cdna3->cdna4). "
             "Default: all supported pairs from the loaded ISAs.",
    )
    arg_parser.add_argument(
        "--gen-encoding-translators",
        action="store_true",
        help="Generate C++ encoding translator functions for DBT (requires --multi with exactly 2 ISAs).",
    )
    arg_parser.add_argument(
        "--encoding-translator-output",
        metavar="DIR",
        help="Output directory for encoding translators (defaults to -o value).",
    )
    arg_parser.add_argument(
        "--encoding-pair",
        metavar="SRC->DST",
        help="Source->target ISA pair for encoding translators (e.g., cdna4->rdna4). "
             "Field structs are always generated from all ISAs in --multi.",
    )
    args = arg_parser.parse_args()

    # Multi-ISA mode.
    if args.multi:
        _run_multi(args)
        return

    gen_flags = [
        args.gen_all, args.gen_decoder, args.gen_isa, args.gen_opr_types,
        args.gen_encodings, args.gen_insts, args.gen_operand,
        args.gen_machine_inst_encodings,
    ]
    if not any(gen_flags):
        print('warning: no --gen-* flag specified; nothing to generate', file=sys.stderr)
        sys.exit(0)

    if not args.isafile:
        print('error: isafile required in single-ISA mode', file=sys.stderr)
        sys.exit(1)

    profile_key = args.profile or _detect_profile(args.isafile)
    profile = _PROFILES[profile_key]()
    isa = Parser(args.isafile, profile).parse()
    semantics = derive_all_semantics(isa)
    config = CodegenConfig(use_shared=args.use_shared)
    code_gen = CodeGenerator(isa, args.output, semantics, config=config)

    if args.gen_all:
        code_gen.gen_all()
    else:
        if args.gen_decoder:
            code_gen.gen_decoder()
        if args.gen_isa:
            code_gen.gen_isa_types()
        if args.gen_opr_types:
            code_gen.gen_operand_types()
        if args.gen_encodings:
            code_gen.gen_encodings()
        if args.gen_insts:
            code_gen.gen_insts()
        if args.gen_operand:
            code_gen.gen_operand()
        if args.gen_machine_inst_encodings:
            code_gen.gen_machine_inst_encodings()


if __name__ == '__main__':
    main()
