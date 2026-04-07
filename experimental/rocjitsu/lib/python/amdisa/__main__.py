# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point: ``python -m amdisa``."""

import argparse
import sys
import xml.etree.ElementTree as elem_tree

from amdisa import (
    CdnaProfile,
    CodeGenerator,
    Parser,
    Rdna1Profile,
    Rdna3Profile,
    Rdna4Profile,
)
from amdisa import xml_schema as xs
from amdisa.semantics import derive_all_semantics

_PROFILES = {
    'cdna': CdnaProfile,
    'cdna1': CdnaProfile,
    'cdna2': CdnaProfile,
    'cdna3': CdnaProfile,
    'cdna4': CdnaProfile,
    'rdna1': Rdna1Profile,
    'rdna2': Rdna1Profile,
    'rdna3': Rdna3Profile,
    'rdna3.5': Rdna3Profile,
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


def main() -> None:
    """Parse an AMD GPU ISA XML spec and generate C++ sources."""
    arg_parser = argparse.ArgumentParser(
        description="Parse a machine-readable AMD GPU ISA specification and generate C++ sources"
    )
    arg_parser.add_argument(
        "isafile", help="XML file with machine-readable AMD GPU ISA specification"
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
    args = arg_parser.parse_args()

    gen_flags = [
        args.gen_all, args.gen_decoder, args.gen_isa, args.gen_opr_types,
        args.gen_encodings, args.gen_insts, args.gen_operand,
        args.gen_machine_inst_encodings,
    ]
    if not any(gen_flags):
        print('warning: no --gen-* flag specified; nothing to generate', file=sys.stderr)
        sys.exit(0)

    profile_key = args.profile or _detect_profile(args.isafile)
    profile = _PROFILES[profile_key]()
    isa = Parser(args.isafile, profile).parse()
    semantics = derive_all_semantics(isa)
    code_gen = CodeGenerator(isa, args.output, semantics)

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
