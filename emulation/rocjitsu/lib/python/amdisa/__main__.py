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
from amdisa.encoding_translator_codegen import (
    generate_encoding_fields,
    generate_encoding_translators,
)
from amdisa.legalization import LegalizationGenerator
from amdisa.legalization_codegen import emit_all as emit_legalization
from amdisa.semantics import derive_all_semantics

_ENCODING_TRANSLATOR_PAIRS = [
    ("cdna4", "cdna3"),
    ("cdna4", "rdna4"),
    ("cdna4", "rdna3"),
]

_PROFILES = {
    "cdna": CdnaProfile,
    "cdna1": Cdna1Profile,
    "cdna2": Cdna2Profile,
    "cdna3": CdnaProfile,
    "cdna4": CdnaProfile,
    "rdna1": Rdna1Profile,
    "rdna2": Rdna2Profile,
    "rdna3": Rdna3Profile,
    "rdna3.5": Rdna3_5Profile,
    "rdna4": Rdna4Profile,
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
    key = f"{family}{version}"
    if key in _PROFILES:
        return key
    key_underscore = f'{family}{version.replace(".", "_")}'
    if key_underscore in _PROFILES:
        return key_underscore
    if family == "cdna":
        return "cdna"
    if family == "rdna":
        major = int(version.split(".")[0])
        if major >= 4:
            return "rdna4"
        if major >= 3:
            return "rdna3"
        return "rdna1"
    return "cdna"


def _run_multi(args) -> None:
    """Multi-ISA mode: parse all XMLs, run CrossIsaAnalyzer, generate shared + per-ISA."""
    specs = []
    for entry in args.multi:
        if ":" not in entry:
            print(
                f"error: --multi entry must be name:xml_path, got: {entry}",
                file=sys.stderr,
            )
            sys.exit(1)
        name, xml_path = entry.split(":", 1)
        profile_key = name.replace(".", "_")
        if profile_key not in _PROFILES:
            profile_key = _detect_profile(xml_path)
        profile = _PROFILES[profile_key]()
        spec = Parser(xml_path, profile).parse()
        sem = derive_all_semantics(spec)
        specs.append((name, spec, sem))

    analyzer = CrossIsaAnalyzer()
    plan = analyzer.analyze(specs)

    print(
        f"Cross-ISA analysis: {plan.total_universal} universal, "
        f"{plan.total_family_shared} family-shared, "
        f"{plan.total_exclusive} exclusive",
        file=sys.stderr,
    )

    config = CodegenConfig()

    # Generate per-ISA files, accumulating shared execute bodies.
    if args.gen_isas:
        all_shared_bodies: dict[tuple[str, str], tuple] = {}
        for name, spec, sem in specs:
            code_gen = CodeGenerator(
                spec, args.isa_output, sem, config=config, shared_plan=plan
            )
            code_gen.gen_all()
            for key, data in code_gen._shared_execute_bodies.items():
                if key not in all_shared_bodies:
                    all_shared_bodies[key] = data

        if all_shared_bodies:
            first_spec = specs[0][1]
            first_sem = specs[0][2]
            writer = CodeGenerator(
                first_spec, args.isa_output, first_sem, config=config, shared_plan=plan
            )
            writer._shared_execute_bodies = all_shared_bodies
            writer._write_shared_execute_templates()

    # DBT legalization tables and encoding translators.
    if args.gen_dbt:
        dbt_output = args.dbt_output
        if not dbt_output:
            print(
                "error: --dbt-output is required when generating DBT tables with --multi",
                file=sys.stderr,
            )
            sys.exit(1)

        leg_gen = LegalizationGenerator(specs)
        results = leg_gen.generate_all()
        generated = emit_legalization(dbt_output, results)
        for src, dst, entries in results:
            counts = leg_gen.summary(entries)
            print(
                f"  {src} -> {dst}: {len(entries)} entries "
                f'({counts["identity"]} identity, {counts["substitute"]} substitute, '
                f'{counts["lower"]} lower, {counts["expand"]} expand, '
                f'{counts["illegal"]} illegal)',
                file=sys.stderr,
            )
        print(f"Generated {len(generated)} files in {dbt_output}", file=sys.stderr)

        generate_encoding_fields(specs, dbt_output)
        spec_map = {name: (spec, sem) for name, spec, sem in specs}
        for src_n, dst_n in _ENCODING_TRANSLATOR_PAIRS:
            if src_n in spec_map and dst_n in spec_map:
                src_spec, _ = spec_map[src_n]
                dst_spec, _ = spec_map[dst_n]
                generate_encoding_translators(
                    src_spec, dst_spec, src_n, dst_n, dbt_output
                )


def main() -> None:
    """Parse an AMD GPU ISA XML spec and generate C++ sources."""
    arg_parser = argparse.ArgumentParser(
        description="Parse a machine-readable AMD GPU ISA specification and generate C++ sources"
    )
    arg_parser.add_argument(
        "isafile",
        nargs="?",
        default=None,
        help="XML file with machine-readable AMD GPU ISA specification",
    )
    arg_parser.add_argument(
        "--multi",
        nargs="+",
        metavar="NAME:XML",
        help="Multi-ISA mode: parse all XMLs and generate shared execute() templates. "
        "Each argument is name:xml_path (e.g., cdna1:/path/to/cdna1.xml).",
    )
    arg_parser.add_argument(
        "--gen-isas",
        action="store_true",
        default=True,
        help="Generate ISA C++ files (decoders, encodings, execute bodies). Default.",
    )
    arg_parser.add_argument(
        "--gen-dbt",
        action="store_true",
        default=True,
        help="Generate DBT legalization tables and encoding translators. Default.",
    )
    arg_parser.add_argument(
        "--isa-output", help="Output path for generated ISA C++ files"
    )
    arg_parser.add_argument(
        "--dbt-output",
        metavar="DIR",
        help="Output directory for DBT tables (defaults to --isa-output).",
    )
    args = arg_parser.parse_args()

    # Multi-ISA mode.
    if args.multi:
        _run_multi(args)
        return

    if not args.isafile:
        print("error: isafile required in single-ISA mode", file=sys.stderr)
        sys.exit(1)

    profile_key = _detect_profile(args.isafile)
    profile = _PROFILES[profile_key]()
    isa = Parser(args.isafile, profile).parse()
    semantics = derive_all_semantics(isa)
    config = CodegenConfig()
    if args.gen_isas:
        code_gen = CodeGenerator(isa, args.isa_output, semantics, config=config)
        code_gen.gen_all()


if __name__ == "__main__":
    main()
