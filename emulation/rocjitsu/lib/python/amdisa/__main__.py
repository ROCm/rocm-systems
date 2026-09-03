# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point: ``python -m amdisa``."""

import argparse
import re
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as elem_tree

from amdisa import (
    Cdna1Profile,
    Cdna2Profile,
    Cdna4Profile,
    CdnaProfile,
    CodegenConfig,
    CodeGenerator,
    Cdna5Profile,
    Parser,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
    load_isa_variants,
)
from amdisa import xml_schema as xs
from amdisa.cross_isa import CrossIsaAnalyzer
from amdisa.encoding_translator_codegen import (
    generate_encoding_fields,
    generate_encoding_translators,
)
from amdisa.legalization import LegalizationGenerator
from amdisa.legalization_codegen import emit_all as emit_legalization
from amdisa.isa_properties_codegen import emit_isa_properties
from amdisa.semantics import derive_all_semantics

_ENCODING_TRANSLATOR_PAIRS = [
    ('cdna4', 'cdna3'),
    ('cdna4', 'rdna4'),
    ('cdna4', 'rdna3'),
]

_PROFILES = {
    'cdna': CdnaProfile,
    'cdna1': Cdna1Profile,
    'cdna2': Cdna2Profile,
    'cdna3': CdnaProfile,
    'cdna4': Cdna4Profile,
    'rdna1': Rdna1Profile,
    'rdna2': Rdna2Profile,
    'rdna3': Rdna3Profile,
    'rdna3.5': Rdna3_5Profile,
    'rdna4': Rdna4Profile,
    'cdna5': Cdna5Profile,
}

_PROFILE_ALIASES = {
    'rdna3_5': 'rdna3.5',
}


def _group_isa_additions_args(entries: list[str] | None) -> dict[str, list[str]]:
    """Group repeated ``NAME:XML`` additions arguments by logical ISA name."""
    grouped: dict[str, list[str]] = {}
    for entry in entries or []:
        if ':' not in entry:
            raise ValueError(
                f'--isa-additions entry must be name:xml_path, got: {entry}'
            )
        name, xml_path = entry.split(':', 1)
        if not name or not xml_path:
            raise ValueError(
                f'--isa-additions entry must have non-empty name and path, got: {entry}'
            )
        grouped.setdefault(name, []).append(xml_path)
    return grouped


def _group_isa_variant_args(entries: list[str] | None) -> dict[str, str]:
    """Group one ``NAME:JSON`` variant manifest per logical ISA name."""
    grouped: dict[str, str] = {}
    for entry in entries or []:
        if ':' not in entry:
            raise ValueError(
                f'--isa-variants entry must be name:json_path, got: {entry}'
            )
        name, json_path = entry.split(':', 1)
        if not name or not json_path:
            raise ValueError(
                '--isa-variants entry must have non-empty name and path, '
                f'got: {entry}'
            )
        if name in grouped:
            raise ValueError(f'--isa-variants repeated for ISA {name!r}')
        grouped[name] = json_path
    return grouped


def _collect_shared_execute_body_variants(specs, plan):
    """Collect candidate shared execute bodies from each ISA.

    The cross-ISA analyzer proves structural compatibility, but the final
    generated body can still depend on architecture-specific lowering choices.
    This preflight keeps those generated bodies visible before the real output
    pass decides which keys can remain shared.
    """
    variants: dict[tuple[str, str], dict[str, tuple]] = {}
    with TemporaryDirectory(prefix='amdisa-shared-preflight-') as out_dir:
        config = CodegenConfig()
        for name, spec, sem in specs:
            code_gen = CodeGenerator(
                spec, out_dir, sem, config=config, shared_plan=plan
            )
            code_gen.gen_all()
            for key, data in code_gen._shared_execute_bodies.items():
                variants.setdefault(key, {})[name] = data
    return variants


def _unshared_execute_keys_from_variants(
    variants: dict[tuple[str, str], dict[str, tuple]],
) -> frozenset[tuple[str, str]]:
    """Return keys whose generated shared bodies differ between ISAs."""
    divergent = set()
    for key, by_isa in variants.items():
        bodies = {data[2] for data in by_isa.values()}
        if len(bodies) > 1:
            divergent.add(key)
    return frozenset(divergent)


def _merge_shared_execute_body(
    merged: dict[tuple[str, str], tuple],
    key: tuple[str, str],
    data: tuple,
    isa_name: str,
) -> None:
    """Merge a final shared body and reject unexpected divergence."""
    existing = merged.get(key)
    if existing is None:
        merged[key] = data
        return
    if existing[2] != data[2]:
        raise AssertionError(
            'shared execute body collision after divergence preflight: '
            f'isa={isa_name!r} mnemonic={key[0]!r} enc={key[1]!r} produced '
            'a different body for a key that was still marked shareable.'
            f'\n--- first writer body ---\n{existing[2]}'
            f'\n--- this writer body ---\n{data[2]}'
        )


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


def _normalize_codegen_identity(name: str) -> str:
    """Normalize and validate an explicit generated directory and namespace."""
    identity = name.replace('.', '_')
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', identity) is None:
        raise ValueError(f'invalid ISA name {name!r}: expected a single C++ identifier')
    return identity


def _parse_isa_arg(value: str) -> tuple[str | None, str, str]:
    """Parse ``[NAME:]XML`` and resolve its ISA profile."""
    if ':' in value:
        name, xml_path = value.split(':', 1)
        _normalize_codegen_identity(name)
    else:
        name, xml_path = None, value

    profile_key = _PROFILE_ALIASES.get(name, name) if name else None
    if profile_key not in _PROFILES:
        profile_key = _detect_profile(xml_path)
    return name, xml_path, profile_key


def _apply_codegen_identity(spec, name: str | None) -> None:
    """Use an explicit CLI name for generated paths and C++ namespaces."""
    if name is None:
        return
    identity = _normalize_codegen_identity(name)
    spec.generated_dir_name = identity
    spec.cpp_namespace = identity


def _codegen_config(
    isa_output: str | None,
    *,
    include_root: str | None = None,
    use_shared_execute_helpers: bool = True,
) -> CodegenConfig:
    if isa_output is None:
        return CodegenConfig(use_shared_execute_helpers=use_shared_execute_helpers)
    return CodegenConfig.for_output(
        isa_output,
        include_root=include_root,
        use_shared_execute_helpers=use_shared_execute_helpers,
    )


def _run(args) -> None:
    """Parse the input XMLs and generate the requested outputs."""
    try:
        addition_xmls = _group_isa_additions_args(getattr(args, 'isa_additions', None))
        variant_jsons = _group_isa_variant_args(getattr(args, 'isa_variants', None))
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)

    parsed_inputs = []
    logical_names = set()
    for entry in args.isafiles:
        name, xml_path, profile_key = _parse_isa_arg(entry)
        profile = _PROFILES[profile_key]()
        logical_name = (
            name or profile.generated_arch_name or profile_key.replace('.', '_')
        )
        parsed_inputs.append((name, logical_name, xml_path, profile))
        logical_names.add(logical_name)

    unknown_addition_names = set(addition_xmls) - logical_names
    if unknown_addition_names:
        names = ', '.join(sorted(unknown_addition_names))
        print(
            f'error: --isa-additions names not present in ISA inputs: {names}',
            file=sys.stderr,
        )
        sys.exit(1)
    unknown_variant_names = set(variant_jsons) - logical_names
    if unknown_variant_names:
        names = ', '.join(sorted(unknown_variant_names))
        print(
            f'error: --isa-variants names not present in ISA inputs: {names}',
            file=sys.stderr,
        )
        sys.exit(1)

    specs = []
    for name, logical_name, xml_path, profile in parsed_inputs:
        spec = Parser(xml_path, profile, addition_xmls.get(logical_name, ())).parse()
        _apply_codegen_identity(spec, name)
        if variant_path := variant_jsons.get(logical_name):
            load_isa_variants(spec, variant_path)
        sem = derive_all_semantics(spec)
        specs.append((spec.arch_name, spec, sem))

    analyzer = CrossIsaAnalyzer()
    plan = analyzer.analyze(specs)

    print(
        f'Cross-ISA analysis: {plan.total_universal} universal, '
        f'{plan.total_family_shared} family-shared, '
        f'{plan.total_exclusive} exclusive',
        file=sys.stderr,
    )
    config = _codegen_config(
        args.isa_output,
        include_root=getattr(args, 'include_root', None),
        use_shared_execute_helpers=len(specs) > 1,
    )

    # Generate per-ISA files, accumulating shared execute bodies.
    if args.gen_isas:
        emit_isa_properties(args.isa_output, specs)
        body_variants = _collect_shared_execute_body_variants(specs, plan)
        unshared_keys = _unshared_execute_keys_from_variants(body_variants)
        config.unshared_execute_keys = unshared_keys
        if unshared_keys:
            print(
                f'Keeping {len(unshared_keys)} arch-dependent shared execute '
                f'keys ISA-local after body preflight',
                file=sys.stderr,
            )

        all_shared_bodies: dict[tuple[str, str], tuple] = {}
        for name, spec, sem in specs:
            code_gen = CodeGenerator(
                spec, args.isa_output, sem, config=config, shared_plan=plan
            )
            code_gen.gen_all()
            for key, data in code_gen._shared_execute_bodies.items():
                _merge_shared_execute_body(all_shared_bodies, key, data, name)

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
        if len(specs) < 2:
            print(
                'Skipping DBT generation: at least two ISAs are required.',
                file=sys.stderr,
            )
            return
        dbt_output = args.dbt_output or args.isa_output
        if not dbt_output:
            print(
                'error: --dbt-output or --isa-output is required when '
                'generating DBT tables',
                file=sys.stderr,
            )
            sys.exit(1)

        leg_gen = LegalizationGenerator(specs)
        results = leg_gen.generate_all()
        generated = emit_legalization(dbt_output, results)
        for src, dst, entries in results:
            counts = leg_gen.summary(entries)
            identity = counts['identity']
            substitute = counts['substitute']
            lower = counts['lower']
            expand = counts['expand']
            illegal = counts['illegal']
            print(
                f'  {src} -> {dst}: {len(entries)} entries '
                f'({identity} identity, {substitute} substitute, '
                f'{lower} lower, {expand} expand, '
                f'{illegal} illegal)',
                file=sys.stderr,
            )
        print(f'Generated {len(generated)} files in {dbt_output}', file=sys.stderr)

        generate_encoding_fields(specs, dbt_output)
        spec_map = {name: (spec, sem) for name, spec, sem in specs}
        for src_n, dst_n in _ENCODING_TRANSLATOR_PAIRS:
            if src_n in spec_map and dst_n in spec_map:
                src_spec, _ = spec_map[src_n]
                dst_spec, _ = spec_map[dst_n]
                generate_encoding_translators(
                    src_spec,
                    dst_spec,
                    src_n,
                    dst_n,
                    dbt_output,
                    config=config,
                )


def main() -> None:
    """Parse an AMD GPU ISA XML spec and generate C++ sources."""
    arg_parser = argparse.ArgumentParser(
        description='Parse a machine-readable AMD GPU ISA specification and generate C++ sources'
    )
    arg_parser.add_argument(
        'isafiles',
        nargs='+',
        metavar='[NAME:]XML',
        help='Machine-readable AMD GPU ISA XML specifications. Each path may '
        'have a generated identity prefix (e.g., cdna1:/path/to/cdna1.xml).',
    )
    arg_parser.add_argument(
        '--isa-additions',
        action='append',
        default=[],
        metavar='NAME:XML',
        help='apply an ISA additions XML file to the named ISA. May be repeated; '
        'files are applied in command-line order.',
    )
    arg_parser.add_argument(
        '--isa-variants',
        action='append',
        default=[],
        metavar='NAME:JSON',
        help='attach one target-feature/legality manifest to the named ISA.',
    )
    arg_parser.add_argument(
        '--gen-isas',
        action='store_true',
        help='Generate ISA C++ files (decoders, encodings, execute bodies).',
    )
    arg_parser.add_argument(
        '--gen-dbt',
        action='store_true',
        help='Generate DBT legalization tables and encoding translators.',
    )
    arg_parser.add_argument(
        '--isa-output', help='Output path for generated ISA C++ files'
    )
    arg_parser.add_argument(
        '--include-root',
        help='Compiler include root used to spell generated include paths. '
        'Absolute paths are emitted when omitted.',
    )
    arg_parser.add_argument(
        '--dbt-output',
        metavar='DIR',
        help='Output directory for DBT tables (defaults to --isa-output).',
    )
    args = arg_parser.parse_args()
    if not args.gen_isas and not args.gen_dbt:
        args.gen_isas = True
        args.gen_dbt = True
    _run(args)


if __name__ == '__main__':
    main()
