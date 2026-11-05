# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validated target-feature requirements for variants of one ISA family."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from amdisa.gpuisa import IsaSpec


class IsaVariantsError(ValueError):
    """An ISA-variant manifest is malformed or inconsistent with its ISA."""


_ROOT_KEYS = frozenset(
    {
        'schema_version',
        'features',
        'variants',
        'instructions',
        'encodings',
        'model_only_instructions',
    }
)
_CAPABILITY_NAME = re.compile(r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$')


def _supports_encoding_requirement(encoding: str) -> bool:
    """Whether schema version 1 can attach a runtime requirement here."""
    return '_VOP_DPP16' in encoding.upper()


def _cpp_name(name: str) -> str:
    return ''.join(part[:1].upper() + part[1:] for part in name.split('_'))


def _validate_names(names: list[str], context: str) -> None:
    generated_names: set[str] = set()
    for name in names:
        if _CAPABILITY_NAME.fullmatch(name) is None:
            raise IsaVariantsError(
                f'{context}: {name!r} is not a lowercase capability identifier'
            )
        generated_name = _cpp_name(name)
        if generated_name in generated_names:
            raise IsaVariantsError(
                f'{context}: {name!r} collides in the generated C++ API'
            )
        generated_names.add(generated_name)


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise IsaVariantsError(f'{context}: expected a list of non-empty strings')
    if len(value) != len(set(value)):
        raise IsaVariantsError(f'{context}: duplicate entry')
    return value


def load_isa_variants(isa_spec: IsaSpec, manifest_path: str) -> None:
    """Attach validated feature requirements from ``manifest_path`` to a spec."""
    path = Path(manifest_path)
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise IsaVariantsError(f'{path}: cannot read ISA variants: {error}') from error

    if not isinstance(document, dict):
        raise IsaVariantsError(f'{path}: expected a JSON object')
    unknown_keys = set(document) - _ROOT_KEYS
    if unknown_keys:
        raise IsaVariantsError(
            f'{path}: unknown root keys: {", ".join(sorted(unknown_keys))}'
        )
    missing_keys = _ROOT_KEYS - set(document)
    if missing_keys:
        raise IsaVariantsError(
            f'{path}: missing root keys: {", ".join(sorted(missing_keys))}'
        )
    if document['schema_version'] != 1:
        raise IsaVariantsError(
            f'{path}: unsupported schema_version {document["schema_version"]!r}'
        )

    features = _string_list(document['features'], f'{path}: features')
    _validate_names(features, f'{path}: features')
    if len(features) > 32:
        raise IsaVariantsError(f'{path}: at most 32 instruction features are supported')
    feature_bits = {name: 1 << index for index, name in enumerate(features)}

    def feature_mask(names: Any, context: str) -> int:
        mask = 0
        for name in _string_list(names, context):
            if name not in feature_bits:
                raise IsaVariantsError(f'{context}: unknown feature {name!r}')
            mask |= feature_bits[name]
        return mask

    variants_node = document['variants']
    if not isinstance(variants_node, dict) or not variants_node:
        raise IsaVariantsError(f'{path}: variants must be a non-empty object')
    variants: dict[str, int] = {}
    for name, names in variants_node.items():
        if not isinstance(name, str) or _CAPABILITY_NAME.fullmatch(name) is None:
            raise IsaVariantsError(
                f'{path}: variant names must be lowercase capability identifiers'
            )
        variants[name] = feature_mask(names, f'{path}: variant {name!r}')
    _validate_names(list(variants), f'{path}: variants')

    instructions_by_name: dict[str, list] = {}
    for encoding in isa_spec.inst_encodings:
        for inst in encoding.insts:
            instructions_by_name.setdefault(inst.name, []).append(inst)
    instruction_node = document['instructions']
    if not isinstance(instruction_node, list):
        raise IsaVariantsError(f'{path}: instructions must be a list')
    seen_instruction_requirements: set[tuple[str, str]] = set()
    instruction_updates: list[tuple[Any, int]] = []
    for index, entry in enumerate(instruction_node):
        context = f'{path}: instructions[{index}]'
        if not isinstance(entry, dict) or set(entry) != {'feature', 'names'}:
            raise IsaVariantsError(f'{context}: expected only feature and names keys')
        feature = entry['feature']
        if not isinstance(feature, str) or feature not in feature_bits:
            raise IsaVariantsError(f'{context}: unknown feature {feature!r}')
        for name in _string_list(entry['names'], f'{context}: names'):
            insts = instructions_by_name.get(name)
            if insts is None:
                raise IsaVariantsError(f'{context}: unknown instruction {name!r}')
            requirement = (feature, name)
            if requirement in seen_instruction_requirements:
                raise IsaVariantsError(f'{context}: duplicate requirement for {name!r}')
            seen_instruction_requirements.add(requirement)
            for inst in insts:
                instruction_updates.append((inst, feature_bits[feature]))

    model_only_updates: list[Any] = []
    for name in _string_list(
        document['model_only_instructions'],
        f'{path}: model_only_instructions',
    ):
        insts = instructions_by_name.get(name)
        if insts is None:
            raise IsaVariantsError(
                f'{path}: model_only_instructions: unknown instruction {name!r}'
            )
        if not any(
            requirement_name == name
            for _, requirement_name in seen_instruction_requirements
        ):
            raise IsaVariantsError(
                f'{path}: model-only instruction {name!r} has no target capability requirement'
            )
        for inst in insts:
            model_only_updates.append(inst)

    encoding_node = document['encodings']
    if not isinstance(encoding_node, list):
        raise IsaVariantsError(f'{path}: encodings must be a list')
    seen_encoding_requirements: set[tuple[str, str, str]] = set()
    encoding_updates: list[tuple[Any, str, int]] = []
    for index, entry in enumerate(encoding_node):
        context = f'{path}: encodings[{index}]'
        if not isinstance(entry, dict) or set(entry) != {
            'feature',
            'encoding',
            'instructions',
        }:
            raise IsaVariantsError(
                f'{context}: expected only feature, encoding, and instructions keys'
            )
        feature = entry['feature']
        encoding = entry['encoding']
        if not isinstance(feature, str) or feature not in feature_bits:
            raise IsaVariantsError(f'{context}: unknown feature {feature!r}')
        if not isinstance(encoding, str) or not encoding:
            raise IsaVariantsError(f'{context}: encoding must be a non-empty string')
        if not _supports_encoding_requirement(encoding):
            raise IsaVariantsError(
                f'{context}: encoding requirements support only DPP16 forms in schema version 1; '
                f'got {encoding!r}'
            )
        for name in _string_list(entry['instructions'], f'{context}: instructions'):
            insts = instructions_by_name.get(name)
            if insts is None:
                raise IsaVariantsError(f'{context}: unknown instruction {name!r}')
            if any(
                inst.available_encodings is None
                or encoding not in inst.available_encodings
                for inst in insts
            ):
                raise IsaVariantsError(
                    f'{context}: instruction {name!r} does not list encoding {encoding!r}'
                )
            requirement = (feature, encoding, name)
            if requirement in seen_encoding_requirements:
                raise IsaVariantsError(
                    f'{context}: duplicate requirement for {name!r} / {encoding!r}'
                )
            seen_encoding_requirements.add(requirement)
            for inst in insts:
                encoding_updates.append((inst, encoding, feature_bits[feature]))

    # Commit only after every manifest entry validates, so a malformed later
    # requirement cannot leave a partially modified parser-owned spec.
    for inst, mask in instruction_updates:
        inst.required_feature_mask |= mask
    for inst in model_only_updates:
        inst.model_only = True
    for inst, encoding, mask in encoding_updates:
        inst.encoding_feature_masks[encoding] = (
            inst.encoding_feature_masks.get(encoding, 0) | mask
        )
    isa_spec.isa_features = tuple(features)
    isa_spec.isa_variants = variants
