# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from amdisa import IsaVariantsError, load_isa_variants
from amdisa.__main__ import _group_isa_variant_args
from amdisa.gpuisa import InstEncoding, Instruction, IsaSpec
from amdisa.isa_profile import Cdna5Profile


def _spec() -> IsaSpec:
    spec = IsaSpec('cdna5', '1.2.0', Cdna5Profile())
    instruction = Instruction(
        'V_COMMON',
        'ENC_VOP1',
        1,
        [],
        available_encodings=frozenset({'ENC_VOP1', 'VOP1_VOP_DPP16'}),
    )
    encoding = InstEncoding('ENC_VOP1', 0, 32, 7, 8, [], [])
    encoding.insts.append(instruction)
    spec.inst_encodings.append(encoding)
    alternate_parent = Instruction(
        'V_COMMON',
        'ENC_VOP3',
        2,
        [],
        available_encodings=instruction.available_encodings,
    )
    vop3 = InstEncoding('ENC_VOP3', 1, 64, 6, 10, [], [])
    vop3.insts.append(alternate_parent)
    spec.inst_encodings.append(vop3)
    return spec


def _write_manifest(path: Path, **overrides) -> Path:
    document = {
        'schema_version': 1,
        'features': ['enhanced'],
        'variants': {'base': [], 'plus': ['enhanced']},
        'instructions': [],
        'model_only_instructions': [],
        'encodings': [
            {
                'feature': 'enhanced',
                'encoding': 'VOP1_VOP_DPP16',
                'instructions': ['V_COMMON'],
            }
        ],
    }
    document.update(overrides)
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def test_loads_instruction_and_encoding_requirements(tmp_path: Path) -> None:
    spec = _spec()
    manifest = _write_manifest(
        tmp_path / 'variants.json',
        instructions=[{'feature': 'enhanced', 'names': ['V_COMMON']}],
    )

    load_isa_variants(spec, str(manifest))

    instructions = [encoding.insts[0] for encoding in spec.inst_encodings]
    assert spec.isa_features == ('enhanced',)
    assert spec.isa_variants == {'base': 0, 'plus': 1}
    assert all(instruction.required_feature_mask == 1 for instruction in instructions)
    assert all(
        instruction.encoding_feature_masks == {'VOP1_VOP_DPP16': 1}
        for instruction in instructions
    )


def test_rejects_more_features_than_instruction_mask_can_represent(
    tmp_path: Path,
) -> None:
    spec = _spec()
    manifest = _write_manifest(
        tmp_path / 'variants.json',
        features=[f'feature_{index}' for index in range(33)],
    )

    with pytest.raises(IsaVariantsError, match='at most 32 instruction features'):
        load_isa_variants(spec, str(manifest))


def test_marks_capability_gated_instructions_model_only(tmp_path: Path) -> None:
    spec = _spec()
    manifest = _write_manifest(
        tmp_path / 'variants.json',
        instructions=[{'feature': 'enhanced', 'names': ['V_COMMON']}],
        model_only_instructions=['V_COMMON'],
    )

    load_isa_variants(spec, str(manifest))

    assert all(
        instruction.model_only
        for encoding in spec.inst_encodings
        for instruction in encoding.insts
    )


def test_invalid_later_requirement_does_not_partially_mutate_spec(
    tmp_path: Path,
) -> None:
    spec = _spec()
    manifest = _write_manifest(
        tmp_path / 'variants.json',
        instructions=[{'feature': 'enhanced', 'names': ['V_COMMON']}],
        model_only_instructions=['V_COMMON'],
        encodings=[
            {
                'feature': 'enhanced',
                'encoding': 'VOP1_OTHER_VOP_DPP16',
                'instructions': ['V_COMMON'],
            }
        ],
    )

    with pytest.raises(IsaVariantsError, match='does not list encoding'):
        load_isa_variants(spec, str(manifest))

    for encoding in spec.inst_encodings:
        for instruction in encoding.insts:
            assert instruction.required_feature_mask == 0
            assert instruction.encoding_feature_masks == {}
            assert not instruction.model_only
    assert spec.isa_features == ()
    assert spec.isa_variants == {}


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'schema_version': 2}, 'unsupported schema_version'),
        ({'features': ['enhanced', 'enhanced']}, 'duplicate entry'),
        ({'features': ['Enhanced']}, 'lowercase capability identifier'),
        (
            {'features': ['feature_1', 'feature1']},
            r'collides in the generated C\+\+ API',
        ),
        ({'variants': {'plus': ['missing']}}, 'unknown feature'),
        ({'variants': {'gfx-1251': []}}, 'lowercase capability identifiers'),
        (
            {
                'encodings': [
                    {
                        'feature': 'enhanced',
                        'encoding': 'VOP1_OTHER_VOP_DPP16',
                        'instructions': ['V_COMMON'],
                    }
                ]
            },
            'does not list encoding',
        ),
        (
            {
                'encodings': [
                    {
                        'feature': 'enhanced',
                        'encoding': 'VOP1_VOP_DPP8',
                        'instructions': ['V_COMMON'],
                    }
                ]
            },
            'support only DPP16 forms',
        ),
        (
            {
                'encodings': [
                    {
                        'feature': 'enhanced',
                        'encoding': 'ENC_VOP1',
                        'instructions': ['V_COMMON'],
                    }
                ]
            },
            'support only DPP16 forms',
        ),
        (
            {'instructions': [{'feature': 'enhanced', 'names': ['V_MISSING']}]},
            'unknown instruction',
        ),
        ({'model_only_instructions': ['V_MISSING']}, 'unknown instruction'),
        (
            {'model_only_instructions': ['V_COMMON']},
            'has no target capability requirement',
        ),
    ],
)
def test_rejects_invalid_manifests(tmp_path: Path, overrides, message: str) -> None:
    spec = _spec()
    manifest = _write_manifest(tmp_path / 'variants.json', **overrides)

    with pytest.raises(IsaVariantsError, match=message):
        load_isa_variants(spec, str(manifest))


def test_groups_one_variant_manifest_per_isa() -> None:
    assert _group_isa_variant_args(
        ['cdna5:/tmp/cdna5.json', 'rdna4:/tmp/rdna4.json']
    ) == {
        'cdna5': '/tmp/cdna5.json',
        'rdna4': '/tmp/rdna4.json',
    }


@pytest.mark.parametrize(
    'entry',
    ['cdna5', ':manifest.json', 'cdna5:'],
)
def test_rejects_malformed_variant_cli_entries(entry: str) -> None:
    with pytest.raises(ValueError, match='must'):
        _group_isa_variant_args([entry])


def test_rejects_repeated_variant_cli_name() -> None:
    with pytest.raises(ValueError, match='repeated'):
        _group_isa_variant_args(['cdna5:first.json', 'cdna5:second.json'])
