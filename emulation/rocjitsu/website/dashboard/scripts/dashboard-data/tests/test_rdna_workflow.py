# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RDNA benchmark workflow wiring; no hosted or native execution is implied."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[7]
WORKFLOW = ROOT / ".github/workflows/rocjitsu-benchmarks.yml"


def test_performance_step_maps_both_rdna_configs():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["benchmark"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "benchmark")
    for target, filename in (
        ("gfx1100", "gfx1100_w7900.json"),
        ("gfx1201", "gfx1201_r9700.json"),
    ):
        variable = f"{target.upper()}_CONFIG_FILENAME"
        assert workflow["env"][variable] == filename
        assert (ROOT / "emulation/rocjitsu/configs" / filename).is_file()
        assert (
            f'--target-config "{target}=${{ROCJITSU_SOURCE_DIR}}/configs/${{{variable}}}"'
            in run
        )


def test_rdna_preparation_uses_separate_sources_and_shared_manifest():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    env = workflow['env']
    assert env['BENCHMARK_MANIFEST'] == 'benchmarks/suites/performance-v2.toml'
    assert env['TENSILE_SHA'] == 'dd77374194a3ea1a7258cd54f87af6747b6362a5'
    steps = workflow['jobs']['benchmark']['steps']
    for name, repository, pin, path in (
        (
            'FULL_TENSILE',
            'ROCm/rocm-libraries',
            '1c63e26254a357007ce2d963266085a30b39515b',
            'full-tensile-source',
        ),
        (
            'LLAMA',
            'ggml-org/llama.cpp',
            '26d52b0f9543f441fc2bfa383557c992a4fbf4b9',
            'llama-source',
        ),
    ):
        assert env[name + '_SHA'] == pin
        assert env[name + '_SOURCE'] == '${{ github.workspace }}/' + path
        checkout = next(s for s in steps if s.get('with', {}).get('path') == path)
        assert checkout['with']['repository'] == repository
        assert checkout['with']['ref'] == '${{ env.' + name + '_SHA }}'
        assert checkout['with']['persist-credentials'] is False
    generate = next(i for i, s in enumerate(steps) if s.get('id') == 'prepare')
    rdna = next(i for i, s in enumerate(steps) if s.get('id') == 'rdna')
    performance = next(i for i, s in enumerate(steps) if s.get('id') == 'benchmark')
    assert generate < rdna < performance
    assert 'rdna --manifest "${BENCHMARK_MANIFEST}"' in steps[rdna]['run']
    assert "steps.rdna.outcome == 'success'" in steps[performance]['if']
    apt = next(s['run'] for s in steps if s['name'] == 'Install base dependencies')
    for package in (
        'libmsgpack-dev',
        'libmsgpack-cxx-dev',
        'libboost-dev',
        'libzstd-dev',
    ):
        assert package in apt
