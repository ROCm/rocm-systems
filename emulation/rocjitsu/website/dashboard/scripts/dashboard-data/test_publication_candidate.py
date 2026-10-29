# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Publication-only gate: run with CANDIDATE_FILE after workflow enrichment."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def candidate():
    filename = os.environ.get('CANDIDATE_FILE')
    if not filename:
        pytest.skip('CANDIDATE_FILE is supplied by the publication workflow')
    return json.loads(Path(filename).read_text())


@pytest.mark.parametrize(
    'field',
    [
        'publication.id',
        'publication.source.branch',
        'publication.source.message',
        'publication.trigger',
        'publication.machine_label',
        'publication.workflow_commit_sha',
        'publication.tensile_commit_sha',
        'provenance.rocjitsu.rocjitsu_commit_sha',
        'provenance.rocjitsu.rocjitsu_commit_timestamp',
    ],
)
def test_required_publication_field(candidate, field):
    value = candidate
    for key in field.split('.'):
        assert isinstance(value, dict) and key in value, f'{field} is required'
        value = value[key]
    assert (
        isinstance(value, str) and value.strip()
    ), f'{field} must be a nonempty string (not null)'
