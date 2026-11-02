# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Execute the authored early resolver with fictional GitHub API responses."""

import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[7] / '.github/workflows/rocjitsu-benchmarks.yml'
)
SOURCE = 'a' * 40
TIP = 'b' * 40
WORKFLOW_SHA = 'c' * 40
REPO = 'ROCm/rocm-systems'


def resolve(
    tmp_path,
    branch='develop',
    commit=SOURCE,
    pr='',
    responses=None,
    event='workflow_dispatch',
    workflow_ref=None,
    ref_type='branch',
):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(
        s
        for s in workflow['jobs']['source-metadata']['steps']
        if s.get('id') == 'resolve'
    )
    # gh is the network boundary only; run the real resolver, git validation and jq.
    gh = tmp_path / 'gh'
    gh.write_text(
        f'#!{sys.executable}\n'
        'import json, os, subprocess, sys\n'
        'args = sys.argv[1:]\n'
        'with open(os.environ["API_CALLS"], "a") as calls: calls.write(args[1] + "\\n")\n'
        'responses = json.loads(os.environ["API_RESPONSES"])\n'
        'if args[1] not in responses: sys.exit("Unexpected API call: " + args[1])\n'
        'data = json.dumps(responses[args[1]])\n'
        'if "--jq" in args:\n'
        '    sys.exit(subprocess.run(["jq", "-r", args[args.index("--jq") + 1]], input=data, text=True).returncode)\n'
        'print(data)\n'
    )
    gh.chmod(0o755)
    output = tmp_path / 'output'
    calls = tmp_path / 'calls'
    api = {
        f'repos/{REPO}/git/ref/heads/{quote(branch, safe="")}': {
            'object': {'sha': TIP}
        },
        f'repos/{REPO}/compare/{SOURCE}...{TIP}': {
            'merge_base_commit': {'sha': SOURCE}
        },
        **(responses or {}),
    }
    result = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', step['run']],
        env={
            **os.environ,
            'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
            'GITHUB_REPOSITORY': REPO,
            'GITHUB_EVENT_NAME': event,
            'GITHUB_REF': (
                workflow_ref if workflow_ref is not None else f'refs/heads/{branch}'
            ),
            'GITHUB_REF_TYPE': ref_type,
            'GITHUB_SHA': WORKFLOW_SHA,
            'REQUESTED_SOURCE_COMMIT': commit,
            'REQUESTED_PULL_REQUEST': pr,
            'GITHUB_OUTPUT': str(output),
            'API_RESPONSES': json.dumps(api),
            'API_CALLS': str(calls),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    values = (
        dict(line.split('=', 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, values, calls.read_text().splitlines() if calls.exists() else []


def test_manual_source_is_independent_of_workflow_revision(tmp_path):
    result, values, calls = resolve(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['source_branch'] == 'develop'
    assert values['source_sha'] == SOURCE != WORKFLOW_SHA
    assert calls == [
        f'repos/{REPO}/git/ref/heads/develop',
        f'repos/{REPO}/compare/{SOURCE}...{TIP}',
    ]


def side_responses(**pr_changes):
    pr = {
        'state': 'open',
        'base': {'ref': 'develop'},
        'head': {'repo': {'full_name': REPO}, 'ref': 'feature/source', 'sha': 'e' * 40},
        'html_url': f'https://github.com/{REPO}/pull/42',
        **pr_changes,
    }
    return {
        f'repos/{REPO}/git/ref/heads/develop': {'object': {'sha': 'd' * 40}},
        f'repos/{REPO}/compare/{"d" * 40}...{SOURCE}': {
            'merge_base_commit': {'sha': 'f' * 40}
        },
        f'repos/{REPO}/pulls/42': pr,
        f'repos/{REPO}/compare/{SOURCE}...{"e" * 40}': {
            'merge_base_commit': {'sha': SOURCE}
        },
    }


def test_side_source_old_commit_is_valid_pr_metadata(tmp_path):
    result, values, calls = resolve(
        tmp_path, branch='feature/source', pr='42', responses=side_responses()
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {
        'source_branch': 'feature/source',
        'source_sha': SOURCE,
        'base_branch': 'develop',
        'base_sha': 'f' * 40,
        'pull_request_number': '42',
        'pull_request_url': f'https://github.com/{REPO}/pull/42',
    }
    assert f'repos/{REPO}/compare/{SOURCE}...{"e" * 40}' in calls


@pytest.mark.parametrize(
    'changes',
    [
        {'state': 'closed'},
        {'base': {'ref': 'release'}},
        {
            'head': {
                'repo': {'full_name': 'fork/repo'},
                'ref': 'feature/source',
                'sha': 'e' * 40,
            }
        },
        {'head': {'repo': {'full_name': REPO}, 'ref': 'other', 'sha': 'e' * 40}},
        {'html_url': 'https://example.invalid/pull/42'},
    ],
)
def test_invalid_pr_rejects_early(tmp_path, changes):
    result, values, _ = resolve(
        tmp_path, branch='feature/source', pr='42', responses=side_responses(**changes)
    )
    assert result.returncode != 0
    assert 'pull_request_number' not in values


def test_selected_commit_must_also_belong_to_pr_head_history(tmp_path):
    responses = side_responses()
    responses[f'repos/{REPO}/compare/{SOURCE}...{"e" * 40}'] = {
        'merge_base_commit': {'sha': 'f' * 40}
    }
    result, _, _ = resolve(
        tmp_path, branch='feature/source', pr='42', responses=responses
    )
    assert result.returncode != 0


@pytest.mark.parametrize('checked_sha', [SOURCE, WORKFLOW_SHA])
def test_recorded_source_checks_measurement_sha_and_workflow_output(
    tmp_path, checked_sha
):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    script = next(
        s['run']
        for s in workflow['jobs']['benchmark']['steps']
        if s['name'] == 'Record benchmark source'
    )
    output = tmp_path / 'output'
    result = subprocess.run(
        [
            'bash',
            '-eu',
            '-c',
            'git() { if [[ "$1" == show ]]; then printf "Measured subject\\n"; else printf "%s\\n" "$CHECKED_SHA"; fi; };\n'
            + script,
        ],
        env={
            **os.environ,
            'CHECKED_SHA': checked_sha,
            'BENCHMARK_SHA': SOURCE,
            'BENCHMARK_BRANCH': 'feature/source',
            'GITHUB_SHA': WORKFLOW_SHA,
            'GITHUB_RUN_ATTEMPT': '1',
            'GITHUB_OUTPUT': str(output),
            'BENCHMARK_DIR': str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == (checked_sha == SOURCE), result.stderr
    if result.returncode == 0:
        outputs = dict(line.split('=', 1) for line in output.read_text().splitlines())
        assert outputs['workflow_commit_sha'] == WORKFLOW_SHA


def test_push_uses_event_sha_without_resolving_a_later_branch_tip(tmp_path):
    result, values, calls = resolve(
        tmp_path, event='push', commit='invalid-unused-manual-input'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {'source_branch': 'develop', 'source_sha': WORKFLOW_SHA}
    assert calls == []


def test_blank_commit_freezes_selected_tip_once(tmp_path):
    result, values, calls = resolve(
        tmp_path,
        commit='',
        responses={
            f'repos/{REPO}/compare/{TIP}...{TIP}': {'merge_base_commit': {'sha': TIP}},
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['source_sha'] == TIP
    assert calls.count(f'repos/{REPO}/git/ref/heads/develop') == 1


def test_short_commit_resolves_to_full_sha_before_branch_validation(tmp_path):
    prefix = SOURCE[:7]
    result, values, calls = resolve(
        tmp_path,
        commit=prefix,
        responses={f'repos/{REPO}/commits/{prefix}': {'sha': SOURCE}},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['source_sha'] == SOURCE
    assert calls.index(f'repos/{REPO}/commits/{prefix}') < calls.index(
        f'repos/{REPO}/compare/{SOURCE}...{TIP}'
    )


@pytest.mark.parametrize('resolved', [None, 'ambiguous', 'b' * 40])
def test_unresolved_or_mismatched_short_sha_rejects(tmp_path, resolved):
    prefix = SOURCE[:7]
    result, values, _ = resolve(
        tmp_path,
        commit=prefix,
        responses={f'repos/{REPO}/commits/{prefix}': {'sha': resolved}},
    )
    assert result.returncode != 0
    assert values == {}


@pytest.mark.parametrize(
    'commit', ['develop', 'a' * 6, 'a' * 41, 'g' * 40, SOURCE + '\n', ' ' + SOURCE]
)
def test_malformed_commit_rejects_before_api(tmp_path, commit):
    result, values, calls = resolve(tmp_path, commit=commit)
    assert result.returncode != 0
    assert values == {} and calls == []


@pytest.mark.parametrize(
    'branch',
    [
        '',
        '-develop',
        'feature..bad',
        'bad\nbranch',
        'x?query',
        'branch@{1}',
        'feature/../develop',
    ],
)
def test_invalid_branch_rejects_before_api(tmp_path, branch):
    result, values, calls = resolve(tmp_path, branch=branch)
    assert result.returncode != 0
    assert values == {} and calls == []


@pytest.mark.parametrize(
    'response', [None, {'object': {'sha': 'not-a-sha'}}, {'object': {'sha': 'b' * 39}}]
)
def test_missing_or_invalid_branch_tip_rejects(tmp_path, response):
    result, values, _ = resolve(
        tmp_path, responses={f'repos/{REPO}/git/ref/heads/develop': response}
    )
    assert result.returncode != 0
    assert values == {}


@pytest.mark.parametrize('branch', ['develop', 'feature/source'])
def test_shared_object_outside_selected_branch_history_rejects(tmp_path, branch):
    result, values, _ = resolve(
        tmp_path,
        branch=branch,
        responses={
            f'repos/{REPO}/compare/{SOURCE}...{TIP}': {
                'merge_base_commit': {'sha': 'f' * 40}
            },
        },
    )
    assert result.returncode != 0
    assert values == {}
    assert 'selected workflow branch' in result.stderr


@pytest.mark.parametrize('pr', ['0', '-1', 'abc', '1\n', '01'])
def test_invalid_pr_number_rejects_before_pull_lookup(tmp_path, pr):
    result, _, calls = resolve(
        tmp_path, branch='feature/source', pr=pr, responses=side_responses()
    )
    assert result.returncode != 0
    assert not any('/pulls/' in call for call in calls)


def test_develop_source_rejects_pr_metadata(tmp_path):
    result, _, _ = resolve(tmp_path, pr='42')
    assert result.returncode != 0


def test_side_source_without_pr_still_has_selected_commit_merge_base(tmp_path):
    result, values, calls = resolve(
        tmp_path, branch='feature/source', responses=side_responses()
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['base_sha'] == 'f' * 40
    assert 'pull_request_number' not in values
    assert not any('/pulls/' in call for call in calls)


def test_uppercase_hex_commit_is_normalized_after_validation(tmp_path):
    result, values, _ = resolve(tmp_path, commit=SOURCE.upper())
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['source_sha'] == SOURCE


@pytest.mark.parametrize('branch', ['develop', 'feature/source'])
def test_selected_workflow_branch_is_the_source_branch(tmp_path, branch):
    result, values, _ = resolve(
        tmp_path,
        branch=branch,
        responses=side_responses() if branch != 'develop' else {},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['source_branch'] == branch
    assert values['source_sha'] == SOURCE


def test_blank_commit_uses_selected_side_branch_tip(tmp_path):
    responses = side_responses()
    responses.update(
        {
            f'repos/{REPO}/compare/{TIP}...{TIP}': {'merge_base_commit': {'sha': TIP}},
            f'repos/{REPO}/compare/{"d" * 40}...{TIP}': {
                'merge_base_commit': {'sha': 'f' * 40}
            },
        }
    )
    result, values, calls = resolve(
        tmp_path,
        branch='feature/source',
        commit='',
        responses=responses,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert values['source_sha'] == TIP
    assert values['source_branch'] == 'feature/source'
    assert calls.count(f'repos/{REPO}/git/ref/heads/feature%2Fsource') == 1


def test_tag_workflow_rejects_before_api(tmp_path):
    result, values, calls = resolve(
        tmp_path, workflow_ref='refs/tags/v1', ref_type='tag'
    )
    assert result.returncode != 0
    assert values == {} and calls == []
