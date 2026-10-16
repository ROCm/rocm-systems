# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixture-based conversion and local file publication tests; no remote writes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[7]
WORKFLOW = ROOT / '.github/workflows/rocjitsu-benchmarks.yml'
HELPER = Path(__file__).resolve().parents[1] / 'prepare-benchmark-result.py'
SHA = 'b' * 40
CORPUS = '6186d5c8bcd5e9bc93f7406e7cb50ed2a7698621'
TENSILE = 'dd77374194a3ea1a7258cd54f87af6747b6362a5'


def raw_run():
    result = {
        'id': 'fixture.case',
        'suite': 'Fixture',
        'name': 'Fixture case',
        'operation': 'copy',
        'target': 'gfx950',
        'thread_policy': 'default',
        'problem': {'dtype': 'fp32', 'n': 64},
        'status': 'completed',
        'exit_code': 0,
        'error': None,
        'timing_results_s': [0.3, 0.1, 0.2],
    }
    return {
        'execution_summary': {
            'timestamp': '2026-10-01T00:00:00Z',
            'finished_at': '2026-10-01T00:01:00Z',
            'wall_time_s': 59.25,
            'warmups': 1,
            'samples': 3,
            'timeout_seconds': 60,
            'benchmark_suite': 'fixture-suite',
        },
        'provenance': {
            'rocjitsu': {
                'rocjitsu_commit_sha': SHA,
                'rocjitsu_commit_timestamp': '2026-09-30T00:00:00Z',
            },
            'corpus': {'corpus_commit_sha': CORPUS},
            'machine': {'hostname': 'fixture-host'},
            'auxiliary': {'optional': None},
        },
        'benchmark_results': [
            result,
            {**result, 'thread_policy': 'single', 'timing_results_s': [9.0, 8.0, 7.0]},
        ],
    }


def context(
    run_id='123',
    attempt='1',
    subject='Source "subject" with spaces = literal $(touch SHOULD_NOT_EXIST)',
):
    return {
        'repository': 'ROCm/rocm-systems',
        'branch': 'develop',
        'ref': 'refs/heads/develop',
        'source_sha': SHA,
        'source_subject': subject,
        'run_id': run_id,
        'run_attempt': attempt,
        'run_url': f'https://github.com/ROCm/rocm-systems/actions/runs/{run_id}/attempts/{attempt}',
        'event': 'workflow_dispatch',
        'trigger': 'manual',
        'runner': 'benchmark machine',
        'corpus_sha': CORPUS,
        'tensile_sha': TENSILE,
    }


def manifest():
    return {
        'name': 'fixture-suite',
        'cases': [
            {
                'id': 'fixture.case',
                'suite': 'Fixture',
                'name': 'Fixture case',
                'operation': 'copy',
                'params': {'dtype': 'fp32', 'n': 64},
                'targets': ['gfx950'],
                'thread_policies': ['default', 'single'],
            }
        ],
    }


def convert(tmp_path, attempt='1', raw=None):
    source = tmp_path / 'raw.json'
    original = (
        json.dumps(raw_run() if raw is None else raw, indent=1) + '\n  '
    ).encode()
    source.write_bytes(original)
    output = tmp_path / f'github-123-attempt-{attempt}-default.json'
    command = [
        sys.executable,
        str(HELPER),
        '--raw-run',
        str(source),
        '--output',
        str(output),
        '--run-id',
        '123',
        '--run-attempt',
        attempt,
        '--repository',
        'ROCm/rocm-systems',
        '--event',
        'workflow_dispatch',
        '--commit-message',
        context()['source_subject'],
        '--machine',
        'benchmark machine',
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert source.read_bytes() == original
    return result, output, original


def test_conversion(tmp_path):
    result, output, _ = convert(tmp_path)
    assert result.returncode == 0, result.stderr
    run = json.loads(output.read_text())
    assert output.stem == run['id']
    assert run['targets'][0]['results'][0]['durationSeconds'] == 0.2
    assert len(run['targets'][0]['results']) == 1
    assert run['source']['message'] == context()['source_subject']
    assert (
        run['execution']['completedAt'] == raw_run()['execution_summary']['finished_at']
    )
    assert run['testCatalog'].startswith('test-catalogs/default-')
    assert {p.name for p in tmp_path.iterdir()} == {'raw.json', output.name}


def test_attempt_identity(tmp_path):
    first, a, _ = convert(tmp_path)
    second, b, _ = convert(tmp_path, '2')
    assert first.returncode == second.returncode == 0
    assert json.loads(a.read_text())['id'] != json.loads(b.read_text())['id']


def test_invalid_input(tmp_path):
    raw = raw_run()
    raw['benchmark_results'][0]['timing_results_s'] = [-1]
    result, output, _ = convert(tmp_path, raw=raw)
    assert result.returncode != 0
    assert not output.exists()


def steps():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']['publish']['steps']


@pytest.mark.parametrize('existing', [False, True])
def test_copy_and_index_preserve_history(tmp_path, existing):
    result, output, original = convert(tmp_path)
    assert result.returncode == 0, result.stderr
    download = tmp_path / 'download'
    (download / 'raw').mkdir(parents=True)
    (download / 'dashboard').mkdir()
    (download / 'raw/run.json').write_bytes(original)
    (download / 'dashboard' / output.name).write_bytes(output.read_bytes())
    publication = tmp_path / 'publication'
    publication.mkdir()
    subprocess.run(['git', 'init', '-q', str(publication)], check=True)
    (publication / 'unrelated.txt').write_text('preserved')
    data = publication / 'rocjitsu-dashboard/data'
    if existing:
        data.mkdir(parents=True)
        (data / 'index.json').write_text(
            json.dumps({'runFiles': ['runs/older.json'], 'custom': 'keep'})
        )
    env = {
        **os.environ,
        'RESULTS_DOWNLOAD_DIR': str(download),
        'GITHUB_RUN_ID': '123',
        'GITHUB_RUN_ATTEMPT': '1',
    }
    body = next(
        step['run'] for step in steps() if step['name'] == 'Publish benchmark files'
    )
    copy_body = body[
        body.index('  DESTINATION=') : body.index('  if git diff --cached')
    ]
    copies = [{'run': copy_body}]
    env['DASHBOARD_READY'] = 'true'
    for _ in range(2):
        for step in copies:
            run = subprocess.run(
                ['bash', '-euo', 'pipefail', '-c', step['run']],
                cwd=publication,
                env=env,
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
    index = json.loads((data / 'index.json').read_text())
    assert index['runFiles'].count('runs/' + output.name) == 1
    if existing:
        assert 'runs/older.json' in index['runFiles']
        assert index['custom'] == 'keep'
    assert (data / 'runs' / output.name).read_bytes() == output.read_bytes()
    assert (
        publication / 'raw-benchmarks/github-123-attempt-1/raw.json'
    ).read_bytes() == original
    assert (publication / 'unrelated.txt').read_text() == 'preserved'
    assert not (data / 'test-catalogs').exists()
    assert not (data / 'metadata.json').exists()


def test_workflow_shell_and_removed_paths():
    doc = yaml.safe_load(WORKFLOW.read_text())
    for job in doc['jobs'].values():
        for step in job['steps']:
            if 'run' in step:
                result = subprocess.run(
                    ['bash', '-n'], input=step['run'], text=True, capture_output=True
                )
                assert result.returncode == 0, result.stderr
    text = WORKFLOW.read_text()
    for removed in (
        'workflow.txt',
        'enriched',
        'publish-raw-benchmark.py',
        'prepare-dashboard-data.py',
        'commit-tree',
        'test-catalogs',
    ):
        assert removed not in text
    assert "github.ref == 'refs/heads/develop'" in text
    assert 'publish_results' not in text
    assert doc['on' if 'on' in doc else True]['workflow_dispatch']['inputs'][
        'commit'
    ] == {
        'description': 'Rocjitsu commit SHA to benchmark',
        'required': True,
        'type': 'string',
    }
    assert doc['env']['ROCJITSU_SHA'] == '${{ inputs.commit || github.sha }}'
    benchmark = {step['name']: step for step in doc['jobs']['benchmark']['steps']}
    assert (
        benchmark['Checkout rocm-systems']['with']['ref'] == '${{ env.ROCJITSU_SHA }}'
    )
    assert benchmark['Checkout workflow tools']['with']['ref'] == '${{ github.sha }}'
    assert '/workflow-source/' in benchmark['Prepare dashboard benchmark result']['run']


@pytest.mark.parametrize(
    'always_fail,dashboard_ready', [(False, True), (False, False), (True, True)]
)
def test_publication_retry(tmp_path, always_fail, dashboard_ready):
    # All Git operations are local. The git wrapper intercepts push before Git
    # can execute it and simulates a branch race using local fetch/update-ref.
    import shutil

    real_git = shutil.which('git')
    assert real_git is not None
    seed = tmp_path / 'origin'
    publication = tmp_path / 'publication'

    def git(cwd, *args):
        return subprocess.run(
            [real_git, '-C', str(cwd), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    seed.mkdir()
    git(seed, 'init', '-q', '-b', 'gh-pages-rocjitsu')
    git(seed, 'config', 'user.name', 'Fixture')
    git(seed, 'config', 'user.email', 'fixture@example.invalid')
    data = seed / 'rocjitsu-dashboard/data'
    data.mkdir(parents=True)
    (data / 'index.json').write_text(
        json.dumps({'runFiles': ['runs/older.json'], 'custom': 'keep'})
    )
    git(seed, 'add', '.')
    git(seed, 'commit', '-qm', 'Initial fixture')
    subprocess.run([real_git, 'clone', '-q', str(seed), str(publication)], check=True)
    git(seed, 'checkout', '-qb', 'race')
    (data / 'index.json').write_text(
        json.dumps(
            {'runFiles': ['runs/older.json', 'runs/concurrent.json'], 'custom': 'keep'}
        )
    )
    (seed / 'concurrent.txt').write_text('preserve concurrent writer')
    git(seed, 'add', '.')
    git(seed, 'commit', '-qm', 'Concurrent writer fixture')
    race = git(seed, 'rev-parse', 'HEAD')
    result, output, original = convert(tmp_path)
    assert result.returncode == 0, result.stderr
    download = tmp_path / 'download'
    (download / 'raw').mkdir(parents=True)
    (download / 'raw/run.json').write_bytes(original)
    if dashboard_ready:
        (download / 'dashboard').mkdir()
        shutil.copy2(output, download / 'dashboard' / output.name)
    wrapper = tmp_path / 'bin'
    wrapper.mkdir()
    wrapper_code = r"""import os, pathlib, subprocess, sys
args = sys.argv[1:]
real = os.environ['REAL_GIT']
if args[0] != 'push':
    os.execv(real, [real, *args])
counter = pathlib.Path(os.environ['PUSH_COUNTER'])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
origin = os.environ['TEST_ORIGIN']
ref = 'refs/heads/gh-pages-rocjitsu'
def git(*args):
    return subprocess.check_output([real, '-C', origin, *args], text=True).strip()
if count == 1:
    git('update-ref', ref, 'refs/heads/race')
    sys.exit(1)
if os.environ['ALWAYS_FAIL'] == 'true':
    sys.exit(1)
old = git('rev-parse', ref)
git('fetch', os.getcwd(), 'HEAD')
new = git('rev-parse', 'FETCH_HEAD')
subprocess.run([real, '-C', origin, 'merge-base', '--is-ancestor', old, new], check=True)
git('update-ref', ref, new, old)
"""
    (wrapper / 'git').write_text('#!' + sys.executable + '\n' + wrapper_code)
    (wrapper / 'git').chmod(0o755)
    env = {
        **os.environ,
        'PATH': str(wrapper) + os.pathsep + os.environ['PATH'],
        'REAL_GIT': real_git,
        'TEST_ORIGIN': str(seed),
        'PUSH_COUNTER': str(tmp_path / 'push-count'),
        'ALWAYS_FAIL': str(always_fail).lower(),
        'DASHBOARD_READY': str(dashboard_ready).lower(),
        'RESULTS_DOWNLOAD_DIR': str(download),
        'RESULTS_BRANCH': 'gh-pages-rocjitsu',
        'GITHUB_RUN_ID': '123',
        'GITHUB_RUN_ATTEMPT': '1',
    }
    body = next(
        step['run'] for step in steps() if step['name'] == 'Publish benchmark files'
    )
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', body],
        cwd=publication,
        env=env,
        capture_output=True,
        text=True,
    )
    assert int((tmp_path / 'push-count').read_text()) == (3 if always_fail else 2)
    assert result.returncode == (1 if always_fail else 0), result.stderr
    if always_fail:
        assert git(seed, 'rev-parse', 'gh-pages-rocjitsu') == race
        assert 'Publication failed after 3 attempts' in result.stderr
    else:
        index = json.loads(
            git(seed, 'show', 'gh-pages-rocjitsu:rocjitsu-dashboard/data/index.json')
        )
        assert (
            'runs/concurrent.json' in index['runFiles']
            and 'runs/older.json' in index['runFiles']
        )
        assert index['custom'] == 'keep'
        assert ('runs/' + output.name in index['runFiles']) == dashboard_ready
        assert (
            git(seed, 'show', 'gh-pages-rocjitsu:concurrent.txt')
            == 'preserve concurrent writer'
        )
        published = subprocess.check_output(
            [
                real_git,
                '-C',
                str(seed),
                'show',
                'gh-pages-rocjitsu:raw-benchmarks/github-123-attempt-1/raw.json',
            ]
        )
        assert published == original


def test_develop_history_gate(tmp_path):
    repository = tmp_path / 'source'
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ['git', '-C', str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git('init', '-q', '-b', 'develop')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('commit', '--allow-empty', '-qm', 'Initial develop')
    ancestor = git('rev-parse', 'HEAD')
    git('commit', '--allow-empty', '-qm', 'Develop tip')
    tip = git('rev-parse', 'HEAD')
    git('update-ref', 'refs/remotes/origin/develop', tip)
    git('checkout', '-qb', 'feature', ancestor)
    git('commit', '--allow-empty', '-qm', 'Unmerged feature')
    feature = git('rev-parse', 'HEAD')
    doc = yaml.safe_load(WORKFLOW.read_text())
    benchmark = {step['name']: step for step in doc['jobs']['benchmark']['steps']}
    assert benchmark['Checkout rocm-systems']['with']['fetch-depth'] == 0
    body = benchmark['Require a commit from develop']['run']
    for revision, expected in [(ancestor, 0), (tip, 0), (feature, 1)]:
        git('checkout', '--detach', revision)
        result = subprocess.run(
            ['bash', '-euo', 'pipefail', '-c', body],
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected, result.stderr
        if expected:
            assert "must be in develop's history" in result.stderr


def test_failed_generation_log_is_visible(tmp_path):
    steps = yaml.safe_load(WORKFLOW.read_text())['jobs']['benchmark']['steps']
    diagnostics = next(
        (step for step in steps if step['name'] == 'Print failed Tensile generation logs'),
        None,
    )
    assert diagnostics is not None, 'Failed generator output must reach the Actions console'
    assert diagnostics['if'] == "${{ failure() && steps.prepare.outcome == 'failure' }}"
    benchmark_dir = tmp_path / 'benchmark with spaces'
    log = benchmark_dir / 'tensile-artifacts/bf16_streamk/generate.log'
    log.parent.mkdir(parents=True)
    log.write_text('Fixture: underlying Tensile error\n')
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', diagnostics['run']],
        env={**os.environ, 'BENCHMARK_DIR': str(benchmark_dir)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text() in result.stdout
    log.unlink()
    missing = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', diagnostics['run']],
        env={**os.environ, 'BENCHMARK_DIR': str(benchmark_dir)},
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 0, missing.stderr
    prepare = next(step for step in steps if step.get('id') == 'prepare')
    assert not prepare.get('continue-on-error', False)


def test_failed_preparation_logs_are_uploaded():
    steps = yaml.safe_load(WORKFLOW.read_text())['jobs']['benchmark']['steps']
    upload = next(
        (step for step in steps if step['name'] == 'Upload preparation failure logs'),
        None,
    )
    assert upload is not None, 'Preparation logs must survive a failed job'
    assert upload['if'] == "${{ failure() && env.BENCHMARK_DIR != '' }}"
    raw_upload = next(step for step in steps if step.get('id') == 'upload_raw')
    assert upload['uses'] == raw_upload['uses']
    assert upload['with']['name'] == '${{ env.RESULTS_ARTIFACT }}-preparation-logs'
    assert upload['with']['if-no-files-found'] == 'warn'
    assert upload['with']['retention-days'] == 14
    assert upload['with']['path'].splitlines() == [
        '${{ env.BENCHMARK_DIR }}/tensile-artifacts/*/generate.log',
        '${{ env.BENCHMARK_DIR }}/build/CMakeFiles/CMakeConfigureLog.yaml',
        '${{ env.BENCHMARK_DIR }}/build/CMakeFiles/*.log',
        '${{ env.BENCHMARK_DIR }}/rocisa-build/CMakeFiles/CMakeConfigureLog.yaml',
        '${{ env.BENCHMARK_DIR }}/rocisa-build/CMakeFiles/*.log',
        '${{ env.BENCHMARK_DIR }}/tensile-native/CMakeFiles/CMakeConfigureLog.yaml',
        '${{ env.BENCHMARK_DIR }}/tensile-native/CMakeFiles/*.log',
    ]
