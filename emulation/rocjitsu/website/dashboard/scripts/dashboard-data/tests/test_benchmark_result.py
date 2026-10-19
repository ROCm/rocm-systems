# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixture-based conversion and local file publication tests; no remote writes."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[7]
WORKFLOW = REPO_ROOT / '.github/workflows/rocjitsu-benchmarks.yml'
HELPER = Path(__file__).resolve().parents[1] / 'prepare-benchmark-result.py'
SOURCE_SHA = 'b' * 40
WORKFLOW_SHA = 'c' * 40
TENSILE_SHA = 'd' * 40
CORPUS_SHA = '6186d5c8bcd5e9bc93f7406e7cb50ed2a7698621'
SOURCE_SUBJECT = 'Source "subject" with spaces = literal $(touch SHOULD_NOT_EXIST)'


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
                'rocjitsu_commit_sha': SOURCE_SHA,
                'rocjitsu_commit_timestamp': '2026-09-30T00:00:00Z',
            },
            'corpus': {'corpus_commit_sha': CORPUS_SHA},
            'machine': {'hostname': 'fixture-host'},
            'auxiliary': {'optional': None},
        },
        'benchmark_results': [
            result,
            {**result, 'thread_policy': 'single', 'timing_results_s': [9.0, 8.0, 7.0]},
        ],
    }


def convert(
    tmp_path, attempt='1', raw=None, workflow_sha=WORKFLOW_SHA, tensile_sha=TENSILE_SHA
):
    source = tmp_path / 'raw.json'
    # Unusual indentation and trailing whitespace detect raw JSON reserialization.
    # Publication must archive these exact bytes, not just equivalent JSON.
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
        SOURCE_SUBJECT,
        '--machine',
        'benchmark machine',
        '--workflow-sha',
        workflow_sha,
        '--tensile-sha',
        tensile_sha,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert source.read_bytes() == original
    return result, output, original


@pytest.fixture
def publication_catalog():
    return {
        'id': 'default-67daf36efa1cd84a9f2beb1f0fa30860edbecf572f2b733c86642cdbbabea8cd',
        'tests': [
            {
                'id': 'fixture.case',
                'suite': 'Fixture',
                'name': 'Fixture case',
                'problem': {'dtype': 'fp32', 'n': 64, 'operation': 'copy'},
            }
        ],
        'targets': {'gfx950': ['fixture.case']},
    }


def test_conversion(tmp_path):
    result, output, _ = convert(tmp_path)
    assert result.returncode == 0, result.stderr
    run = json.loads(output.read_text())
    assert output.stem == run['id']
    assert run['targets'][0]['results'][0]['durationSeconds'] == 0.2
    assert len(run['targets'][0]['results']) == 1
    assert run['source']['message'] == SOURCE_SUBJECT
    assert (
        run['execution']['completedAt'] == raw_run()['execution_summary']['finished_at']
    )
    assert run['testCatalog'].startswith('test-catalogs/default-')
    assert {p.name for p in tmp_path.iterdir()} == {'raw.json', output.name}


def test_environment_labels_in_converter_output(tmp_path):
    raw = raw_run()
    raw['provenance']['auxiliary']['custom_option'] = {'nested': [1, None]}
    result, output, _ = convert(tmp_path, raw=raw)
    assert result.returncode == 0, result.stderr
    run = json.loads(output.read_text())
    details = {detail['key']: detail for detail in run['environment']}
    assert details['thread_policy']['label'] == 'Thread policy'
    assert details['corpus.corpus_commit_sha']['label'] == 'Corpus commit'
    assert details['execution.warmups']['label'] == 'Warmups'
    assert details['machine.hostname']['label'] == 'Hostname'
    assert details['auxiliary.custom_option']['label'] == 'Auxiliary Custom Option'
    assert details['auxiliary.custom_option']['value'] == '{"nested": [1, null]}'
    assert list(details) == sorted(details)

    assert [(detail['key'], detail['value']) for detail in run['environment']] == [
        ('auxiliary.custom_option', '{"nested": [1, null]}'),
        ('auxiliary.optional', 'null'),
        ('corpus.corpus_commit_sha', CORPUS_SHA),
        ('execution.benchmark_suite', 'fixture-suite'),
        ('execution.samples', 3),
        ('execution.timeout_seconds', 60),
        ('execution.warmups', 1),
        ('machine.hostname', 'fixture-host'),
        ('tensile.commit_sha', TENSILE_SHA),
        ('thread_policy', 'default'),
        ('workflow.commit_sha', WORKFLOW_SHA),
    ]
    assert run['targets'][0]['results'][0]['timing_results_s'] == [0.3, 0.1, 0.2]


def test_converter_revision_metadata(tmp_path):
    first, output, original = convert(tmp_path)
    assert first.returncode == 0, first.stderr
    run = json.loads(output.read_text())
    details = {detail['key']: detail for detail in run['environment']}
    assert details['workflow.commit_sha'] == {
        'key': 'workflow.commit_sha',
        'label': 'Workflow commit',
        'value': WORKFLOW_SHA,
    }
    assert details['tensile.commit_sha'] == {
        'key': 'tensile.commit_sha',
        'label': 'Tensile commit',
        'value': TENSILE_SHA,
    }
    assert len({run['source']['commit'], WORKFLOW_SHA, TENSILE_SHA}) == 3
    assert (tmp_path / 'raw.json').read_bytes() == original
    second, changed_output, _ = convert(
        tmp_path,
        workflow_sha='e' * 40,
        tensile_sha='f' * 40,
    )
    assert second.returncode == 0, second.stderr
    changed = json.loads(changed_output.read_text())
    for key in ('id', 'comparisonId', 'testCatalog', 'source', 'execution', 'targets'):
        assert changed[key] == run[key]
    changed_details = {detail['key']: detail for detail in changed['environment']}
    assert changed_details['workflow.commit_sha']['value'] == 'e' * 40
    assert changed_details['tensile.commit_sha']['value'] == 'f' * 40
    for key in details.keys() - {'workflow.commit_sha', 'tensile.commit_sha'}:
        assert changed_details[key] == details[key]


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


def test_dashboard_data_ci():
    doc = yaml.safe_load(
        (REPO_ROOT / '.github/workflows/rocjitsu-website-test.yml').read_text()
    )
    # Covers path filters and ordinary pytest wiring, not workflow execution.
    # PyYAML safe_load treats the unquoted YAML key on as boolean True.
    triggers = doc['on' if 'on' in doc else True]
    for event in ('pull_request', 'push'):
        assert '.github/workflows/rocjitsu-benchmarks.yml' in triggers[event]['paths']
    assert any(
        shlex.split(step.get('run', ''))
        == ['python', '-m', 'pytest', 'scripts/dashboard-data/tests']
        for step in doc['jobs']['dashboard-data']['steps']
    )


@pytest.mark.parametrize(
    'status,timings,duration',
    [
        ('completed', [0.3, 0.1, 0.2], 0.2),
        ('failed', [0.3], None),
        ('timeout', [], None),
    ],
)
def test_converter_result_status(tmp_path, status, timings, duration):
    raw = raw_run()
    measurement = raw['benchmark_results'][0]
    measurement.update(
        status=status,
        timing_results_s=timings,
        error=None if status == 'completed' else 'fixture workload failure',
        exit_code=0 if status == 'completed' else None,
    )
    result, output, _ = convert(tmp_path, raw=raw)
    assert result.returncode == 0, result.stderr
    published = json.loads(output.read_text())['targets'][0]['results'][0]
    assert published['status'] == status
    assert published['durationSeconds'] == duration
    assert published['error'] == measurement['error']
    assert published['timing_results_s'] == timings


def test_workload_failure_diagnostics(tmp_path):
    benchmark = yaml.safe_load(WORKFLOW.read_text())['jobs']['benchmark']
    logs = next(
        step
        for step in benchmark['steps']
        if step['name'] == 'Upload workload failure diagnostics'
    )
    assert logs['if'] == (
        "${{ !cancelled() && (steps.benchmark.outcome == 'failure' "
        "|| steps.benchmark.outputs.case_failures == 'true') }}"
    )
    assert logs['uses'].startswith('actions/upload-artifact@')
    assert logs['with'] == {
        'name': '${{ env.RESULTS_ARTIFACT }}-diagnostics',
        'path': (
            '${{ env.ROCJITSU_RESULT_DIR }}/*/*/cases/*/stdout.txt\n'
            '${{ env.ROCJITSU_RESULT_DIR }}/*/*/cases/*/stderr.txt\n'
        ),
        'if-no-files-found': 'ignore',
        'retention-days': 14,
    }
    # Match only current-run workload streams, not setup logs or other runs.
    paths = [
        'current/gfx950/default/cases/fixture.case/stdout.txt',
        'current/gfx1250/single/cases/fixture.case/stderr.txt',
        'current/gfx950/default/cases/fixture.case/config.json',
        'current/setup/stderr.txt',
        'older/gfx950/default/cases/fixture.case/stderr.txt',
    ]
    for path in paths:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    matched = {
        str(file.relative_to(tmp_path))
        for pattern in logs['with']['path'].splitlines()
        for file in (tmp_path / 'current').glob(
            pattern.removeprefix('${{ env.ROCJITSU_RESULT_DIR }}/')
        )
    }
    assert matched == set(paths[:2])


def steps():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']['publish']['steps']


@pytest.mark.parametrize('existing', [False, True])
def test_copy_and_index_preserve_history(tmp_path, existing, publication_catalog):
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
    catalog = data / 'test-catalogs' / (publication_catalog['id'] + '.json')
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps(publication_catalog))
    assert json.loads(output.read_text())['testCatalog'] == str(
        catalog.relative_to(data)
    )
    if existing:
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
    assert json.loads(catalog.read_text()) == publication_catalog
    assert not (data / 'metadata.json').exists()


@pytest.mark.parametrize(
    'catalog_case',
    [
        'present',
        'missing',
        'traversal',
        'absolute',
        'symlink',
        'directory',
        'not-string',
        'newline',
    ],
)
def test_publication_catalog_path(tmp_path, publication_catalog, catalog_case):
    result, output, _ = convert(tmp_path)
    assert result.returncode == 0, result.stderr
    run = json.loads(output.read_text())
    publication = tmp_path / 'publication'
    publication.mkdir()
    subprocess.run(['git', 'init', '-q', str(publication)], check=True)
    data = publication / 'rocjitsu-dashboard/data'
    catalog = data / run['testCatalog']
    catalog.parent.mkdir(parents=True)
    outside = publication / 'rocjitsu-dashboard/outside.json'
    outside.write_text(json.dumps(publication_catalog))
    if catalog_case == 'directory':
        catalog.mkdir()
    elif catalog_case == 'symlink':
        catalog.symlink_to(outside)
    elif catalog_case != 'missing':
        catalog.write_text(json.dumps(publication_catalog))
    if catalog_case == 'traversal':
        run['testCatalog'] = '../outside.json'
    elif catalog_case == 'absolute':
        run['testCatalog'] = str(outside)
    elif catalog_case == 'not-string':
        run['testCatalog'] = 123
        (data / '123').write_text(json.dumps(publication_catalog))
    elif catalog_case == 'newline':
        run['testCatalog'] += '\n'
    original_index = b'{"runFiles":["runs/older.json"],"custom":"keep"}\n'
    (data / 'index.json').write_bytes(original_index)
    download = tmp_path / 'download/dashboard'
    download.mkdir(parents=True)
    (download / output.name).write_text(json.dumps(run))
    body = next(
        step['run'] for step in steps() if step['name'] == 'Publish benchmark files'
    )
    dashboard_body = body[
        body.index('  if [[ "${DASHBOARD_READY}"') : body.index(
            '  if git diff --cached'
        )
    ]
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', dashboard_body],
        cwd=publication,
        env={
            **os.environ,
            'DASHBOARD_READY': 'true',
            'RESULTS_DOWNLOAD_DIR': str(download.parent),
            'GITHUB_RUN_ID': '123',
            'GITHUB_RUN_ATTEMPT': '1',
        },
        capture_output=True,
        text=True,
    )
    if catalog_case == 'present':
        assert result.returncode == 0, result.stderr
        assert (
            'runs/' + output.name
            in json.loads((data / 'index.json').read_text())['runFiles']
        )
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        assert (data / 'index.json').read_bytes() == original_index


def test_benchmark_artifact_contract():
    doc = yaml.safe_load(WORKFLOW.read_text())
    benchmark = {step['name']: step for step in doc['jobs']['benchmark']['steps']}
    uploads = {
        step['name']
        for step in benchmark.values()
        if step.get('uses', '').startswith('actions/upload-artifact@')
    }
    assert uploads == {
        'Upload original raw benchmark JSON',
        'Upload dashboard benchmark JSON',
        'Upload workload failure diagnostics',
    }
    raw = benchmark['Upload original raw benchmark JSON']
    assert raw['if'] == (
        "${{ !cancelled() && (steps.benchmark.outcome == 'success' "
        "|| steps.benchmark.outcome == 'failure') }}"
    )
    assert raw['with'] == {
        'name': '${{ env.RESULTS_ARTIFACT }}-raw',
        'path': '${{ env.ROCJITSU_RESULT_DIR }}/run.json',
        'if-no-files-found': 'error',
        'retention-days': 14,
    }
    dashboard = benchmark['Upload dashboard benchmark JSON']
    assert (
        dashboard['if'] == "${{ !cancelled() && steps.dashboard.outcome == 'success' }}"
    )
    assert dashboard['with'] == {
        'name': '${{ env.RESULTS_ARTIFACT }}-dashboard',
        'path': '${{ env.BENCHMARK_DIR }}/dashboard/*.json',
        'if-no-files-found': 'error',
        'retention-days': 14,
    }
    converter = benchmark['Prepare dashboard benchmark result']
    assert (
        converter['if'] == "${{ !cancelled() && steps.benchmark.outcome == 'success' }}"
    )
    assert converter['env'] == {
        'BENCHMARK_COMMIT_MESSAGE': '${{ steps.metadata.outputs.commit_message }}',
        'BENCHMARK_MACHINE': '${{ steps.metadata.outputs.machine }}',
        'BENCHMARK_WORKFLOW_SHA': '${{ github.sha }}',
    }
    assert doc['jobs']['benchmark']['outputs'] == {
        'raw_ready': "${{ steps.upload_raw.outcome == 'success' }}",
        'dashboard_ready': "${{ steps.upload_dashboard.outcome == 'success' }}",
    }
    downloads = {
        step['name']: step
        for step in doc['jobs']['publish']['steps']
        if step.get('uses', '').startswith('actions/download-artifact@')
    }
    assert set(downloads) == {'Download original raw JSON', 'Download dashboard JSON'}
    assert downloads['Download original raw JSON']['with'] == {
        'name': '${{ env.RESULTS_ARTIFACT }}-raw',
        'path': '${{ env.RESULTS_DOWNLOAD_DIR }}/raw',
    }
    assert downloads['Download dashboard JSON']['with'] == {
        'name': '${{ env.RESULTS_ARTIFACT }}-dashboard',
        'path': '${{ env.RESULTS_DOWNLOAD_DIR }}/dashboard',
    }
    assert downloads['Download dashboard JSON']['if'] == (
        "${{ needs.benchmark.outputs.dashboard_ready == 'true' }}"
    )


@pytest.mark.parametrize('symlinked', [False, True])
def test_completion_marker_output_path(tmp_path, symlinked):
    workspace = tmp_path / 'real workspace'
    results = workspace / 'results'
    results.mkdir(parents=True)
    (results / 'run.json').write_text('{}')
    if symlinked:
        link = tmp_path / 'workspace link'
        link.symlink_to(workspace, target_is_directory=True)
        selected_results = link / 'results'
    else:
        selected_results = results
    # Match the pinned runner's resolved artifact path without running a runner.
    (tmp_path / 'runner-output.txt').write_text(
        f'run failed: {selected_results.resolve()}/run.json\n'
    )
    body = next(
        step['run']
        for step in yaml.safe_load(WORKFLOW.read_text())['jobs']['benchmark']['steps']
        if step['name'] == 'Run performance suite'
    )
    comparison = body.split('esac\n', 1)[1].split('if ! python -m json.tool', 1)[0]
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', comparison],
        env={
            **os.environ,
            'BENCHMARK_DIR': str(tmp_path),
            'ROCJITSU_RESULT_DIR': str(selected_results),
            'SUITE_STATUS': 'failed',
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_workflow_shell_and_source_contracts():
    doc = yaml.safe_load(WORKFLOW.read_text())
    for job in doc['jobs'].values():
        for step in job['steps']:
            if 'run' in step:
                result = subprocess.run(
                    ['bash', '-n'], input=step['run'], text=True, capture_output=True
                )
                assert result.returncode == 0, result.stderr
    assert doc['jobs']['benchmark']['if'] == "${{ github.ref_type == 'branch' }}"
    assert doc['jobs']['publish']['if'] == (
        "${{ !cancelled() && github.ref == 'refs/heads/develop' "
        "&& needs.benchmark.outputs.raw_ready == 'true' }}"
    )
    # PyYAML safe_load treats the unquoted YAML key on as boolean True.
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
    'always_fail,dashboard_ready,catalog_state',
    [
        (False, True, 'present'),
        (False, False, 'missing'),
        (True, True, 'present'),
        (False, True, 'missing'),
        (False, True, 'removed-on-race'),
    ],
)
def test_publication_retry(
    tmp_path, always_fail, dashboard_ready, catalog_state, publication_catalog
):
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
    catalog = data / 'test-catalogs' / (publication_catalog['id'] + '.json')
    if catalog_state != 'missing':
        catalog.parent.mkdir()
        catalog.write_text(json.dumps(publication_catalog))
    (data / 'index.json').write_text(
        json.dumps({'runFiles': ['runs/older.json'], 'custom': 'keep'})
    )
    git(seed, 'add', '.')
    git(seed, 'commit', '-qm', 'Initial fixture')
    initial = git(seed, 'rev-parse', 'HEAD')
    initial_index = (data / 'index.json').read_bytes()
    subprocess.run([real_git, 'clone', '-q', str(seed), str(publication)], check=True)
    git(seed, 'checkout', '-qb', 'race')
    (data / 'index.json').write_text(
        json.dumps(
            {'runFiles': ['runs/older.json', 'runs/concurrent.json'], 'custom': 'keep'}
        )
    )
    (seed / 'concurrent.txt').write_text('preserve concurrent writer')
    if catalog_state == 'removed-on-race':
        catalog.unlink()
    race_index = (data / 'index.json').read_bytes()
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
    if dashboard_ready and catalog_state != 'present':
        assert result.returncode != 0, result.stdout + result.stderr
        assert 'catalog' in result.stderr.lower()
        counter = tmp_path / 'push-count'
        if catalog_state == 'removed-on-race':
            assert counter.read_text() == '1'
            expected_head, expected_index = race, race_index
        else:
            assert not counter.exists()
            expected_head, expected_index = initial, initial_index
        assert git(publication, 'rev-parse', 'HEAD') == expected_head
        assert git(seed, 'rev-parse', 'gh-pages-rocjitsu') == expected_head
        assert (
            publication / 'rocjitsu-dashboard/data/index.json'
        ).read_bytes() == expected_index
        return
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
