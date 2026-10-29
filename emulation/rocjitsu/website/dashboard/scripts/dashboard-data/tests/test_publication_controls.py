# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the actual workflow shell against fictional data and local Git only."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE.parents[2]
WORKFLOW = HERE.parents[6] / '.github/workflows/rocjitsu-benchmarks.yml'


def workflow_step(name):
    steps = yaml.safe_load(WORKFLOW.read_text())['jobs']['publish']['steps']
    return next(step['run'] for step in steps if step['name'] == name)


def execute(names, ctx):
    # Resolve the small set of Actions contexts used by the authored shared env.
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow['jobs']['publish']
    contexts = {
        'github.workspace': str(ctx.repo.parent),
        'github.run_id': ctx.env['GITHUB_RUN_ID'],
        'github.run_attempt': ctx.env['GITHUB_RUN_ATTEMPT'],
        'needs.benchmark.outputs.run_attempt': ctx.env['BENCHMARK_RUN_ATTEMPT'],
        'needs.benchmark.outputs.results_artifact': ctx.env[
            'BENCHMARK_RESULTS_ARTIFACT'
        ],
    }
    for key in ('RUN_ID', 'DATA_DIR', 'RESULTS_ARTIFACT'):
        value = job['env'].get(key, workflow['env'].get(key))
        for context, replacement in contexts.items():
            value = value.replace('${{ ' + context + ' }}', replacement)
        assert '${{' not in value
        ctx.env[key] = value
    result = None
    for name in (names,) if isinstance(names, str) else names:
        result = subprocess.run(
            ['bash', '-e', '-o', 'pipefail', '-c', workflow_step(name)],
            cwd=ctx.repo,
            env=ctx.env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return result
        # Actions exports GITHUB_ENV values to subsequent steps, not the current shell.
        env_file = Path(ctx.env['GITHUB_ENV'])
        if env_file.exists():
            ctx.env.update(
                line.split('=', 1) for line in env_file.read_text().splitlines()
            )
    assert result is not None, 'At least one actual workflow step is required'
    return result


PUBLICATION_STEPS = (
    'Validate publication candidate',
    'Store immutable run',
    'Update run index',
    'Commit and push publication',
)


def git(repo, *args, check=True):
    return subprocess.run(
        ['git', *args], cwd=repo, capture_output=True, text=True, check=check
    )


@pytest.fixture
def publication(tmp_path):
    raw = json.loads(
        (DASHBOARD / 'tests/fixtures/fictional-producer-run.json').read_text()
    )
    raw['benchmark_results'][0]['timing_results_s'] = [3e-9, 1e-9, 2e-9]
    raw['benchmark_results'][1].update(
        status='failed', exit_code=7, timing_results_s=[], error='Fictional failure'
    )
    raw['extra'] = {'zero': 0, 'false': False, 'null': None}
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'run.json').write_text(json.dumps(raw))
    repo = tmp_path / 'publication'
    repo.mkdir()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    env = {
        **os.environ,
        'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
        'ROCJITSU_SOURCE_DIR': str(DASHBOARD.parents[1]),
        'RESULTS_DOWNLOAD_DIR': str(results),
        'RUNNER_TEMP': str(scratch),
        'GITHUB_RUN_ID': '123',
        'GITHUB_RUN_ATTEMPT': '1',
        'BENCHMARK_RUN_ATTEMPT': '1',
        'BENCHMARK_RESULTS_ARTIFACT': 'rocjitsu-benchmark-123-1',
        'GITHUB_EVENT_NAME': 'push',
        'BENCHMARK_BRANCH': 'develop',
        'BENCHMARK_COMMIT_MESSAGE': 'Measured "subject" $(false)\nbody',
        'BENCHMARK_MACHINE': 'fictional-machine',
        'WORKFLOW_COMMIT_SHA': 'c' * 40,
        'BENCHMARK_SHA': raw['provenance']['rocjitsu']['rocjitsu_commit_sha'],
        'TENSILE_COMMIT_SHA': 'd' * 40,
        'SOURCE_BASE_BRANCH': '',
        'SOURCE_BASE_SHA': '',
        'SOURCE_PULL_REQUEST_NUMBER': '',
        'SOURCE_PULL_REQUEST_URL': '',
        'RESULTS_BRANCH': 'gh-pages-rocjitsu',
        'GITHUB_ENV': str(tmp_path / 'github-env'),
    }
    return SimpleNamespace(
        raw=raw,
        repo=repo,
        env=env,
        candidate=scratch / 'dashboard-publication-diagnostics/candidate.json',
        data=repo / 'rocjitsu-dashboard/data',
    )


@pytest.mark.parametrize(
    'artifact_name',
    ['rocjitsu-benchmark-123-1', 'renamed-benchmark-results-123-1'],
)
def test_publication_only_rerun_keeps_benchmark_artifact_and_identity(
    publication, tmp_path, artifact_name
):
    ctx = publication
    initialize_results(ctx, tmp_path)
    output = tmp_path / 'benchmark-output'
    source = next(
        step['run']
        for step in yaml.safe_load(WORKFLOW.read_text())['jobs']['benchmark']['steps']
        if step['name'] == 'Record benchmark source'
    )
    recorded = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', source],
        cwd=ctx.repo,
        env={
            **ctx.env,
            'GITHUB_OUTPUT': str(output),
            'GITHUB_SHA': ctx.env['WORKFLOW_COMMIT_SHA'],
            'BENCHMARK_SHA': git(ctx.repo, 'rev-parse', 'HEAD').stdout.strip(),
            'BENCHMARK_DIR': ctx.env['RUNNER_TEMP'],
        },
        capture_output=True,
        text=True,
    )
    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    outputs = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert outputs['run_attempt'] == '1'
    ctx.env['BENCHMARK_RUN_ATTEMPT'] = outputs['run_attempt']
    ctx.env['BENCHMARK_RESULTS_ARTIFACT'] = artifact_name
    ctx.env['GITHUB_RUN_ATTEMPT'] = '2'
    assert execute('Fill publication metadata', ctx).returncode == 0
    assert ctx.env['RESULTS_ARTIFACT'] == artifact_name
    assert (
        json.loads(ctx.candidate.read_text())['publication']['id']
        == 'github-123-attempt-1'
    )


def test_enrichment_preserves_raw_and_types_optional_metadata(publication):
    ctx = publication
    ctx.env.update(
        BENCHMARK_BRANCH='feature/quoted',
        GITHUB_EVENT_NAME='workflow_dispatch',
        SOURCE_BASE_BRANCH='develop',
        SOURCE_BASE_SHA='b' * 40,
        SOURCE_PULL_REQUEST_NUMBER='42',
        SOURCE_PULL_REQUEST_URL='https://example.invalid/pull/42',
    )
    result = execute('Fill publication metadata', ctx)
    assert result.returncode == 0, result.stdout + result.stderr
    candidate = json.loads(ctx.candidate.read_text())
    assert {k: v for k, v in candidate.items() if k != 'publication'} == {
        k: v for k, v in ctx.raw.items() if k != 'publication'
    }
    assert candidate['publication'] == {
        'id': 'github-123-attempt-1',
        'machine_label': 'fictional-machine',
        'trigger': 'manual',
        'workflow_commit_sha': ctx.env['WORKFLOW_COMMIT_SHA'],
        'tensile_commit_sha': 'd' * 40,
        'source': {
            'branch': 'feature/quoted',
            'message': ctx.env['BENCHMARK_COMMIT_MESSAGE'],
            'base': {'branch': 'develop', 'commit': 'b' * 40},
            'pull_request': {'number': 42, 'url': 'https://example.invalid/pull/42'},
        },
    }


@pytest.mark.parametrize('status', ['completed', 'failed', 'missing'])
def test_artifact_records_workflow_pin_without_publication(publication, status):
    ctx = publication
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow['jobs']['benchmark']['steps']
    record = next(
        (s for s in steps if s['name'] == 'Record result workflow revision'),
        {'run': ''},
    )
    result_dir = Path(ctx.env['RESULTS_DOWNLOAD_DIR'])
    result_file = result_dir / 'run.json'
    raw = json.loads(result_file.read_text())
    if status == 'missing':
        result_file.unlink()
    else:
        raw['execution_summary']['status'] = status
        result_file.write_text(json.dumps(raw))
    result = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', record['run']],
        env={
            **ctx.env,
            'ROCJITSU_RESULT_DIR': str(result_dir),
            'GITHUB_SHA': ctx.env['WORKFLOW_COMMIT_SHA'],
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if status == 'missing':
        assert not result_file.exists()
    else:
        observed = json.loads(result_file.read_text())
        expected = json.loads(json.dumps(raw))
        expected['publication']['workflow_commit_sha'] = ctx.env['WORKFLOW_COMMIT_SHA']
        assert observed == expected
        assert observed['provenance'] == ctx.raw['provenance']
        assert (
            observed['publication']['workflow_commit_sha'] != ctx.env['BENCHMARK_SHA']
        )
    # This step must also run after suite failure, independent of publication.
    assert (
        record['if']
        == "${{ always() && (steps.benchmark.outcome == 'success' || steps.benchmark.outcome == 'failure') }}"
    )
    names = [s['name'] for s in steps]
    assert (
        names.index('Run performance suite')
        < names.index(record['name'])
        < names.index('Upload benchmark diagnostics')
    )
    assert not ctx.candidate.exists()


def test_selected_source_and_workflow_revisions_reach_real_consumer(publication):
    ctx = publication
    benchmark = yaml.safe_load(WORKFLOW.read_text())['jobs']['benchmark']
    record = next(
        s for s in benchmark['steps'] if s['name'] == 'Record result workflow revision'
    )
    recorded = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', record['run']],
        env={
            **ctx.env,
            'ROCJITSU_RESULT_DIR': ctx.env['RESULTS_DOWNLOAD_DIR'],
            'GITHUB_SHA': ctx.env['WORKFLOW_COMMIT_SHA'],
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    artifact = json.loads(
        (Path(ctx.env['RESULTS_DOWNLOAD_DIR']) / 'run.json').read_text()
    )
    assert (
        artifact['publication']['workflow_commit_sha'] == ctx.env['WORKFLOW_COMMIT_SHA']
    )
    ctx.env.update(
        BENCHMARK_BRANCH='feature/source',
        GITHUB_EVENT_NAME='workflow_dispatch',
        SOURCE_BASE_BRANCH='develop',
        SOURCE_BASE_SHA='b' * 40,
        SOURCE_PULL_REQUEST_NUMBER='42',
        SOURCE_PULL_REQUEST_URL='https://github.com/ROCm/rocm-systems/pull/42',
    )
    result = execute(
        ('Fill publication metadata', 'Validate publication candidate'), ctx
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [
            'node',
            '--input-type=module',
            '-e',
            '''
import fs from 'node:fs';
const { adaptPublishedDashboardData } = await import(process.argv[1]);
const run = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const index = { runFiles: ['runs/side-branches/github-123-attempt-1.json'] };
const { data, sourceData } = adaptPublishedDashboardData({ index, runs: [run] });
console.log(JSON.stringify({ normalized: data.allRuns[0], raw: sourceData.runs[0] }));
''',
            (DASHBOARD / 'src/data/dashboardDataAdapter.js').as_uri(),
            str(ctx.candidate),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    consumed = json.loads(result.stdout)
    native = consumed['normalized']['provenance']
    assert (
        native['rocjitsuCommitSha']
        == ctx.env['BENCHMARK_SHA']
        != ctx.env['WORKFLOW_COMMIT_SHA']
    )
    assert native['commitMessage'] == ctx.env['BENCHMARK_COMMIT_MESSAGE']
    assert (
        next(
            item['value']
            for item in native['details']
            if item['key'] == 'rocjitsu.workflow_commit_sha'
        )
        == ctx.env['WORKFLOW_COMMIT_SHA']
    )
    assert consumed['normalized']['branch'] == 'feature/source'
    assert consumed['normalized']['sourceBase'] == {
        'branch': 'develop',
        'commit': 'b' * 40,
    }
    assert consumed['normalized']['pullRequest']['number'] == 42
    assert consumed['raw'] == json.loads(ctx.candidate.read_text())
    assert consumed['raw']['provenance'] == ctx.raw['provenance']


@pytest.mark.parametrize('optional', ['null', 'absent', 'populated'])
def test_enriched_candidate_passes_gate_without_changing_bytes(publication, optional):
    ctx = publication
    if optional == 'populated':
        ctx.env.update(
            SOURCE_BASE_BRANCH='develop',
            SOURCE_BASE_SHA='b' * 40,
            SOURCE_PULL_REQUEST_NUMBER='42',
        )
    assert execute('Fill publication metadata', ctx).returncode == 0
    candidate = json.loads(ctx.candidate.read_text())
    if optional == 'absent':
        del candidate['publication']['source']['base']
        del candidate['publication']['source']['pull_request']
        ctx.candidate.write_text(json.dumps(candidate))
    before = ctx.candidate.read_bytes()
    result = execute('Validate publication candidate', ctx)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '9 passed' in result.stdout
    assert ctx.candidate.read_bytes() == before
    assert not ctx.data.exists()


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
@pytest.mark.parametrize('invalid', ['missing', None, '', '   '])
def test_gate_rejects_required_field_before_publication_writes(
    publication, field, invalid
):
    ctx = publication
    assert execute('Fill publication metadata', ctx).returncode == 0
    candidate = json.loads(ctx.candidate.read_text())
    container = candidate
    *parents, key = field.split('.')
    for parent in parents:
        container = container[parent]
    if invalid == 'missing':
        del container[key]
    else:
        container[key] = invalid
    ctx.candidate.write_text(json.dumps(candidate))
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode == 1, result.stdout + result.stderr
    assert field in result.stdout
    assert (
        'is required' in result.stdout or 'must be a nonempty string' in result.stdout
    )
    assert not ctx.data.exists()
    assert not (ctx.candidate.parent / 'commit.log').exists()
    assert not (ctx.candidate.parent / 'push.log').exists()


def test_gate_requires_candidate_path(publication):
    ctx = publication
    ctx.env.pop('CANDIDATE_FILE', None)
    result = execute('Validate publication candidate', ctx)
    assert result.returncode != 0
    assert 'CANDIDATE_FILE' in result.stderr
    assert not ctx.data.exists()


def test_gate_rejects_nulls_from_actual_workflow_enrichment(publication):
    ctx = publication
    ctx.env.update(BENCHMARK_MACHINE='', BENCHMARK_SHA='e' * 40)
    result = execute(('Fill publication metadata', *PUBLICATION_STEPS), ctx)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'publication.machine_label' in result.stdout
    assert 'publication.source.message' in result.stdout
    assert not ctx.data.exists()


def initialize_results(ctx, tmp_path):
    remote = tmp_path / 'results.git'
    git(tmp_path, 'init', '--bare', str(remote))
    git(ctx.repo, 'init', '--initial-branch=gh-pages-rocjitsu')
    git(ctx.repo, 'config', 'user.name', 'Fictional publication')
    git(ctx.repo, 'config', 'user.email', 'fixture@example.invalid')
    git(ctx.repo, 'config', 'commit.gpgsign', 'false')
    (ctx.repo / 'unrelated.txt').write_text('keep me\n')
    git(ctx.repo, 'add', 'unrelated.txt')
    git(ctx.repo, 'commit', '-m', 'Seed local fixture')
    git(ctx.repo, 'remote', 'add', 'origin', str(remote))
    git(ctx.repo, 'push', 'origin', 'HEAD:refs/heads/gh-pages-rocjitsu')
    return remote


def test_publication_places_new_run_and_identical_retry_is_noop(publication, tmp_path):
    ctx = publication
    remote = initialize_results(ctx, tmp_path)
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode == 0, result.stdout + result.stderr
    path = 'runs/default-branch/github-123-attempt-1.json'
    assert (ctx.data / path).read_bytes() == ctx.candidate.read_bytes()
    index = ctx.data / 'index.json'
    before = index.read_bytes()
    assert json.loads(before)['runFiles'] == [path]
    head = git(ctx.repo, 'rev-parse', 'HEAD').stdout.strip()
    assert (
        git(
            tmp_path,
            '--git-dir',
            str(remote),
            'rev-parse',
            'refs/heads/gh-pages-rocjitsu',
        ).stdout.strip()
        == head
    )
    (ctx.candidate.parent / 'push.log').unlink()
    ctx.env['GITHUB_RUN_ATTEMPT'] = '2'
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (ctx.candidate.parent / 'push.log').exists()
    assert index.read_bytes() == before
    assert git(ctx.repo, 'rev-parse', 'HEAD').stdout.strip() == head
    assert (ctx.repo / 'unrelated.txt').read_text() == 'keep me\n'
    assert git(ctx.repo, 'status', '--porcelain').stdout == ''
    original_run = (ctx.data / path).read_bytes()
    ctx.env['GITHUB_RUN_ATTEMPT'] = '2'
    ctx.env['BENCHMARK_RUN_ATTEMPT'] = '2'
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode == 0, result.stdout + result.stderr
    next_path = 'runs/default-branch/github-123-attempt-2.json'
    assert json.loads(index.read_text())['runFiles'] == [path, next_path]
    assert (ctx.data / next_path).read_bytes() == ctx.candidate.read_bytes()
    assert (ctx.data / path).read_bytes() == original_run
    assert (
        git(
            tmp_path,
            '--git-dir',
            str(remote),
            'rev-parse',
            'refs/heads/gh-pages-rocjitsu',
        ).stdout
        == git(ctx.repo, 'rev-parse', 'HEAD').stdout
    )


def test_unknown_metadata_and_unaligned_message_stay_null(publication):
    ctx = publication
    ctx.env.update(
        BENCHMARK_MACHINE='',
        WORKFLOW_COMMIT_SHA='c' * 40,
        BENCHMARK_SHA='e' * 40,
        TENSILE_COMMIT_SHA='',
        GITHUB_EVENT_NAME='unknown',
    )
    assert execute('Fill publication metadata', ctx).returncode == 0
    pub = json.loads(ctx.candidate.read_text())['publication']
    assert pub['source'] == {
        'branch': 'develop',
        'message': None,
        'base': None,
        'pull_request': None,
    }
    assert pub['trigger'] is pub['machine_label'] is pub['tensile_commit_sha'] is None
    ctx.env.update(SOURCE_PULL_REQUEST_NUMBER='42', WORKFLOW_COMMIT_SHA='')
    assert execute('Fill publication metadata', ctx).returncode == 0
    pub = json.loads(ctx.candidate.read_text())['publication']
    assert pub['source']['pull_request'] == {'number': 42}
    assert pub['workflow_commit_sha'] is None
    raw_path = Path(ctx.env['RESULTS_DOWNLOAD_DIR']) / 'run.json'
    raw = json.loads(raw_path.read_text())
    raw['provenance']['rocjitsu']['rocjitsu_commit_sha'] = None
    raw_path.write_text(json.dumps(raw))
    assert execute('Fill publication metadata', ctx).returncode == 0
    assert (
        json.loads(ctx.candidate.read_text())['publication']['source']['message']
        is None
    )


@pytest.mark.parametrize('identical', [False, True])
def test_existing_run_is_never_overwritten(publication, tmp_path, identical):
    ctx = publication
    remote = initialize_results(ctx, tmp_path)
    assert execute('Fill publication metadata', ctx).returncode == 0
    existing = ctx.data / 'runs/default-branch/github-123-attempt-1.json'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(
        ctx.candidate.read_bytes() if identical else b'existing conflicting run\n'
    )
    git(ctx.repo, 'add', 'rocjitsu-dashboard')
    git(ctx.repo, 'commit', '-m', 'Seed conflict')
    before = existing.read_bytes()
    remote_head = git(
        tmp_path, '--git-dir', str(remote), 'rev-parse', 'refs/heads/gh-pages-rocjitsu'
    ).stdout
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    if identical:
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((ctx.data / 'index.json').read_text())['runFiles'] == [
            'runs/default-branch/github-123-attempt-1.json'
        ]
    else:
        assert result.returncode != 0
        assert not (ctx.data / 'index.json').exists()
    assert existing.read_bytes() == before
    assert git(
        tmp_path,
        '--git-dir',
        str(remote),
        'rev-parse',
        'refs/heads/gh-pages-rocjitsu',
    ).stdout == (
        git(ctx.repo, 'rev-parse', 'HEAD').stdout if identical else remote_head
    )


@pytest.mark.parametrize(
    'component',
    [
        'rocjitsu-dashboard',
        'rocjitsu-dashboard/data/runs',
        'rocjitsu-dashboard/data/runs/default-branch',
        'rocjitsu-dashboard/data/index.json',
        'rocjitsu-dashboard/data/runs/default-branch/github-123-attempt-1.json',
    ],
)
def test_publication_rejects_symbolic_paths(publication, tmp_path, component):
    ctx = publication
    initialize_results(ctx, tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'keep'
    marker.write_bytes(b'keep\n')
    link = ctx.repo / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    git(ctx.repo, 'add', 'rocjitsu-dashboard')
    git(ctx.repo, 'commit', '-m', 'Seed symbolic path')
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode != 0
    assert sorted(p.name for p in outside.iterdir()) == ['keep']
    assert marker.read_bytes() == b'keep\n'


def test_index_append_preserves_history_and_stages_only_run_and_index(
    publication, tmp_path
):
    ctx = publication
    initialize_results(ctx, tmp_path)
    ctx.env['BENCHMARK_BRANCH'] = 'feature/side'
    ctx.data.mkdir(parents=True)
    index = ctx.data / 'index.json'
    index.write_text(
        json.dumps(
            {
                'generatedAt': '2026-01-01T00:00:00Z',
                'runFiles': ['runs/default-branch/old.json'],
            }
        )
    )
    git(ctx.repo, 'add', 'rocjitsu-dashboard/data/index.json')
    git(ctx.repo, 'commit', '-m', 'Seed index')
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(index.read_text())['runFiles'] == [
        'runs/default-branch/old.json',
        'runs/side-branches/github-123-attempt-1.json',
    ]
    assert set(
        git(
            ctx.repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'
        ).stdout.splitlines()
    ) == {
        'rocjitsu-dashboard/data/index.json',
        'rocjitsu-dashboard/data/runs/side-branches/github-123-attempt-1.json',
    }
    before = index.read_bytes()
    assert execute('Update run index', ctx).returncode == 0
    assert index.read_bytes() == before


@pytest.mark.parametrize(
    'paths', [['../escape.json'], ['runs/default-branch/old.json'] * 2, None]
)
def test_unusable_index_is_not_rewritten(publication, tmp_path, paths):
    ctx = publication
    initialize_results(ctx, tmp_path)
    ctx.data.mkdir(parents=True)
    index = ctx.data / 'index.json'
    before = json.dumps({'generatedAt': 'unchanged', 'runFiles': paths}).encode()
    index.write_bytes(before)
    git(ctx.repo, 'add', 'rocjitsu-dashboard')
    git(ctx.repo, 'commit', '-m', 'Seed invalid index')
    assert execute('Fill publication metadata', ctx).returncode == 0
    assert execute(PUBLICATION_STEPS, ctx).returncode != 0
    assert index.read_bytes() == before


def test_non_fast_forward_push_fails_without_overwriting_peer(publication, tmp_path):
    ctx = publication
    remote = initialize_results(ctx, tmp_path)
    peer = tmp_path / 'peer'
    git(tmp_path, 'clone', '--branch', 'gh-pages-rocjitsu', str(remote), str(peer))
    git(peer, 'config', 'user.name', 'Fictional peer')
    git(peer, 'config', 'user.email', 'peer@example.invalid')
    git(peer, 'config', 'commit.gpgsign', 'false')
    (peer / 'peer.txt').write_text('preserve peer\n')
    git(peer, 'add', 'peer.txt')
    git(peer, 'commit', '-m', 'Concurrent local update')
    git(peer, 'push', 'origin', 'HEAD:refs/heads/gh-pages-rocjitsu')
    remote_head = git(peer, 'rev-parse', 'HEAD').stdout
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode != 0
    assert 'rejected' in result.stdout + result.stderr
    assert (
        git(
            tmp_path,
            '--git-dir',
            str(remote),
            'rev-parse',
            'refs/heads/gh-pages-rocjitsu',
        ).stdout
        == remote_head
    )
    assert (ctx.candidate.parent / 'push.log').is_file()


def test_commit_failure_is_visible_and_never_pushes(publication, tmp_path):
    ctx = publication
    remote = initialize_results(ctx, tmp_path)
    before = git(
        tmp_path, '--git-dir', str(remote), 'rev-parse', 'refs/heads/gh-pages-rocjitsu'
    ).stdout
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    hook = hooks / 'pre-commit'
    hook.write_text(
        '#!/bin/sh\nprintf "deliberate local commit failure\\n" >&2\nexit 1\n'
    )
    hook.chmod(0o755)
    git(ctx.repo, 'config', 'core.hooksPath', str(hooks))
    assert execute('Fill publication metadata', ctx).returncode == 0
    result = execute(PUBLICATION_STEPS, ctx)
    assert result.returncode != 0
    assert 'deliberate local commit failure' in result.stdout + result.stderr
    assert (
        git(
            tmp_path,
            '--git-dir',
            str(remote),
            'rev-parse',
            'refs/heads/gh-pages-rocjitsu',
        ).stdout
        == before
    )
    assert not (ctx.candidate.parent / 'push.log').exists()
