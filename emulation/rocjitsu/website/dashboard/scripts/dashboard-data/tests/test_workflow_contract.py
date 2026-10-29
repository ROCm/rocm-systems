# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Workflow wiring and shell syntax; publication behavior runs in local Git tests."""

from pathlib import Path
import subprocess

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[7] / '.github/workflows'


@pytest.mark.parametrize('event', ['pull_request', 'push'])
def test_website_ci_watches_benchmark_preparation(event):
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-website-test.yml').read_text())
    # PyYAML's YAML 1.1 loader reads the unquoted Actions `on` key as True.
    triggers = workflow[True]
    assert (
        'emulation/rocjitsu/scripts/prepare-benchmarks.py' in triggers[event]['paths']
    )


def test_benchmark_pr_trigger_matches_fuzz_opt_in_events_and_paths():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    trigger = workflow[True]['pull_request']
    assert trigger['branches'] == ['develop']
    assert trigger['types'] == ['labeled', 'synchronize']
    assert trigger['paths'] == workflow[True]['push']['paths']


@pytest.mark.parametrize(
    'event,action,label,labels,head_repo,repo,allowed',
    [
        (
            'pull_request',
            'labeled',
            'rocjitsu-presubmit-benchmarks',
            ['rocjitsu-presubmit-benchmarks'],
            'ROCm/rocm-systems',
            'ROCm/rocm-systems',
            True,
        ),
        (
            'pull_request',
            'synchronize',
            '',
            ['rocjitsu-presubmit-benchmarks'],
            'ROCm/rocm-systems',
            'ROCm/rocm-systems',
            True,
        ),
        (
            'pull_request',
            'labeled',
            'other',
            ['rocjitsu-presubmit-benchmarks', 'other'],
            'ROCm/rocm-systems',
            'ROCm/rocm-systems',
            False,
        ),
        (
            'pull_request',
            'synchronize',
            '',
            [],
            'ROCm/rocm-systems',
            'ROCm/rocm-systems',
            False,
        ),
        (
            'pull_request',
            'labeled',
            'rocjitsu-presubmit-benchmarks',
            ['rocjitsu-presubmit-benchmarks'],
            'fork/repo',
            'ROCm/rocm-systems',
            False,
        ),
        ('workflow_dispatch', '', '', [], '', 'ROCm/rocm-systems', True),
        ('push', '', '', [], '', 'ROCm/rocm-systems', True),
        ('workflow_dispatch', '', '', [], '', 'fork/repo', False),
    ],
)
def test_benchmark_opt_in_gate(event, action, label, labels, head_repo, repo, allowed):
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    condition = workflow['jobs']['source-metadata']['if']
    for key, value in {
        "contains(github.event.pull_request.labels.*.name, 'rocjitsu-presubmit-benchmarks')": repr(
            'rocjitsu-presubmit-benchmarks' in labels
        ),
        'github.event.pull_request.head.repo.full_name': repr(head_repo),
        'github.event.label.name': repr(label),
        'github.event.action': repr(action),
        'github.event_name': repr(event),
        'github.repository': repr(repo),
        'github.ref': repr('refs/heads/develop'),
    }.items():
        condition = condition.replace(key, value)
    assert (
        eval(
            condition.replace('&&', 'and').replace('||', 'or'), {'__builtins__': {}}, {}
        )
        is allowed
    )


@pytest.mark.parametrize(
    'event,source,benchmark,cancelled,expected',
    [
        ('pull_request', 'success', 'success', False, True),
        ('pull_request', 'success', 'failure', False, True),
        ('pull_request', 'success', 'skipped', False, False),
        ('pull_request', 'success', 'cancelled', False, False),
        ('pull_request', 'failure', 'skipped', False, False),
        ('pull_request', 'success', 'success', True, False),
        ('push', 'success', 'success', False, False),
        ('workflow_dispatch', 'success', 'success', False, False),
    ],
)
def test_report_only_runs_for_executed_pr_benchmarks(
    event, source, benchmark, cancelled, expected
):
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    job = workflow['jobs']['report']
    assert job['needs'] == ['source-metadata', 'benchmark', 'publish']
    assert job['permissions'] == {
        'contents': 'read',
        'actions': 'read',
        'pull-requests': 'write',
    }
    assert job['concurrency'] == {
        'group': 'rocjitsu-benchmark-report-${{ github.event.pull_request.number }}',
        'cancel-in-progress': False,
        'queue': 'max',
    }
    condition = job['if']
    for key, value in {
        'always()': 'True',
        '!cancelled()': repr(not cancelled),
        'github.event_name': repr(event),
        'needs.source-metadata.result': repr(source),
        'needs.benchmark.result': repr(benchmark),
    }.items():
        condition = condition.replace(key, value)
    assert (
        eval(
            condition.replace('&&', 'and').replace('||', 'or'), {'__builtins__': {}}, {}
        )
        is expected
    )
    report = next(s for s in job['steps'] if s['name'] == 'Comment on the pull request')
    assert report['env'] == {
        'BENCHMARK_RESULT': '${{ needs.benchmark.result }}',
        'PUBLICATION_RESULT': '${{ needs.publish.result }}',
        'SOURCE_SHA': '${{ needs.source-metadata.outputs.source_sha }}',
    }
    assert 'runNumber: process.env.GITHUB_RUN_NUMBER' in report['with']['script']
    assert 'runAttempt: process.env.GITHUB_RUN_ATTEMPT' in report['with']['script']
    assert 'headSha: process.env.SOURCE_SHA' in report['with']['script']
    assert 'reportBenchmark' in report['with']['script']


def test_report_helper_changes_trigger_their_node_ci_tests():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-website-test.yml').read_text())
    helper_paths = [
        '.github/scripts/rocjitsu_benchmark_report.js',
        '.github/scripts/tests/test_rocjitsu_benchmark_report.js',
    ]
    for event in ('push', 'pull_request'):
        for path in helper_paths:
            assert path in workflow[True][event]['paths']
    step = next(
        s
        for s in workflow['jobs']['verify']['steps']
        if s['name'] == 'Test benchmark PR reporting'
    )
    assert step['working-directory'] == '.'
    assert (
        step['run']
        == 'node --test .github/scripts/tests/test_rocjitsu_benchmark_report.js'
    )


def test_dispatch_form_explains_source_and_pr_constraints():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    inputs = workflow[True]['workflow_dispatch']['inputs']
    source = inputs['source_commit']['description']
    assert 'Use workflow from' in source
    assert 'blank' in source and 'branch tip' in source
    assert '7-40' in source and 'hex' in source and 'history' in source
    assert 'side-branch SHA' in source and 'develop' in source
    pr = inputs['pull_request_number']['description']
    assert 'open' in pr and 'same-repository' in pr
    assert 'selected side branch' in pr and 'develop' in pr


def test_python_ci_names_its_current_workflow_responsibilities():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-website-test.yml').read_text())
    job = workflow['jobs']['dashboard-data']
    assert job['name'] == 'Verify benchmark preparation and publication'
    step = next(s for s in job['steps'] if 'python -m pytest' in s.get('run', ''))
    assert step['name'] == 'Test benchmark preparation and publication'
    assert step['run'] == 'python -m pytest scripts/dashboard-data/tests'


def test_website_ci_runs_opt_in_published_data_production_smoke():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-website-test.yml').read_text())
    job = workflow['jobs']['verify']
    steps = job['steps']
    smoke = next(
        (s for s in steps if s['name'] == 'Smoke test published dashboard data'), None
    )
    assert (
        smoke is not None
    ), 'Website CI must test the new build against published data'
    assert (
        smoke['run']
        == 'npm run test:e2e:production -- tests/production/published-data.e2e.js'
    )
    assert smoke['env'] == {'PLAYWRIGHT_PUBLISHED_DATA': '1'}
    assert 'if' not in smoke and 'continue-on-error' not in smoke
    assert not job.get('continue-on-error', False)
    assert steps.index(smoke) > next(
        i for i, s in enumerate(steps) if s['name'] == 'Verify website'
    )
    assert steps.index(smoke) < next(
        i for i, s in enumerate(steps) if s['name'] == 'Upload Playwright diagnostics'
    )
    assert 'PLAYWRIGHT_PUBLISHED_DATA' not in workflow.get('env', {})
    assert 'PLAYWRIGHT_PUBLISHED_DATA' not in job.get('env', {})


def test_publication_serializes_shared_branch_writes_without_dropping_pending_jobs():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    assert workflow['jobs']['publish']['concurrency'] == {
        'group': 'rocjitsu-publish-gh-pages-rocjitsu',
        'cancel-in-progress': False,
        'queue': 'max',
    }
    assert 'concurrency' not in workflow


@pytest.mark.parametrize(
    'event,pr,run_id,cancel',
    [
        ('pull_request', 42, 100, True),
        ('pull_request', 43, 101, True),
        ('push', None, 102, False),
        ('workflow_dispatch', None, 103, False),
    ],
)
def test_only_pr_benchmarks_coalesce(event, pr, run_id, cancel):
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    concurrency = workflow['jobs']['benchmark']['concurrency']
    assert concurrency['group'] == (
        '${{ github.workflow }}-benchmark-${{ github.event.pull_request.number || github.run_id }}'
    )
    group = (
        concurrency['group']
        .replace('${{ github.workflow }}', 'rocjitsu-benchmarks')
        .replace(
            '${{ github.event.pull_request.number || github.run_id }}',
            str(pr or run_id),
        )
    )
    assert group == f'rocjitsu-benchmarks-benchmark-{pr if pr is not None else run_id}'
    expression = (
        concurrency['cancel-in-progress'].removeprefix('${{').removesuffix('}}').strip()
    )
    assert (
        eval(
            expression.replace('github.event_name', repr(event)),
            {'__builtins__': {}},
            {},
        )
        is cancel
    )


def test_publication_steps_follow_responsibility_order():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    assert [step['name'] for step in workflow['jobs']['publish']['steps']] == [
        'Download benchmark diagnostics',
        'Checkout repository for publication',
        'Checkout publication validation source',
        'Set up publication validation Python',
        'Install publication validation dependencies',
        'Fill publication metadata',
        'Validate publication candidate',
        'Store immutable run',
        'Update run index',
        'Commit and push publication',
        'Upload publication diagnostics',
    ]


def test_publication_validation_uses_workflow_source_and_declared_dependencies():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    steps = {step['name']: step for step in workflow['jobs']['publish']['steps']}
    checkout = steps['Checkout publication validation source']
    assert checkout['uses'].startswith('actions/checkout@')
    assert checkout['with'] == {
        'path': 'rocm-systems',
        'ref': '${{ github.sha }}',
        'persist-credentials': False,
    }
    assert workflow['env']['ROCJITSU_SOURCE_DIR'] == (
        '${{ github.workspace }}/rocm-systems/emulation/rocjitsu'
    )
    python = steps['Set up publication validation Python']
    assert python['uses'].startswith('actions/setup-python@')
    assert python['with']['python-version'] == '3.12'
    assert steps['Install publication validation dependencies']['run'] == (
        'python -m pip install -r '
        '"${ROCJITSU_SOURCE_DIR}/website/dashboard/scripts/dashboard-data/requirements-test.txt"'
    )
    validation = steps['Validate publication candidate']
    assert ': "${CANDIDATE_FILE:?}"' in validation['run']
    assert 'python -m pytest -q ' in validation['run']
    assert (
        '"${ROCJITSU_SOURCE_DIR}/website/dashboard/scripts/dashboard-data/test_publication_candidate.py"'
        in validation['run']
    )
    assert 'if' not in validation and 'continue-on-error' not in validation


def test_diagnostics_path_uses_runtime_context_and_upload_survives_early_failure():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    publish = workflow['jobs']['publish']
    # runner is unavailable in job env, but available in a step's with values.
    assert 'DIAGNOSTICS_DIR' not in publish['env']
    fill = next(s for s in publish['steps'] if s['name'] == 'Fill publication metadata')
    assert (
        'DIAGNOSTICS_DIR="${RUNNER_TEMP}/dashboard-publication-diagnostics"'
        in fill['run']
    )
    assert 'DIAGNOSTICS_DIR=%s' in fill['run'] and '"${GITHUB_ENV}"' in fill['run']
    upload = next(
        s for s in publish['steps'] if s['name'] == 'Upload publication diagnostics'
    )
    assert upload['if'] == 'always()'
    assert (
        upload['with']['path'] == '${{ runner.temp }}/dashboard-publication-diagnostics'
    )


@pytest.mark.parametrize('matches', [False, True])
def test_tensile_revision_shell_consumes_declared_pin(tmp_path, matches):
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    script = next(
        s['run']
        for s in workflow['jobs']['benchmark']['steps']
        if s['name'] == 'Record checked Tensile revision'
    )
    output = tmp_path / 'output'
    result = subprocess.run(
        ['bash', '-eu', '-c', 'git() { printf "%s\\n" "$CHECKED_SHA"; };\n' + script],
        env={
            'TENSILE_SOURCE': str(tmp_path / 'tensile'),
            'TENSILE_SHA': 'c' * 40,
            'CHECKED_SHA': ('c' if matches else 'd') * 40,
            'GITHUB_OUTPUT': str(output),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is matches, result.stderr
    if matches:
        assert output.read_text() == 'commit_sha=' + 'c' * 40 + '\n'
    else:
        assert not output.exists()


def test_benchmark_pin_and_unfiltered_execution():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    assert workflow['env']['CORPUS_SHA'] == 'b95ea549e1f500ce4ce83ae511ae5e4434930374'
    benchmark = workflow['jobs']['benchmark']
    command = next(
        s['run'] for s in benchmark['steps'] if s['name'] == 'Run performance suite'
    )
    assert command.count('-m benchmarks.runner') == 1
    assert (
        workflow['env']['BENCHMARK_MANIFEST'] == 'benchmarks/suites/performance-v1.toml'
    )
    assert '--manifest "${BENCHMARK_MANIFEST}"' in command
    generation = next(
        s['run']
        for s in benchmark['steps']
        if s['name'] == 'Generate pinned Tensile artifacts'
    )
    assert (
        'prepare-benchmarks.py" generate --manifest "${BENCHMARK_MANIFEST}"'
        in generation
    )
    workflow_text = (WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text()
    assert workflow_text.count('benchmarks/suites/performance-v1.toml') == 1
    helper = WORKFLOWS.parents[1] / 'emulation/rocjitsu/scripts/prepare-benchmarks.py'
    assert 'benchmarks/suites/performance-v1.toml' not in helper.read_text()
    assert command.count('--target-config ') == 2
    assert '--output "${ROCJITSU_RESULT_DIR}"' in command
    for selector in (
        '--thread-policy',
        '--target ',
        '--case ',
        '--warmups',
        '--samples',
    ):
        assert selector not in command


def test_publication_metadata_uses_existing_checked_outputs():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    benchmark = workflow['jobs']['benchmark']
    assert (
        benchmark['outputs']['workflow_commit_sha']
        == '${{ steps.source.outputs.workflow_commit_sha }}'
    )
    assert (
        benchmark['outputs']['tensile_commit_sha']
        == '${{ steps.tensile.outputs.commit_sha }}'
    )
    publish = workflow['jobs']['publish']
    assert (
        publish['env']['WORKFLOW_COMMIT_SHA']
        == '${{ needs.benchmark.outputs.workflow_commit_sha }}'
    )
    assert (
        publish['env']['TENSILE_COMMIT_SHA']
        == '${{ needs.benchmark.outputs.tensile_commit_sha }}'
    )
    assert (
        publish['env']['BENCHMARK_COMMIT_MESSAGE']
        == '${{ needs.benchmark.outputs.source_commit_message }}'
    )
    for field in ('base_branch', 'base_sha', 'pull_request_number', 'pull_request_url'):
        assert (
            '${{ needs.source-metadata.outputs.' + field + ' }}'
            in publish['env'].values()
        )


def test_publication_identity_uses_the_recorded_benchmark_attempt():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    assert (
        workflow['jobs']['benchmark']['outputs']['run_attempt']
        == '${{ steps.source.outputs.run_attempt }}'
    )
    benchmark = workflow['jobs']['benchmark']
    upload = next(
        step
        for step in benchmark['steps']
        if step['name'] == 'Upload benchmark diagnostics'
    )
    assert (
        benchmark['outputs']['results_artifact']
        == upload['with']['name']
        == '${{ env.RESULTS_ARTIFACT }}'
    )
    publish = workflow['jobs']['publish']
    assert (
        publish['env']['RUN_ID']
        == 'github-${{ github.run_id }}-attempt-${{ needs.benchmark.outputs.run_attempt }}'
    )
    assert (
        publish['env']['RESULTS_ARTIFACT']
        == '${{ needs.benchmark.outputs.results_artifact }}'
    )
    assert (
        publish['env']['BENCHMARK_RUN_ATTEMPT']
        == '${{ needs.benchmark.outputs.run_attempt }}'
    )
    download = next(
        step
        for step in publish['steps']
        if step['name'] == 'Download benchmark diagnostics'
    )
    assert download['with']['name'] == '${{ env.RESULTS_ARTIFACT }}'
    commit = next(
        step
        for step in publish['steps']
        if step['name'] == 'Commit and push publication'
    )
    assert (
        'Publish rocjitsu benchmark run ${GITHUB_RUN_ID}.${BENCHMARK_RUN_ATTEMPT}'
        in commit['run']
    )
    diagnostics = next(
        step
        for step in publish['steps']
        if step['name'] == 'Upload publication diagnostics'
    )
    assert (
        diagnostics['with']['name']
        == 'rocjitsu-publication-${{ github.run_id }}-${{ github.run_attempt }}'
    )


def test_dispatch_source_defaults_and_safe_input_binding():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    inputs = workflow[True]['workflow_dispatch']['inputs']
    assert list(inputs) == ['source_commit', 'pull_request_number', 'publish_results']
    assert inputs['source_commit'].get('default', '') == ''
    assert inputs['source_commit']['required'] is False
    assert inputs['publish_results']['default'] is True
    assert (
        inputs['publish_results']['description']
        == 'Publish to shared dashboard dataset. Warning: side-branch workflows use side-branch tooling.'
    )
    metadata = workflow['jobs']['source-metadata']
    assert metadata['permissions'] == {'contents': 'read', 'pull-requests': 'read'}
    step = next(s for s in metadata['steps'] if s.get('id') == 'resolve')
    assert step['env']['REQUESTED_SOURCE_COMMIT'] == '${{ inputs.source_commit }}'
    assert '${{' not in step['run']
    assert workflow['jobs']['benchmark']['needs'] == 'source-metadata'


@pytest.mark.parametrize(
    'workflow_ref', ['refs/heads/develop', 'refs/heads/feature/workflow']
)
@pytest.mark.parametrize(
    'event,publish',
    [
        ('push', False),
        ('workflow_dispatch', True),
        ('workflow_dispatch', False),
        ('pull_request', False),
    ],
)
def test_publication_gate_allows_enabled_side_workflow(workflow_ref, event, publish):
    job = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())['jobs'][
        'publish'
    ]
    condition = job['if'].removeprefix('${{').removesuffix('}}').strip()
    for key, value in {
        '!cancelled()': 'True',
        'needs.benchmark.outputs.ran': repr('true'),
        'github.ref': repr(workflow_ref),
        'github.event_name': repr(event),
        'inputs.publish_results': repr(publish),
    }.items():
        condition = condition.replace(key, value)
    condition = (
        condition.replace('== true', '== True').replace('&&', 'and').replace('||', 'or')
    )
    allowed = eval(condition, {'__builtins__': {}}, {})
    assert allowed == (
        (event == 'push' and workflow_ref == 'refs/heads/develop')
        or (event == 'workflow_dispatch' and publish)
        or event == 'pull_request'
    )
    assert job['permissions'] == {'contents': 'write'}


def test_benchmark_and_tooling_checkouts_have_separate_owners():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    benchmark = workflow['jobs']['benchmark']
    steps = {s['name']: s for s in benchmark['steps']}
    assert (
        benchmark['env']['BENCHMARK_SHA']
        == '${{ needs.source-metadata.outputs.source_sha }}'
    )
    assert (
        benchmark['env']['BENCHMARK_BRANCH']
        == '${{ needs.source-metadata.outputs.source_branch }}'
    )
    assert (
        steps['Checkout rocm-systems']['with']['ref']
        == '${{ needs.source-metadata.outputs.source_sha }}'
    )
    assert steps['Checkout benchmark workflow tooling']['with'] == {
        'path': 'workflow-source',
        'ref': '${{ github.sha }}',
        'persist-credentials': False,
        'sparse-checkout': 'emulation/rocjitsu/scripts',
    }
    assert (
        workflow['env']['WORKFLOW_SOURCE_DIR']
        == '${{ github.workspace }}/workflow-source/emulation/rocjitsu'
    )
    for name in (
        'Install benchmark dependencies',
        'Build rocjitsu',
        'Prepare Tensile wheel and native adapter',
        'Generate pinned Tensile artifacts',
        'Collect preparation diagnostics',
        'Clean owned benchmark scratch',
    ):
        assert (
            '${WORKFLOW_SOURCE_DIR}/scripts/prepare-benchmarks.py' in steps[name]['run']
        )
    runner = steps['Run performance suite']['run']
    assert '--rocjitsu-source-dir "${ROCJITSU_SOURCE_DIR}"' in runner
    publish = workflow['jobs']['publish']
    assert (
        publish['env']['BENCHMARK_BRANCH']
        == '${{ needs.source-metadata.outputs.source_branch }}'
    )
    assert (
        publish['env']['BENCHMARK_SHA']
        == '${{ needs.source-metadata.outputs.source_sha }}'
    )


def test_manual_publication_guard_and_diagnostics_remain_independent():
    workflow = yaml.safe_load((WORKFLOWS / 'rocjitsu-benchmarks.yml').read_text())
    condition = workflow['jobs']['publish']['if']
    assert "!cancelled() && needs.benchmark.outputs.ran == 'true'" in condition
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_results == true"
        in condition
    )
    for job, name in [
        ('benchmark', 'Upload benchmark diagnostics'),
        ('publish', 'Upload publication diagnostics'),
    ]:
        step = next(s for s in workflow['jobs'][job]['steps'] if s['name'] == name)
        assert step['if'] == 'always()'
        assert step['with']['retention-days'] == 14


@pytest.mark.parametrize(
    'name',
    [
        'rocjitsu-benchmarks.yml',
        'rocjitsu-website-test.yml',
        'rocjitsu-publish-website.yml',
    ],
)
def test_workflow_yaml_and_bash_syntax(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    count = 0
    for job in workflow['jobs'].values():
        for step in job['steps']:
            if 'run' not in step:
                continue
            result = subprocess.run(
                ['bash', '-n'], input=step['run'], capture_output=True, text=True
            )
            assert result.returncode == 0, f"{name}: {step['name']}: {result.stderr}"
            count += 1
    assert count > 0
