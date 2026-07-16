#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local sanity checks for rocprofiler-systems GitHub workflows."""

import argparse
import base64
import gzip
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install the repository Python requirements before "
        "running this check."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "projects" / "rocprofiler-systems"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

BUILD_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-build.yml"
BUILD_GROUP_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-build-group.yml"
COVERAGE_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-code-coverage.yml"
CONTINUOUS_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-continuous-integration.yml"
SANITIZER_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-ubuntu-noble-sanitizers.yml"
RUN_CI = PROJECT_ROOT / "scripts" / "run-ci.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "summarize-junit-results.py"
MATRIX_HELPER = PROJECT_ROOT / "scripts" / "generate-ci-build-matrix.py"
MATRIX_FILE = PROJECT_ROOT / ".github" / "ci-build-matrix.json"

TARGET_WORKFLOWS = [
    BUILD_WORKFLOW,
    BUILD_GROUP_WORKFLOW,
    COVERAGE_WORKFLOW,
    CONTINUOUS_WORKFLOW,
    SANITIZER_WORKFLOW,
]


class CheckFailure(RuntimeError):
    """Raised when a workflow contract check fails."""


class GitHubActionsLoader(yaml.SafeLoader):
    """YAML loader that keeps GitHub Actions keys such as `on` as strings."""


GitHubActionsLoader.yaml_implicit_resolvers = {
    key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for first_char in "OoYyNn":
    GitHubActionsLoader.yaml_implicit_resolvers[first_char] = [
        resolver
        for resolver in GitHubActionsLoader.yaml_implicit_resolvers.get(first_char, [])
        if resolver[0] != "tag:yaml.org,2002:bool"
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local rocprofiler-systems workflow sanity checks."
    )
    parser.add_argument(
        "--strict-actionlint",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Control optional actionlint execution. auto warns when actionlint "
            "is missing, on requires it, off skips it."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra details for checks that pass.",
    )
    return parser.parse_args(argv)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_workflow(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.load(read_text(path), Loader=GitHubActionsLoader)
    except yaml.YAMLError as exc:
        raise CheckFailure(f"{path.relative_to(REPO_ROOT)} is invalid YAML: {exc}")

    require(isinstance(data, dict), f"{path.relative_to(REPO_ROOT)} is not a mapping")
    require("on" in data, f"{path.relative_to(REPO_ROOT)} is missing top-level 'on'")
    ok(f"YAML parsed: {path.relative_to(REPO_ROOT)}")
    return data


def as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{context} must be a mapping")
    return value


def as_list(value: Any, context: str) -> List[Any]:
    require(isinstance(value, list), f"{context} must be a list")
    return value


def jobs(workflow: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    return as_mapping(workflow.get("jobs"), f"{path.relative_to(REPO_ROOT)} jobs")


def job_steps(job: Mapping[str, Any], job_name: str) -> List[Mapping[str, Any]]:
    steps = as_list(job.get("steps"), f"job '{job_name}' steps")
    result: List[Mapping[str, Any]] = []
    for index, step in enumerate(steps):
        if isinstance(step, str):
            continue
        result.append(as_mapping(step, f"job '{job_name}' step {index}"))
    return result


def step_uses(step: Mapping[str, Any], action_prefix: str) -> bool:
    uses = step.get("uses")
    return isinstance(uses, str) and uses.startswith(action_prefix)


def step_with_name(
    steps: Iterable[Mapping[str, Any]], name: str
) -> Optional[Mapping[str, Any]]:
    for step in steps:
        if step.get("name") == name:
            return step
    return None


def with_mapping(step: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    return as_mapping(step.get("with"), context)


def check_actionlint(paths: Sequence[Path], strict_actionlint: str) -> None:
    if strict_actionlint == "off":
        warn("actionlint skipped by --strict-actionlint=off")
        return

    actionlint = shutil.which("actionlint")
    if not actionlint:
        message = "actionlint is not installed"
        if strict_actionlint == "on":
            raise CheckFailure(message)
        warn(f"{message}; skipping optional static workflow lint")
        return

    command = [actionlint] + [str(path.relative_to(REPO_ROOT)) for path in paths]
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    ok("actionlint passed for rocprofiler-systems workflows")


def check_junit_publication(
    build_workflow: Mapping[str, Any], build_group_workflow: Mapping[str, Any]
) -> None:
    orchestrator_jobs = jobs(build_workflow, BUILD_WORKFLOW)
    workflow_jobs = jobs(build_group_workflow, BUILD_GROUP_WORKFLOW)
    expected_top_level = (
        "matrix-setup",
        "ubuntu-2204",
        "ubuntu-2404",
        "debian",
        "rhel",
        "code-coverage",
        "sanitizers",
        "publish-test-results",
    )
    for job_name in expected_top_level:
        require(
            job_name in orchestrator_jobs,
            f"build workflow is missing top-level job '{job_name}'",
        )

    for job_name in ("ubuntu-2204", "ubuntu-2404", "debian", "rhel"):
        job = as_mapping(orchestrator_jobs[job_name], job_name)
        require(
            job.get("needs") == "matrix-setup",
            f"top-level job '{job_name}' must need matrix-setup",
        )
        require(
            job.get("uses") == "./.github/workflows/rocprofiler-systems-build-group.yml",
            f"top-level job '{job_name}' must call the build-group workflow",
        )

    require(
        as_mapping(orchestrator_jobs["sanitizers"], "sanitizers job").get("uses")
        == "./.github/workflows/rocprofiler-systems-ubuntu-noble-sanitizers.yml",
        "top-level sanitizers job must call the sanitizer reusable workflow",
    )
    require(
        as_mapping(orchestrator_jobs["code-coverage"], "code-coverage job").get("uses")
        == "./.github/workflows/rocprofiler-systems-code-coverage.yml",
        "top-level code-coverage job must call the coverage reusable workflow",
    )
    require(
        "prepare-matrix" not in workflow_jobs,
        "build-group workflow must not contain an internal matrix setup job",
    )
    require(
        "primary-build" in workflow_jobs,
        "build group workflow is missing job 'primary-build'",
    )
    require(
        "system-deps" in workflow_jobs,
        "build group workflow is missing job 'system-deps'",
    )

    for job_name in ("primary-build", "system-deps"):
        job = as_mapping(workflow_jobs[job_name], job_name)
        matrix = str(
            as_mapping(job.get("strategy"), f"job '{job_name}' strategy").get(
                "matrix", ""
            )
        )
        require(
            "inputs." in matrix,
            f"job '{job_name}' matrix must come from workflow_call inputs",
        )
        steps = job_steps(job, job_name)
        publishers = [
            step
            for step in steps
            if step_uses(step, "EnricoMi/publish-unit-test-result-action@")
        ]
        require(
            not publishers,
            f"job '{job_name}' must not publish JUnit results directly",
        )

        uploads = [step for step in steps if step_uses(step, "actions/upload-artifact@")]
        junit_uploads = [
            step
            for step in uploads
            if str(with_mapping(step, "upload artifact step").get("name", "")).startswith(
                "junit-build-"
            )
            and "test-results.xml"
            in str(with_mapping(step, "upload artifact step").get("path"))
        ]
        require(
            junit_uploads,
            f"job '{job_name}' must upload test-results.xml as a build JUnit artifact",
        )

    aggregate = as_mapping(
        orchestrator_jobs["publish-test-results"], "publish-test-results job"
    )
    needs = aggregate.get("needs")
    require(
        isinstance(needs, list)
        and set(needs)
        == {
            "ubuntu-2204",
            "ubuntu-2404",
            "debian",
            "rhel",
            "code-coverage",
            "sanitizers",
        },
        "publish-test-results must need build groups, coverage, and sanitizers",
    )
    steps = job_steps(aggregate, "publish-test-results")

    download_steps = [
        step for step in steps if step_uses(step, "actions/download-artifact@")
    ]
    download_patterns = {
        str(with_mapping(step, "download artifact step").get("pattern"))
        for step in download_steps
    }
    require(
        "junit-build-*" in download_patterns,
        "summary job must download only build JUnit artifacts with junit-build-*",
    )
    require(
        "junit-sanitizer-*" in download_patterns,
        "summary job must download sanitizer JUnit artifacts separately",
    )
    require(
        "sanitizer-summary-*" in download_patterns,
        "summary job must download sanitizer status artifacts",
    )

    publish_steps = [
        step
        for step in steps
        if step_uses(step, "EnricoMi/publish-unit-test-result-action@")
    ]
    require(
        not publish_steps,
        "publish-test-results must use the custom Markdown summary, not EnricoMi",
    )

    build_summary_step = step_with_name(steps, "Generate test result summary")
    require(
        build_summary_step is not None,
        "publish-test-results must generate a custom test result summary",
    )
    build_summary_run = str(build_summary_step.get("run", ""))
    require(
        "summarize-junit-results.py" in build_summary_run
        and "--mode build" in build_summary_run,
        "build summary must be generated by summarize-junit-results.py --mode build",
    )
    require(
        "--commit-sha" in build_summary_run and "github.sha" in build_summary_run,
        "build summary must include the commit SHA represented by the results",
    )
    require(
        "--commit-url" in build_summary_run and "github.server_url" in build_summary_run,
        "build summary must link the commit represented by the results",
    )

    sanitizer_summary_step = step_with_name(steps, "Generate sanitizer result summary")
    require(
        sanitizer_summary_step is not None,
        "publish-test-results must generate a sanitizer summary",
    )
    sanitizer_summary_run = str(sanitizer_summary_step.get("run", ""))
    require(
        "summarize-junit-results.py" in sanitizer_summary_run
        and "--mode sanitizer" in sanitizer_summary_run,
        "sanitizer summary must be generated by summarize-junit-results.py --mode sanitizer",
    )

    comment_steps = [step for step in steps if step_uses(step, "actions/github-script@")]
    require(
        len(comment_steps) == 1,
        "publish-test-results must have exactly one github-script comment updater",
    )
    comment_script = str(
        with_mapping(comment_steps[0], "github-script step").get("script", "")
    )
    require(
        "rocprofiler-systems-ci-summary" in comment_script,
        "comment updater must use the stable summary marker",
    )
    require(
        "rocprofiler-systems-build-summary:start" in comment_script
        and "rocprofiler-systems-build-summary:end" in comment_script,
        "comment updater must replace the build summary section",
    )
    require(
        "rocprofiler-systems-sanitizer-summary:start" in comment_script
        and "rocprofiler-systems-sanitizer-summary:end" in comment_script,
        "comment updater must replace the sanitizer summary section",
    )
    ok("JUnit publication uses one top-level aggregate Markdown summary")


def check_ccache_keys(build_group_workflow: Mapping[str, Any]) -> None:
    workflow_jobs = jobs(build_group_workflow, BUILD_GROUP_WORKFLOW)
    expected_tokens = {
        "primary-build": [
            "matrix.ccache_key_distro",
            "matrix.image",
            "github.job",
            "matrix.distro",
            "matrix.compiler",
            "github.sha",
        ],
        "system-deps": [
            "matrix.ccache_key_distro",
            "matrix.image",
            "github.job",
            "matrix.compiler",
            "github.sha",
        ],
    }

    for job_name, tokens in expected_tokens.items():
        job = as_mapping(workflow_jobs[job_name], job_name)
        restore_step = step_with_name(job_steps(job, job_name), "Restore ccache")
        require(restore_step is not None, f"job '{job_name}' is missing ccache restore")
        cache_with = with_mapping(restore_step, f"job '{job_name}' ccache step")
        key = str(cache_with.get("key", ""))
        restore_keys = str(cache_with.get("restore-keys", ""))
        for token in tokens:
            require(
                token in key,
                f"job '{job_name}' ccache key is missing {token}",
            )
        for token in tokens[:-1]:
            require(
                token in restore_keys,
                f"job '{job_name}' ccache restore-keys are missing {token}",
            )
    ok("ccache keys include image and expected matrix dimensions")


def check_build_matrix_file() -> None:
    data = json.loads(MATRIX_FILE.read_text(encoding="utf-8"))
    primary = as_list(data.get("primary"), "primary build matrix")
    system_deps = as_list(data.get("system_deps"), "system-deps build matrix")

    expected_counts = {
        "ubuntu-22.04": (5, 2),
        "ubuntu-24.04": (5, 2),
        "debian": (4, 0),
        "rhel": (12, 4),
    }
    for group, (primary_count, system_deps_count) in expected_counts.items():
        actual_primary = [entry for entry in primary if entry.get("group") == group]
        actual_system_deps = [
            entry for entry in system_deps if entry.get("group") == group
        ]
        require(
            len(actual_primary) == primary_count,
            f"group '{group}' must have {primary_count} primary matrix entries",
        )
        require(
            len(actual_system_deps) == system_deps_count,
            f"group '{group}' must have {system_deps_count} system-deps entries",
        )

    ok("build matrix data preserves expected group entry counts")


def check_continuous_tarball_install(workflow_text: str) -> None:
    require(
        "TARBALL_ROCM_VERSION=$(basename" in workflow_text,
        "continuous workflow must derive TARBALL_ROCM_VERSION from the tarball",
    )
    require(
        'echo "ROCM_VERSION=${TARBALL_ROCM_VERSION}" >> "${GITHUB_ENV}"' in workflow_text,
        "continuous workflow must export ROCM_VERSION from TARBALL_ROCM_VERSION",
    )
    require(
        'echo "${ROCM_PATH}/bin" >> "${GITHUB_PATH}"' in workflow_text,
        "continuous workflow must add ROCm bin to GITHUB_PATH",
    )
    require(
        'echo "${ROCM_PATH}/llvm/bin" >> "${GITHUB_PATH}"' in workflow_text,
        "continuous workflow must add ROCm llvm/bin to GITHUB_PATH",
    )
    install_block = workflow_text.split("Install Latest Nightly ROCm", 1)[-1].split(
        "Output TheRock Manifest", 1
    )[0]
    require(
        "${{ env.ROCM_VERSION }}" not in install_block,
        "ROCm tarball install block must not use stale env.ROCM_VERSION",
    )
    ok("continuous CI ROCm tarball install exports the tarball version")


def check_coverage_workflow(
    coverage_workflow: Mapping[str, Any], workflow_text: str
) -> None:
    require(
        "workflow_call" in as_mapping(coverage_workflow.get("on"), "coverage on"),
        "coverage workflow must be reusable through workflow_call",
    )
    require(
        "push:" not in workflow_text and "pull_request:" not in workflow_text,
        "coverage workflow must not run independently from the orchestrator",
    )
    workflow_jobs = jobs(coverage_workflow, COVERAGE_WORKFLOW)
    require(
        "rocprofiler-systems-code-coverage" in workflow_jobs,
        "coverage workflow must keep the code coverage job",
    )
    ok("coverage workflow is reusable through the top-level orchestrator")


def check_sanitizer_workflow(
    sanitizer_workflow: Mapping[str, Any], workflow_text: str
) -> None:
    require(
        "vname:" not in workflow_text,
        "sanitizer workflow must not contain the misspelled top-level vname key",
    )
    require(
        sanitizer_workflow.get("name") is not None,
        "sanitizer workflow must have a top-level name",
    )
    require(
        "workflow_call" in as_mapping(sanitizer_workflow.get("on"), "sanitizer on"),
        "sanitizer workflow must be reusable through workflow_call",
    )
    require(
        "push:" not in workflow_text and "pull_request:" not in workflow_text,
        "sanitizer workflow must not run independently from the orchestrator",
    )
    require(
        "python3 ./scripts/run-ci.py --stage generate" in workflow_text,
        "sanitizer workflow must generate CI scripts through run-ci.py",
    )
    require(
        "EnricoMi/publish-unit-test-result-action" not in workflow_text,
        "sanitizer workflow must use the custom summary path, not EnricoMi",
    )

    workflow_jobs = jobs(sanitizer_workflow, SANITIZER_WORKFLOW)
    require(
        "ubuntu-noble-sanitizers" in workflow_jobs,
        "sanitizer workflow is missing the matrix sanitizer job",
    )
    require(
        "publish-sanitizer-test-results" not in workflow_jobs,
        "sanitizer summary must be published by the top-level summary job",
    )

    matrix_job = as_mapping(
        workflow_jobs["ubuntu-noble-sanitizers"], "ubuntu-noble-sanitizers job"
    )
    matrix_steps = job_steps(matrix_job, "ubuntu-noble-sanitizers")

    generate_step = step_with_name(matrix_steps, "Generate CI scripts")
    require(
        generate_step is not None,
        "sanitizer matrix job must have a 'Generate CI scripts' step",
    )
    require(
        "--repeat-until-pass" not in str(generate_step.get("run", "")),
        "sanitizer generate stage must not pass --repeat-until-pass "
        "(ignored at generate; run-ci.py applies --repeat only at the test stage)",
    )

    test_step = step_with_name(matrix_steps, "Test")
    require(test_step is not None, "sanitizer matrix job must have a 'Test' step")
    test_run = str(test_step.get("run", ""))
    require(
        "--repeat-until-pass 1" in test_run and "--repeat-after-timeout 1" in test_run,
        "sanitizer test stage must pass --repeat-until-pass 1 and "
        "--repeat-after-timeout 1 so sanitizer findings are not masked by retries",
    )

    junit_upload = step_with_name(matrix_steps, "Upload JUnit test results")
    require(
        junit_upload is not None,
        "sanitizer matrix job must upload JUnit test result artifacts",
    )
    junit_with = with_mapping(junit_upload, "sanitizer JUnit upload step")
    require(
        str(junit_with.get("name", "")).startswith("junit-sanitizer-"),
        "sanitizer JUnit artifact name must start with junit-sanitizer-",
    )
    require(
        "test-results.xml" in str(junit_with.get("path", "")),
        "sanitizer JUnit upload must include test-results.xml",
    )
    require(
        step_with_name(matrix_steps, "Write sanitizer summary status") is not None,
        "sanitizer matrix job must write a status artifact payload",
    )
    require(
        step_with_name(matrix_steps, "Upload sanitizer summary status") is not None,
        "sanitizer matrix job must upload status artifacts",
    )
    ok("sanitizer workflow is reusable and reports through artifacts")


def write_fake_tool(bin_dir: Path, name: str, log_path: Path) -> None:
    tool_path = bin_dir / name
    tool_path.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\0\' "$0" "$@" >> '
        f"{str(log_path)!r}\n"
        "printf '\\n' >> "
        f"{str(log_path)!r}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    tool_path.chmod(tool_path.stat().st_mode | stat.S_IXUSR)


def fake_tool_invocations(log_path: Path) -> List[List[str]]:
    invocations: List[List[str]] = []
    if not log_path.exists():
        return invocations

    for line in log_path.read_bytes().splitlines():
        if not line:
            continue
        invocations.append([part.decode("utf-8") for part in line.split(b"\0") if part])
    return invocations


def run_ci_command(
    args: Sequence[str],
    binary_dir: Path,
    fake_bin_dir: Path,
    env: Mapping[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUN_CI)] + list(args)
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            **env,
            "PATH": f"{fake_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "NO_COLOR": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def _load_run_ci_module():
    """Import run-ci.py as a module (its filename isn't a valid identifier,
    so it can't use a normal `import`) to unit test its pure functions
    directly, without a subprocess."""
    spec = importlib.util.spec_from_file_location("run_ci", RUN_CI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gzip_base64_value(text: str) -> str:
    """Encode `text` the way CDash measurement values are encoded: gzip then
    base64, matching what run-ci.py's _decode_ctest_value expects to
    reverse."""
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def check_summarize_skipped_tests_unit(verbose: bool) -> None:
    """Unit-test the pure skip-reason resolution functions directly against
    constructed XML fixtures — no subprocess needed, unlike every other
    check in this file, since these are pure functions over ElementTree
    elements and bytes (see run-ci.py's "Skipped-test reporting" section).
    """
    run_ci = _load_run_ci_module()

    def test_el(xml: str):
        return ET.fromstring(xml)

    # Completion Status carries a real reason: used directly, no fallback.
    reason = run_ci._skip_reason(
        test_el("""<Test Status="notrun"><Name>test_a</Name><Results>
                 <NamedMeasurement name="Completion Status">
                   <Value>Julia not available</Value>
                 </NamedMeasurement>
               </Results></Test>""")
    )
    require(
        reason == "Julia not available", "expected the direct Completion Status reason"
    )

    # SKIP_RETURN_CODE placeholder + a matching captured-output pattern: the
    # regex-extracted reason wins.
    encoded = _gzip_base64_value(
        "some pytest noise\nSKIPPED [1] test_b.py:12: real skip reason\nmore noise"
    )
    reason = run_ci._skip_reason(
        test_el(f"""<Test Status="notrun"><Name>test_b</Name><Results>
                 <NamedMeasurement name="Completion Status"><Value>SKIP_RETURN_CODE: 5</Value></NamedMeasurement>
                 <Measurement><Value encoding="base64" compression="gzip">{encoded}</Value></Measurement>
               </Results></Test>""")
    )
    require(reason == "real skip reason", "expected the regex-extracted reason to win")

    # SKIP_RETURN_CODE placeholder + no matching pattern anywhere: keep the
    # SKIP_RETURN_CODE text rather than falling back to the generic default.
    encoded = _gzip_base64_value("no recognizable skip marker in this output")
    reason = run_ci._skip_reason(
        test_el(f"""<Test Status="notrun"><Name>test_c</Name><Results>
                 <NamedMeasurement name="Completion Status"><Value>SKIP_RETURN_CODE: 5</Value></NamedMeasurement>
                 <Measurement><Value encoding="base64" compression="gzip">{encoded}</Value></Measurement>
               </Results></Test>""")
    )
    require(
        reason == "SKIP_RETURN_CODE: 5",
        "expected the SKIP_RETURN_CODE placeholder to survive when nothing better is found",
    )

    # No Completion Status measurement and no matching output at all: fall
    # back to the generic default.
    reason = run_ci._skip_reason(
        test_el("<Test Status='notrun'><Name>test_d</Name></Test>")
    )
    require(reason == "(no reason captured)", "expected the generic fallback reason")

    # _collect_skip_reasons only reports "notrun" tests.
    with tempfile.TemporaryDirectory(prefix="rocprofsys-skip-unit-") as temp_dir:
        xml_path = Path(temp_dir) / "Test.xml"
        xml_path.write_text(
            """<Site><Testing>
                 <Test Status="notrun"><Name>skipped_one</Name><Results>
                   <NamedMeasurement name="Completion Status"><Value>no GPU</Value></NamedMeasurement>
                 </Results></Test>
                 <Test Status="passed"><Name>passed_one</Name></Test>
               </Testing></Site>""",
            encoding="utf-8",
        )
        reasons = run_ci._collect_skip_reasons(str(xml_path))
    require(
        reasons == {"skipped_one": "no GPU"},
        f"expected only the notrun test to be collected, got {reasons}",
    )

    if verbose:
        print(reasons)
    ok("run-ci.py skip-reason resolution functions behave correctly")


_SKIP_TEST_SENTINEL_NAME = "annotate-parallel-overhead-sys-run"
_SKIP_TEST_SENTINEL_REASON = "Requires perf_event_paranoid at most 2 or CAP_SYS_ADMIN"


def write_fake_test_xml_with_skip(bin_dir: Path, binary_dir: Path) -> None:
    """Fake ctest producing Testing/TAG + Testing/<TAG>/Test.xml with one
    skipped test (Status="notrun") carrying a Completion Status reason, and
    one passed test that must NOT show up in the skip summary.
    """
    tag_dir = binary_dir / "Testing" / "20260202-0000"
    tool_path = bin_dir / "ctest"
    xml_path = tag_dir / "Test.xml"
    xml_content = (
        "<Site><Testing>"
        f'<Test Status="notrun"><Name>{_SKIP_TEST_SENTINEL_NAME}</Name><Results>'
        '<NamedMeasurement name="Completion Status">'
        f"<Value>{_SKIP_TEST_SENTINEL_REASON}</Value>"
        "</NamedMeasurement>"
        "</Results></Test>"
        '<Test Status="passed"><Name>some-other-test</Name></Test>'
        "</Testing></Site>\n"
    )
    tool_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f"mkdir -p {str(tag_dir)!r}\n"
        f"printf '20260202-0000\\n' > {str(binary_dir / 'Testing' / 'TAG')!r}\n"
        f"cat > {str(xml_path)!r} <<'FAKE_TEST_XML_EOF'\n"
        f"{xml_content}"
        "FAKE_TEST_XML_EOF\n"
        "exit 0\n",
        encoding="utf-8",
    )
    tool_path.chmod(tool_path.stat().st_mode | stat.S_IXUSR)


def check_run_ci_skipped_tests_summary(verbose: bool) -> None:
    """A test-stage run must report skipped tests with their resolved
    reason (see run-ci.py summarize_skipped_tests()), and must not list
    tests that actually passed."""
    with tempfile.TemporaryDirectory(prefix="rocprofsys-ci-skip-check-") as temp_dir:
        temp_path = Path(temp_dir)
        binary_dir = temp_path / "build"
        fake_bin_dir = temp_path / "bin"
        fake_bin_dir.mkdir()
        fake_tool_log = temp_path / "fake-tools.log"

        for tool in ("cmake", "git", "gcov"):
            write_fake_tool(fake_bin_dir, tool, fake_tool_log)
        write_fake_test_xml_with_skip(fake_bin_dir, binary_dir)

        common_args = [
            "--name",
            "local-ci-skip-check",
            "--site",
            "Local",
            "-B",
            str(binary_dir),
        ]
        env = {"TERM": "dumb", "GITHUB_ACTIONS": "true"}
        run_ci_command(
            ["--stage", "generate", *common_args], binary_dir, fake_bin_dir, env
        )

        test = run_ci_command(
            ["--stage", "test", *common_args], binary_dir, fake_bin_dir, env
        )
        require(
            "Skipped tests (1)" in test.stdout,
            "run-ci.py must report exactly one skipped test",
        )
        # Scope the remaining assertions to the "Skipped tests" group only —
        # collect_test_artifacts separately dumps the raw Test.xml (which
        # legitimately mentions the passed test) elsewhere in stdout.
        section_start = test.stdout.index("Skipped tests (1)")
        section_end = test.stdout.find("::endgroup::", section_start)
        skip_section = test.stdout[section_start:section_end]
        require(
            _SKIP_TEST_SENTINEL_NAME in skip_section
            and _SKIP_TEST_SENTINEL_REASON in skip_section,
            "run-ci.py must print the skipped test's name and resolved reason",
        )
        require(
            "some-other-test" not in skip_section,
            "run-ci.py must not list a passed test in the skip summary",
        )

        if verbose:
            print(test.stdout)
    ok("run-ci.py reports skipped tests with their resolved reason")


def _load_summary_module():
    """Import summarize-junit-results.py as a module (its filename isn't a
    valid identifier) to unit test its pure functions directly."""
    spec = importlib.util.spec_from_file_location("summarize_junit", SUMMARY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_summarize_junit_results_unit(verbose: bool) -> None:
    summarize = _load_summary_module()

    expected_groups = {
        "junit-build-rhel-system-deps-0": "system-deps",
        "junit-build-ubuntu-22.04-system-deps-1": "system-deps",
        "junit-build-rhel-primary-build-3": "rhel",
        "junit-build-ubuntu-24.04-primary-build-0": "ubuntu-24.04",
        "junit-build-debian-primary-build-0": "debian",
    }
    for artifact, expected in expected_groups.items():
        actual = summarize.build_group_from_artifact(artifact)
        require(
            actual == expected,
            f"build_group_from_artifact({artifact!r}) returned {actual!r}, expected {expected!r}",
        )

    with tempfile.TemporaryDirectory(prefix="rocprofsys-summary-unit-") as temp_dir:
        root = Path(temp_dir) / "junit-results"

        def write_artifact(name: str, content: str) -> None:
            artifact_dir = root / name
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "test-results.xml").write_text(content, encoding="utf-8")

        write_artifact(
            "junit-build-rhel-primary-build-0",
            '<testsuite tests="5" failures="1" skipped="0" time="12.5">'
            '<testcase classname="t" name="a"><failure/></testcase></testsuite>',
        )
        # Truncated file — ctest killed mid-write (timeout/OOM), uploaded anyway.
        write_artifact(
            "junit-build-debian-primary-build-0", '<testsuites><testsuite tests="12"'
        )
        # Zero-byte file — upload raced the write.
        write_artifact("junit-build-ubuntu-22.04-primary-build-0", "")

        paths = summarize.collect_paths([str(root / "**" / "*.xml")])
        results = [
            result
            for result in (summarize.parse_junit(path, "build") for path in paths)
            if result is not None
        ]
        require(
            len(results) == 1,
            f"expected only the well-formed artifact to parse, got {len(results)}",
        )
        require(
            results[0].group_key == "rhel" and results[0].failed == 1,
            "the well-formed artifact must still parse correctly when malformed "
            "artifacts are present alongside it",
        )

        markdown = summarize.render_build_summary(results)
        require(
            "System deps" not in markdown,
            "no system-deps artifact was supplied, so the summary must not invent one",
        )

        if verbose:
            print(markdown)
    ok("summarize-junit-results.py buckets system-deps and skips malformed XML")


def check_run_ci_split_stage_contract(verbose: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="rocprofsys-ci-check-") as temp_dir:
        temp_path = Path(temp_dir)
        binary_dir = temp_path / "build"
        fake_bin_dir = temp_path / "bin"
        fake_bin_dir.mkdir()
        fake_tool_log = temp_path / "fake-tools.log"

        for tool in ("ctest", "cmake", "git", "gcov"):
            write_fake_tool(fake_bin_dir, tool, fake_tool_log)

        common_args = [
            "--name",
            "local-ci-check",
            "--site",
            "Local",
            "-B",
            str(binary_dir),
        ]
        env = {"TERM": "dumb"}
        generate = run_ci_command(
            [
                "--stage",
                "generate",
                *common_args,
                "--",
                "-DCMAKE_C_COMPILER=gcc",
                "-DCMAKE_CXX_COMPILER=g++",
                "--",
                "-LE",
                "network|gpu",
            ],
            binary_dir,
            fake_bin_dir,
            env,
        )
        test_script = binary_dir / "dashboard_test.cmake"
        require(
            test_script.exists(), "run-ci.py generate did not write dashboard_test.cmake"
        )
        test_script_text = read_text(test_script)
        require(
            'OUTPUT_JUNIT "' in test_script_text,
            "dashboard_test.cmake must set OUTPUT_JUNIT",
        )
        require(
            'EXCLUDE_LABEL "network|gpu"' in test_script_text,
            "dashboard_test.cmake must preserve -LE as EXCLUDE_LABEL",
        )
        require(
            "CTEST_REPEAT_" not in test_script_text,
            "dashboard_test.cmake must not use invalid CTEST_REPEAT_* variables",
        )

        test = run_ci_command(
            ["--stage", "test", *common_args],
            binary_dir,
            fake_bin_dir,
            env,
        )
        invocations = fake_tool_invocations(fake_tool_log)
        ctest_invocations = [
            invocation
            for invocation in invocations
            if invocation and Path(invocation[0]).name == "ctest"
        ]
        require(ctest_invocations, "run-ci.py test did not invoke ctest")
        test_invocation = ctest_invocations[-1]
        expected_repeat = ["--repeat", "until-pass:3", "after-timeout:2"]
        require(
            all(token in test_invocation for token in expected_repeat),
            "split test stage must pass --repeat until-pass:3 after-timeout:2 "
            "to the outer ctest command",
        )

        if verbose:
            print(generate.stdout)
            print(test.stdout)
            ok(f"captured ctest command: {' '.join(test_invocation)}")
    ok("run-ci.py split test stage preserves JUnit and repeat behavior")


_FAILURE_SENTINEL = "SENTINEL_BUILD_FAILURE"
_WARNING_SENTINEL = "SENTINEL_BUILD_WARNING"
_NON_XML_ESC = "[NON-XML-CHAR-0x1B]"


def write_fake_build_xml_failure(bin_dir: Path, binary_dir: Path) -> None:
    """Fake ctest reproducing a real CTEST_USE_LAUNCHERS build failure:
    NO LastBuild*.log at all, only Testing/<TAG>/Build.xml with the failing rule's
    output pre-escaped exactly the way real CTest escapes it, then exits non-zero.
    """
    tag_dir = binary_dir / "Testing" / "20260101-0000"
    tool_path = bin_dir / "ctest"
    xml_path = tag_dir / "Build.xml"
    stdout_text = (
        f"file.cpp:2: {_NON_XML_ESC}[01;31m{_NON_XML_ESC}[Kerror: "
        f"{_NON_XML_ESC}[m{_NON_XML_ESC}[K{_FAILURE_SENTINEL}"
    )
    xml_content = (
        "<Site><Build>"
        '<Failure type="Error"><Result>'
        f"<StdOut>{stdout_text}</StdOut><StdErr/>"
        "<ExitCondition>1</ExitCondition>"
        "</Result></Failure>"
        "</Build></Site>\n"
    )
    tool_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f"mkdir -p {str(tag_dir)!r}\n"
        f"printf '20260101-0000\\n' > {str(binary_dir / 'Testing' / 'TAG')!r}\n"
        f"cat > {str(xml_path)!r} <<'FAKE_BUILD_XML_EOF'\n"
        f"{xml_content}"
        "FAKE_BUILD_XML_EOF\n"
        "exit 1\n",
        encoding="utf-8",
    )
    tool_path.chmod(tool_path.stat().st_mode | stat.S_IXUSR)


def write_fake_build_success_log(bin_dir: Path, binary_dir: Path) -> None:
    """Fake ctest reproducing a successful build that still emitted a
    warning: LastBuild*.log is written (real behavior on success, unlike
    failure), with real ANSI escape bytes exactly as gcc's
    -fdiagnostics-color emits them.
    """
    temp_dir = binary_dir / "Testing" / "Temporary"
    tool_path = bin_dir / "ctest"
    log_path = temp_dir / "LastBuild_20260101-0000.log"
    tool_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f"mkdir -p {str(temp_dir)!r}\n"
        "printf 'file.cpp:1: \\x1b[01;35m\\x1b[Kwarning: \\x1b[m\\x1b[K"
        f"{_WARNING_SENTINEL}\\n' > {str(log_path)!r}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    tool_path.chmod(tool_path.stat().st_mode | stat.S_IXUSR)


def check_run_ci_failure_colored_log(verbose: bool) -> None:
    """A failed build stage must recover colored output from CTest's
    launcher XML report (Testing/<TAG>/Build.xml) since no LastBuild*.log
    exists on failure, and must annotate the correct error/warning counts
    (see run-ci.py print_failure_log() / annotate_build_diagnostics())."""
    with tempfile.TemporaryDirectory(prefix="rocprofsys-ci-fail-check-") as temp_dir:
        temp_path = Path(temp_dir)
        binary_dir = temp_path / "build"
        fake_bin_dir = temp_path / "bin"
        fake_bin_dir.mkdir()
        fake_tool_log = temp_path / "fake-tools.log"

        for tool in ("cmake", "git", "gcov"):
            write_fake_tool(fake_bin_dir, tool, fake_tool_log)
        write_fake_build_xml_failure(fake_bin_dir, binary_dir)

        common_args = [
            "--name",
            "local-ci-fail-check",
            "--site",
            "Local",
            "-B",
            str(binary_dir),
        ]
        # GITHUB_ACTIONS=true so run-ci.py's log() emits the real "::warning::"
        # annotation format instead of the local "[WARNING]" fallback.
        env = {"TERM": "dumb", "GITHUB_ACTIONS": "true"}
        run_ci_command(
            ["--stage", "generate", *common_args], binary_dir, fake_bin_dir, env
        )

        build = run_ci_command(
            ["--stage", "build", *common_args],
            binary_dir,
            fake_bin_dir,
            env,
            check=False,
        )
        require(
            build.returncode != 0,
            "fake failing ctest did not cause the run-ci.py build stage to fail",
        )
        require(
            _FAILURE_SENTINEL in build.stdout,
            "run-ci.py must recover and print the build failure text from Build.xml",
        )
        require(
            "NON-XML-CHAR" not in build.stdout,
            "run-ci.py must un-escape CTest's [NON-XML-CHAR-...] tokens before printing",
        )
        require(
            "::warning::Build: 1 error(s), 0 warning(s)" in build.stdout,
            "run-ci.py must annotate the correct error/warning counts on build failure",
        )

        if verbose:
            print(build.stdout)
    ok("run-ci.py recovers colored build-failure output and annotates counts correctly")


def check_run_ci_build_success_annotation(verbose: bool) -> None:
    """A successful build stage that still emitted a warning must annotate
    it from LastBuild*.log (which does exist on success), and must not
    print a failure-log color group since nothing failed."""
    with tempfile.TemporaryDirectory(prefix="rocprofsys-ci-warn-check-") as temp_dir:
        temp_path = Path(temp_dir)
        binary_dir = temp_path / "build"
        fake_bin_dir = temp_path / "bin"
        fake_bin_dir.mkdir()
        fake_tool_log = temp_path / "fake-tools.log"

        for tool in ("cmake", "git", "gcov"):
            write_fake_tool(fake_bin_dir, tool, fake_tool_log)
        write_fake_build_success_log(fake_bin_dir, binary_dir)

        common_args = [
            "--name",
            "local-ci-warn-check",
            "--site",
            "Local",
            "-B",
            str(binary_dir),
        ]
        env = {"TERM": "dumb", "GITHUB_ACTIONS": "true"}
        run_ci_command(
            ["--stage", "generate", *common_args], binary_dir, fake_bin_dir, env
        )

        build = run_ci_command(
            ["--stage", "build", *common_args], binary_dir, fake_bin_dir, env
        )
        require(
            "::warning::Build: 0 error(s), 1 warning(s)" in build.stdout,
            "run-ci.py must annotate warning counts from LastBuild*.log on success",
        )
        require(
            "build log:" not in build.stdout,
            "run-ci.py must not print a failure-log color group on a successful build",
        )

        if verbose:
            print(build.stdout)
    ok("run-ci.py annotates build warnings correctly on a successful build")


def run_checks(args: argparse.Namespace) -> None:
    for path in TARGET_WORKFLOWS + [RUN_CI, SUMMARY_SCRIPT, MATRIX_HELPER, MATRIX_FILE]:
        require(path.exists(), f"required file is missing: {path.relative_to(REPO_ROOT)}")

    parsed_workflows = {path: load_workflow(path) for path in TARGET_WORKFLOWS}

    check_actionlint(TARGET_WORKFLOWS, args.strict_actionlint)
    check_junit_publication(
        parsed_workflows[BUILD_WORKFLOW], parsed_workflows[BUILD_GROUP_WORKFLOW]
    )
    check_ccache_keys(parsed_workflows[BUILD_GROUP_WORKFLOW])
    check_build_matrix_file()
    check_continuous_tarball_install(read_text(CONTINUOUS_WORKFLOW))
    check_coverage_workflow(
        parsed_workflows[COVERAGE_WORKFLOW], read_text(COVERAGE_WORKFLOW)
    )
    check_sanitizer_workflow(
        parsed_workflows[SANITIZER_WORKFLOW], read_text(SANITIZER_WORKFLOW)
    )
    check_run_ci_split_stage_contract(args.verbose)
    check_run_ci_failure_colored_log(args.verbose)
    check_run_ci_build_success_annotation(args.verbose)
    check_summarize_skipped_tests_unit(args.verbose)
    check_run_ci_skipped_tests_summary(args.verbose)
    check_summarize_junit_results_unit(args.verbose)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_checks(args)
    except CheckFailure as exc:
        fail(str(exc))
        return 1
    except subprocess.CalledProcessError as exc:
        fail(f"command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        if exc.stdout:
            print(exc.stdout)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
