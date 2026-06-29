#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local sanity checks for rocprofiler-systems GitHub workflows."""

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
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
CONTINUOUS_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-continuous-integration.yml"
SANITIZER_WORKFLOW = WORKFLOW_DIR / "rocprofiler-systems-ubuntu-noble-sanitizers.yml"
RUN_CI = PROJECT_ROOT / "scripts" / "run-ci.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "summarize-junit-results.py"

TARGET_WORKFLOWS = [
    BUILD_WORKFLOW,
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


def step_run_contains(step: Mapping[str, Any], needle: str) -> bool:
    run = step.get("run")
    return isinstance(run, str) and needle in run


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


def check_junit_publication(build_workflow: Mapping[str, Any]) -> None:
    workflow_jobs = jobs(build_workflow, BUILD_WORKFLOW)
    require("build" in workflow_jobs, "build workflow is missing job 'build'")
    require(
        "system-deps" in workflow_jobs,
        "build workflow is missing job 'system-deps'",
    )
    require(
        "publish-test-results" in workflow_jobs,
        "build workflow is missing aggregate job 'publish-test-results'",
    )

    for job_name in ("build", "system-deps"):
        steps = job_steps(as_mapping(workflow_jobs[job_name], job_name), job_name)
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
            if "junit-" in str(with_mapping(step, "upload artifact step").get("name"))
            and "test-results.xml"
            in str(with_mapping(step, "upload artifact step").get("path"))
        ]
        require(
            junit_uploads,
            f"job '{job_name}' must upload test-results.xml as a JUnit artifact",
        )

    aggregate = as_mapping(
        workflow_jobs["publish-test-results"], "publish-test-results job"
    )
    needs = aggregate.get("needs")
    require(
        isinstance(needs, list) and set(needs) == {"build", "system-deps"},
        "publish-test-results must need exactly build and system-deps",
    )
    steps = job_steps(aggregate, "publish-test-results")

    download_steps = [
        step for step in steps if step_uses(step, "actions/download-artifact@")
    ]
    require(download_steps, "publish-test-results must download JUnit artifacts")
    download_with = with_mapping(download_steps[0], "download artifact step")
    require(
        download_with.get("pattern") == "junit-*",
        "download-artifact pattern must be junit-*",
    )
    require(
        download_with.get("path") == "junit-results",
        "download-artifact path must be junit-results",
    )
    require(
        download_with.get("merge-multiple") is False,
        "download-artifact must keep merge-multiple: false",
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

    summary_step = step_with_name(steps, "Generate test result summary")
    require(
        summary_step is not None,
        "publish-test-results must generate a custom test result summary",
    )
    summary_run = str(summary_step.get("run", ""))
    require(
        "summarize-junit-results.py" in summary_run and "--mode build" in summary_run,
        "build summary must be generated by summarize-junit-results.py --mode build",
    )
    require(
        "--commit-sha" in summary_run and "github.sha" in summary_run,
        "build summary must include the commit SHA represented by the results",
    )
    require(
        "--commit-url" in summary_run and "github.server_url" in summary_run,
        "build summary must link the commit represented by the results",
    )
    require(
        "cat junit-summary.md" in summary_run and "GITHUB_STEP_SUMMARY" in summary_run,
        "build summary must be written to GITHUB_STEP_SUMMARY",
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
        "build comment updater must use the stable summary marker",
    )
    require(
        "rocprofiler-systems-build-summary:start" in comment_script
        and "rocprofiler-systems-build-summary:end" in comment_script,
        "build comment updater must replace the build summary section",
    )
    require(
        "rocprofiler-systems-sanitizer-summary:start" in comment_script,
        "build comment updater must preserve/create the sanitizer summary section",
    )
    ok("JUnit publication uses the custom aggregate Markdown summary")


def check_ccache_keys(build_workflow: Mapping[str, Any]) -> None:
    workflow_jobs = jobs(build_workflow, BUILD_WORKFLOW)
    expected_tokens = {
        "build": [
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
        "python3 ./scripts/run-ci.py --stage generate" in workflow_text,
        "sanitizer workflow must generate CI scripts through run-ci.py",
    )
    require(
        "--repeat-until-pass 1" in workflow_text,
        "sanitizer workflow must keep its explicit repeat-until-pass override",
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
        "publish-sanitizer-test-results" in workflow_jobs,
        "sanitizer workflow is missing aggregate sanitizer summary job",
    )

    matrix_job = as_mapping(
        workflow_jobs["ubuntu-noble-sanitizers"], "ubuntu-noble-sanitizers job"
    )
    matrix_steps = job_steps(matrix_job, "ubuntu-noble-sanitizers")
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

    aggregate = as_mapping(
        workflow_jobs["publish-sanitizer-test-results"],
        "publish-sanitizer-test-results job",
    )
    require(
        aggregate.get("needs") == ["ubuntu-noble-sanitizers"],
        "sanitizer summary job must need ubuntu-noble-sanitizers",
    )
    aggregate_steps = job_steps(aggregate, "publish-sanitizer-test-results")
    summary_step = step_with_name(aggregate_steps, "Generate sanitizer result summary")
    require(
        summary_step is not None,
        "sanitizer summary job must generate a custom sanitizer summary",
    )
    summary_run = str(summary_step.get("run", ""))
    require(
        "summarize-junit-results.py" in summary_run and "--mode sanitizer" in summary_run,
        "sanitizer summary must be generated by summarize-junit-results.py --mode sanitizer",
    )
    comment_steps = [
        step for step in aggregate_steps if step_uses(step, "actions/github-script@")
    ]
    require(
        len(comment_steps) == 1,
        "sanitizer summary job must have exactly one github-script comment updater",
    )
    comment_script = str(
        with_mapping(comment_steps[0], "sanitizer github-script step").get("script", "")
    )
    require(
        "rocprofiler-systems-ci-summary" in comment_script,
        "sanitizer comment updater must use the stable summary marker",
    )
    require(
        "rocprofiler-systems-sanitizer-summary:start" in comment_script
        and "rocprofiler-systems-sanitizer-summary:end" in comment_script,
        "sanitizer comment updater must replace the sanitizer summary section",
    )
    ok("sanitizer workflow keeps expected run-ci and custom summary contracts")


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
    args: Sequence[str], binary_dir: Path, fake_bin_dir: Path, env: Mapping[str, str]
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
        check=True,
    )


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


def run_checks(args: argparse.Namespace) -> None:
    for path in TARGET_WORKFLOWS + [RUN_CI, SUMMARY_SCRIPT]:
        require(path.exists(), f"required file is missing: {path.relative_to(REPO_ROOT)}")

    parsed_workflows = {path: load_workflow(path) for path in TARGET_WORKFLOWS}

    check_actionlint(TARGET_WORKFLOWS, args.strict_actionlint)
    check_junit_publication(parsed_workflows[BUILD_WORKFLOW])
    check_ccache_keys(parsed_workflows[BUILD_WORKFLOW])
    check_continuous_tarball_install(read_text(CONTINUOUS_WORKFLOW))
    check_sanitizer_workflow(
        parsed_workflows[SANITIZER_WORKFLOW], read_text(SANITIZER_WORKFLOW)
    )
    check_run_ci_split_stage_contract(args.verbose)


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
