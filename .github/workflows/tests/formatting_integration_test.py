# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run real formatter configurations with the matrix's sparse checkouts."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

REPOSITORY = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load(
    (REPOSITORY / ".github/workflows/scoped-formatting.yml").read_text()
)
SCOPES = WORKFLOW["jobs"]["pre-commit"]["strategy"]["matrix"]["include"]


class FormattingIntegrationTests(unittest.TestCase):
    def check_scope(self, name: str) -> None:
        scope = next(scope for scope in SCOPES if scope["scope"] == name)
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        # Temporary clones share immutable objects; neither fixtures nor hook
        # fixes can modify the source checkout.
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"

            def git(*arguments: str, input: str | None = None) -> str:
                return subprocess.check_output(
                    ["git", *arguments],
                    cwd=repository if repository.exists() else REPOSITORY,
                    input=input,
                    text=True,
                    env=environment,
                    stderr=subprocess.PIPE,
                ).strip()

            git("clone", "--shared", "--no-checkout", str(REPOSITORY), str(repository))
            git("config", "user.name", "Formatting Test")
            git("config", "user.email", "formatting@example.com")
            git("sparse-checkout", "init", "--cone")
            git("sparse-checkout", "set", *scope["sparse-checkout"].splitlines())
            git("checkout", "--detach", "HEAD")
            base = git("rev-parse", "HEAD")
            config = scope.get("config", ".pre-commit-config.yaml")
            self.assertTrue((repository / config).is_file(), config)
            source = {
                "runtimes": "runtimes",
                "rocjitsu": "emulation/rocjitsu",
                "cuid": "shared/cuid",
                "rocprofiler-compute": "projects/rocprofiler-compute",
            }[name]
            fixture = f"{source}/formatting-integration-probe.txt"
            malformed = "hello   \n"
            (repository / fixture).write_text(malformed)
            git("add", fixture)
            outside = "unrelated/formatting-integration-probe.txt"
            blob = git("hash-object", "-w", "--stdin", input=malformed)
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{outside}")
            git("commit", "-m", "formatting probes")
            git("sparse-checkout", "reapply")
            head = git("rev-parse", "HEAD")
            self.assertFalse((repository / outside).exists())
            output = Path(directory) / "scope-output"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(REPOSITORY / ".github/actions/changed-files/scope.py"),
                    "--base",
                    base,
                    "--head",
                    head,
                    "--include",
                    scope["include"],
                    "--exclude",
                    scope.get("exclude", ""),
                    "--output",
                    str(output),
                ],
                cwd=repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            selected = dict(
                line.split("=", 1) for line in output.read_text().splitlines()
            )
            self.assertEqual(selected["affected"], "true")
            arguments = [
                sys.executable,
                "-m",
                "pre_commit",
                "run",
                "--show-diff-on-failure",
                "--config",
                config,
                "--from-ref",
                selected["base"],
                "--to-ref",
                selected["head"],
            ]
            result = subprocess.run(
                arguments,
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("trailing-whitespace", result.stdout)
            self.assertEqual((repository / fixture).read_text(), "hello\n")
            self.assertEqual(git("diff", "--name-only").splitlines(), [fixture])
            self.assertEqual(git("rev-parse", f"HEAD:{outside}"), blob)
            self.assertFalse((repository / outside).exists())
            # Model the author's corrected commit, so pre-commit sees a clean
            # checkout just as it does in Actions instead of stashing fixes.
            git("add", fixture)
            git("commit", "-m", "format scoped probe")
            arguments[-1] = git("rev-parse", "HEAD")
            result = subprocess.run(
                arguments,
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtimes(self) -> None:
        self.check_scope("runtimes")

    def test_rocjitsu(self) -> None:
        self.check_scope("rocjitsu")

    def test_cuid(self) -> None:
        self.check_scope("cuid")

    def test_rocprofiler_compute(self) -> None:
        self.check_scope("rocprofiler-compute")


if __name__ == "__main__":
    unittest.main()
