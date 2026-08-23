"""
Unit tests for run_pre_commit_by_subtree: file grouping, config resolution, and
the guard that refuses to fall back to the root pre-commit config.

The fall-back guard matters more than it looks: onboarded subtrees are listed in
the repo-root config's `exclude:` block, so running the root config over their
files reports "no files to check" and exits 0. A silent pass on a merge-gating
check is worse than a failure, so a missing subtree config must be fatal.
"""

import os
from unittest.mock import patch

SUBTREES = ["projects/rccl"]


class TestGroupFilesBySubtree:
    """Grouping changed paths by the onboarded subtree that owns them."""

    def test_rccl_only(self):
        from run_pre_commit_by_subtree import group_files_by_subtree

        files = ["projects/rccl/src/init.cc", "projects/rccl/.pre-commit-config.yaml"]
        assert group_files_by_subtree(files, SUBTREES) == {"projects/rccl": files}

    def test_mixed_rccl_and_unonboarded_drops_the_rest(self):
        from run_pre_commit_by_subtree import group_files_by_subtree

        files = [
            "projects/rccl/src/init.cc",
            ".github/workflows/pre-formatting.yml",
            "projects/hip/src/foo.cpp",
            "docs/readme.md",
        ]
        # Only rccl is onboarded; everything else is ignored rather than being
        # handed to the root config.
        assert group_files_by_subtree(files, SUBTREES) == {
            "projects/rccl": ["projects/rccl/src/init.cc"]
        }

    def test_nothing_onboarded_touched_yields_no_groups(self):
        from run_pre_commit_by_subtree import group_files_by_subtree

        files = [".github/workflows/pre-formatting.yml", "docs/readme.md"]
        assert group_files_by_subtree(files, SUBTREES) == {}

    def test_filename_with_spaces_is_kept_intact(self):
        from run_pre_commit_by_subtree import group_files_by_subtree

        files = ["projects/rccl/docs/some file.md"]
        assert group_files_by_subtree(files, SUBTREES) == {"projects/rccl": files}

    def test_sibling_prefix_is_not_captured(self):
        from run_pre_commit_by_subtree import group_files_by_subtree

        # projects/rccl-tests must not be swallowed by the projects/rccl entry.
        files = ["projects/rccl-tests/src/x.cc", "projects/rccl/src/init.cc"]
        assert group_files_by_subtree(files, SUBTREES) == {
            "projects/rccl": ["projects/rccl/src/init.cc"]
        }

    def test_longest_prefix_wins_for_nested_subtrees(self):
        from run_pre_commit_by_subtree import group_files_by_subtree

        subtrees = ["projects/rccl", "projects/rccl/plugin"]
        files = ["projects/rccl/plugin/a.cc", "projects/rccl/src/b.cc"]
        assert group_files_by_subtree(files, subtrees) == {
            "projects/rccl/plugin": ["projects/rccl/plugin/a.cc"],
            "projects/rccl": ["projects/rccl/src/b.cc"],
        }


class TestConfigFor:
    """Config path resolution."""

    def test_config_path_is_subtree_relative_to_repo_root(self):
        from run_pre_commit_by_subtree import config_for

        assert config_for("projects/rccl") == "projects/rccl/.pre-commit-config.yaml"


class TestRunGroup:
    """Invocation and the no-root-fallback guard."""

    def test_missing_subtree_config_fails_instead_of_falling_back(self):
        from run_pre_commit_by_subtree import run_group

        with patch("run_pre_commit_by_subtree.os.path.isfile", return_value=False):
            with patch("run_pre_commit_by_subtree.subprocess.run") as run:
                ok = run_group("projects/nope", ["projects/nope/a.cc"], dry_run=False)

        assert ok is False
        # Critically: it must not have shelled out at all. Running the root
        # config here would exit 0 having checked nothing.
        run.assert_not_called()

    def test_passes_subtree_config_and_diff_flag(self):
        from run_pre_commit_by_subtree import run_group

        files = ["projects/rccl/src/init.cc"]
        with patch("run_pre_commit_by_subtree.os.path.isfile", return_value=True):
            with patch("run_pre_commit_by_subtree.subprocess.run") as run:
                run.return_value.returncode = 0
                ok = run_group("projects/rccl", files, dry_run=False)

        assert ok is True
        cmd = run.call_args[0][0]
        assert cmd[:4] == [
            "pre-commit",
            "run",
            "-c",
            "projects/rccl/.pre-commit-config.yaml",
        ]
        assert "--show-diff-on-failure" in cmd
        assert cmd[-1] == "projects/rccl/src/init.cc"

    def test_nonzero_exit_is_reported_as_failure(self):
        from run_pre_commit_by_subtree import run_group

        with patch("run_pre_commit_by_subtree.os.path.isfile", return_value=True):
            with patch("run_pre_commit_by_subtree.subprocess.run") as run:
                run.return_value.returncode = 1
                ok = run_group("projects/rccl", ["projects/rccl/src/a.cc"], False)

        assert ok is False

    def test_dry_run_does_not_execute(self):
        from run_pre_commit_by_subtree import run_group

        with patch("run_pre_commit_by_subtree.os.path.isfile", return_value=True):
            with patch("run_pre_commit_by_subtree.subprocess.run") as run:
                ok = run_group("projects/rccl", ["projects/rccl/src/a.cc"], True)

        assert ok is True
        run.assert_not_called()


class TestReadFilesList:
    """Reading the changed-file list, including sparse-checkout absences."""

    def test_blank_lines_and_absent_paths_are_dropped(self, tmp_path):
        from run_pre_commit_by_subtree import read_files_list

        present = tmp_path / "present.cc"
        present.write_text("int main(){}\n", encoding="utf-8")
        listing = tmp_path / "files.txt"
        listing.write_text(f"{present}\n\n{tmp_path / 'absent.cc'}\n", encoding="utf-8")

        assert read_files_list(str(listing)) == [str(present)]

    def test_path_with_spaces_survives(self, tmp_path):
        from run_pre_commit_by_subtree import read_files_list

        spaced = tmp_path / "a file.md"
        spaced.write_text("hi\n", encoding="utf-8")
        listing = tmp_path / "files.txt"
        listing.write_text(f"{spaced}\n", encoding="utf-8")

        assert read_files_list(str(listing)) == [str(spaced)]


class TestMain:
    """End-to-end wiring, with pre-commit itself mocked out."""

    def test_no_onboarded_files_exits_zero_without_running(self, tmp_path):
        from run_pre_commit_by_subtree import main

        other = tmp_path / "readme.md"
        other.write_text("x\n", encoding="utf-8")
        listing = tmp_path / "files.txt"
        listing.write_text(f"{other}\n", encoding="utf-8")

        with patch("run_pre_commit_by_subtree.subprocess.run") as run:
            main(["--files-from", str(listing)])  # must not raise SystemExit
        run.assert_not_called()

    def test_failing_group_exits_one(self, tmp_path, monkeypatch):
        import pytest

        from run_pre_commit_by_subtree import main

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "projects" / "rccl" / "src"
        src.mkdir(parents=True)
        (src / "init.cc").write_text("int main(){}\n", encoding="utf-8")
        (tmp_path / "projects" / "rccl" / ".pre-commit-config.yaml").write_text(
            "repos: []\n", encoding="utf-8"
        )
        listing = tmp_path / "files.txt"
        listing.write_text("projects/rccl/src/init.cc\n", encoding="utf-8")

        with patch("run_pre_commit_by_subtree.subprocess.run") as run:
            run.return_value.returncode = 1
            with pytest.raises(SystemExit) as exc:
                main(["--files-from", str(listing)])

        assert exc.value.code == 1
