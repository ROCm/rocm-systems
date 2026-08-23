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

import pytest

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

    def test_every_file_reaches_pre_commit_after_the_files_flag(self):
        """The whole list must follow --files, not just the first entry.

        With a single-file list, `*files` and `*files[:1]` are the same
        command, so a one-file test cannot see the difference. Truncating
        there checks file #1, skips the rest, and still logs "N file(s)" and
        exits 0 -- a green gate over unchecked code, which this suite exists
        to prevent. Three or more files, and an exact tail comparison, is what
        makes that visible.
        """
        from run_pre_commit_by_subtree import run_group

        files = [
            "projects/rccl/src/init.cc",
            "projects/rccl/src/misc/argcheck.cc",
            "projects/rccl/src/include/comm.h",
        ]
        with patch("run_pre_commit_by_subtree.os.path.isfile", return_value=True):
            with patch("run_pre_commit_by_subtree.subprocess.run") as run:
                run.return_value.returncode = 0
                assert run_group("projects/rccl", files, dry_run=False) is True

        cmd = run.call_args[0][0]
        assert "--files" in cmd, "paths must be flagged, not passed as hook ids"
        # Exact tail: catches truncation, reordering, and dropped entries.
        assert cmd[cmd.index("--files") + 1 :] == files
        assert "--all-files" not in cmd, "must stay scoped to the PR's files"

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


class TestValidateSubtrees:
    """Subtree spellings that would silently own nothing."""

    def test_plain_path_is_accepted(self):
        from run_pre_commit_by_subtree import validate_subtrees

        validate_subtrees(["projects/rccl", "emulation/rocjitsu"])

    def test_the_shipped_list_is_valid(self):
        from run_pre_commit_by_subtree import ONBOARDED_SUBTREES, validate_subtrees

        validate_subtrees(ONBOARDED_SUBTREES)

    @pytest.mark.parametrize(
        "bad", ["projects/rccl/", "./projects/rccl", "/projects/rccl", " projects/rccl"]
    )
    def test_spellings_that_own_no_files_are_rejected(self, bad):
        from run_pre_commit_by_subtree import group_files_by_subtree, validate_subtrees

        # Establish the harm first: each of these groups nothing, so without
        # the guard the gate is permanently green and says nothing about it.
        assert group_files_by_subtree(["projects/rccl/src/init.cc"], [bad]) == {}
        with pytest.raises(ValueError):
            validate_subtrees([bad])


class TestCheckSubtreesMaterialised:
    """The guard against `detect` under-reporting and silently checking nothing."""

    def test_absent_subtree_is_reported(self, tmp_path, monkeypatch):
        from run_pre_commit_by_subtree import check_subtrees_materialised

        monkeypatch.chdir(tmp_path)  # nothing checked out at all
        assert check_subtrees_materialised(
            ["projects/rccl/src/init.cc"], ["projects/rccl"]
        ) == ["projects/rccl"]

    def test_checked_out_subtree_is_fine(self, tmp_path, monkeypatch):
        from run_pre_commit_by_subtree import check_subtrees_materialised

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "projects" / "rccl" / ".pre-commit-config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("repos: []\n", encoding="utf-8")

        assert (
            check_subtrees_materialised(
                ["projects/rccl/src/init.cc"], ["projects/rccl"]
            )
            == []
        )

    def test_deleting_files_is_not_mistaken_for_a_missing_checkout(
        self, tmp_path, monkeypatch
    ):
        # A PR that only deletes rccl files names paths that are correctly
        # absent. The subtree IS checked out, so this must not hard-fail.
        from run_pre_commit_by_subtree import check_subtrees_materialised

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "projects" / "rccl" / ".pre-commit-config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("repos: []\n", encoding="utf-8")

        assert (
            check_subtrees_materialised(
                ["projects/rccl/src/gone.cc"], ["projects/rccl"]
            )
            == []
        )

    def test_untouched_subtrees_are_not_required_on_disk(self, tmp_path, monkeypatch):
        from run_pre_commit_by_subtree import check_subtrees_materialised

        monkeypatch.chdir(tmp_path)
        assert check_subtrees_materialised(["docs/readme.md"], ["projects/rccl"]) == []


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

    def test_a_malformed_subtree_stops_the_run_rather_than_matching_nothing(
        self, tmp_path, monkeypatch
    ):
        from run_pre_commit_by_subtree import main

        monkeypatch.chdir(tmp_path)
        listing = tmp_path / "files.txt"
        listing.write_text("projects/rccl/src/init.cc\n", encoding="utf-8")

        with patch("run_pre_commit_by_subtree.subprocess.run") as run:
            with pytest.raises(ValueError):
                main(["--files-from", str(listing), "--subtree", "projects/rccl/"])
        run.assert_not_called()

    def test_an_unchecked_out_subtree_fails_instead_of_passing_vacuously(
        self, tmp_path, monkeypatch
    ):
        """The whole silent-green path, end to end.

        `detect` fails to name projects/rccl -- because it is missing from
        repos-config.json, or get_changed_files returned a partial page -- so
        the second checkout never widens the sparse cone. The tree diff still
        names the rccl files, they are all dropped as absent, and the old code
        logged "nothing to check" and exited 0 on a PR full of rccl changes.
        """
        from run_pre_commit_by_subtree import main

        monkeypatch.chdir(tmp_path)  # workspace holds .github and nothing else
        (tmp_path / ".github").mkdir()
        listing = tmp_path / "files.txt"
        listing.write_text(
            "projects/rccl/src/init.cc\nprojects/rccl/src/comm.h\n", encoding="utf-8"
        )

        with patch("run_pre_commit_by_subtree.subprocess.run") as run:
            with pytest.raises(SystemExit) as exc:
                main(["--files-from", str(listing)])

        assert exc.value.code == 1
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


class TestExcludedProjectListsAgree:
    """The three places that list root-excluded projects must not drift apart.

    A project in the root `exclude:` is invisible to the root config, so the bot's failure comment
    must name it or its developers get sent to a command that checks nothing and exits 0.
    """

    @staticmethod
    def _repo_root():
        from pathlib import Path

        return Path(__file__).resolve().parents[3]

    def _root_excluded_projects(self):
        import re

        import yaml

        cfg = yaml.safe_load(
            (self._repo_root() / ".pre-commit-config.yaml").read_text()
        )
        return set(re.findall(r"(projects/[A-Za-z0-9._-]+)/", cfg["exclude"]))

    def _projects_named_in_failure_comment(self):
        import re

        import yaml

        policy = yaml.safe_load(
            (self._repo_root() / "tools/systems_pr_bot/policy.yml").read_text()
        )
        body = policy["checks"]["failure_comments"]["pre-commit"]["body"]
        return set(re.findall(r"`(projects/[A-Za-z0-9._-]+)`", body))

    def test_failure_comment_names_every_root_excluded_project(self):
        excluded = self._root_excluded_projects()
        named = self._projects_named_in_failure_comment()
        assert excluded, "parsed no projects out of the root exclude block"
        assert named == excluded, (
            "tools/systems_pr_bot/policy.yml's pre-commit failure comment is out of step with "
            "the root .pre-commit-config.yaml `exclude:` block.\n"
            f"  excluded but not named: {sorted(excluded - named)}\n"
            f"  named but not excluded: {sorted(named - excluded)}"
        )

    def test_every_onboarded_subtree_is_root_excluded(self):
        from run_pre_commit_by_subtree import ONBOARDED_SUBTREES

        missing = set(ONBOARDED_SUBTREES) - self._root_excluded_projects()
        assert not missing, (
            f"{sorted(missing)} are checked by this script but not excluded from the root config, "
            "so both configs claim their files (onboarding step 4)."
        )
