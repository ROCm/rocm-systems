"""Unit tests for run_pre_commit_by_subtree: grouping, config resolution, and the
no-root-fallback guard (a missing subtree config must be fatal, not silently skipped).
"""

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
        """Three files, exact tail compare: a one-file list can't see `*files[:1]`."""
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

    def test_config_missing_is_not_a_materialisation_failure(
        self, tmp_path, monkeypatch
    ):
        # Distinct from an absent subtree: the directory IS on disk, only its
        # config is missing. That must reach run_group's own error, not this one's.
        from run_pre_commit_by_subtree import check_subtrees_materialised

        monkeypatch.chdir(tmp_path)
        (tmp_path / "projects" / "rccl").mkdir(parents=True)

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
        """detect under-reporting rccl must be fatal, not a silent exit 0."""
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

    def test_checked_out_but_unconfigured_subtree_reaches_run_groups_error(
        self, tmp_path, monkeypatch, caplog
    ):
        # The directory IS on disk, only its config is missing. Must fail via
        # run_group's own message, not be misreported as "never checked out".
        import logging

        from run_pre_commit_by_subtree import main

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "projects" / "rccl" / "src"
        src.mkdir(parents=True)
        (src / "init.cc").write_text("int main(){}\n", encoding="utf-8")
        listing = tmp_path / "files.txt"
        listing.write_text("projects/rccl/src/init.cc\n", encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            with patch("run_pre_commit_by_subtree.subprocess.run") as run:
                with pytest.raises(SystemExit) as exc:
                    main(["--files-from", str(listing)])

        assert exc.value.code == 1
        run.assert_not_called()
        assert "refusing to fall back" in caplog.text
        assert "never checked out" not in caplog.text

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
    """Places naming root-excluded projects (failure comment, FAQ) must match `exclude:`."""

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
        # /? : some root-excluded entries have no trailing slash (see "shared/amdgpu-...").
        return set(re.findall(r"(projects/[A-Za-z0-9._-]+)/?", cfg["exclude"]))

    def _projects_named_in_failure_comment(self):
        import re

        import yaml

        policy = yaml.safe_load(
            (self._repo_root() / "tools/systems_pr_bot/policy.yml").read_text()
        )
        body = policy["checks"]["failure_comments"]["pre-commit"]["body"]
        return re.findall(r"`(projects/[A-Za-z0-9._-]+)`", body)

    def _faq_callout_text(self):
        # Scoped to the "own config" blockquote, not the whole FAQ: an unrelated
        # `projects/x` mention elsewhere in the file must not affect this guard.
        lines = (
            (self._repo_root() / "docs/SYSTEMS_PR_BOT_FAQ.md").read_text().splitlines()
        )
        start = next(i for i, line in enumerate(lines) if "own config" in line)
        end = next(i for i in range(start, len(lines)) if not lines[i].startswith(">"))
        return "\n".join(lines[start:end])

    def _projects_named_in_faq(self):
        import re

        return re.findall(r"`(projects/[A-Za-z0-9._-]+)`", self._faq_callout_text())

    @staticmethod
    def _assert_no_repeats(named_list, where):
        dupes = sorted({p for p in named_list if named_list.count(p) > 1})
        assert not dupes, f"{where} names a project more than once: {dupes}"

    def test_failure_comment_names_every_root_excluded_project(self):
        excluded = self._root_excluded_projects()
        named_list = self._projects_named_in_failure_comment()
        named = set(named_list)
        assert excluded, "parsed no projects out of the root exclude block"
        assert named == excluded, (
            "tools/systems_pr_bot/policy.yml's pre-commit failure comment is out of step with "
            "the root .pre-commit-config.yaml `exclude:` block.\n"
            f"  excluded but not named: {sorted(excluded - named)}\n"
            f"  named but not excluded: {sorted(named - excluded)}"
        )
        self._assert_no_repeats(
            named_list, "tools/systems_pr_bot/policy.yml's failure comment"
        )

    def test_faq_names_every_root_excluded_project(self):
        excluded = self._root_excluded_projects()
        named_list = self._projects_named_in_faq()
        named = set(named_list)
        assert excluded, "parsed no projects out of the root exclude block"
        assert named == excluded, (
            "docs/SYSTEMS_PR_BOT_FAQ.md's own-config callout is out of step with the root "
            ".pre-commit-config.yaml `exclude:` block.\n"
            f"  excluded but not named: {sorted(excluded - named)}\n"
            f"  named but not excluded: {sorted(named - excluded)}"
        )
        self._assert_no_repeats(
            named_list, "docs/SYSTEMS_PR_BOT_FAQ.md's own-config callout"
        )

    def test_every_onboarded_subtree_is_root_excluded(self):
        from run_pre_commit_by_subtree import ONBOARDED_SUBTREES

        missing = set(ONBOARDED_SUBTREES) - self._root_excluded_projects()
        assert not missing, (
            f"{sorted(missing)} are checked by this script but not excluded from the root config, "
            "so both configs claim their files (onboarding step 4)."
        )

    def test_every_onboarded_subtree_is_in_the_workflow_paths_filter(self):
        # Onboarding step 2: without a `{subtree}/**` entry, the gate never
        # triggers on that subtree's own PRs.
        import yaml

        from run_pre_commit_by_subtree import ONBOARDED_SUBTREES

        workflow = yaml.safe_load(
            (self._repo_root() / ".github/workflows/pre-formatting.yml").read_text()
        )
        on = workflow.get(True) or workflow.get("on")
        paths = on["pull_request"]["paths"]
        missing = [s for s in ONBOARDED_SUBTREES if f"{s}/**" not in paths]
        assert not missing, (
            f"{missing} are onboarded but pre-formatting.yml's paths: has no '{{subtree}}/**' "
            "entry for them (onboarding step 2), so the gate never runs on their PRs."
        )

    def test_every_onboarded_subtree_is_in_repos_config(self):
        # Onboarding step 3: pr_detect_changed_subtrees.py can only ever emit what
        # is in repos-config.json, and that is what materialises the subtree.
        import json

        from run_pre_commit_by_subtree import ONBOARDED_SUBTREES

        repos = json.loads(
            (self._repo_root() / ".github/repos-config.json").read_text()
        )
        known = {f"{r['category']}/{r['name']}" for r in repos["repositories"]}
        missing = set(ONBOARDED_SUBTREES) - known
        assert not missing, (
            f"{sorted(missing)} are onboarded but not a category/name entry in "
            ".github/repos-config.json (onboarding step 3), so their files are never checked out."
        )


class TestRcclConfigRegexes:
    """Pins projects/rccl/.pre-commit-config.yaml's hook scoping and exclude regexes."""

    @staticmethod
    def _read_repo_file(relative_path):
        # projects/rccl/ is outside this gate's own sparse-checkout cone, so in CI this step
        # runs before the widening checkout and the working-tree branch below never fires; the
        # git object store is the actual path taken, not a rare fallback (blobs are still fetched
        # in full since neither checkout step sets a partial-clone `filter:`).
        import subprocess
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        on_disk = repo_root / relative_path
        if on_disk.is_file():
            return on_disk.read_text()
        return subprocess.run(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout

    @classmethod
    def _rccl_config(cls):
        import re

        import yaml

        cfg = yaml.safe_load(
            cls._read_repo_file("projects/rccl/.pre-commit-config.yaml")
        )
        clang_format = next(
            hook
            for repo in cfg["repos"]
            for hook in repo["hooks"]
            if hook["id"] == "clang-format"
        )
        # No external re.VERBOSE: pre-commit compiles `exclude` with no flags too, relying
        # entirely on the pattern's own inline (?x). Forcing VERBOSE here would hide a typo
        # that drops (?x) from the real config, since pre-commit would then compile it differently.
        return re.compile(clang_format["files"]), re.compile(cfg["exclude"])

    @pytest.mark.parametrize(
        "path",
        [
            "projects/rccl/src/init.cc",
            "projects/rccl/src/include/comm.h",
            "projects/rccl/src/graph/paths.cu",
            "projects/rccl/src/device/prims_simple.cuh",
        ],
    )
    def test_files_regex_matches_rccl_source(self, path):
        files_re, _ = self._rccl_config()
        assert files_re.search(path)

    @pytest.mark.parametrize(
        "path",
        [
            "projects/rccl/test/common.cc",  # not under src/
            "projects/rccl/src/tools/foo.py",  # wrong extension
            "docs/readme.md",  # wrong project entirely
        ],
    )
    def test_files_regex_excludes_non_source(self, path):
        files_re, _ = self._rccl_config()
        assert not files_re.search(path)

    @pytest.mark.parametrize(
        "path",
        [
            "projects/rccl/src/include/nvtx3/nvtx3.hpp",
            "projects/rccl/src/transport/net_ib/gdaki/doca-gpunetio/x.h",
            "projects/rccl/src/include/gdrwrap.h",
            "projects/rccl/src/include/mlx5/mlx5dvcore.h",
            "projects/rccl/src/include/ibvcore.h",
            "projects/rccl/src/nccl.h.in",
        ],
    )
    def test_vendored_paths_are_excluded(self, path):
        _, exclude_re = self._rccl_config()
        assert exclude_re.search(path)

    def test_regular_source_is_not_excluded(self):
        _, exclude_re = self._rccl_config()
        assert not exclude_re.search("projects/rccl/src/init.cc")

    def test_whitespace_hooks_stay_scoped_to_rccl(self):
        # Only clang-format's files: is pinned above. policy.yml and the FAQ both
        # document running this config with --all-files, so a hook that lost its
        # scope would then apply to the whole monorepo, not just rccl.
        import yaml

        cfg = yaml.safe_load(
            self._read_repo_file("projects/rccl/.pre-commit-config.yaml")
        )
        hooks = [
            hook
            for repo in cfg["repos"]
            for hook in repo["hooks"]
            if hook["id"] != "clang-format"
        ]
        assert hooks, "expected at least one non-clang-format hook to check"
        for hook in hooks:
            assert (
                hook.get("files") == "^projects/rccl/"
            ), f"{hook['id']} is missing or has the wrong files: scope: {hook.get('files')!r}"

    def test_exclude_block_matches_clang_format_ignores_vendored_entries(self):
        # This exclude: block re-states .clang-format-ignore's "external"/"special
        # files" entries so the whitespace hooks respect them too. Nothing else
        # keeps the two lists in step, so a vendored dir added to one and not the
        # other is silently unprotected here (or clang-format there).
        import re

        ignore_lines = self._read_repo_file(
            "projects/rccl/.clang-format-ignore"
        ).splitlines()
        end = next(
            i for i, line in enumerate(ignore_lines) if "generated codes" in line
        )
        vendored_globs = {
            line.strip()
            for line in ignore_lines[:end]
            if line.strip() and not line.strip().startswith("#")
        }
        from_ignore = {g[:-3] if g.endswith("/**") else g for g in vendored_globs}

        import yaml

        cfg = yaml.safe_load(
            self._read_repo_file("projects/rccl/.pre-commit-config.yaml")
        )
        inner = re.search(
            r"\(\?x\)\^projects/rccl/\((.*)\)", cfg["exclude"], re.DOTALL
        ).group(1)
        from_config = {
            alt.strip().rstrip("$").replace("\\.", ".").rstrip("/")
            for alt in inner.split("|")
        }

        assert from_ignore == from_config, (
            "projects/rccl/.clang-format-ignore and .pre-commit-config.yaml's exclude: "
            "block disagree on the vendored/special-file entries.\n"
            f"  in .clang-format-ignore only: {sorted(from_ignore - from_config)}\n"
            f"  in .pre-commit-config.yaml only: {sorted(from_config - from_ignore)}"
        )
