# Maintenance Guide

A how-to for the routine, config-level changes a **project developer** makes in
`.github/scripts` and its sibling config files. These are knob-turning tasks —
adding your project to a list, changing when CI runs, toggling a sync flag — not
changes to how the scripts themselves work.

Each task below lists the file(s) to edit, what to change, and the tests to run.

> **Before you start:** install the dependencies the scripts need.
> ```bash
> pip install -r .github/requirements.txt
> ```
> **Run the tests** (from the repo root) after any change:
> ```bash
> python3 -m pytest .github/scripts/tests/
> ```
> Most CI-config changes should come with a matching test case — see
> [Updating the tests](#updating-the-tests).

---

## Table of contents

1. [Onboard a new sub-project (subtree)](#1-onboard-a-new-sub-project-subtree)
2. [Change a project's sync behavior (push/pull/source-of-truth)](#2-change-a-projects-sync-behavior)
3. [Point a project at a different upstream branch](#3-point-a-project-at-a-different-upstream-branch)
4. [Wire a project into TheRock CI](#4-wire-a-project-into-therock-ci)
5. [Define or adjust a project's build flags and tests](#5-define-or-adjust-a-projects-build-flags-and-tests)
6. [Opt a project into (or out of) Windows CI](#6-opt-a-project-into-or-out-of-windows-ci)
7. [Adjust nightly test coverage](#7-adjust-nightly-test-coverage)
8. [Skip CI for certain files or paths](#8-skip-ci-for-certain-files-or-paths)
9. [Add or update a label](#9-add-or-update-a-label)
10. [Updating the tests](#updating-the-tests)

---

## 1. Onboard a new sub-project (subtree)

**When:** You're migrating a project repo into the monorepo as a subtree under
`projects/<name>` or `shared/<name>`.

**File:** `.github/repos-config.json`

Add an object to the `repositories` array:

```json
{
  "name": "myproject",
  "url": "ROCm/myProject",
  "branch": "develop",
  "category": "projects",
  "auto_subtree_pull": false,
  "auto_subtree_push": true,
  "monorepo_source_of_truth": true
}
```

Field notes (see `repo_config_model.py` for the authoritative schema — **all fields
are required**):

| Field | Meaning |
|-------|---------|
| `name` | Lower-cased, no underscores; matches the subtree directory name (e.g. `rocblas`). |
| `url` | `org/repo` of the upstream, in its real casing (e.g. `ROCm/rocBLAS`). |
| `branch` | The upstream branch to sync against (e.g. `develop`, `amd-staging`). |
| `category` | The top-level directory: `projects` or `shared`. |
| `auto_subtree_pull` | Auto-import upstream changes into the monorepo. |
| `auto_subtree_push` | Auto-patch monorepo changes back out to the upstream ("patch back"). |
| `monorepo_source_of_truth` | The monorepo is now authoritative for this project. |

**Notes:**
- The PR category label (`project: myproject` / `shared: myproject`) is derived
  automatically from the directory path by `pr_category_label.py` — no extra step.
  Just make sure the label exists (see [task 9](#9-add-or-update-a-label)).
- This step only registers the project for **syncing**. To get it built/tested in
  CI, also do [task 4](#4-wire-a-project-into-therock-ci).

---

## 2. Change a project's sync behavior

**When:** You want to stop (or start) automatically patching a project's changes
back to its old upstream repo, or its migration status changes.

**File:** `.github/repos-config.json`

Flip the relevant boolean on the project's entry:

```diff
-      "auto_subtree_push": true,
+      "auto_subtree_push": false,
```

- `auto_subtree_push: false` → changes merged in the monorepo are **no longer**
  patched back out to the upstream repo.
- `auto_subtree_pull` and `monorepo_source_of_truth` work the same way.

These flags are read by `pr_detect_changed_subtrees.py` (via its
`--require-auto-pull` / `--require-auto-push` / `--require-monorepo-source`
filters) to decide which subtrees a sync job acts on.

---

## 3. Point a project at a different upstream branch

**When:** The upstream project changes its base branch (e.g. `amd-staging` → `develop`).

**File:** `.github/repos-config.json`

Change the `branch` field on the project's entry:

```diff
-      "branch": "amd-staging",
+      "branch": "develop",
```

This branch is what the sync scripts clone and push to (`pr_merge_sync_patches.py`)
and import from (`import_subrepo_prs.py`).

---

## 4. Wire a project into TheRock CI

**When:** You want changes under your subtree to trigger builds/tests.

**File:** `.github/scripts/therock_matrix.py`

Add an entry to `subtree_to_project_map` mapping your subtree path to a *logical
project group*:

```diff
 subtree_to_project_map = {
+    "projects/myproject": "core",
     "projects/amdsmi": "core",
     ...
 }
```

The value (`"core"`, `"profiler"`, `"runtimes"`, …) must be a key in `project_map`.
- If your project fits an existing group, reuse its name — you're done.
- If it needs its own build flags or test list, create a new group
  ([task 5](#5-define-or-adjust-a-projects-build-flags-and-tests)).

---

## 5. Define or adjust a project's build flags and tests

**When:** Your project needs build flags or a test list that no existing group provides.

**File:** `.github/scripts/therock_matrix.py`

Add (or edit) an entry in `project_map`:

```diff
 project_map = {
+    "myproject": {
+        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_MYPROJECT=ON"],
+        "projects_to_test": "myproject-tests, some-dependent-test",
+    },
     "core": { ... },
 }
```

- `cmake_options`: a list of flags passed to TheRock's build. (A bare string also
  works for backward compatibility, but prefer a list.)
- `projects_to_test`: a **comma-separated string** of test targets. Leave empty
  (`""`) to build but not run dedicated tests.
- `-DTHEROCK_ENABLE_CORE=ON` and the shared Python-executable options are appended
  automatically by `therock_configure_ci.py` — don't add them here.
- If any selected project uses `-DTHEROCK_ENABLE_ALL=ON`, that wins and overrides
  the individually merged flags.

---

## 6. Opt a project into (or out of) Windows CI

**File:** `.github/scripts/therock_matrix.py`

By default a subtree is **Linux-only**. To make it also run on Windows, add a glob
to `trigger_windows_ci_for_subtrees_paths`:

```diff
 trigger_windows_ci_for_subtrees_paths = [
     "projects/clr/*",
     "projects/hip/*",
+    "projects/myproject/**",
     ...
 ]
```

To make a subtree **Windows-only** (skip Linux CI), add it to `windows_only_subtrees`:

```diff
 windows_only_subtrees = {
     "shared/amdgpu-windows-interop",
+    "shared/my-windows-thing",
 }
```

---

## 7. Adjust nightly test coverage

**When:** Add or remove components (e.g. math libs) from the scheduled nightly run.

**File:** `.github/scripts/therock_matrix.py`

Edit the `nightly` entry in `project_map`. Nightly runs full coverage
(`-DTHEROCK_ENABLE_ALL=ON`); the knob you usually turn is `projects_to_test`:

```diff
     "nightly": {
         "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
-        "projects_to_test": "hip-tests, rocrtst, ..., rocwmma",
+        "projects_to_test": "hip-tests, rocrtst, ..., rocwmma, mynewtest",
     },
```

The nightly path is selected in `therock_configure_ci.py` when the event is a
`schedule` run (or a `workflow_dispatch` of the "TheRock CI Nightly" workflow).

---

## 8. Skip CI for certain files or paths

**When:** Docs-only, config-only, or otherwise build-irrelevant changes shouldn't
trigger a full CI run.

**File:** `.github/scripts/therock_configure_ci.py`

Add an `fnmatch`-style glob to `SKIPPABLE_PATH_PATTERNS`:

```diff
 SKIPPABLE_PATH_PATTERNS = [
     "docs/*",
     "*.md",
+    "projects/myproject/docs/*",
     ...
 ]
```

If **every** changed file in a push/PR matches a skippable pattern, CI is skipped.
A single non-matching file is enough to run CI.

---

## 9. Add or update a label

**When:** You need a new label, or want to recolor/redescribe an existing one. The
`project:`/`shared:` category labels are applied automatically, but the label must
**exist in the repo** first.

**File:** `.github/labels.yml`

Add or edit a list entry:

```yaml
- name: "project: myproject"
  color: 0e8a16
  description: Changes under projects/myproject
```

- `color` is a 6-digit hex string **without** the leading `#`.
- `description` may be an empty string or `null`.
- `apply-labels.py` reads this file and creates/updates labels on the target repo
  (it skips ones already up to date). `collect-labels.py` does the reverse —
  harvesting labels from many repos into this file — so prefer editing here rather
  than hand-editing labels in the GitHub UI.

---

## 10. Updating the tests

The CI-config scripts are covered by unit tests; project-onboarding changes
generally add a matching case.

| You changed | Add/adjust a test in |
|-------------|----------------------|
| `therock_matrix.py` or `therock_configure_ci.py` | `tests/therock_configure_ci_test.py` |
| `pr_merge_sync_patches.py` | `tests/test_pr_merge_sync_patches.py` |

The configure-CI tests mock `subprocess.run` (the `git diff` call) and assert on the
projects/flags `retrieve_projects()` returns. To cover a new mapping, add a case
that feeds a changed path under your subtree and checks the expected project group
is selected — for example:

```python
@patch("subprocess.run")
def test_myproject_changes(self, mock_run):
    args = {"is_pull_request": True, "base_ref": "HEAD^"}
    mock_process = MagicMock()
    mock_process.stdout = "projects/myproject/src/main.cpp"
    mock_run.return_value = mock_process

    project_to_run = therock_configure_ci.retrieve_projects(args)
    self.assertGreaterEqual(len(project_to_run), 1)
```

Run them with:

```bash
python3 -m pytest .github/scripts/tests/
# or the focused file:
python3 .github/scripts/tests/therock_configure_ci_test.py
```
