# Scoped formatting and lightweight unit tests

`.github/workflows/scoped-formatting.yml` schedules four checks on every pull
request, regardless of its target branch or changed paths:

| Scope | Check name | Pre-commit configuration |
| --- | --- | --- |
| runtimes | `pre-commit / runtimes` | `.pre-commit-config.yaml` |
| RocJITsu | `pre-commit` | `.pre-commit-config.yaml` |
| CUID | `pre-commit / cuid` | `projects/cuid/.pre-commit-config.yaml` |
| rocprofiler-compute | `pre-commit / rocprofiler-compute` | `projects/rocprofiler-compute/.pre-commit-config.yaml` |

RocJITsu retains the `pre-commit` name required by the base branch's PR policy.
The workflow title identifies the PR number, target branch, and target commit.
PR edits trigger a fresh evaluation, including a change of target branch with
an unchanged head commit. Each matrix row reports its own result; a failure
does not cancel the other rows.

Each row uses `.github/actions/scoped-formatting`, which calls the shared
`.github/actions/changed-files` selector. The selector inspects the complete
local Git diff between the event's merge base and head. It does not use
GitHub's capped path filters or PR-files API. It includes deleted paths and
both sides of renames, preserves unusual filename bytes, and fails if Git
cannot establish an unambiguous comparison. The formatter uses the same
validated comparison commits.

The matrix's include/exclude regular expressions decide whether a row needs
formatting. An unaffected row succeeds without installing formatters or
checking out source and submodules. Changes to either shared action or the
formatting workflow activate all four scopes. The initial sparse, blobless
checkout retrieves commit/tree history and the actions; source contents are
fetched only when needed.

Pushes to `develop` that change CUID or rocprofiler-compute paths also trigger
the workflow. Documentation exclusions apply only to pull requests. Project
documentation, including Python files under `docs/`, is checked on these pushes.

Run a scope locally with its configuration:

```bash
python -m pip install pre-commit
pre-commit run --config .pre-commit-config.yaml --all-files --show-diff-on-failure
```

Use `--config projects/<project>/.pre-commit-config.yaml` for a project configuration.

## Registering lightweight unit tests

`.github/workflows/unit-tests.yml` owns the shared test matrix. Add a row with
a unique `suite`, an `include` regular expression covering its implementation
and test inputs, a `sparse-checkout` list, optional space-separated Python
`dependencies`, and a `run` command. Include every directory the tests read in
the sparse checkout. The runner provides Python 3.11 and reports each suite
separately as `unit-tests / <suite>`.

The workflow runs on every PR and `develop` push. Each row uses the complete
Git-diff selector before provisioning test inputs or installing dependencies.
Changes to the selector or test workflow activate every suite. Formatting
regression tests check the workflow wiring and scope patterns; selector tests
build real Git histories, including large diffs and blobless sparse checkouts.
These tests are separate from the formatting jobs that run pre-commit hooks.
