<!-- Copyright Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Scoped formatting and lightweight unit tests

`.github/workflows/scoped-formatting.yml` schedules four checks on every pull
request, regardless of its target branch or changed paths:

| Scope | Check name | Pre-commit configuration |
| --- | --- | --- |
| runtimes | `pre-commit / runtimes` | `.pre-commit-config.yaml` |
| RocJITsu | `pre-commit` | `.pre-commit-config.yaml` |
| CUID | `pre-commit / cuid` | `shared/cuid/.pre-commit-config.yaml` |
| rocprofiler-compute | `pre-commit / rocprofiler-compute` | `projects/rocprofiler-compute/.pre-commit-config.yaml` |

The PR policy requires all four formatting checks. RocJITsu retains its
`pre-commit` name. The workflow title identifies the PR number, target branch,
and target commit. A change of target branch triggers fresh validation even
when the head commit is unchanged. Title and description edits skip validation
under a separate, nonrequired check name and use isolated concurrency groups,
so they cannot cancel validation or replace its results. Each matrix row
reports its own result; a failure does not cancel the other rows.

Formatting steps live directly in the workflow. Each row calls the shared
`.github/actions/changed-files` selector, also used by the unit-test workflow.
The selector inspects the complete local Git diff between the event's merge
base and head. It does not use GitHub's capped path filters or PR-files API.
It includes deleted paths and both sides of renames, preserves unusual filename
bytes, and fails if Git cannot establish an unambiguous comparison. The
formatter uses the same validated comparison commits.

The matrix's include/exclude regular expressions decide whether a row needs
formatting. An unaffected row succeeds without installing formatters or
checking out source and submodules. Changes to the shared selector or the
formatting workflow activate all four scopes. The initial sparse, blobless
checkout retrieves commit/tree history and the selector; source contents are
fetched only for the row's explicit sparse checkout when needed. Formatting
jobs have a 15-minute timeout, with a 10-minute timeout for runtimes.

Pushes to `develop` that change CUID or rocprofiler-compute paths also trigger
the workflow. Scope-selection documentation exclusions apply only to pull
requests, so documentation changes activate these rows on pushes. Each
formatter configuration's own file filters still apply.

Run a scope locally with its configuration:

```bash
python -m pip install pre-commit
pre-commit run --config .pre-commit-config.yaml --all-files --show-diff-on-failure
```

Use the configuration path listed in the table for CUID or rocprofiler-compute.

## Registering lightweight unit tests

`.github/workflows/unit-tests.yml` owns the shared test matrix. Add a row with
a unique `suite`, an `include` regular expression covering its implementation
and test inputs, a `sparse-checkout` list, optional space-separated Python
`dependencies`, and a `run` command. Multiline commands can install other
suite-specific dependencies, as the formatting suite does for Node tests.
Suites using Node declare `node-version` to provision it explicitly.
Include every directory the tests read in the sparse checkout. The runner
provides Python 3.11 and reports each suite separately as `unit-tests / <suite>`.

The workflow validates source changes and retargets on every PR, plus `develop`
pushes. Metadata-only edits use the same isolation as formatting. Each row
uses the complete Git-diff selector before provisioning test inputs or
installing dependencies.
Changes to the selector or test workflow activate every suite. Formatting
unit tests exercise the actual policy evaluator and scope patterns. Node tests
use GitHub's expression evaluator to check metadata edits, retargeting, check
names, and concurrency. Selector tests build real Git histories, including
large diffs and blobless sparse checkouts.

The separate `formatting-integration` suite uses temporary sparse clones and
each row's real pre-commit configuration. It verifies that malformed scoped
files are fixed, out-of-scope files stay untouched, and the same formatter
command passes once the scoped violation is fixed. This suite runs when
formatting infrastructure or its formatter inputs change.

Run the lightweight suites locally:

```bash
python -B -m unittest discover -s .github/actions/changed-files/tests -v
python -B -m unittest discover -s .github/workflows/tests -v
npm ci --prefix .github/workflows/tests --ignore-scripts --no-audit --no-fund
npm test --prefix .github/workflows/tests
```

With `pyyaml`, `requests`, and `pre-commit` installed, run the formatter
integration suite separately:

```bash
python -B .github/workflows/tests/formatting_integration_test.py -v
```

The integration suite needs the source Git objects for all formatting scopes
and network access to install the pinned pre-commit hook environments.
