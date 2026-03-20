# Contributing to ROCm Compute Profiler

## Getting Started

> Sourced from [ROCm Systems CONTRIBUTING.md](https://github.com/ROCm/rocm-systems/blob/develop/CONTRIBUTING.md#contributing-to-the-rocm-libraries).

ROCm Compute Profiler lives under `projects/rocprofiler-compute` in the [ROCm Systems super-repo](https://github.com/ROCm/rocm-systems). You have two options for cloning it locally.

### Option A: Full Clone

Clone the entire super-repo and navigate to the project:

```bash
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems/projects/rocprofiler-compute
```

### Option B: Sparse Checkout (Recommended for most contributors)

If you only need to work on `rocprofiler-compute`, a sparse checkout significantly reduces how much data is downloaded and checked out:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-compute
git checkout develop
cd projects/rocprofiler-compute
```

This uses Git's partial clone feature to avoid downloading blobs you don't need. For more background on partial and shallow clones, see GitHub's [blog post on the topic](https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone).

### Working Alongside Other Projects

If your changes span multiple projects in the super-repo, expand your sparse checkout:

```bash
git sparse-checkout set projects/rocprofiler-compute projects/rocprofiler-sdk
```

## Reporting Issues and Bugs

- Search [existing issues](https://github.com/ROCm/rocm-systems/issues) before filing a new one — your bug may already be tracked.
- If you don't find an existing issue, [open a new one](https://github.com/ROCm/rocm-systems/issues/new) with a clear description of the problem and steps to reproduce it.

## Submitting a Pull Request

> Sourced from [ROCm Systems CONTRIBUTING.md](https://github.com/ROCm/rocm-systems/blob/develop/CONTRIBUTING.md#pull-request-guidelines).

### Branch Naming

Name your branch using the following convention:

```
users/<github-username>/<branch-name>
```

Keep names descriptive but concise. If your PR addresses a GitHub issue, include the issue number in the branch name.

> **Note:** Branches within the super-repo have full CI/CD access. Pull requests from forks are welcome, but may have limited access to certain build and test infrastructure while we work toward full parity.

### Pushing and Opening the PR

When your changes are ready:

```bash
git push origin users/<github-username>/<branch-name>
```

Then open a pull request against the `develop` branch in the [ROCm Systems repository](https://github.com/ROCm/rocm-systems/compare). In your PR description:

- Summarize what the change does and why.
- Reference any related GitHub issues.
- Note any areas that need special attention during review.

### Review and Labeling

Labels and reviewer assignments are handled automatically based on the files you've changed. Reviewers for `projects/rocprofiler-compute` are defined in the top-level [CODEOWNERS](https://github.com/ROCm/rocm-systems/blob/develop/CODEOWNERS) file.

### CI Requirements

All pull requests must pass CI checks before merging. For `rocprofiler-compute`, these currently include compilation checks, with correctness and performance tests being added over time. See the [CI documentation](https://github.com/ROCm/rocm-systems/blob/develop/docs/continuous-integration.md) for a full breakdown of what runs on each PR.

> [!TIP]
> Run our pre-commit hooks locally before pushing to catch formatting issues early. See [Using Pre-Commit Hooks](#using-pre-commit-hooks) below for setup instructions.

## Adding Experimental Features

New features that aren't yet stable can be introduced behind the `--experimental` flag. This lets users opt in to preview functionality while keeping the default experience stable.

### How It Works

The `--experimental` flag acts as a master toggle:

- Experimental options are **hidden** from help output unless `--experimental` is passed.
- Attempting to use an experimental flag without `--experimental` raises a clear error.
- A warning is displayed when an experimental feature is active.

To see available experimental features:

```bash
rocprof-compute profile --experimental --help
```

### Adding a New Experimental Feature

Follow these three steps to add a new experimental flag.

**Step 1 — Register it in the `--experimental` help text**

In `src/argparser.py`, update the `add_general_group()` function:

```python
general_group.add_argument(
    "--experimental",
    action="store_true",
    default=False,
    help=(
        "Enable experimental feature(s):\n"
        "   Your feature name (--your-flag)\n"  # Add this line
    ),
)
```

**Step 2 — Add the argument using `ExperimentalAction`**

Add your flag to the relevant parser group (profile, analyze, etc.):

```python
# For a flag that accepts a value
profile_group.add_argument(
    "--your-flag",
    dest="your_flag",
    required=False,
    default=None,
    action=ExperimentalAction,
    experimental_enabled=experimental_enabled,
    feature_label="Your feature name",
    base_action="store",  # Required — see supported actions below
    type=str,
    nargs="*",
    metavar="",
    help="\t\t\tDescription of your feature",
)

# For a boolean toggle flag
analyze_group.add_argument(
    "--your-flag",
    dest="your_flag",
    required=False,
    default=False,
    action=ExperimentalAction,
    experimental_enabled=experimental_enabled,
    feature_label="Your feature description",
    base_action="store_const",  # Required — see supported actions below
    nargs=0,
    const=True,
    help="\t\tDescription of your feature",
)
```

The `base_action` parameter is required and must be one of:

| Value | Behavior |
|---|---|
| `store` | Store a value (standard argparse default) |
| `store_const` | Store a fixed constant; consumes no arguments |
| `store_true` | Store `True` when the flag is present |
| `store_false` | Store `False` when the flag is present |
| `append` | Append each value to a list |
| `append_const` | Append a constant to a list |
| `count` | Count occurrences (e.g. `-vvv`) |
| `extend` | Extend a list with multiple values |

**Step 3 — Verify behavior**

Confirm the flag is hidden without `--experimental` and visible with it:

```bash
# Should not appear
rocprof-compute profile --help

# Should appear with EXPERIMENTAL: prefix
rocprof-compute profile --experimental --help
```

### Promoting a Feature to Stable

When a feature is ready for general availability:

1. Remove it from the `--experimental` help text in `src/argparser.py`.
2. Replace `action=ExperimentalAction` with a standard argparse action (e.g. `action="store"`).
3. Remove the `experimental_enabled`, `feature_label`, and `base_action` parameters.
4. Update documentation and tests accordingly.

## Using Pre-Commit Hooks

Pre-commit hooks automatically check your code for formatting issues before each commit, helping you catch problems before they reach CI.

**Setup:**

```bash
python3 -m pip install pre-commit
cd rocprofiler-compute
pre-commit install
```

Once installed, every commit will run the configured checks automatically:

![A screen capture showing terminal output from a pre-commit hook](docs/data/contributing/pre-commit-hook.png)

See the [pre-commit documentation](https://pre-commit.com/#quick-start) for more details.

## Code Style and Formatting

ROCm Compute Profiler uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. All contributions to `src/` must pass Ruff checks before merging. Pre-commit hooks handle this automatically, but you can also run Ruff manually.

### Running Ruff

Install Ruff:

```bash
pip install ruff
```

Check for issues:

```bash
ruff check .
ruff format --check .
```

Auto-fix most issues:

```bash
ruff check --fix .
ruff format .
```

### Type Annotations

All new functions in `src/` must include type annotations on arguments and return values (except `self` and `cls`). When modifying an existing function, please annotate any parameters you touch.

```python
# Correct
def process_kernel_data(kernel_name: str, metrics: list[float]) -> dict[str, Any]:
    return {"kernel": kernel_name, "avg": sum(metrics) / len(metrics)}

# Will be flagged by Ruff (ANN rules)
def process_kernel_data(kernel_name, metrics):
    return {"kernel": kernel_name, "avg": sum(metrics) / len(metrics)}
```

To check specifically for missing annotations:

```bash
ruff check --select ANN .
```

### String Formatting

Use f-strings for all string interpolation. Older-style `.format()` and `%` formatting will be flagged by Ruff.

```python
# Correct
message = f"Processing {name} with {count} metrics"

# Will be flagged by Ruff (UP rules)
message = "Processing {} with {} metrics".format(name, count)
message = "Processing %s with %s metrics" % (name, count)
```

### Path Handling

Use `pathlib.Path` for all file system operations instead of `os.path`.

```python
# Correct
config_path = Path.cwd() / "config" / "settings.yaml"
if config_path.exists() and config_path.is_file():
    ...

# Will be flagged by Ruff (PTH rules)
config_path = os.path.join(os.getcwd(), "config", "settings.yaml")
if os.path.exists(config_path) and os.path.isfile(config_path):
    ...
```

### Disabling Formatting Rules Locally

If you have a specific reason to suppress formatting in a section of code:

- `# fmt: off` / `# fmt: on` — disable/re-enable formatting for a block.
- `# fmt: skip` — skip formatting for a single line.
- `# noqa: <RULE>` — suppress a specific lint rule for a line.

Use these sparingly and only when genuinely necessary.

## Documentation Changes

For instructions on building and testing changes to files under the `docs/` folder, see the [ROCm documentation contributing guide](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Metrics Management

If your PR modifies **metric configurations** — panel YAMLs under `src/rocprof_compute_soc/analysis_configs/gfx<arch>/*.yaml`, config deltas, or metric descriptions in `docs/data/metrics_description.yaml` — follow the metric management workflow:

1. Edit the relevant panel YAMLs.
2. Where appropriate, generate and apply a config delta using the [workflow script](tools/config_management/master_config_workflow_script.py).
3. Verify that hashes are updated and CI tests pass.

For full details, see the [metric config management README](./tools/config_management/README.md).