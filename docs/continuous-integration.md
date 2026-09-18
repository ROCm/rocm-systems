# Continuous Integration

> [!IMPORTANT]
> This document is currently in **draft** and may be subject to change.

This document is to detail the various continuous integration (CI) systems that are run on the rocm-systems super-repo.

## Table of Contents
1. [Azure Pipelines](#azure-pipelines)
    1. [Overview](#az-overview)
    2. [PR Workflow](#az-workflow)
    3. [Interpreting Results](#az-results)
    4. [Build and Test Coverage](#az-coverage)
    5. [Downstream Job Triggers](#az-downstream)
2. [Math CI](#math-ci)
    1. [Overview](#math-overview)
3. [Windows CI](#windows-ci)
    1. [Overview](#win-overview)
4. [TheRock CI](#therock-ci)
    1. [Overview](#rock-overview)
    2. [Maintaining the pinned TheRock ref](#rock-ref)

## Azure Pipelines

### Overview <a id="az-overview"></a>

The ROCm Azure Pipelines CI (also known as External CI) is a public-facing CI system that builds and tests against latest public source code. It encompasses almost all of the ROCm stack, typically pulling source code from the `develop` or `amd-staging` branch on a component's GitHub repository. The CI's main source is publicly available at [ROCm/ROCm/.azuredevops](https://github.com/ROCm/ROCm/tree/develop/.azuredevops).

See the [Azure super-repo dashboard](https://dev.azure.com/ROCm-CI/ROCm-CI/_build?definitionScope=%5Csuper-repo) for a full list of pipelines running in the super-repo.

For commits, the pipelines will run based on the conditions defined in the trigger files under [/.azuredevops](https://github.com/ROCm/rocm-systems/tree/develop/.azuredevops).

For PRs, the [`Dispatch Azure CI`](https://github.com/ROCm/rocm-systems/blob/develop/.github/workflows/azure-ci-dispatcher.yml) GitHub Action will be run, which will analyze a PR's contents and determine which pipelines to run. This action will report the final results of each Azure run it dispatches.

### PR Workflow <a id="az-workflow"></a>

1. PR is submitted
2. `Dispatch Azure CI` is run on the PR
    1. Analyzes the PR's contents, determines which pipelines to run
    2. Sends request(s) to Azure API to start runs
3. Azure CI builds and tests the PR against latest public source
4. `Dispatch Azure CI` waits until all runs are finished and reports their overall status

URLs for individual Azure runs can be found in the logs of the `Dispatch Azure CI` action, under the `Wait for and report Azure CI` step.

### Interpreting Results <a id="az-results"></a>

Any errors or warnings during a run will be highlighted on the run's main page on Azure, and clicking on those will bring you directly to the offending logs.

Azure runs can have the following statuses: `Success`, `Failed`, or `Warning`. This corresponds to GitHub status checks as follows:

| Azure Status | GitHub PR Status | Explanation |
|-|-|-|
| ✅ Success | ✅ Succeeded | The job was successful. |
| ⚠️ Warning | ✅ Succeeded with issues | An allowed failure occurred and the job continued on without further issue. |
| ❌ Failed | ❌ Failing | The job failed. |
| Did not run | ⬛ Neutral | The job did not run, likely due to not fulfilling the trigger requirements. |

Warnings can occur if a step fails but was marked as being allowed to fail, so a job will continue running in the event of a warning.

In particular, steps are allowed to fail if they have the property `continueOnError: true` ([reference](https://learn.microsoft.com/en-us/azure/devops/pipelines/process/tasks?view=azure-devops&tabs=yaml#task-control-options)).

### Build and Test Coverage <a id="az-coverage"></a>

Azure CI builds and tests primarily on Ubuntu 22.04 LTS and for `gfx942` and `gfx90a` architectures, and adding build support for more architectures and operating systems is in progress.

Build coverage:
| | Ubuntu 22.04 | Almalinux 8 |
|-|-|-|
| **gfx942** | ✅ Supported | ✅ Supported |
| **gfx90a** | ✅ Supported | ✅ Supported |
| **gfx1201** | 🚧 In progress | 🚧 In progress |
| **gfx1100** | 🚧 In progress | 🚧 In progress |
| **gfx1030** | 🚧 In progress | 🚧 In progress |

Test coverage:
| | Ubuntu 22.04 | Almalinux 8 |
|-|-|-|
| **gfx942** | ✅ Supported | ❌ Unsupported |
| **gfx90a** | ✅ Supported | ❌ Unsupported |
| **gfx1201** | ❌ Unsupported | ❌ Unsupported |
| **gfx1100** | ❌ Unsupported | ❌ Unsupported |
| **gfx1030** | ❌ Unsupported | ❌ Unsupported |

For testing, the majority of components use `ctest` or `gtest`. Component-specific details such as build flags and test configurations can be viewed in a component's main pipeline file in [ROCm/ROCm/.azuredevops/components](https://github.com/ROCm/ROCm/tree/develop/.azuredevops/components).

### Downstream Job Triggers <a id="az-downstream"></a>

Azure CI runs for a component will trigger runs for downstream components (provided that they are fully migrated onto the super-repo). The end goal is to catch upstream breaking changes before they are merged and to ensure the super-repo is always in a valid state.

For example: a rocPRIM PR will trigger a rocPRIM job. If successful, it will then continue to run hipCUB and rocThrust jobs.

Currently, the following downstream trigger paths are enabled:

```mermaid
graph TD;
  rocPRIM-->hipCUB;
  rocPRIM-->rocThrust;
  rocRAND-->hipRAND;
  hipBLAS-common-->hipBLASLt
```

## Math CI

### Overview <a id="math-overview"></a>

## Windows CI

### Overview <a id="win-overview"></a>

## TheRock CI

### Overview <a id="rock-overview"></a>

CI workflows under `.github/workflows/therock-*.yml` build and test against
[ROCm/TheRock](https://github.com/ROCm/TheRock) pinned to a specific commit,
rather than tracking its default branch, so that `rocm-systems` CI doesn't
break every time TheRock's `main` changes.

### Maintaining the pinned TheRock ref <a id="rock-ref"></a>

The pinned commit is a single source of truth: `.github/therock_ref.json`.
Every workflow that checks out TheRock resolves the ref from that file at
runtime via the `.github/actions/therock-ref` composite action, instead of
hardcoding a SHA in the workflow itself.

The one exception is the three `.github/workflows/_therock_*.yml` internal
wrapper workflows, which call TheRock's own reusable workflows
(`setup_multi_arch.yml`, `multi_arch_ci_linux.yml`, `multi_arch_ci_windows.yml`)
via `jobs.<id>.uses:`. GitHub Actions requires that target to be a static
string literal, so it can't be resolved dynamically — these three files are
the only place in the repo where the commit SHA is hardcoded.

**To bump the pinned ref, do not hand-edit any of these files.** Instead run:

```bash
python .github/scripts/update_therock_ref.py <new_sha> [--commit-date YYYY-MM-DD]
```

This updates `.github/therock_ref.json` and re-pins all three wrapper
workflows in one step, keeping them in sync. If `--commit-date` is omitted,
it's fetched automatically from the GitHub API.

`.github/scripts/tests/therock_ref_drift_test.py` runs as part of
`therock-ci.yml` and fails CI if the wrappers' pins drift from
`therock_ref.json`, or if a hardcoded TheRock commit SHA reappears anywhere
else under `.github/workflows/`.
