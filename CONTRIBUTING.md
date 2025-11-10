# Contributing to the ROCm Libraries

Thank you for contributing! This guide outlines the development workflow, contribution standards, and best practices when working in the super-repo.

## Getting Started

### Option A: Clone the super-repo

```bash
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems
```

### Option B: Clone the super-repo with Sparse-Checkout

To limit your local checkout to only the project(s) you work on and improve performance with a large codebase, you can configure sparse-checkout prior to cloning:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-sdk shared/rocprofiler-compute
git checkout develop # or the branch you are starting from
```

This uses Git’s partial clone feature (`--filter=blob:none`) to reduce how much data is downloaded, and sparse-checkout to limit what is checked out to disk. For more background, including guidance on tree-less clones (`--filter=tree:0`) and shallow clones (`--depth=1`), see GitHub’s [blog post on partial and shallow clones](https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone).

With the source tree as of June 19th, 2025, the clone command lasted 4 seconds in one test run.
The checkout command of the two projects lasted less than 90 seconds.

## Working on Multiple Projects

If your work involves changing projects or introducing new projects, you can update your sparse-checkout environment:

```bash
git sparse-checkout set projects/hip projects/clr projects/hip-tests
```

This keeps your working directory clean and fast, as you won't need to clone the entire super-repo.

---

## Directory Structure

- `.github/`: CI workflows, scripts, and configuration files for synchronizing repositories during the migration period.
- `docs/`: Documentation, including this guide and other helpful resources.
- `projects/<name>/`: Each folder corresponds to a ROCm library that was previously maintained in its own GitHub repository and released as distinct packages.
- `shared/<name>/`: Shared components that existed in their own repository, used as dependencies by multiple libraries, but do not produce distinct packages in previous ROCm releases.

Further changes to the structure may be made to improve development efficiency and minimize redundancy.

---

## Ignored commits for git blame

There were two major styling change commits, which impacted almost every C++ file in the repository. If you want to avoid seeing those changes in `git blame` output, you can run `git config blame.ignoreRevsFile .git-blame-ignore-revs`. This will exclude style changes from git blame.

---

## Making Changes

### From a Developer's Perspective

You can continue working inside your project's folder as you did before the super-repo migration.
This process is intended to remain as familiar as possible, though some adjustments may be made to improve efficiency based on feedback.

#### Example: hipblaslt Developer

```bash
cd projects/hipblaslt
# Edit, build, test as usual
```

---

## Keeping Your Branch in Sync

To stay up to date with the latest changes in the super-repo:

```bash
git fetch origin
git rebase origin/develop
```

Avoid using git merge to keep history clean and maintain a linear progression.

---

## New Product Introduction (NPI) and New Technology Introduction (NTI) Development

A mirror of this super-repo will be on GitHub Enterprise Managed User (EMU) and available only on the AMD intranet.
Please reach out within the AMD intranet if you need the link and permissions.

A primary development branch will be created for a new product or new technology.
This branch will remain private until it is cleared to be shared to the public, where it be pushed to the public repo and merged with `develop`.
It will have a subset of CI/CD in place, relative to the public repo.
There will be automation setup to regularly to rebase the branch in the EMU repo with latest `develop` from the public repo.

---

## Branching Model

We are transitioning to trunk-based development, with the tentative plan happening after the next major version release (7.0).
Until the switch is fully implemented, we will continue to sync changes to individual repositories following their existing development model (e.g., `develop` -> `staging` -> `mainline` -> `release`).
However, once trunk-based development is in place, feature branches will be created directly from the default branch, `develop`.
During this period, a high priority will be placed on keeping the `develop` branch healthy.

## Pull Request Guidelines

### 1. Branch Naming and Forks

When creating a branch for your work, use the following convention to make branch names informative and consistent: `users/<github-username>/<branch-name>`.

Try to keep branch names descriptive yet concise to reflect the purpose of the branch. For example, referencing the GitHub Issue number if the pull request is related.

The build and test infrastructure has some tasks where pull requests from forks have fewer privileges than pull requests from branches within this repo. Thus, branches in this repo are encouraged but you are welcome to use forks and their potential gaps. We are actively working towards achieving feature parity between pull requests from branches and pull requests from forks. Please stay tuned.

### 2. Opening the PR

Once you're ready:

```bash
git push origin branch-name-like-above
```

### 3. Auto-Labeling and Review Routing

The super-repo uses automation to assign labels and reviewers based on the changed files. Reviewers are designated via the top-level CODEOWNERS file.

### 4. Tests and CI

Existing testing and CI infrastructure will be updated to directly point to the super-repo.
Specific checks will become mandatory for pull requests before merging. Initially, these will be limited to compilation, but will expand to correctness tests and eventually performance tests.
Hardware and operating system coverage will also expand for these checks over time.
Please refer to [this documentation](/docs/continuous-integration.md) for further details on the current signals that will be provided through CI for pull requests and commits.

---

## Large File Storage

[Data Version Control](https://dvc.org) is the system for large file storage in this super-repo. It provides staging capabilities on top of what Git LFS typically provides that ROCm CI/CD workflows can make use of. Files are stored in an AWS S3 bucket that has public-read access.

Currently, `dvc` utilization is limited to the `pal` libraries in the `shared/amdgpu-windows-interop` directory.
If your development does not involve these files, you do not need to install `dvc`.

### Installing DVC

`dvc` can be installed as a python module via pip and is cross-platform. Visit the [dvc installation page](https://dvc.org/doc/install) if you want to use another method of installation. Due to our use of an AWS S3 bucket with `dvc`, the `dvc[s3]` module should be installed. The configuration to download the large files from the AWS S3 bucket is already set in this repository.

```bash
pip install dvc[s3]
```

### Retrieving large files:

```bash
git pull
dvc pull
```

### Switching to versions in other branches or commits:

```bash
git checkout feature-branch
dvc checkout
```

### Update large files

Write-access requires authentication. Please reach out to a project lead for credentials. To make updates to files maintained by `DVC`:

```bash
dvc add [path-to-large-file-modified]
dvc push
git status
git add [dvc-files-mentioned-from-status-output]
git commit -m "commit message"
git push
```

---

## Gardener Rotation

In order to achieve the goal of keeping the `develop` branch healthy, a team of ROCm engineers will be dedicated towards monitoring and triaging issues that arise.
This team will collaborate to identify offending commits to isolate what changes need to be reverted.
There may be occassions where bulk reverts may need to occur for more complex issues.

See [docs/gardening.md](docs/gardening.md) for more information.

---

## Developer Communications

As this super-repo continues to evolve, weekly office hour sessions with a wide audience of ROCm engineers and managers will occur.
Focused meetings with smaller project teams will be also be scheduled regularly.
These discussions can go over any topic of the super-repo important to the different teams.
If you want to be looped into these syncs, please reach out to project leadership.

---

## Project Versioning

Project versioning will adhere to the principles of [Semantic Versioning (semver)](https://semver.org/):

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes.
> 2. MINOR version when you add functionality in a backward compatible manner.
> 3. PATCH version when you make backward compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

### Symbol Visibility

Every software project should use the following CMake code in the top-level CMakeLists.txt after the first `project(...)` call before
any CMake build targets are created:

```cmake
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
```

This ensures that all shared object symbols have hidden visibility by default.
Any public API functions must then be explicitly exported. It is recommended for projects
to defined a project-specific preprocessor definition of the form: `<PROJECT_NAME>_API`.
Here is a [simplified sample derived from rocprofiler-sdk](projects/rocprofiler-sdk/source/include/rocprofiler-sdk/defines.h):

```cpp
// macro for attribute, useful since Windows uses __declspec
#define ROCPROFILER_SDK_ATTRIBUTE(...) __attribute__((__VA_ARGS__))

// always defined for export
#define ROCPROFILER_SDK_PUBLIC_API ROCPROFILER_SDK_ATTRIBUTE(visibility("default"))

// always defined for suppressing export
#define ROCPROFILER_SDK_HIDDEN_API ROCPROFILER_SDK_ATTRIBUTE(visibility("hidden"))

// See CMake code below for rocprofiler_sdk_EXPORTS
#if defined(rocprofiler_sdk_EXPORTS)
#    define ROCPROFILER_SDK_API ROCPROFILER_SDK_PUBLIC_API
#else
#    define ROCPROFILER_SDK_API
#endif
```

In the above code, we defined a `ROCPROFILER_SDK_PUBLIC_API` for exporting a symbol. However, we only
want these symbols to be marked as exported when _building_ the shared library -- when a user of the
API is including the project's headers and linking to the shared library, the symbol should not be marked as
exported[^1]. Thus, we define `rocprofiler_sdk_EXPORTS` when building the shared library -- with the expectation
that consumer software linking to the shared library will not define `rocprofiler_sdk_EXPORTS`.
CMake, by default, when building shared libraries, defines `<target-name>_EXPORTS`. It can be overridden via
the `DEFINE_SYMBOL` target property (which is useful overriding the output library name via the `OUTPUT_NAME` target property):

```cmake
set_target_properties(
    rocprofiler-sdk-shared-library
    PROPERTIES
        OUTPUT_NAME   rocprofiler-sdk
        DEFINE_SYMBOL rocprofiler_sdk_EXPORTS
)
```

Alternatively, one can simply define it explicitly via target compile definitions (with special emphasis on the use of `PRIVATE`)
as needed:

```cmake
target_compile_definitions(
    rocprofiler-sdk-object-library
    PRIVATE
        rocprofiler_sdk_EXPORTS=1
)
```

However, the `DEFINE_SYMBOL` approach is the preferred method since there is no chance of using `PUBLIC` or `INTERFACE` instead of `PRIVATE`.

[^1]: This is mostly a best practice on Linux, but is quite important on Windows, which requires symbols
to be marked as imported when defined in a external shared library.

#### Project Versioning File Structure

- `projects/<project-name>/VERSION`
  - This file contains the major, minor, and patch version of the project on the first line.
  - The VERSION in this file should be the sole source of truth regarding the version. Anything needing the version number should read from this file.
  - It also contains, on the second line, `# hash: <md5sum>` which is a hash of the files at the time of the last version bump. More on this later.
  - This file should be installed to `<prefix>/share/<project-name>/VERSION`
- `projects/<project-name>/versioning.yml`
  - Every `VERSION` file should be accompanied by a `versioning.yml` file.
  - This file contains the information needed for performing versioning checks and updating the version
  - In general it has three sections per project: `source-tree`, `build-tree`, and `install-tree`.
    - Each one of these sections defines the source files for generating the md5sum hash (`sources` and `headers`), the public API headers (`headers`), and which binary files should be checked for ABI and API changes (`abi-check`).
  - This file should be installed to `<prefix>/share/<project-name>/VERSION`

#### Useful Utilities

- `cmake/rocm_versioning.cmake`
  - This is a CMake module with various functions to assist with compliance to our versioning standards.
- `scripts/abi-guard.py`
  - This is a Python3 command-line tool for generating/updating the project's VERSION file, generating a versioning YAML spec, listing the files included or excluded from API/ABI checks, executing API/ABI checks, etc.
  A quick start guide is provided in [scripts/abi-guard-README.md](scripts/abi-guard-README.md).

#### Versioning Workflows

A reusable workflow is provided in `.github/workflows/abi-guard.yml`.
This workflow should be used to quickly integrate checks for versioning compliance.
In general, this workflow automates using [libabigail](https://man7.org/linux/man-pages/man7/libabigail.7.html) to detect changes in the API/ABI of a project.

- If this workflow detects added/removed functions/variables in the public API, it fails without at least a minor version bump.
- If this workflow detects changed functions in the public API, it strongly recommends an alternative approach since this breaks the ABI and/or ABI and, in general, a minor version bumps for a breaking change is not [Semantic Versioning (semver)](https://semver.org/) compliant.
- If this workflow detects no API/ABI breaks but the version remains unchanged, it recommends a patch version bump.
  - Note: we probably want to be able to avoid this failing for documentation changes.

Usage of this workflow is added on a per-project basis at present. Ideally, this workflow will be automatically applied in the future.
[Sample using the abi-guard.yml workflow on rocprofiler-sdk project](.github/workflows/rocprofiler-sdk-abi-guard.yml).

---

## Integration with TheRock

[TheRock](https://github.com/rocm/therock) is our new open-source build system for ROCm. It is designed to significantly enhance our support and scalability for ROCm 7.0 and beyond, and it is actively welcoming community contributions. TheRock currently supports a subset of AMD GPU targets, with ongoing efforts from our team and the community to expand this further, as detailed in TheRock [roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md).

As part of this mono-repo, TheRock is leveraged to extend our CI to add faster support for more testing and more targets with faster builds speeds. While some of these improvements will be seen with the existing CI, some will be exclusive with the TheRock CI targets given the changes in the high-level CMake system and specific patches that still remain within TheRock. Post ROCm 7.0, our goal is to unify our build system to one to ensure all of our CI has the benefits of the new build system.

---

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

Happy contributing!
