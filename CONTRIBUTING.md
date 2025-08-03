# Contributing to the ROCm Libraries

Thank you for contributing! This guide outlines the development workflow, contribution standards, and best practices when working in the monorepo.

## Getting Started

### Option A: Clone the Monorepo

```bash
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems
```

### Option B: Clone the Monorepo with Sparse-Checkout

To limit your local checkout to only the project(s) you work on and improve performance with a large codebase, you can configure sparse-checkout prior to cloning:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas shared/tensile
git checkout develop # or the branch you are starting from
```

This uses Git’s partial clone feature (`--filter=blob:none`) to reduce how much data is downloaded, and sparse-checkout to limit what is checked out to disk. For more background, including guidance on tree-less clones (`--filter=tree:0`) and shallow clones (`--depth=1`), see GitHub’s [blog post on partial and shallow clones](https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone).

With the source tree as of June 19th, 2025, the clone command lasted 4 seconds in one test run.
The checkout command of the two projects lasted less than 90 seconds.

## Working on Multiple Projects

If your work involves changing projects or introducing new projects, you can update your sparse-checkout environment:

```bash
git sparse-checkout set projects/hipsparse projects/rocsparse
```

This keeps your working directory clean and fast, as you won't need to clone the entire monorepo.

---

## Directory Structure

- `.github/`: CI workflows, scripts, and configuration files for synchronizing repositories during the migration period.
- `docs/`: Documentation, including this guide and other helpful resources.
- `projects/<name>/`: Each folder corresponds to a ROCm library that was previously maintained in its own GitHub repository and released as distinct packages.
- `shared/<name>/`: Shared components that existed in their own repository, used as dependencies by multiple libraries, but do not produce distinct packages in previous ROCm releases.

Further changes to the structure may be made to improve development efficiency and minimize redundancy.

---

## Making Changes

### From a Developer's Perspective

You can continue working inside your project's folder as you did before the monorepo migration.
This process is intended to remain as familiar as possible, though some adjustments may be made to improve efficiency based on feedback.

#### Example: hipblaslt Developer

```bash
cd projects/hipblaslt
# Edit, build, test as usual
```

---

## Keeping Your Branch in Sync

To stay up to date with the latest changes in the monorepo:

```bash
git fetch origin
git rebase origin/develop
```

Avoid using git merge to keep history clean and maintain a linear progression.

---

## New Product Introduction (NPI) and New Technology Introduction (NTI) Development

A mirror of this monorepo will be on GitHub Enterprise Managed User (EMU) and available only on the AMD intranet.
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

The monorepo uses automation to assign labels and reviewers based on the changed files. Reviewers are designated via the top-level CODEOWNERS file.

### 4. Tests and CI

Existing testing and CI infrastructure will be updated to directly point to the monorepo.
Specific checks will become mandatory for pull requests before merging. Initially, these will be limited to compilation, but will expand to correctness tests and eventually performance tests.
Hardware and operating system coverage will also expand for these checks over time.
Please refer to [this documentation](/docs/continuous-integration.md) for further details on the current signals that will be provided through CI for pull requests and commits.

---

## Gardener Rotation

In order to achieve the goal of keeping the `develop` branch healthy, a team of ROCm engineers will be dedicated towards monitoring and triaging issues that arise.
This team will collaborate to identify offending commits to isolate what changes need to be reverted.
There may be occassions where bulk reverts may need to occur for more complex issues.

See [docs/gardening.md](docs/gardening.md) for more information.

---

## Developer Communications

As this monorepo continues to evolve, weekly office hour sessions with a wide audience of ROCm engineers and managers will occur.
Focused meetings with smaller project teams will be also be scheduled regularly.
These discussions can go over any topic of the monorepo important to the different teams.
If you want to be looped into these syncs, please reach out to project leadership.

---

## Integration with TheRock

[TheRock](https://github.com/rocm/therock) is our new open-source build system for ROCm. It is designed to significantly enhance our support and scalability for ROCm 7.0 and beyond, and it is actively welcoming community contributions. TheRock currently supports a subset of AMD GPU targets, with ongoing efforts from our team and the community to expand this further, as detailed in TheRock [roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md).

As part of this mono-repo, TheRock is leveraged to extend our CI to add faster support for more testing and more targets with faster builds speeds. While some of these improvements will be seen with the existing CI, some will be exclusive with the TheRock CI targets given the changes in the high-level CMake system and specific patches that still remain within TheRock. Post ROCm 7.0, our goal is to unify our build system to one to ensure all of our CI has the benefits of the new build system.

---

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

Happy contributing!
