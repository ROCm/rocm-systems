# ROCm Systems

Welcome to the ROCm Systems super-repo. This repository consolidates multiple ROCm systems projects into a single repository to streamline development, CI, and integration. The first set of projects focuses on requirements for building PyTorch.

# Super-repo Status and CI Health

This table provides the current status of the migration of specific ROCm systems projects as well as a pointer to their current CI health.

**Key:**
- **Completed**: Fully migrated and integrated. This super-repo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the super-repo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component              | Source of Truth | Migration Status | Azure CI Status                       | Component CI Status                   |
|------------------------|-----------------|------------------|---------------------------------------|---------------------------------------|
| `amdsmi`               | EMU             | Pending          |                                       |                                       |
| `aqlprofile`           | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Faqlprofile?repoName=ROCm%2Frocm-systems&branchName=refs%2Fpull%2F368%2Fmerge)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=365&repoName=ROCm%2Frocm-systems&branchName=refs%2Fpull%2F368%2Fmerge) | [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml) |
| `clr`                  | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Fhip-clr?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=335&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `hip`                  | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Fhip-clr?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=335&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `hipother`             | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Fhip-clr?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=335&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `hip-tests`            | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Fhip-tests?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=362&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `rdc`                  | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frdc?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=360&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `rocm-core`            | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocm-core?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=349&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `rocminfo`             | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocminfo?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=356&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `rocm-smi-lib`         | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocm-smi-lib?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=358&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `rocprofiler`          | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocprofiler?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=329&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `rocprofiler-compute`  | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocprofiler-compute?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=344&repoName=ROCm%2Frocm-systems&branchName=develop) | [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml) <br> [![ rhel-8](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel-8.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel-8.yml) <br> [![tarball](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-tarball.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-tarball.yml) <br> [![ubuntu jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml) |
| `rocprofiler-register` | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocprofiler-register?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=327&repoName=ROCm%2Frocm-systems&branchName=develop) | [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml) |
| `rocprofiler-sdk`      | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocprofiler-sdk?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=347&repoName=ROCm%2Frocm-systems&branchName=develop) | [![Code Coverage Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml) <br> [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml) <br> [![Documentation](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml) <br> [![Restrictions](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml) <br> [![Release Compatibility](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml) |
| `rocprofiler-systems`  | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocprofiler-systems?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=345&repoName=ROCm%2Frocm-systems&branchName=develop) | [![Containers](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml) <br> [![rocprofiler-systems GHCR Packages for CI Images](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ghcr.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ghcr.yml) <br> [![CPack](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-cpack.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-cpack.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml) <br> [![OpenSUSE](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-opensuse.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-opensuse.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml) <br> [![RedHat Linux](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml) <br> [![Ubuntu Jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml) <br> [![Ubuntu Noble](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-noble.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-noble.yml) |
| `rocr-runtime`         | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocr-runtime?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=354&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |
| `roctracer`            | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Froctracer?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=331&repoName=ROCm%2Frocm-systems&branchName=develop) |                                       |


## Tentative migration schedule

| Component              | Tentative Date |
|------------------------|----------------|


*Remaining schedule to be determined.

# TheRock CI Status

Note TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.

[![The Rock CI](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml/badge.svg?branch%3Adevelop+event%3Apush)](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml?query=branch%3Adevelop+event%3Apush)

---

## Nomenclature

Project names have been standardized to match the casing and punctuation of released packages. This removes inconsistent camel-casing and underscores used in legacy repositories.

## Structure

The repository is organized as follows:

```
projects/
  amdsmi/
  aqlprofile/
  clr/
  hip/
  hipother/
  hip-tests/
  rccl/
  rdc/
  rocm-core
  rocminfo/
  rocmsmilib/
  rocprofiler/
  rocprofiler-compute/
  rocprofiler-register/
  rocprofiler-sdk/
  rocprofiler-systems/
  rocrruntime/
  rocshmem/
  roctracer/
```

- Each folder under `projects/` corresponds to a ROCm systems project that was previously maintained in a standalone GitHub repository and released as distinct packages.
- Each folder under `shared/` contains code that existed in its own repository and is used as a dependency by multiple projects, but does not produce its own distinct packages in previous ROCm releases.

## Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Getting Started

To begin contributing or building, see the [CONTRIBUTING.md](./CONTRIBUTING.md) guide. It includes setup instructions, sparse-checkout configuration, development workflow, and pull request guidelines.

## License

This super-repo contains multiple subprojects, each of which retains the license under which it was originally published.

📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.
📄 Refer to the header notice in individual files outside `projects/` or `shared/` folders for their specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

We're happy to help!
