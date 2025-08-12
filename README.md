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
| `aqlprofile`           | Public          | Completed        |                                       | [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml) |
| `clr`                  | EMU             | Pending          |                                       |                                       |
| `hip`                  | EMU             | Pending          |                                       |                                       |
| `hipother`             | EMU             | Pending          |                                       |                                       |
| `hip-tests`            | EMU             | Pending          |                                       |                                       |
| `rccl`                 | Public          | Pending          |                                       |                                       |
| `rdc`                  | Public          | Completed        |                                       |                                       |
| `rocm-core`            | Public          | Completed        |                                       |                                       |
| `rocminfo`             | Public          | Completed        |                                       |                                       |
| `rocm-smi-lib`         | Public          | Completed        |                                       |                                       |
| `rocprofiler`          | Public          | Completed        |                                       |                                       |
| `rocprofiler-compute`  | Public          | Completed        |                                       | [![Documentation](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-docs.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-docs.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml) <br> [![mi-rhel9](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-mi-rhel9.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-mi-rhel9.yml) <br> [![packaging](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-packaging.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-packaging.yml) <br> [![ rhel-8](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel-8.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel-8.yml) <br> [![tarball](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-tarball.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-tarball.yml) <br> [![ubuntu jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml) |
| `rocprofiler-register` | Public          | Completed        | [![Azure Pipelines](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2Frocprofiler-register?repoName=ROCm%2Frocm-systems&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=327&repoName=ROCm%2Frocm-systems&branchName=develop) | [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml) |
| `rocprofiler-sdk`      | Public          | Completed        |                                       | [![Code Coverage Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml) <br> [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml) <br> [![Documentation](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml) <br> [![Restrictions](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml) <br> [![Release Compatibility](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml) |
| `rocprofiler-systems`  | Public          | Completed        |                                       | [![Containers](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml) <br> [![CPack](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-cpack.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-cpack.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml) <br> [![OpenSUSE](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-opensuse.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-opensuse.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml) <br> [![RedHat Linux](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml) <br> [![Release](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-release.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-release.yml) <br> [![Ubuntu Focal](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-focal.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-focal.yml) <br> [![Ubuntu Jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml) |
| `rocr-runtime`         | Public          | Completed        |                                       |                                       |
| `rocshmem`             | Public          | Pending          |                                       |                                       |
| `roctracer`            | Public          | Completed        |                                       |                                       |


## Tentative migration schedule

| Component              | Tentative Date |
|------------------------|----------------|
| `clr`                  | 8/21           |
| `hip`                  | 8/21           |
| `hipother`             | 8/21           |
| `hip-tests`            | 8/21           |

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

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

We're happy to help!
