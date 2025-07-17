# ROCm Systems

Welcome to the ROCm Systems monorepo. This repository consolidates multiple ROCm systems projects into a single repository to streamline development, CI, and integration. The first set of projects focuses on requirements for building PyTorch.

# Monorepo Status and CI Health

This table provides the current status of the migration of specific ROCm systems projects as well as a pointer to their current CI health.

**Key:**
- **Completed**: Fully migrated and integrated. This monorepo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the monorepo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component              | Source of Truth | Migration Status | Azure CI Status                       | Component CI Status                   |
|------------------------|-----------------|------------------|---------------------------------------|---------------------------------------|
| `amdsmi`               | EMU             | Pending          |                                       |                                       |
| `aqlprofile`           | EMU             | Pending          |                                       |                                       |
| `clr`                  | EMU             | Pending          |                                       |                                       |
| `hip`                  | EMU             | Pending          |                                       |                                       |
| `hipother`             | EMU             | Pending          |                                       |                                       |
| `hip-tests`            | EMU             | Pending          |                                       |                                       |
| `rccl`                 | Public          | Pending          |                                       |                                       |
| `rdc`                  | EMU             | Pending          |                                       |                                       |
| `rocm-core`            | EMU             | Pending          |                                       |                                       |
| `rocminfo`             | EMU             | Pending          |                                       |                                       |
| `rocmsmilib`           | EMU             | Pending          |                                       |                                       |
| `rocprofiler`          | EMU             | Pending          |                                       |                                       |
| `rocprofiler-compute`  | Public          | Pending          |                                       |                                       |
| `rocprofiler-register` | EMU             | Pending          |                                       |                                       |
| `rocprofiler-sdk`      | EMU             | Pending          |                                       |                                       |
| `rocprofiler-systems`  | Public          | Pending          |                                       |                                       |
| `rocr-runtime`         | EMU             | Pending          |                                       |                                       |
| `rocshmem`             | Public          | Pending          |                                       |                                       |
| `roctracer`            | EMU             | Pending          |                                       |                                       |


## Tentative migration schedule

| Component              | Tentative Date |
|------------------------|----------------|
| `aqlprofile`           | 8/5            |
| `rocprofiler`          | 8/5            |
| `rocprofiler-compute`  | 8/5            |
| `rocprofiler-register` | 8/5            |
| `rocprofiler-sdk`      | 8/5            |
| `rocprofiler-systems`  | 8/5            |
| `roctracer`            | 8/5            |
| `amdsmi`               | 8/8            |
| `rdc`                  | 8/8            |
| `rocmsmilib`           | 8/8            |
| `rocminfo`             | 8/11           |
| `rocr-runtime`         | 8/11           |
| `rocm-core`            | 8/12           |
| `clr`                  | 8/13           |
| `hip`                  | 8/13           |
| `hipother`             | 8/13           |
| `hip-tests`            | 8/13           |

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

This monorepo contains multiple subprojects, each of which retains the license under which it was originally published.

📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

We're happy to help!
