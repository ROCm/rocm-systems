
# Migrating ROCm Systems projects to a monorepo

## Overview

This document describes the process for migrating multiple ROCm systems software repositories into a unified monorepo using `git subtree`. This migration aims to improve development efficiency, streamline build and CI processes, and eventually consolidate release and packaging workflows.

The monorepo will adopt a trunk-based development model, enabling fast iteration and collaboration. Legacy repositories will retain their waterfall-style development, and synchronization mechanisms will ensure continuity.

## Goals and Benefits

- Simplify cross-project collaboration
- Consolidate build and CI systems
- Enable shared dependency management
- Provide a single source of truth for active development
- Reduce integration friction between teams

## Projects in Scope

The migration is employing a divide and conquer strategy. The division within this repo includes below:

- amdsmi
- aqlprofile
- rdc
- rocm_smi_lib
- rocprofiler
- rocprofiler-compute
- rocprofiler-register
- rocprofiler-sdk
- rocprofiler-systems
- ROCR-Runtime
- roctracer

## Migration Strategy

The projects in scope will be imported into the monorepo under a dedicated subdirectory using `git subtree`. This preserves commit history and allows for future bidirectional sync with the original repositories. GitHub Actions have been created to run the git subtree commands.

## Development Model Changes

### Legacy Repositories

Continue using the current branching models.

Automatic updates from the monorepo to these repositories is disabled in the beginning.
Updates must be done manually through `git subtree push` commands.

### Monorepo

Use trunk-based development:
- Short-lived feature branches
- Regular integration into main
- Rely on feature flags or conditional logic for partial implementations

## Build System Consolidation

### Short-Term

- Each subproject retains its own CMakeLists.txt and build configuration.
- Wrapper scripts at the root level allow building individual or all projects.

### Long-Term

- Introduce a monorepo-level superbuild using CMake and/or Ninja to orchestrate all subprojects.
- Normalize build targets and output structure
- Provide developer-friendly aliases or presets for common build workflows.

## CI Integration

### Short-Term

- CI definitions from original repositories are preserved and adjusted for new paths.
- Each subproject may retain its own configuration files target GitHub Actions, Azure Pipelines, Jenkins, etc.

### Monorepo-Wide CI

- Introduce top-level CI pipelines.
- Configure path-based triggers to only build/test changed subprojects.
- Use build caching and artifact reuse to improve performance.

## Release and Packaging Transition

### Legacy Repositories

- Continue current packaging and release workflows as-is.

### Monorepo

- Gradually migrate to centralized release definitions.
- Standardize versioning, changelogs, and metadata across subprojects.
- Utilize CMake packaging modules (e.g., CPACK) to produce consistent artifacts.
- Store packaging specs and release scripts at the monorepo root or within each subproject as needed.

## Synchronization with Legacy Repositories

A GitHub action runs every hour to pull in updates from the legacy repositories into this monorepo.

Updates from this monorepo to the individual legacy repositories must be done manually for now. Future plan is to automate this direction of the synchronization, and to create pull requests automatically when merge conflicts are created.

## Tooling and Automation

- Scripts for:
  - Subtree add/split/push operations
  - Selective build/test triggering
- Optional: CMake presets for unified configuration
- Linters and formatters applied at monorepo level
- Git hooks or bots to enforce monorepo policies (e.g., directory structure, commit standards)

## Access Control and Permissions

- Monorepo will use centralized access management through a consolidated CODEOWNERS file to maintain project-specific review rules.
- Legacy repository permissions will remain unchanged.

## Communication Plan

- Share migration plans with all stakeholders.
- Provide training resources for:
  - Git subtree usage
  - New trunk-based development workflow
  - Monorepo CI and build tools
  - Maintain channels for questions, feedback, and issue reporting

11. Known Limitations and Risks

- Subtree operations are slower on large histories
- Cannot do shallow clones or partial history fetches with git subtree
- Risk of conflicts during sync if legacy repos diverge significantly
- Potential for increased CI cost and time without selective workflows
