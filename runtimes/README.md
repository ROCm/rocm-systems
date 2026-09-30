# ROCm Runtime Components

The `runtimes` directory is the future home of ROCm runtime components that are
developed together in this repository.

ROCm Systems is currently in a transitional state as key runtime components
move into a monorepo layout. This directory will be built out over time as that
migration progresses. Components here are early-access software and are not
part of the repository's default build or installation yet.

## Contents

- [`api-headers`](api-headers/README.md) contains the vendored AMDF, HSA, DRM,
  KFD, and UDMABUF API headers used by runtime implementations. Its README
  records their primary sources and synchronization policy.
- [`rocddi`](rocddi/README.md) is the early-access, private,
  implementation-neutral AMD GPU device-interface layer.

Each component directory owns its architecture, build, validation, deployment,
and developer documentation. New runtime projects should follow the same
structure instead of adding project-specific build metadata or scripts directly
to this directory.
