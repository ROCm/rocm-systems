# Contributing to hipFile

Thank you for contributing! This guide outlines the development workflow,
contribution standards, and best practices when working on hipFile.

> [!NOTE]
> The ROCm libraries are transitioning to a monorepo system called [TheRock](https://github.com/ROCm/TheRock). hipFile has not been incorporated into this monorepo yet.

## Development Setup

TBD

## Branching Model

The main development trunk of hipFile is the `develop` branch.

hipFile generally uses trunk-based development, where feature branches
are intended to be relatively short-lived. We do not, however, use the
"branch by abstraction" pattern, where in-flight work is switched on
and off by feature flags. Instead, when necessary, feature branches will
be created and prefixed with `feature/`. These feature branches will
be deleted after the feature has been merged to `develop`. Any feature
branches which are not merged to `develop`, but should be kept around
for posterity, will be renamed `feature` --> `inactive`.

External developers must use forks for development. You will sometimes
see branches from AMD staff named `<user>/<branch>`. These will be
very short-lived.

### Release branches

Releases are tagged `vX.Y.Z`, where `X`, `Y`, and `Z` are the major,
minor, and patch versions of the release.

Releases occur from `develop`. Upon release, the tag is created and
the minor version number is bumped. The major version number will
only be bumped on `develop` when making an API/ABI-breaking change.

Release branches are created retroactively and only when it is
necessary to bugfix supported versions of the library. It is our
intent to minimize the number of versions we support and features
will normally not be brought back to minor releases of previous
library versions. Release branches are not kept in sync with
`develop` and just contain cherry-picks of bug and performance
fixes.

Bugfixing should take place on `develop` and be cherry-picked to
any branches that are being maintained. The only exception to this
is when a bug is only present on a release branch.

## Pull Requests

We welcome pull requests from outside contributors. Pull requests
must pass our CI and be approved by at least one code owner.
Outside contributors should fully fill out the ROCm template (for
non-trivial PRs).

Please also ensure that your code complies with this project's
coding style, as detailed in `STYLEGUIDE.md`.

## Issue Reporting

* Non-security issues should be reported as GitHub issues
* Security issues should be reported as directed in `SECURITY.md`
* Feature requests should be made using GitHub discussions
