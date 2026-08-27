# amdsmi-rocm-systems-integration Specification

## Purpose

Defines AMD SMI's place in the `rocm-systems` monorepo: where the authoritative
sources live, how a change reaches TheRock and then a ROCm release, how the
build's version numbers are supplied from outside the project, and which CI jobs
gate the packaging behavior specified by the other capabilities.

Everything downstream — the subproject build ([amdsmi-therock-subproject]), the
artifact ([amdsmi-therock-artifact]), the wheels
([amdsmi-rocm-python-distribution]), and the OS packages
([amdsmi-rocm-os-packages]) — is cut from a single pinned commit of this
repository. This capability owns how that commit is produced, versioned, and
validated; what the commit's contents must do is owned by the behavior
capabilities [amdsmi-c-api-abi], [amdsmi-device-discovery], [amdsmi-python-api]
and [amdsmi-cli].

## Requirements

### Requirement: rocm-systems Is The Source Of Truth

AMD SMI's authoritative sources SHALL live at `projects/amdsmi` in the
`rocm-systems` monorepo. Pull requests SHALL target the `develop` branch. The
standalone pre-monorepo repository SHALL NOT be treated as the source of truth.

#### Scenario: Contributors work only in the monorepo

- **WHEN** a change to AMD SMI is proposed
- **THEN** it is opened against `rocm-systems`, whose migration status for
  `amdsmi` is recorded as complete and whose repository configuration marks the
  monorepo as the source of truth; the legacy repository receives an automatic
  push and is used only for release activities

#### Scenario: A sparse checkout is the normal working mode

- **WHEN** a developer or CI job needs only AMD SMI
- **THEN** they sparse-checkout `projects/amdsmi` (plus `.github/workflows` for
  CI), because a full monorepo clone is disproportionately expensive

#### Scenario: Ownership is scoped to the project path

- **WHEN** a pull request touches `projects/amdsmi`
- **THEN** the AMD SMI reviewers are requested via the monorepo CODEOWNERS
  entry, with documentation paths additionally routed to the documentation
  owners by the project-local CODEOWNERS

### Requirement: TheRock And rocm-systems Pin Each Other

TheRock SHALL consume `rocm-systems` as a git submodule tracking `develop`, and
`rocm-systems` CI SHALL check out TheRock at an explicitly pinned commit.

#### Scenario: A monorepo change is not live in TheRock until the pin moves

- **WHEN** an AMD SMI change merges to `rocm-systems:develop`
- **THEN** TheRock builds continue to use the previously pinned submodule commit
  until it is bumped, so a packaging change lands in TheRock as a deliberate
  step

#### Scenario: A TheRock change is not live in monorepo CI until repinned

- **WHEN** TheRock changes how subprojects are staged or artifacts are sliced
- **THEN** `rocm-systems`' TheRock CI keeps using its pinned TheRock revision
  until that pin is updated, which bounds the blast radius in both directions

#### Scenario: Packaging changes must be validated on both sides

- **WHEN** a change alters AMD SMI's install layout
- **THEN** it can pass AMD SMI's own CI while still breaking artifact slicing,
  so a TheRock-side build is needed before the change is considered complete

### Requirement: Release Branches Are Fed By Cherry-Pick From develop

A change SHALL land on `develop` first and then be cherry-picked to the
appropriate `release-staging/rocm-rel-x.y` branch, so every commit on a release
branch also exists on `develop`.

#### Scenario: Squash merges keep cherry-picks cheap

- **WHEN** a change is merged to `develop`
- **THEN** squashing it into one logical commit is preferred, because that
  commit is what gets cherry-picked into the release branch

#### Scenario: Standalone repositories are updated by a scheduled fan-out

- **WHEN** a commit lands on `develop` or on a release-staging branch in the
  monorepo
- **THEN** an hourly scheduled subtree sync copies it into the corresponding
  standalone repository, so downstream consumers of the old repository see it
  without a separate submission

#### Scenario: Landing directly on a release branch is not permitted

- **WHEN** a fix is needed on a release branch
- **THEN** it must exist on `develop` first and arrive by pull request,
  preventing a release branch from carrying a change that would be lost at the
  next merge

### Requirement: The Package Version Is Supplied From The Build Environment

AMD SMI's own version SHALL come from `include/amd_smi/amdsmi.h`, and the
release-identifying components SHALL be supplied by environment variables at
package time:

| Variable | Effect | Default |
| -------- | ------ | ------- |
| `ROCM_LIBPATCH_VERSION` | fourth component of the CPack package version | `99999` |
| `CPACK_DEBIAN_PACKAGE_RELEASE` / `CPACK_RPM_PACKAGE_RELEASE` | package release field | `local` |
| `ROCM_BUILD_ID` | build identifier in the internal version string | `local-build-0` |

Only these come from outside. The three leading version components are not
settable from the command line at all; the configure-time rule that recomputes
them from the header, shadowing any `-D` switch, is
[amdsmi-build-configuration].

#### Scenario: A locally built package is visibly non-release

- **WHEN** a developer or a CI gate builds packages without the release
  environment set
- **THEN** the artifacts are named with the `99999` libpatch and the `local`
  release, which is what CI matches when collecting build artifacts and which
  makes an accidental release-looking package impossible

#### Scenario: The release harness supplies the ROCm identity

- **WHEN** the official release build runs
- **THEN** it sets the libpatch and release environment variables, so the
  `amd-smi-lib` package version identifies the ROCm release it belongs to

#### Scenario: Three version numbers coexist and mean different things

- **WHEN** a user inspects a ROCm installation
- **THEN** AMD SMI's library version (from `amdsmi.h`), the ROCm SDK version
  (from TheRock's version computation), and the package version each differ, and
  only the library version is bumped by AMD SMI's own changes

### Requirement: Pull Requests Are Gated By The Systems PR Bot

Every pull request SHALL be checked by the monorepo's policy bot, which posts a
single results table. The bot SHALL apply the `Not ready to Review` label only
when the description is missing a tracking reference; every other failing row
SHALL leave the label alone.

#### Scenario: Only the missing tracking reference applies the label

- **WHEN** a pull request description is shorter than 30 characters but does
  cite a JIRA or issue ID
- **THEN** the description row fails and the bot check goes red, but the
  not-ready label is not applied — the label is reserved for the one rule that
  makes a PR untriageable

#### Scenario: Missing tests and forbidden files warn but do not block

- **WHEN** a source change ships without an accompanying test, or a
  credential-shaped file is present
- **THEN** the results table shows a warning row while the bot check stays green

#### Scenario: Titles follow the Conventional Commits form

- **WHEN** a commit or pull request title is written for AMD SMI
- **THEN** it uses the `type(amdsmi): description` form rather than a
  bracket-tag prefix

#### Scenario: Automated dependency bumps bypass the policy

- **WHEN** a pull request is opened by one of the configured bump-bot accounts
- **THEN** every row is auto-approved, so routine submodule bumps neither block
  nor spam the not-ready label

#### Scenario: The bot can be opted out of explicitly

- **WHEN** a pull request description carries the `@skip-pr-bot` tag
- **THEN** no policy checks run and any existing not-ready label is removed

### Requirement: Packaging Behavior Is Gated By Dedicated CI Workflows

The monorepo SHALL run a set of AMD SMI workflows that, between them, cover
every packaging invariant these specifications describe:

| Workflow | Covers |
| -------- | ------ |
| AMD SMI CI | manylinux wheel build gate, plus build/install/test across nine Debian- and RPM-family distros on GPU hardware |
| Manylinux wheels | `auditwheel`-repaired release wheel and a Debian-host wheel, plus the SONAME split and disabled-fallback checks |
| Python versions | loader contract across CPython 3.6 – 3.14, no GPU |
| SLES packaging | the `pythonXY` RPM interpreter dependency form and the packaged module path, no GPU |
| Upgrade / downgrade | deb and rpm install → upgrade → downgrade round trip, plus the tests-component removal check, no GPU |

The option combinations each of these workflows configures, and the parts of
the option surface none of them reaches, are [amdsmi-build-configuration].

#### Scenario: Most packaging coverage needs no GPU

- **WHEN** the wheel, Python-version, SLES, and upgrade/downgrade jobs run
- **THEN** they execute on hosted runners in containers, so packaging
  regressions are caught without competing for GPU hardware

#### Scenario: The distro matrix covers what container-only CI cannot

- **WHEN** the main CI job runs
- **THEN** it builds, installs, and tests on real GPUs across Ubuntu 20/22,
  Debian 10, SLES, RHEL 8/9/10, AlmaLinux 8, and Azure Linux 3, covering the
  site-packages detection and install paths that a container-only build never
  exercises

#### Scenario: Untrusted fork code never reaches the privileged runners

- **WHEN** a pull request originates from a fork
- **THEN** the self-hosted GPU matrix job is skipped, because those runners are
  privileged with `SYS_MODULE` and the host module tree mounted

#### Scenario: The distro-specific build logic lives in a reusable script

- **WHEN** a developer needs to reproduce a distro-specific CI failure
- **THEN** they run `tests/amdsmi_build/run_amdsmi_build.py` locally or in the
  same container; it autodetects the package manager, package format, and distro
  quirks from `/etc/os-release` rather than encoding them in workflow YAML

#### Scenario: Path filters keep the gates targeted, unevenly

- **WHEN** a change touches only documentation under `projects/amdsmi/docs`
- **THEN** the wheel, Python-version, SLES, and upgrade/downgrade jobs do not
  run, but the main AMD SMI CI job still does — its path filter is the whole
  `projects/amdsmi/**` tree with no documentation exclusions
