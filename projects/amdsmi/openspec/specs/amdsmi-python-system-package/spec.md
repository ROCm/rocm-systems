# amdsmi-python-system-package Specification

## Purpose

Defines how the `amd-smi-lib` deb/rpm delivers the `amdsmi` Python module: where
the module is installed, how that destination is detected, what interpreter
dependency the package declares, and how the maintainer scriptlets behave across
install, upgrade, downgrade, and removal.

Since ROCm 7.14 the package no longer runs `pip install` from `postinst`. The
module is part of the package payload and is installed directly into the system
interpreter's site-packages, so removing the package removes the module. Getting
the destination wrong is silent: the package installs "successfully" and
`import amdsmi` fails with `ModuleNotFoundError`. Most requirements here exist
to make that failure impossible or loud.

This capability owns the site-packages copy and the maintainer scriptlets. The
prefix-relative `share/amd_smi` copy the same package installs is specified in
[amdsmi-install-layout]; loader behavior is in [amdsmi-python-loader]. The
`amdrocm-amdsmi` package built by TheRock is an unrelated package family — see
[amdsmi-rocm-os-packages].

## Requirements

### Requirement: Module Installed For The Default System Interpreter

The build SHALL detect the site-packages directory of the host's **default**
`python3` — the interpreter that a plain `import amdsmi` and the `amd-smi` CLI
shebang resolve to — and install the module there. Detection SHALL prefer
`/usr/libexec/platform-python` when it exists, and otherwise use
`/usr/bin/python3`.

#### Scenario: RHEL build host with a repointed python3

- **WHEN** the build image has repointed `/usr/bin/python3` to a newer
  interpreter via `alternatives`, as RHEL CI images commonly do
- **THEN** detection uses `/usr/libexec/platform-python`, which `alternatives`
  does not repoint, so the module lands where the OS's own `python3` and the CLI
  actually look

#### Scenario: Debian, SUSE, and Azure Linux hosts

- **WHEN** the host ships no `platform-python`
- **THEN** detection falls back to `/usr/bin/python3`, which is the default
  interpreter on those distros

#### Scenario: A packager overrides the destination

- **WHEN** `AMDSMI_SYSTEM_PYTHON_SITELIB` is set explicitly
- **THEN** auto-detection is skipped and that path is used

### Requirement: Site-Packages Detection Uses On-Path Directories

Detection SHALL query `site.getsitepackages()` and choose, in order:
`/usr/lib/python3/dist-packages`, any `*/dist-packages`, then any
`*/site-packages`, excluding `/usr/local`. It SHALL NOT use
`sysconfig.get_paths()['purelib']` as the primary source.

#### Scenario: Debian's version-agnostic path is preferred

- **WHEN** detection runs on Debian or Ubuntu
- **THEN** `/usr/lib/python3/dist-packages` is chosen — `sysconfig`'s "purelib"
  (`/usr/lib/python3.X/site-packages`) is not on those interpreters' `sys.path`
  at all

#### Scenario: /usr/local is avoided

- **WHEN** `site.getsitepackages()` reports a `/usr/local` entry
- **THEN** it is not chosen, because the system package owns `/usr` and pip
  manages `/usr/local` separately

#### Scenario: An unusual layout still lands on sys.path

- **WHEN** no `/usr` candidate matches
- **THEN** the first `site.getsitepackages()` entry is used, since every entry
  it reports is guaranteed to be on `sys.path`

### Requirement: Undetectable Destination Fails The Build

When the destination cannot be determined for a system-package build, configure
SHALL fail rather than fall back to an interpreter the target host does not use.

#### Scenario: No default python3 on the build host

- **WHEN** neither `/usr/libexec/platform-python` nor `/usr/bin/python3` exists
  and `BUILD_PYTHON_WHEEL` is `OFF`
- **THEN** configuration fails with a fatal error instructing the packager to
  set `AMDSMI_SYSTEM_PYTHON_SITELIB` explicitly or to build the wheel instead

#### Scenario: Query failure with no usable fallback

- **WHEN** the detection query fails or returns empty and no `Python3_SITELIB`
  is available (a C-only build never runs `find_package(Python3)`)
- **THEN** configuration fails, rather than setting an empty destination that
  would install the module to the prefix root

#### Scenario: Wheel builds are exempt

- **WHEN** `BUILD_PYTHON_WHEEL` is `ON` — typically inside a manylinux container
  with no system python3
- **THEN** the missing-default-interpreter case is not fatal, because the wheel
  ships its own tree

### Requirement: Site-Packages Install Is Staged Through DESTDIR

The site-packages destination is an absolute path outside the package prefix, so
it is redirected only by `DESTDIR`. The install SHALL be performed at install
time and only when `DESTDIR` is set. The bundled `libamd_smi*.so` SHALL be
excluded. Permissions SHALL be pinned to 0644 for files and 0755 for
directories.

#### Scenario: CPack and distro packagers get the module

- **WHEN** CPack or a distro packaging harness runs the install with `DESTDIR`
  set
- **THEN** the module is staged under `$DESTDIR/<sitelib>/amdsmi` and lands in
  the package payload

#### Scenario: A plain cmake --install does not touch the host /usr

- **WHEN** `cmake --install` runs with no `DESTDIR`
- **THEN** the site-packages copy is skipped with a status message, avoiding an
  `EACCES` against the host's real `/usr`; the module is still installed under
  `share/amd_smi`. This is also what makes TheRock builds skip it — see
  [amdsmi-therock-subproject]

#### Scenario: A permissive build umask cannot leak

- **WHEN** the build runs under a umask that would otherwise produce
  group/other-writable files
- **THEN** the installed `.py` files under the system directory are 0644 and
  directories are 0755

### Requirement: The Two Installed Module Copies Must Not Drift

This package is the only channel that installs the module twice: into the
interpreter's site-packages and into `<prefix>/share/amd_smi/amdsmi` (the tree
specified in [amdsmi-install-layout]). Both copies are required, and they SHALL
be byte-identical.

#### Scenario: Both copies serve distinct consumers

- **WHEN** the package is installed
- **THEN** the site-packages copy makes a plain `import amdsmi` work, and the
  `share/amd_smi` copy is what the TheRock artifact flow captures and what
  downstream tools reach via `sys.path.insert(0, ROCM_PATH + "/share/amd_smi")`

#### Scenario: A symlink or redirector is not an acceptable substitute

- **WHEN** replacing one copy with a link to the other is considered
- **THEN** it is not viable, because TheRock ships only the ROCm prefix and
  cannot reach the `/usr` site-packages tree

#### Scenario: Drift fails the build instead of shipping

- **WHEN** the two installed copies differ
- **THEN** `tests/run_amdsmi_dual_copy_test.py` fails, so a partial upgrade or a
  stray edit cannot ship a tree where `import amdsmi` resolves to a different
  version depending on `sys.path` order

### Requirement: RPM Declares A Versioned Interpreter Dependency

On RPM distros the module path is version-specific, so the RPM SHALL declare a
dependency on the matching interpreter. The Python major.minor SHALL be derived
from the detected site-packages path, and the dependency token SHALL be chosen
from the `ID`/`ID_LIKE` fields of `/etc/os-release`:

| Distro family match | Dependency token |
| ------------------- | ---------------- |
| `rhel`, `centos`, `fedora`, `almalinux`, `rocky`, `mariner`, `azurelinux` | `python(abi) = X.Y` |
| `suse`, `sles` | `pythonXY` (no dot, for example `python311`) |
| anything else, or no ABI derived | `python3 >= 3.6.8` |

The deb SHALL keep the loose `python3 (>= 3.6.8)` dependency in all cases.

#### Scenario: The RPM only installs where its baked path exists

- **WHEN** an RPM built against one interpreter is offered to a host whose
  default python3 is a different minor
- **THEN** the versioned dependency prevents the install, instead of installing
  the module where that host's python3 never looks

#### Scenario: The dependency is set by hand because the generator does not fire

- **WHEN** the RPM is built
- **THEN** the requirement is set explicitly, because CPack installs raw `.py`
  files with no `.dist-info` metadata and the RHEL8 build image lacks the
  path-based python dependency generator

#### Scenario: Debian's path carries no minor

- **WHEN** the detected path is Debian's version-agnostic
  `/usr/lib/python3/dist-packages`
- **THEN** no ABI is derived and the dependency is left as the loose
  `python3 (>= 3.6.8)`, which every python3 minor satisfies

#### Scenario: The SLES token is verified without a GPU

- **WHEN** the SLES packaging CI job builds the RPM in an openSUSE Leap
  container
- **THEN** it asserts a `pythonXY` requirement is present and that zypper finds
  a package *providing* it — matching the loose `python3` fallback instead would
  let a broken detection pass the check it exists to make

### Requirement: The RPM Does Not Co-Own System Python Directories

The RPM SHALL own only its own `<sitelib>/amdsmi` subtree. The site-packages
directory and each of its ancestors up to (but excluding) `/usr` SHALL be
excluded from the auto filelist.

#### Scenario: Shared python directories keep their OS-package ownership

- **WHEN** the RPM is built
- **THEN** it does not declare `%dir` ownership of `/usr/lib64/python3.X` or
  `.../site-packages`, and therefore cannot apply its default group-writable
  mode to directories the OS python package owns

### Requirement: Maintainer Scriptlets Are Scoped To Their Component

Maintainer scriptlets SHALL be assigned per component group. The `runtime` group
(the main package) SHALL own the RPM post-install, pre-uninstall, and
post-uninstall scriptlets and the deb `postinst` and `prerm`. The `tests` group
SHALL own only its own pre-uninstall. No unprefixed
`CPACK_RPM_<KIND>_SCRIPT_FILE` or generic `CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA`
SHALL be set.

#### Scenario: Removing the tests package leaves the main package intact

- **WHEN** `amd-smi-lib-tests` is removed while `amd-smi-lib` is still installed
- **THEN** the main package's linker registration, log directory, and logrotate
  configuration survive, verified by
  `tests/run_amdsmi_component_removal_test.py`

#### Scenario: A component added later inherits nothing

- **WHEN** a new CPack component is introduced without an explicit scriptlet
  assignment
- **THEN** it ships no scriptlets, because assignment is explicit rather than
  inherited — an unprefixed assignment would be copied into every component and
  let the asan or tests package clobber the main package's absolute-path state

### Requirement: Post-Install Behavior

On install and upgrade the package SHALL, when `ENABLE_LDCONFIG` is on, write
the library directory into `/etc/ld.so.conf.d/x86_64-libamd_smi_lib.conf` and
run `ldconfig`. It SHALL warn when a pip-installed copy shadows the packaged
module. It SHALL activate `argcomplete` global completion on a best-effort
basis. It SHALL create the operational log file and a logrotate drop-in.

#### Scenario: The SONAME becomes resolvable

- **WHEN** post-install completes
- **THEN** the dynamic linker resolves `libamd_smi.so.<MAJOR>`, which is what
  the loader's final resolution step depends on

#### Scenario: A shadowing pip copy is reported, not removed

- **WHEN** `import amdsmi` in the system python3 resolves outside the packaged
  site-packages directory
- **THEN** post-install prints a warning naming both locations and the
  `pip uninstall amdsmi` remedy, and still exits successfully

#### Scenario: Missing optional tooling does not fail the install

- **WHEN** `logrotate` or the argcomplete activation helper is absent
- **THEN** the step is skipped with an informational message and the install
  succeeds

#### Scenario: System-wide logrotate scheduling is not modified

- **WHEN** the logrotate drop-in is written
- **THEN** `logrotate.timer` and `cron.daily/logrotate` are left untouched,
  because they are shared with every other package that uses logrotate

### Requirement: Removal Cleanup Distinguishes Removal From Upgrade

Destructive cleanup — removing the `ld.so.conf.d` entry, the log directory, the
logrotate drop-in, the installed tests directory, and stale build leftovers —
SHALL run only on a full removal, never on an upgrade. Generated bytecode SHALL
be removed from every location the module was installed to, including the
absolute site-packages path outside the package prefix.

The linker entry SHALL be removed from a different scriptlet on each format: the
deb removes it from `prerm`, the RPM from `%postun`.

#### Scenario: An RPM upgrade keeps the incoming package's linker entry

- **WHEN** the RPM is upgraded
- **THEN** the entry survives, because RPM runs the new package's `%post` —
  which re-creates it — *before* the old package's `%postun`; doing the removal
  in `%preun` instead would delete the entry the new package still needs

#### Scenario: A deb upgrade keeps the incoming package working

- **WHEN** the deb is upgraded
- **THEN** `prerm` is invoked with `upgrade` rather than `remove`, so it does
  not delete the linker entry, logs, tests directory, or logrotate configuration

#### Scenario: No unowned bytecode is left behind

- **WHEN** the package is removed
- **THEN** `__pycache__` directories under the CLI's `libexec` tree, the
  `share/amd_smi` copy, and `<sitelib>/amdsmi` are removed — the package owns
  the `.py` files but not the `.pyc` Python generated on first import

### Requirement: Upgrade And Downgrade Transitions

Moving between package versions SHALL leave the module importable, the CLI
runnable, and the linker registration correct at every step.

| Transition | Required behavior |
| ---------- | ----------------- |
| pre-7.14 (pip-era) → 7.14+ | the old package's pre-removal still `pip uninstall`s the legacy module and removes its `.pth`; the new package owns the site-packages files |
| 7.14+ → 7.14+ | plain file replacement by the package manager |
| 7.14+ → pre-7.14 (downgrade) | the old package re-adds the pip install; a user-installed PyPI wheel still wins and survives |
| removal, then a PyPI wheel | removal deletes only the package's own files; the wheel is self-contained |

#### Scenario: A self-upgrade round trip is verified in CI

- **WHEN** the upgrade/downgrade job builds the tree once and then repackages it
  twice at two `ROCM_LIBPATCH_VERSION` values in the same distro container
- **THEN** installing old → upgrading to new → reinstalling old keeps
  `import amdsmi`, the `amd-smi` CLI, and the `ld.so.conf.d` entry consistent at
  each
  step; a true cross-release test additionally needs a published prior artifact,
  which this job does not have

#### Scenario: A package must be tested in the distro it was built in

- **WHEN** a package built on one distro is installed on another
- **THEN** the module may not import, because the site-packages path is baked in
  at build time — so the CI job builds and tests in the same image

### Requirement: The Built Package Is Checked Before Installation

The module's destination SHALL be verified by inspecting a built `.deb`/`.rpm`,
before the package is ever installed.

#### Scenario: A misplaced module fails at build time

- **WHEN** `tests/run_amdsmi_cpack_path_test.py` inspects the built package
- **THEN** it asserts the `amdsmi` module files appear under a `site-packages`
  or `dist-packages` directory, catching a detection regression that would
  otherwise only surface as `ModuleNotFoundError` on a user's machine

### Requirement: CLI Shebang Is Preserved

The `amd-smi` CLI SHALL keep its `#!/usr/bin/env python3` shebang in the built
package, so it runs under the same default interpreter whose site-packages the
module was installed into.

#### Scenario: RPM shebang mangling is disabled

- **WHEN** the RPM is built on RHEL8, where the buildroot policy script would
  otherwise rewrite the shebang to `/usr/libexec/platform-python`
- **THEN** `__brp_mangle_shebangs` is explicitly undefined in the spec, leaving
  the shebang intact

#### Scenario: Bytecompile failures do not abort the RPM build

- **WHEN** byte-compiling the packaged Python files fails
- **THEN** the RPM build continues, because bytecode is regenerated on first
  import anyway
