# amdsmi-install-layout Specification

## Purpose

Defines the shape of the tree AMD SMI's own install rules produce, and the parts
of it that are a public interface rather than an implementation detail:
`<root>/lib/libamd_smi.so.<MAJOR>`, `<root>/share/amd_smi/amdsmi/`,
`<root>/libexec/amdsmi_cli/`, and `<root>/bin/amd-smi`.

Every delivery channel inherits this layout — the `amd-smi-lib` deb/rpm installs
it under `/opt/rocm`, TheRock stages it and slices it into the `core-amdsmi`
artifact, and the ROCm pip and `amdrocm-amdsmi` packages relocate it wholesale.
Because the relationships between these four directories are all *relative*, the
tree keeps working wherever it is placed; equally, moving any one of them
relative to the others breaks resolution in channels that do not run the
loader's other steps. Those relationships are load-bearing, so they are
specified here once rather than restated per channel.

This capability owns the tree only — the paths and the relationships between
them, never the behavior of what sits at each path. Library resolution is
specified in [amdsmi-python-loader]; the additional site-packages copy that
accompanies this tree in the deb/rpm is in [amdsmi-python-system-package]; and
artifact capture of the tree is in [amdsmi-therock-artifact].

## Requirements

### Requirement: Prefix-Relative Tree Layout

Every build SHALL install, under the ROCm prefix:

| Path | Contents |
| ---- | -------- |
| `<root>/lib/libamd_smi.so.<MAJOR>` | the native library, reached through the `SOVERSION` symlink chain over the real `libamd_smi.so.<X.Y.Z>` file |
| `<root>/share/amd_smi/amdsmi/` | the Python module, excluding any bundled `libamd_smi*.so` |
| `<root>/share/amd_smi/tests/`, `<root>/share/amd_smi/example/` | the test tree (when tests are enabled) and the example sources |
| `<root>/libexec/amdsmi_cli/` | the CLI modules, including `amdsmi_cli.py` |
| `<root>/bin/amd-smi` | a relative symlink to `../libexec/amdsmi_cli/amdsmi_cli.py` |

The same tree additionally carries `<root>/lib/libgoamdsmi_shim64.so*` and the
Go shim headers `<root>/include/goamdsmi.h` and
`<root>/include/amdsmi_go_shim.h` in any channel that installs the whole build
rather than a selected component set.

#### Scenario: The module resolves the library by relative path

- **WHEN** the wrapper at `<root>/share/amd_smi/amdsmi/amdsmi_wrapper.py` loads
  the library
- **THEN** it finds `<root>/lib/libamd_smi.so.<MAJOR>` at the wrapper file's
  fourth parent, with no absolute path and no dynamic-linker configuration —
  which is why the three path components between the wrapper and the prefix root
  cannot change independently

#### Scenario: The bundled wheel library is never in this tree

- **WHEN** the module is installed under `share/amd_smi`
- **THEN** `libamd_smi*.so` is excluded from that directory, so the tree
  resolves the shared `lib/` library rather than a private copy that could
  diverge from it

#### Scenario: The CLI reaches its own modules relatively

- **WHEN** `bin/amd-smi` runs and needs the CLI modules and then the `amdsmi`
  package
- **THEN** it resolves `libexec/amdsmi_cli` and `share/amd_smi` relative to its
  own location, so the whole tree is relocatable as a unit

#### Scenario: The test tree travels inside share/amd_smi

- **WHEN** AMD SMI is built with `BUILD_TESTS=ON`
- **THEN** the gtest binary and the Python test suite install under
  `share/amd_smi/tests` rather than a sibling directory, which is what makes
  them fall inside the packaging patterns that select `share/amd_smi` (see
  [amdsmi-therock-artifact])

### Requirement: The Layout Is A Public Interface

The `<root>/share/amd_smi` path and the `bin` → `libexec` → `share` → `lib`
relationships SHALL be treated as a compatibility surface, not an internal
detail. Moving any of these directories relative to the others is a breaking
change for consumers that do not go through a plain `import amdsmi`.

#### Scenario: Downstream ROCm tools insert the path directly

- **WHEN** a consumer such as rocprofiler-compute runs
  `sys.path.insert(0, ROCM_PATH + "/share/amd_smi")` against a ROCm install
- **THEN** `import amdsmi` succeeds; without the `share/amd_smi` copy it would
  raise `ModuleNotFoundError`

#### Scenario: Relocating the CLI breaks a channel that nothing else covers

- **WHEN** the CLI moves out of `libexec/amdsmi_cli` relative to `bin/`
- **THEN** the ROCm pip channel's `bin/amd-smi` entry point fails, because its
  import fallback is a fixed `../libexec/amdsmi_cli` hop — while the deb/rpm
  channel, which reaches the module through site-packages, keeps working and
  hides the regression

#### Scenario: The layout survives a prefix change

- **WHEN** the tree is installed under a versioned prefix such as
  `/opt/rocm/core-X.Y`, extracted from a tarball, or relocated into a Python
  platform directory
- **THEN** every hop still resolves, because none of them is anchored to an
  absolute prefix
