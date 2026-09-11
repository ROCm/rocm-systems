---
myst:
  html_meta:
    "description lang=en": "How to install AMD SMI libraries and CLI tool."
    "keywords": "system, management, interface, cpu, gpu, hsmp, versions"
---

# Install the AMD SMI library and CLI tool

AMD SMI is delivered as three pieces that ship together: the `libamd_smi` C
library, the `amdsmi` Python module, and the `amd-smi` CLI. This page covers the
system requirements and every supported way to get them onto a Linux system.

(install_choose)=
## Choose an installation method

Every method below delivers the same AMD SMI release. They differ in what else
they pull in, where the files land, and how much wiring you do yourself. Find
the row that matches your situation and follow its link.

| Method | `amd-smi` CLI | `import amdsmi` | Use this when |
|---|---|---|---|
| {ref}`ROCm Core SDK <install_rocm>` | Included | Ready to use | You want ROCm anyway. **Start here if you are unsure.** |
| {ref}`Standalone package <install_without_rocm>` | Included | Ready to use | You want AMD SMI without the rest of the ROCm libraries and tools |
| {ref}`Nightly build <install_nightly>` | Included | Ready to use | You need a pre-release fix, or you are testing against an unreleased ROCm |
| {ref}`Tarball <install_tarball>` | After you add `bin/` to `PATH` | After you add it to `sys.path` | You have no root access, need versions side by side, or are on an air-gapped host |
| {ref}`PyPI wheel <install_pypi>` | Not included | Ready to use | You only script against the Python API and do not want ROCm on the host |

Set your choice below and the instructions on this page narrow to match. The
page address updates as you choose, so you can link someone the exact procedure
you followed.

:::{install-selector}
:::

:::{note}
Whichever method you pick, the machine you query still needs the `amdgpu` kernel
driver. See {ref}`Driver requirements <install_amdgpu_driver>`.
:::

## Supported platforms

AMD SMI supports:

- {ref}`AMD GPUs <rocm:release-supported-hw>` on Linux bare metal systems
- AMD GPUs in Linux virtual machine guests
- AMD EPYC™ CPUs through the
  [esmi_ib_library](https://github.com/amd/esmi_ib_library) (requires the
  `amd_hsmp` kernel module with HSMP enabled in BIOS at runtime)

For AMD SMI on Linux SR-IOV hosts, refer to
the [AMD SMI for Virtualization documentation](https://instinct.docs.amd.com/projects/amd-smi-virt/en/latest/index.html).

AMD SMI library runs on AMD ROCm supported platforms. Refer to
{ref}`AMD hardware support <rocm:release-supported-hw>` for more information.

(install_reqs)=
## Requirements

Before installing AMD SMI, make sure your system meets the following
requirements.

(install_amdgpu_driver)=
### Driver requirements

To run AMD SMI, the following components need to be installed on your system:

- The `amdgpu-dkms` driver
  - For current amdgpu driver installation instructions, see the [AMD GPU
    Driver (amdgpu)
    documentation](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/install/detailed-install/prerequisites.html).

    :::{note}
    ROCm releases claim support for a {ref}`range of amdgpu driver versions
    <rocm:release-supported-fw>`. Because AMD SMI gets GPU telemetry and
    management data directly from the amdgpu kernel driver, version mismatches
    in either direction can affect which metrics and controls are available:

    - If the amdgpu driver is older than your AMD SMI release expects, some
      features might be unavailable, causing some fields to read N/A.
    - If the amdgpu driver is newer than AMD SMI expects, AMD SMI might not
      recognize new data formats (for example, newer `gpu_metrics` versions)
      and can report N/A for affected fields.

    To maximize compatibility, we recommend using the latest amdgpu driver version
    that matches your AMD SMI or ROCm release. See {ref}`About N/A values
    <cli-output-na>` for more information.
    :::
- The `amd_hsmp` or `hsmp_acpi` driver
  - Required for `amdsmi_init(AMDSMI_INIT_AMD_CPUS)` and `amd-smi` CPU commands.
  - See [amd_hsmp](https://github.com/amd/amd_hsmp) for more information.
  - Without it, CPU discovery is skipped non-fatally and only GPU and NIC data
    is reported.

Also confirm that your Linux kernel version matches the system requirements
described in {ref}`Operating system support <rocm:release-supported-os>`.

### Interface prerequisites

The following prerequisites apply to the AMD SMI library interfaces:

- Python interface and `amd-smi` CLI:
  - Python 3.6.8 or later
  - 64-bit Python
- Go interface:
  - Go 1.20 or later

::::{note}
During the driver installation process on Azure Linux 3, you might encounter
the `ModuleNotFoundError: No module named 'more_itertools'` warning. This
warning is a result of the reintroduction of `python3-wheel` and
`python3-setuptools` dependencies in the CMake of AMD SMI, which requires
`more_itertools` to build these Python libraries. This issue will be fixed in a
future ROCm release. As a workaround, use the following command before
installation:

```bash
sudo python3 -m pip install more_itertools
```
::::

(install_rocm)=
## Install the ROCm Core SDK

:::{install-section}
:method: rocm
:::

**Use this when** you want ROCm on the machine anyway. This is the default path
and the one most users should take.

AMD SMI is included with most installations of the ROCm Core SDK on Linux.

For instructions, see {doc}`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

(install_without_rocm)=
## Install AMD SMI standalone on Linux

:::{install-section}
:method: standalone
:::

**Use this when** you want AMD SMI managed by your package manager, but not the
rest of the ROCm libraries and tools.

Install the `amdrocm-amdsmi` package. It includes AMD SMI and the ROCm system
dependencies it needs, and nothing else.

1. Complete the {doc}`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the AMD SMI package that matches your desired ROCm version. Package
   names use the following format:

   ```
   amdrocm-amdsmi<rocm_version>
   ```

   `<rocm_version>` represents the ROCm Core SDK version to install. Omit this
   suffix to install the latest available version.

   For example, to install the latest ROCm AMD SMI release for supported GPU
   architectures:

   ::::{install-when}
   :distro: debian
   ```bash
   sudo apt install amdrocm-amdsmi
   ```
   ::::
   ::::{install-when}
   :distro: rhel
   ```bash
   sudo dnf install amdrocm-amdsmi
   ```
   ::::
   ::::{install-when}
   :distro: sles
   ```bash
   sudo zypper install amdrocm-amdsmi
   ```
   ::::

3. Prepend the `amd-smi` binary to your PATH, it is not on PATH by default.
   Replace `<major>` and `<minor>` with the appropriate ROCm version.

   ```bash
   export PATH="/opt/rocm/core-<major>.<minor>/bin${PATH:+:${PATH}}"
   ```

   To persist this across shells, append the line to your `~/.bashrc` (or
   equivalent shell config).

4. Verify your installation.

   ```bash
   amd-smi version
   ```

(install_nightly)=
## Install a nightly build

:::{install-section}
:method: nightly
:::

**Use this when** you need a fix that has not shipped in a release yet, or you
are validating against an unreleased ROCm.

Nightly builds of the ROCm Core SDK (including AMD SMI) are published by
[TheRock](https://github.com/ROCm/TheRock) to a unified pip index.

1. Create and activate a Python virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install ROCm with the device extra matching your GPU. For example, for
   gfx942 (MI300X / MI325X):

   ```bash
   pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
       "rocm[libraries,device-gfx942]"
   ```

   For the full list of `device-*` extras and other release options, see
   [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#supported-python-device--install-extras).

3. Verify your installation:

   ```bash
   amd-smi version
   ```

(install_tarball)=
## Install from a tarball

:::{install-section}
:method: tarball
:::

**Use this when** you cannot use a package manager: no root access, multiple
versions side by side, or an air-gapped host.

A tarball has no install step. Extracting it runs no package manager, no
`ldconfig`, and no pip, so nothing is registered with Python. The `amd-smi` CLI
and the `amdsmi` Python module both ship inside the archive and run from
wherever you extract it, so you point your tools at the tree instead of
installing out of it.

1. Extract the tarball. The examples below use `$AMDSMI_ROOT` for the directory
   that contains `bin/`, `lib/`, and `share/`. If the archive expands into a
   versioned top-level directory, point `AMDSMI_ROOT` at that directory.

   ```bash
   mkdir -p ~/rocm-tarball
   tar -xf <rocm-tarball>.tar.gz -C ~/rocm-tarball
   export AMDSMI_ROOT=~/rocm-tarball

   ls "$AMDSMI_ROOT/lib/libamd_smi.so."*
   ls "$AMDSMI_ROOT/share/amd_smi/amdsmi/amdsmi_wrapper.py"
   ```

   Those two paths are what the CLI and the Python loader key on.

2. Run the CLI. It locates its own Python modules, so only `PATH` is needed.

   ```bash
   export PATH="$AMDSMI_ROOT/bin${PATH:+:${PATH}}"
   amd-smi version
   ```

3. Make the module importable, using whichever scope you want.

   **Current shell** — one variable, no files written:

   ```bash
   export PYTHONPATH="$AMDSMI_ROOT/share/amd_smi${PYTHONPATH:+:${PYTHONPATH}}"
   ```

   `PYTHONPATH` outranks every install location, so this also overrides an
   installed `amd-smi-lib` package or a pip `amdsmi` wheel for any Python
   started from this shell. Append the `export` line to your `~/.bashrc` (or
   equivalent) to persist it.

   **One virtual environment** — no environment variable, nothing else on the
   host affected:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   echo "$AMDSMI_ROOT/share/amd_smi" > "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/amdsmi.pth"
   ```

4. Verify. The reported module path should be under `$AMDSMI_ROOT/share/amd_smi`.

   ```bash
   python3 -c "import amdsmi; print(amdsmi.__file__); print(amdsmi.amdsmi_get_lib_version())"
   ```

:::{important}
Reference the module where the tarball put it. Do not copy it elsewhere.

The wrapper finds the library at `<root>/lib/libamd_smi.so.<MAJOR>` by walking
up from its own file, so it pairs with the tarball's own library only while it
stays at `<root>/share/amd_smi/amdsmi`. Copy it into `site-packages` and that
relative path stops resolving: the loader falls through to a bare
`libamd_smi.so.<MAJOR>` lookup and binds whatever the dynamic linker finds
first. On a host that already has ROCm installed that is `/opt/rocm/lib`, so
you silently get a different library than the one you extracted; on a host
without ROCm the import fails outright. The directory also ships no
`pyproject.toml` or `setup.py`, so `pip install` cannot consume it in place.

Keeping the module in the tree is what makes `LD_LIBRARY_PATH` unnecessary: the
library's `RUNPATH` is `$ORIGIN`-relative, so its own ROCm dependencies resolve
from the same extracted tree.
:::

See [Packaging and install paths](../packaging.md) for the full precedence and
coexistence rules.

(install_pypi)=
## Install the Python bindings from PyPI

:::{install-section}
:method: pypi
:::

**Use this when** you only need the Python API, for example to script against
GPU telemetry on a host where you do not want a ROCm installation.

This method installs the `amdsmi` module only. It does not provide the
`amd-smi` CLI. The wheel bundles its own copy of the library
(`libamd_smi_python.so`) next to the wrapper, so it does not require
`/opt/rocm` to be present. The `amdgpu` kernel driver is still required.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the wheel:

   ```bash
   python3 -m pip install amdsmi
   ```

:::{important}
The wheel is self-contained by design: it loads the library it bundled and
never falls back to a system ROCm installation. Two consequences follow.

- If the host already has a newer AMD SMI installed, the wheel does not use it.
  The two can report different library versions from the same machine.
- The PyPI release cadence is independent of ROCm, so the published wheel can
  lag the AMD SMI version in your ROCm installation.

Check the published version against your ROCm version before depending on it,
and prefer one of the ROCm-managed methods above when ROCm is already present.
:::

(install_verify)=
## Verify your installation

These checks apply to every method. Run the ones that match what you installed.

1. Confirm the CLI resolves and can reach the driver:

   ```bash
   amd-smi version
   amd-smi list
   ```

2. Confirm the Python module imports, and check *which* copy answered:

   ```bash
   python3 -c "import amdsmi; print(amdsmi.__file__); print(amdsmi.amdsmi_get_lib_version())"
   ```

   The printed path is the one piece of output worth reading carefully. On a
   host with more than one AMD SMI present it tells you which install actually
   won. If it is not the one you just installed, see
   [Packaging and install paths](../packaging.md) for the resolution order.

## Optional and advanced installation

Use these optional procedures for CLI autocompletion and advanced setups, such
as systems with multiple ROCm instances.

### Enable CLI autocompletion

The `amd-smi` CLI application supports autocompletion. If `argcomplete` is not
installed and enabled already, do so using the following commands.

```shell
python3 -m pip install argcomplete
activate-global-python-argcomplete --user
# restart shell to enable
```

(install-manual-py-lib)=
### Install the Python library for multiple ROCm instances

If multiple ROCm versions are installed and you are not using `pyenv`,
uninstall previous versions of AMD SMI before installing the desired version
from your ROCm instance.

#### Manually install the Python library

Multiple ROCm installations may cause `amd-smi` failures.
Installing multiple versions of ROCm on the same system can result in the `amd-smi` CLI not functioning correctly.

Starting with ROCm 7.14, the `amd-smi-lib` rpm/deb package no longer
runs `pip install` during postinst — it installs the `amdsmi` Python package directly into the system
Python's `site-packages` as part of the package payload. Removing the
package removes those files; only a legacy (pre-7.14) pip-registered
install needs to be uninstalled manually.

1. Remove previous AMD SMI installations.

   ```shell
   # Legacy: pre-7.14 system packages and any user pip install
   python3 -m pip list | grep amd
   python3 -m pip uninstall amdsmi

   # Modern: 7.14+ system package, remove it with your package manager
   sudo apt remove amd-smi-lib   # or: sudo dnf remove amd-smi-lib
   ```

2. Install the AMD SMI Python library. Pick **one** of these paths:

   - **System package** — install or reinstall `amd-smi-lib` from your target
     ROCm instance. The package installs the wrapper into the system Python's
     `site-packages`, so `import amdsmi` resolves it directly. `sudo` is
     usually required.
   - **Tarball** — keeps each version self-contained in its own tree and
     installs nothing system-wide, which is the cleanest way to keep instances
     from colliding. See {ref}`Install from a tarball <install_tarball>`.
   - **PyPI wheel** — isolated inside a virtual environment and independent of
     `/opt/rocm`. See
     {ref}`Install the Python bindings from PyPI <install_pypi>`.

   See `py-interface/README.md` in the source tree, or
   [Packaging and install paths](../packaging.md) for the full install-paths
   matrix, coexistence rules, and the `AMDSMI_LIB_OVERRIDE` override.

3. Confirm the right copy is on your Python path. With several ROCm instances
   installed, the module path matters more than the import succeeding:

   ```bash
   python3 -c "import amdsmi; print(amdsmi.__file__)"
   ```
