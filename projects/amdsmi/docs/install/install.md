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
| {ref}`ROCm Core SDK <install_rocm>` | Included | After you add `share/amd_smi` to `sys.path` | You want ROCm anyway. **Start here if you are unsure.** |
| {ref}`Standalone package <install_without_rocm>` | After you add `bin/` to `PATH` | After you add `share/amd_smi` to `sys.path` | You want AMD SMI without the rest of the ROCm libraries and tools |
| {ref}`Nightly build <install_nightly>` | Included in the virtual environment | After you add `share/amd_smi` to `sys.path` | You need a pre-release fix, or you are testing against an unreleased ROCm |
| {ref}`Tarball <install_tarball>` | After you add `bin/` to `PATH` | After you add `share/amd_smi` to `sys.path` | You have no root access, need versions side by side, or are on an air-gapped host |
| {ref}`PyPI wheel <install_pypi>` | Not included | Ready to use, but it loads the host's `libamd_smi.so` | You only script against the Python API and already have a ROCm library on the host |

:::{note}
Whichever method you pick, the machine you query still needs the `amdgpu` kernel
driver. See {ref}`Driver requirements <install_amdgpu_driver>`.
:::

Every ROCm-managed method above ships the Python module at
`<root>/share/amd_smi/amdsmi` and registers it with no interpreter, so
`import amdsmi` needs one line of setup regardless of which one you choose. The
procedure, and how to find `<root>` for your method, is in
{ref}`Make the Python module importable <install_python_module>`.

:::{note}
ROCm 7.2 and earlier are distributed from `repo.radeon.com` as a separate
package family, in which AMD SMI is the `amd-smi-lib` deb/rpm installed under
`/opt/rocm-<major>.<minor>.<patch>`. That package does register the module with
the system Python, so `import amdsmi` works straight after installing it, and it
drops an `/etc/ld.so.conf.d` entry for its library. The `amdrocm-amdsmi` package
described on this page belongs to the ROCm Core SDK and does neither.
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

**Use this when** you want ROCm on the machine anyway. This is the default path
and the one most users should take.

AMD SMI is included with most installations of the ROCm Core SDK on Linux.

For instructions, see {doc}`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

The SDK installs under a versioned prefix, `/opt/rocm/core-<major>.<minor>`, and
points `/opt/rocm/bin`, `/opt/rocm/lib` and `/opt/rocm/share` at the version it
selects. It also registers `/usr/bin/amd-smi` through `update-alternatives`, so
the CLI is on `PATH` with no further setup. The Python module is staged at
`/opt/rocm/share/amd_smi/amdsmi`; see
{ref}`Make the Python module importable <install_python_module>`.

(install_without_rocm)=
## Install AMD SMI standalone on Linux

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

   :::::{tab-set}
   ::::{tab-item} Debian-based distros
   ```bash
   sudo apt install amdrocm-amdsmi
   ```
   ::::
   ::::{tab-item} RHEL-based distros
   ```bash
   sudo dnf install amdrocm-amdsmi
   ```
   ::::
   ::::{tab-item} SLES
   ```bash
   sudo zypper install amdrocm-amdsmi
   ```
   ::::
   :::::

3. Prepend the `amd-smi` binary to your PATH. The package installs under the
   versioned prefix `/opt/rocm/core-<major>.<minor>`, and on its own it creates
   neither the `/opt/rocm/bin` link nor the `/usr/bin/amd-smi` alternative —
   those come from the rest of the ROCm Core SDK. Replace `<major>` and
   `<minor>` with the ROCm version you installed.

   ```bash
   export PATH="/opt/rocm/core-<major>.<minor>/bin${PATH:+:${PATH}}"
   ```

   To persist this across shells, append the line to your `~/.bashrc` (or
   equivalent shell config).

4. Verify your installation.

   ```bash
   amd-smi version
   ```

To use the Python API as well, see
{ref}`Make the Python module importable <install_python_module>`. The package
stages the module at `/opt/rocm/core-<major>.<minor>/share/amd_smi/amdsmi` and
installs nothing into any interpreter.

(install_nightly)=
## Install a nightly build

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
   pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
       "rocm[libraries,device-gfx942]"
   ```

   For the full list of `device-*` extras, the other release channels
   (stable, prerelease), and the legacy index that serves releases before
   ROCm 10.1, see
   [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#supported-python-device--install-extras).

3. Verify your installation:

   ```bash
   amd-smi version
   ```

   `pip` installs a console script for the CLI, so `amd-smi` is on `PATH` while
   the environment is active. The Python module is *not* installed as a
   top-level package: it is staged inside the `rocm-sdk-core` payload, at
   `<site-packages>/_rocm_sdk_core/share/amd_smi/amdsmi`. See
   {ref}`Make the Python module importable <install_python_module>`.

(install_tarball)=
## Install from a tarball

**Use this when** you cannot use a package manager: no root access, multiple
versions side by side, or an air-gapped host.

A tarball has no install step. Extracting it runs no package manager, no
`ldconfig`, and no pip, so nothing is registered with Python. The `amd-smi` CLI
and the `amdsmi` Python module both ship inside the archive and run from
wherever you extract it, so you point your tools at the tree instead of
installing out of it.

1. Download a ROCm Core tarball,
   `therock-dist-linux-<gpu-family>-<version>.tar.gz`, from the channel you
   want. Nightlies are at <https://nightly.repo.amd.com/rocm/core/tarball/>;
   [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
   lists the stable, prerelease and development channels.

2. Extract it into a directory of its own. The archive has no top-level
   directory — it expands straight into `bin/`, `lib/`, `share/` and the rest of
   a ROCm tree — so extracting it into a populated directory mixes it into
   whatever is already there. The examples below call the extraction directory
   `$AMDSMI_ROOT`.

   ```bash
   mkdir -p ~/rocm-tarball
   tar -xf therock-dist-linux-<gpu-family>-<version>.tar.gz -C ~/rocm-tarball
   export AMDSMI_ROOT=~/rocm-tarball

   ls "$AMDSMI_ROOT/lib/libamd_smi.so."*
   ls "$AMDSMI_ROOT/share/amd_smi/amdsmi/amdsmi_wrapper.py"
   ```

   Those two paths are what the CLI and the Python loader key on.

3. Run the CLI.

   ```bash
   export PATH="$AMDSMI_ROOT/bin${PATH:+:${PATH}}"
   amd-smi version
   ```

   You do not need `LD_LIBRARY_PATH`: `libamd_smi.so` is linked with an
   `$ORIGIN`-relative `RPATH`, so it finds its own dependencies inside the
   extracted tree.

   :::{warning}
   `amd-smi` searches `$ROCM_PATH/share/amd_smi` (or `$ROCM_HOME`) before the
   copy next to its own executable. If either variable points at another ROCm
   installation, the extracted CLI runs *that* installation's Python modules,
   and those modules load *that* installation's library — so `amd-smi version`
   reports a version you did not extract. Unset the variable, or set
   `ROCM_PATH="$AMDSMI_ROOT"`, before running the tarball's CLI.
   :::

4. Make the module importable. Follow
   {ref}`Make the Python module importable <install_python_module>` with
   `<root>` set to `$AMDSMI_ROOT`.

See [Packaging and install paths](../packaging.md) for the full precedence and
coexistence rules.

(install_pypi)=
## Install the Python bindings from PyPI

**Use this when** you want `import amdsmi` to work with no `sys.path` setup on a
host that already has an AMD SMI library, and you do not need the CLI.

This method installs the `amdsmi` module only. It does not provide the
`amd-smi` CLI, and it does not ship a native library: the published wheel is a
pure-Python `py3-none-any` package whose loader looks for `libamd_smi.so` under
`$ROCM_HOME`/`$ROCM_PATH`, then through the dynamic linker, then in
`/opt/rocm/lib`. A ROCm or AMD SMI installation providing that library must
already be on the host, as must the `amdgpu` kernel driver.

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
The PyPI release cadence is independent of ROCm, and the wheel is versioned by
the AMD SMI library version rather than the ROCm version. The published wheel
routinely lags the AMD SMI in a current ROCm installation, and it binds against
whatever `libamd_smi.so` the host provides — which is *not* guaranteed to match
the bindings it ships.

Compare the published version against `amd-smi version` on the host before
depending on it, and prefer one of the ROCm-managed methods above when ROCm is
already installed: they always pair the module with the library it was
generated from.
:::

(install_python_module)=
## Make the Python module importable

Every method except the {ref}`PyPI wheel <install_pypi>` *ships* the `amdsmi`
module without *installing* it: the files land at `<root>/share/amd_smi/amdsmi`,
and no package manager, `ldconfig` or pip step registers that directory with an
interpreter. `import amdsmi` therefore fails until you put its parent on
`sys.path`. The `amd-smi` CLI is unaffected — it adds its own `share/amd_smi` to
`sys.path` at startup.

`<root>` depends on the method:

| Method | `<root>` |
|---|---|
| {ref}`ROCm Core SDK <install_rocm>` | `/opt/rocm` (a link to the selected `/opt/rocm/core-<major>.<minor>`) |
| {ref}`Standalone package <install_without_rocm>` | `/opt/rocm/core-<major>.<minor>` |
| {ref}`Nightly build <install_nightly>` | `<site-packages>/_rocm_sdk_core` in the virtual environment |
| {ref}`Tarball <install_tarball>` | The directory you extracted into |

Pick whichever scope suits you, substituting your `<root>`:

**Current shell** — one variable, no files written:

```bash
export PYTHONPATH="<root>/share/amd_smi${PYTHONPATH:+:${PYTHONPATH}}"
```

`PYTHONPATH` outranks every install location, so this also overrides an
installed `amd-smi-lib` package or a pip `amdsmi` wheel for any Python started
from this shell. Append the `export` line to your `~/.bashrc` (or equivalent) to
persist it.

**One virtual environment** — no environment variable, nothing else on the host
affected:

```bash
python3 -m venv .venv
source .venv/bin/activate
echo "<root>/share/amd_smi" > "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/amdsmi.pth"
```

:::{important}
Reference the module where the installation put it. Do not copy it elsewhere.

The wrapper finds the library at `<root>/lib/libamd_smi.so.<MAJOR>` by walking
up from its own file, so it pairs with its own tree's library only while it
stays at `<root>/share/amd_smi/amdsmi`. Copy it into `site-packages` and that
relative path stops resolving: the loader falls through to a bare
`libamd_smi.so.<MAJOR>` lookup and binds whatever the dynamic linker finds
first. On a host that already has ROCm installed that is `/opt/rocm/lib`, so you
silently get a different library than the one you installed; on a host without
one the import fails outright. The directory also ships no `pyproject.toml` or
`setup.py`, so `pip install` cannot consume it in place either.
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
     usually required. This is the only delivery that claims a system-wide
     `import amdsmi`, which also makes it the one that collides across
     instances.
   - **Tarball** — keeps each version self-contained in its own tree and
     installs nothing system-wide, which is the cleanest way to keep instances
     from colliding. See {ref}`Install from a tarball <install_tarball>`.
   - **PyPI wheel** — isolated inside a virtual environment, but it loads a
     library from the host, so set `ROCM_PATH` to the instance you want it to
     bind against. See
     {ref}`Install the Python bindings from PyPI <install_pypi>`.

   See `py-interface/README.md` in the source tree, or
   [Packaging and install paths](../packaging.md) for the full install-paths
   matrix, coexistence rules, and the `AMDSMI_LIB_OVERRIDE` override.

3. Confirm the right copy is on your Python path. With several ROCm instances
   installed, the module path matters more than the import succeeding:

   ```bash
   python3 -c "import amdsmi; print(amdsmi.__file__)"
   ```
