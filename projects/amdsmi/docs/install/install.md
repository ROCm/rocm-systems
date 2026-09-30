---
myst:
  html_meta:
    "description lang=en": "How to install AMD SMI libraries and CLI tool."
    "keywords": "system, management, interface, cpu, gpu, hsmp, versions"
---

# Install the AMD SMI library and CLI tool

This page describes the system requirements for AMD SMI and explains how to
install the AMD SMI library, Python interface, and `amd-smi` CLI on Linux.

## Supported platforms

AMD SMI supports:

- {ref}`AMD GPUs <rocm:release-supported-hw>` on Linux bare metal systems
- AMD GPUs in Linux virtual machine guests
- AMD EPYC™ CPUs through the
  [esmi_ib_library](https://github.com/amd/esmi_ib_library) (requires an HSMP
  kernel driver; see {ref}`install_amdgpu_driver`)

For AMD SMI on Linux SR-IOV hosts, refer to
the [AMD SMI for Virtualization documentation](https://instinct.docs.amd.com/projects/amd-smi-virt/en/latest/index.html).

(install_reqs)=
## Requirements

Before installing AMD SMI, make sure your system meets the following
requirements.

(install_amdgpu_driver)=
### Driver requirements

To run AMD SMI, the following kernel drivers need to be loaded on your system:

- The `amdgpu` driver
  - On native Linux, `amdsmi_init()` and the `amd-smi` CLI find GPUs only
    while it is loaded. The `/sys/module/amdgpu` directory exists when it is.
  - The driver included in your distribution's kernel works. AMD also
    distributes the driver as the `amdgpu-dkms` package, which builds it out
    of tree with DKMS; for installation instructions, see the [AMD GPU Driver
    (amdgpu)
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
- The `amd_hsmp` or `hsmp_acpi` kernel driver (AMD EPYC CPUs only, optional)
  - Required for `amdsmi_init(AMDSMI_INIT_AMD_CPUS)` and `amd-smi` CPU
    commands. HSMP must also be enabled in the BIOS.
  - Both drivers are part of the upstream Linux kernel: `amd_hsmp` since
    Linux 5.18 and `hsmp_acpi` since Linux 6.13. For older kernels, see
    [amd_hsmp](https://github.com/amd/amd_hsmp).
  - Without it, CPU discovery is skipped non-fatally and only GPU and NIC data
    is reported.

Also confirm that your Linux kernel version matches the system requirements
described in {ref}`Operating system support <rocm:release-supported-os>`.

For the experimental WSL backend, which uses `/dev/dxg` instead of `amdgpu`,
see [Using AMD SMI under WSL](../how-to/amdsmi-wsl-mode.md).

### Interface prerequisites

The following prerequisites apply to the AMD SMI library interfaces:

- Python interface and `amd-smi` CLI:
  - Python 3.6.8 or later
  - 64-bit Python
- Go interface:
  - Go 1.20 or later

::::{note}
When installing an `amd-smi-lib` package older than ROCm 7.14 on Azure Linux
3, you might encounter the `ModuleNotFoundError: No module named
'more_itertools'` warning. Those packages ran `pip` with `setuptools` and
`wheel` during installation, which requires `more_itertools`. ROCm 7.14 and
later packages no longer run `pip` during installation and are not affected.
For older packages, use the following command before installation:

```bash
sudo python3 -m pip install more_itertools
```
::::

(install_rocm)=
## Install the ROCm Core SDK

AMD SMI is included with most installations of the ROCm Core SDK on Linux.

For instructions, see {doc}`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

(install_without_rocm)=
## Install AMD SMI standalone on Linux

Alternatively, if you want to install AMD SMI without additional ROCm libraries
and tools, install the `amdrocm-amdsmi` package. This includes AMD SMI and
ROCm system dependencies.

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

3. Add the `amd-smi` binary directory to your `PATH`; it is not on `PATH` by
   default. To use the Python library, also add its directory to
   `PYTHONPATH`; this package does not install `amdsmi` into the system
   Python's `site-packages`. Replace `<major>` and `<minor>` with the
   appropriate ROCm version.

   ```bash
   export PATH="/opt/rocm/core-<major>.<minor>/bin${PATH:+:${PATH}}"
   export PYTHONPATH="/opt/rocm/core-<major>.<minor>/share/amd_smi${PYTHONPATH:+:${PYTHONPATH}}"
   ```

   To persist these across shells, append the lines to your `~/.bashrc` (or
   equivalent shell config).

4. Verify your installation.

   ```bash
   amd-smi version
   python3 -c "import amdsmi; print(amdsmi.amdsmi_get_lib_version())"
   ```

(install_nightly)=
## Install a nightly build

Nightly builds of the ROCm Core SDK (including AMD SMI) are published by
[TheRock](https://github.com/ROCm/TheRock) to a unified pip index.

1. Create and activate a Python virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install ROCm. AMD SMI is part of the `rocm-sdk-core` package, which the
   `rocm` package always installs, so no extras are needed for `amd-smi`:

   ```bash
   pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ rocm
   ```

   To also install the libraries and device code for your GPU, add the
   matching extras, for example `"rocm[libraries,device-gfx942]"` for gfx942
   (MI300X / MI325X). For the full list of `device-*` extras and other release
   channels, see
   [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#supported-python-device--install-extras).

3. Verify your installation:

   ```bash
   amd-smi version
   ```

   The CLI is added to the virtual environment's `PATH`, but the `amdsmi`
   Python package remains under the SDK's `share/amd_smi` directory. To use
   it directly, add that directory to `PYTHONPATH`:

   ```bash
   AMDSMI_PYTHON_DIR="$(python3 -c 'from pathlib import Path; import _rocm_sdk_core; print(Path(_rocm_sdk_core.__file__).resolve().parent / "share/amd_smi")')"
   export PYTHONPATH="${AMDSMI_PYTHON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
   python3 -c "import amdsmi; print(amdsmi.amdsmi_get_lib_version())"
   ```

   Set this after activating the target virtual environment; a `PYTHONPATH`
   pointing at another ROCm installation would override its Python package.

## Optional and advanced installation

Use these optional procedures for CLI autocompletion and advanced setups, such
as systems with multiple ROCm instances.

### Enable CLI autocompletion

The `amd-smi` CLI application supports bash autocompletion through
`argcomplete`. For a native package, install it with your distribution's
package manager:

```shell
sudo apt install python3-argcomplete      # Debian, Ubuntu
sudo dnf install python3-argcomplete      # RHEL, AlmaLinux, Rocky Linux
sudo zypper install python3-argcomplete   # SLES, openSUSE
```

For a pip-installed ROCm SDK, activate its virtual environment and install
`argcomplete` there:

```shell
python3 -m pip install argcomplete
```

Register the command in your current Bash shell. Some distributions,
including Ubuntu 22.04, suffix the helper name with `3`:

```shell
if command -v register-python-argcomplete >/dev/null 2>&1; then
    eval "$(register-python-argcomplete amd-smi)"
else
    eval "$(register-python-argcomplete3 amd-smi)"
fi
```

To persist registration, add that block to your `~/.bashrc` after any
virtual environment activation. Explicit registration works with both the
native script and TheRock's pip launcher. The `amd-smi-lib` package can also
activate the global hook during installation, but it recognizes only native
CLI builds carrying the `PYTHON_ARGCOMPLETE_OK` marker.

(install-manual-py-lib)=
### Install the Python library for multiple ROCm instances

Installing multiple versions of ROCm on the same system can cause
`import amdsmi` to load a different AMD SMI version than the `amd-smi` CLI or
the library you expect. Make sure only one copy of the `amdsmi` Python package
is visible to your Python interpreter.

The `amdsmi` Python package reaches your interpreter in one of these ways:

- The `amd-smi-lib` rpm/deb package installs it into the system Python's
  `site-packages`. Before ROCm 7.14, the package ran `pip install` during
  installation instead, which leaves a pip-registered copy to uninstall
  manually.
- The `amdrocm-amdsmi` package and ROCm pip installs from TheRock install it
  only under `share/amd_smi`, which Python uses only when it is on
  `PYTHONPATH`.
- `python3 -m pip install amdsmi` installs the pip wheel.

1. Remove other AMD SMI Python installations.

   ```shell
   # Legacy: pre-7.14 amd-smi-lib packages and any user pip install
   python3 -m pip list | grep amd
   python3 -m pip uninstall amdsmi

   # 7.14+ amd-smi-lib package: remove it with your package manager
   sudo apt remove amd-smi-lib   # or: sudo dnf remove amd-smi-lib
   ```

   Also check that `PYTHONPATH` does not point at the `share/amd_smi`
   directory of another ROCm installation.

2. Install the AMD SMI Python library. Pick **one** of these paths:

   - **`amd-smi-lib`**: install or reinstall `amd-smi-lib` from your target
     ROCm instance with `sudo`.
   - **`amdrocm-amdsmi`**: set `PYTHONPATH` to the `share/amd_smi` directory
     of your target ROCm instance, as described in {ref}`install_without_rocm`.
   - **Pip wheel** (recommended for venvs, containers, or when ROCm is not
     installed on the host but the amdgpu kernel driver is present):

     ```shell
     python3 -m pip install amdsmi
     ```

     The wheel ships its own SONAME-renamed `libamd_smi_python.so` next to
     the wrapper, so it does not depend on `/opt/rocm` being present.

   See `py-interface/README.md` in the source tree, or
   [Packaging and install paths](../packaging.md) for the full install-paths
   matrix, coexistence rules, and the `AMDSMI_LIB_OVERRIDE` override.

   For pip, use `--break-system-packages` only if installing into a non-venv
   Python that PEP 668 has marked externally managed.

3. You should now have the AMD SMI Python library in your Python path:

   ```shell-session
   ~$ python3
   Python 3.8.10 (default, May 26 2023, 14:05:08)
   [GCC 9.4.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import amdsmi
   >>>
   ```
