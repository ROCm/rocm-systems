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
- AMD EPYC™ CPUs through the [esmi_ib_library](https://github.com/amd/esmi_ib_library)

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
  - See [amd_hsmp](https://github.com/amd/amd_hsmp) for more information.

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

1. Remove previous AMD SMI installation.

   ```shell
   python3 -m pip list | grep amd
   python3 -m pip uninstall amdsmi
   ```

2. Install the AMD SMI Python library from your target ROCm instance. Replace
   `<major>` and `<minor>` with the appropriate ROCm version.

   ```shell
   # Install from target ROCm instance
   cd /opt/rocm/core-<major>.<minor>/share/amd_smi
   python3 -m pip install --user .
   ```

   > **Note:** `sudo` may be required. On some systems, use `--break-system-packages` if pip installation fails.

3. You should now have the AMD SMI Python library in your Python path:

   ```shell-session
   ~$ python3
   Python 3.8.10 (default, May 26 2023, 14:05:08)
   [GCC 9.4.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import amdsmi
   >>>
   ```
