---
myst:
  html_meta:
    "description lang=en": "How to install AMD SMI libraries and CLI tool."
    "keywords": "system, management, interface, cpu, gpu, hsmp, versions"
---

# Install the AMD SMI library and CLI tool

This section describes how to install the AMD SMI library, Python interface,
and command line tool either as part of the
{doc}`ROCm software stack <rocm:what-is-rocm>` -- or manually.

(install_reqs)=
## Requirements

The following are required to install and use the AMD SMI library through its language interfaces and CLI.

* The `amdgpu` driver must be loaded for AMD SMI initialization to work. See
  [Install the amdgpu driver](#install_amdgpu_driver).

* Export `LD_LIBRARY_PATH` to the `amdsmi` installation directory.

  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64
  ```

### Supported platforms

The AMD SMI library supports Linux bare metal and Linux virtual machine guests
for AMD GPUs, and supports AMD EPYC™ CPUs via the
[esmi_ib_library](https://github.com/amd/esmi_ib_library).
To use AMD SMI for virtualization on Linux SR-IOV hosts, refer to
the [AMD SMI for Virtualization documentation](https://instinct.docs.amd.com/projects/amd-smi-virt/en/latest/index.html).

AMD SMI library can run on AMD ROCm supported platforms. Refer to
{doc}`System requirements (Linux) <rocm-install-on-linux:reference/system-requirements>`
for more information.
<!--https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html-->

To run the AMD SMI library, the `amdgpu` driver and the `amd_hsmp` or `hsmp_acpi` driver need to be installed. Optionally, `libdrm` can be installed to query firmware
information and hardware IPs.

### Python interface and CLI tool prerequisites

* Python version 3.6.8 or greater (64-bit)

::::{note}
During the driver installation process on Azure Linux 3, you might encounter the `ModuleNotFoundError: No module named 'more_itertools'` warning. This warning is a result of the reintroduction of `python3-wheel` and `python3-setuptools` dependencies in the CMake of AMD SMI, which requires `more_itertools` to build these Python libraries. This issue will be fixed in a future ROCm release. As a workaround, use the following command before installation:

```
sudo python3 -m pip install more_itertools
```
::::

### Go interface prerequisites

* Go version 1.20 or greater

(install_amdgpu_driver)=
## Install the amdgpu driver

```{note}
As of ROCm 7.0.0, the `amdgpu` driver is distributed separately from the ROCm
software stack. See
{doc}`rocm-install-on-linux:reference/user-kernel-space-compat-matrix` for
driver to ROCm user space compatibility information.
```

Confirm that your Linux kernel version matches the system requirements described in
{ref}`rocm-install-on-linux:supported_distributions`.

For up-to-date installation instructions, see the [AMD GPU Driver (amdgpu)
documentation](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/install/detailed-install/prerequisites.html).

(install_amdgpu_rocm)=
## Install AMD SMI with ROCm

AMD SMI is included as a core package in the ROCm software stack as part of the
`rocm-developer-tools` meta package. See [ROCm runtime
packages](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/package-manager-integration.html#id3)
for more information.

```{note}
The `amdgpu-install` script is no longer the recommended way to install ROCm.
Install using your supported Linux distribution's package manager instead.
```

For up-to-date installation instructions via package manager, see {doc}`ROCm
installation for Linux <rocm-install-on-linux:install/prerequisites>`.

After installing the `amdgpu` driver and ROCm, verify your AMD SMI installation:

```shell
amd-smi
```

(install_without_rocm)=
## Install AMD SMI without ROCm

The following are example steps to install the AMD SMI libraries and CLI tool on
Ubuntu 22.04.

1. Install the library.

   ```shell
   sudo apt install amd-smi-lib
   ```

2. Add the installation directory to your PATH. If installed with ROCm, ignore
   this step.

   ```shell
   export PATH="${PATH:+${PATH}:}~/opt/rocm/bin"
   ```

3. Verify your installation.

   ```shell
   amd-smi --help
   ```

## Optionally enable CLI autocompletion

The `amd-smi` CLI application supports autocompletion. If `argcomplete` is not
installed and enabled already, do so using the following commands.

```shell
python3 -m pip install argcomplete
activate-global-python-argcomplete --user
# restart shell to enable
```

(install-manual-py-lib)=
## Install the Python library for multiple ROCm instances

If {doc}`multiple ROCm versions are installed
<rocm-install-on-linux:install/install-methods/multi-version-install-index>` and you
are not using `pyenv`, uninstall previous versions of AMD SMI before installing
the desired version from your ROCm instance.

### Manually install the Python library

Multiple ROCm installations may cause `amd-smi` failures.
Installing multiple versions of ROCm on the same system can result in the `amd-smi` CLI not functioning correctly.

1. Remove previous AMD SMI installation.

   ```shell
   python3 -m pip list | grep amd
   python3 -m pip uninstall amdsmi
   ```

2. Install the AMD SMI Python library from your target ROCm instance.

   ```shell
   # Install from target ROCm instance
   cd /opt/rocm/share/amd_smi
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
