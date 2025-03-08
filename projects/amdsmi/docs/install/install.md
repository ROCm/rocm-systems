---
myst:
  html_meta:
    "description lang=en": "How to install AMD SMI libraries and CLI tool."
    "keywords": "system, management, interface, cpu, gpu, hsmp, versions"
---

# AMD SMI library and CLI tool

This section describes how to install the AMD SMI library, Python interface,
and command line tool either as part of the
{doc}`ROCm software stack <rocm:what-is-rocm>` -- or manually.

(install_reqs)=
## Requirements

The following are required to install and use the AMD SMI library through its language interfaces and CLI.

* The `amdgpu` driver must be loaded for AMD SMI initialization to work.

* Export `LD_LIBRARY_PATH` to the `amdsmi` installation directory.

  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64
  ```

### Python interface and CLI tool prerequisites

* Python version 3.6.8 or greater (64-bit)

* Modules:
  * `python3-wheel`

  * `python3-setuptools`

### Go interface prerequisites

* Go version 1.20 or greater

### Supported platforms

At initial release, the AMD SMI library will support Linux bare metal and Linux
virtual machine guest for AMD GPUs. In a future release, the library will be
extended to support AMD EPYC™ CPUs.

AMD SMI library can run on AMD ROCm supported platforms, refer to
{doc}`System requirements (Linux) <rocm-install-on-linux:reference/system-requirements>`
for more information.
<!--https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html-->

To run the AMD SMI library, the `amdgpu` driver and the `amd_hsmp` driver need
to be installed. Optionally, `libdrm` can be installed to query firmware
information and hardware IPs.

(install_amdgpu_rocm)=
## Install amdgpu driver and AMD SMI with ROCm

<!--https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html-->
1. Get the `amdgpu-install` installer following the instructions for your
   Linux distribution at {doc}`rocm-install-on-linux:install/amdgpu-install`.

   See the following example; your desired ROCm release and install URL may be
   different.

   ```shell
   sudo apt update
   wget https://repo.radeon.com/amdgpu-install/6.2.2/ubuntu/noble/amdgpu-install_6.2.60202-1_all.deb
   sudo apt install ./amdgpu-install_6.2.60202-1_all.deb
   ```

2. Use `amdgpu-install` to install the `amdgpu` driver and ROCm packages with
   AMD SMI included.

   ``` shell
   sudo amdgpu-install --usecase=rocm
   ```

   The `amdgpu-install --usecase=rocm` option triggers both an `amdgpu` driver
   update and AMD SMI packages to be installed on your device.

3. Verify your installation.

   ```shell
   amd-smi --help
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
<rocm-install-on-linux:install/install-methods/multi-version-install>` and you
are not using `pyenv`, uninstall previous versions of AMD SMI before installing
the desired version from your ROCm instance.

### Manually install the Python library

The following are example AMD SMI installation steps on Ubuntu 22.04 without
ROCm.

1. Remove previous AMD SMI installation.

   ```shell
   python3 -m pip list | grep amd
   python3 -m pip uninstall amdsmi
   ```

2. Install the AMD SMI Python library from your target ROCm instance.

   ```shell
   apt install amd-smi-lib
   cd /opt/rocm/share/amd_smi
   python3 -m pip install --upgrade pip
   python3 -m pip install --user .
   ```

3. You should now have the AMD SMI Python library in your Python path:

   ```shell-session
   ~$ python3
   Python 3.8.10 (default, May 26 2023, 14:05:08)
   [GCC 9.4.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import amdsmi
   >>>
   ```
