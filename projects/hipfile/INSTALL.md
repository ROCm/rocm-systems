# Install guide for hipFile

## Quick Install Guide

hipFile depends on ROCm 7.2. Install ROCm and amdgpu-dkms, as described in the [ROCm quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

On Ubuntu 24.04, the installation process is as follows. First, install a couple of needed packages.

```
sudo apt install libmount-dev wget
```

Then, install the nightly hipFile packages.

```
wget https://github.com/ROCm/hipFile/releases/download/nightly/hipfile_0.2.0.70200-nightly.9999.24.04_amd64.deb
wget https://github.com/ROCm/hipFile/releases/download/nightly/hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb
sudo dpkg -i hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb hipfile_0.2.0.70200-nightly.9999.24.04_amd64.deb
```

We can verify that the HIP libraries and kernel support AIS by running ais-check.

```
/opt/rocm/bin/ais-check
```

Successful output from ais-check will show True for several checks:

```
AIS support in:
        Kernel P2PDMA support   : True
        HIP runtime             : True
        amdgpu                  : True
```

hipFile currently works on local NVMe disks. To set up a new NVMe device we can partition it, create a filesystem on the partition, and mount it. Below we use the device `/dev/nvme1n1`. Make sure you use the correct device in your system.

```
sudo apt install gdisk
# Add partition spanning entire device
sudo sgdisk -n 1:0:0 /dev/nvme1n1
# Create and mount filesystem
sudo mkfs.ext4 /dev/nvme1n1p1
sudo mkdir /mnt/ext4
sudo mount /dev/nvme1n1p1 /mnt/ext4 -o data=ordered
# Create directory accessible by current user
sudo mkdir /mnt/ext4/"$USER"
sudo chown "$USER": /mnt/ext4/"$USER"
```

Now, we can download and compile the aiscp test program.

```
wget https://raw.githubusercontent.com/ROCm/hipFile/refs/heads/develop/examples/aiscp/aiscp.cpp
amdclang++ -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -I/opt/rocm/include -lamdhip64 -lhipfile aiscp.cpp -o aiscp
```

To verify the fast path is working, copy a file with compatibility mode disabled. This should run successfully if the source and destination paths are on filesystems supporting `O_DIRECT`.

```
# Create a random input file
dd if=/dev/urandom of=/mnt/ext4/"$USER"/source bs=4K count=16
# Copy file
HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/ext4/"$USER"/source /mnt/ext4/"$USER"/dest
md5sum /mnt/ext4/"$USER"/source /mnt/ext4/"$USER"/dest
```

## Building hipFile

> [!NOTE]
> hipFile is alpha software that has undergone testing on limited hardware. It may not work on your system at this time.

Supported hardware:
* hipFile should work with the [GPUs that ROCm supports](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Storage is currently limited to local NVMe drives
* hipFile has only been tested on AMD CPUs/systems

Supported compilers: amdclang++ (preferred), clang++, g++

If CMake can't find hipcc/nvcc, you can set `-DCMAKE_HIP_COMPILER=<path>`

Supported platforms: Linux (Windows may be supported in a future release)

Supported filesystems: Only ext4 and xfs are supported at this time. ext4 must be mounted with the default data=ordered mount option.

Targeting NVIDIA requires cuFile to be installed

The kernel must have `CONFIG_PCI_P2PDMA` enabled

Multipath NVMe devices are not supported at this time. If you are using a multipath-supporting device, you may need to disable multipath in the nvme\_core kernel driver. On Ubuntu 24.04, this can be done by running the following:

```
sudo bash -c 'echo "options nvme_core multipath=N" > /etc/modprobe.d/nvme_core.conf'
sudo update-initramfs -c -k all
sudo systemctl reboot
```

### Prerequisites

#### Build Tools
* CMake >= 3.21
* C++ >= 17 (tested w/ clang++ & g++, we don't use GNU extensions)
* The `ais-check` script requires Python 3.x

#### AMD Components
* ROCm >= 7.2
    * `hip-dev` (Debian/Ubuntu)
    * `hip-devel` (RHEL/Fedora/openSUSE)
* amdgpu-dkms >= 30.20.1

#### Other Packages
* Boost.Program\_options
    * `libboost-program-options-dev` (Debian/Ubuntu)
    * `libboost-program-options-devel` (RHEL/Fedora/openSUSE)
* libmount
    * `libmount-dev` (Debian/Ubuntu)
    * `libmount-devel` (RHEL/Fedora/openSUSE)

### Configure

You do not need to set the `HIP_PLATFORM` environment variable, as
that will be set by CMake.

`cmake -DCMAKE_HIP_PLATFORM=(amd|nvidia) <options> <path/to/source>`

Options
|Option|Default|Purpose|
|------|-------|-------|
|AIS\_BUILD\_DOCS|OFF|Build API documentation (requires Doxygen)|
|AIS\_INSTALL\_EXAMPLES|ON|Install example programs|
|AIS\_USE\_CLANG\_TIDY|OFF|Run the `clang-tidy` tool (clang only)|
|AIS\_USE\_CODE\_COVERAGE|OFF|Generate code coverage information when tests are run (clang only)|
|AIS\_USE\_IWYU|OFF|Run the `include-what-you-use` tool (clang only)|
|AIS\_WARN\_UNSAFE\_BUFFER\_OPS|ON|Scan code for unsafe buffer operations (clang only, OFF for others)|

Sanitizer options (clang only)
|Option|Default|Purpose|
|------|-------|-------|
|AIS\_USE\_SANITIZERS|OFF|Build with -fsanitize=address,float-divide-by-zero,integer,leak,local-bounds,nullability,undefined,vptr|
|AIS\_USE\_THREAD\_SANITIZER|OFF|Build with -fsanitize=thread (not compatible with AIS\_USE\_SANITIZERS)|

### Build
`cmake --build .`

### Run tests

`ctest .` will run all tests.

You can filter on test labels using the `-L` option. We currently have `hipfile` and `internal` labels for the public API and its internals, respectively, and `unit`, `system`, and `stress` labels for the different test types. Multiple `-L` options form an *and*. A `|` can be used for an *or*.

`ctest . -L internal -L unit` runs all the internal library unit tests.
`ctest . -L 'system|unit'` runs all system or unit tests.

You can filter on test names using the `-R` option. This option supports wildcards.

`ctest . -R HipFileBuffer.get_buffer_makes_temporary_buffer` runs the specified test.
`ctest . -R 'HipFileBuffer*'` runs all tests that start with HipFileBuffer.

### Install and package
The hipFile library can be installed with CMake. The default
install prefix is still the CMake default of `/usr/local`.

`cmake --install .`

The libraries can also be packaged with CPack. .rpm, .deb. and .tar.gz
are all supported.

`cpack -G RPM`
`cpack -G DEB`
`cpack -G TGZ`

### Code coverage
The same version of Clang and LLVM must be used in order to generate
coverage results. `rocm-llvm-dev` can be installed to get the LLVM
version matching amdclang++.

PATH may need to be adjusted depending on the system configuration. The
following will add the default ROCm version of the tools to the path.
```
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:"$PATH"
```

To generate code coverage information, the `AIS_USE_CODE_COVERAGE`
flag must be enabled when configuring. Run the build and test steps,
then run the following to generate the summaries.
```
<path/to/repo>/util/llvm-coverage.sh
```

The results will be wrote to `<path/to/repo>/build`, in the
`coverage-report.txt` and `coverage-lines.txt` files.

### Documentation
The API documentation is built using Doxygen. To build it, use the
`AIS_BUILD_DOCS` option. This will build the documentation for any
libraries that have been configured. As a special case, configuring
the documentation without any library will build the documentation
for BOTH libraries, allowing for a docs-only build.

The documentation will be built with the libraries and appear in
`docs/(hip|roc)file`. We build HTML, XML, and LaTeX docs. If you
want a pdf, run `make pdf` in the `latex` directory, which will
create a file named refman.pdf that you can rename.

If you want to build the docs without compiling the libraries,
you can just build the `doc` target (if you've set `AIS_BUILD_DOCS`):

    `cmake --build . --target doc`
