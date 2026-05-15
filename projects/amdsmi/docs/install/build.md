---
myst:
  html_meta:
    "description lang=en": "How to build AMD SMI from source."
    "keywords": "system, management, interface, contribute, contributing, ROCm, develop, testing"
---

# Building AMD SMI

This section describes the prerequisites and steps to build AMD SMI from source.

(build_reqs)=
## Required software

To build the AMD SMI library, the following components are required. Note that
the software versions specified were used during development; earlier
versions are not guaranteed to work.

* CMake (v3.15.0 or later) -- `python3 -m pip install cmake`
* g++ (v5.4.0 or later)
* libdrm-dev (for Ubuntu and Debian)
* libdrm-devel (for RPM-based distributions)

In order to build the AMD SMI Python package, the following components are
required:

* Python (3.6.8 or later)
* virtualenv -- `python3 -m pip install virtualenv`

## Build steps

1. Clone the AMD SMI repository to your local Linux machine.

   ```shell
   git clone https://github.com/ROCm/amdsmi.git
   ```

2. The default installation location for the library and headers is `/opt/rocm`.
   Before installation, any old ROCm directories should be deleted:

   * `/opt/rocm`
   * `/opt/rocm-<version_number>`

3. Build the library by following the typical CMake build sequence (run as root
   user or use `sudo` before `make install` command); for instance:

   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j $(nproc)
   make install
   ```

   The built library is located in the  `build/` directory. To build the `rpm`
   and `deb` packages use the following command:

   ```bash
   make package
   ```

(rebuild_py_wrapper)=
## Rebuild the Python wrapper

The Python wrapper for the AMD SMI library is found in the [auto-generated
file](#py_lib_fs) `py-interface/amdsmi_wrapper.py`. It is essential to
regenerate this wrapper whenever there are changes to the C++ API. It is not
regenerated automatically.

To regenerate the wrapper, use the following command.

```shell
./update_wrapper.sh
```

After this command, the file in `py-interface/amdsmi_wrapper.py` will be updated
on compile.

```{note}
You need Docker installed on your system to regenerate the Python wrapper.
```

(build_tests)=
## Build the tests

To verify the build and capabilities of AMD SMI on your system, as well as to
see practical examples of its usage, you can build and run the available [tests
in the repository](https://github.com/ROCm/amdsmi/tree/amd-staging/tests).
Follow these steps to build the tests:

```bash
mkdir -p build
cd build
cmake -DBUILD_TESTS=ON ..
make -j $(nproc)
```

(run_tests)=
### Run the tests

Once the tests are [built](#build_tests), you can run them by executing the
`amdsmitst` program. The executable can be found at `build/tests/amd_smi_test/`.

(build_docs)=
## Build the docs

To build the documentation, follow the instructions at [Building
documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html).

