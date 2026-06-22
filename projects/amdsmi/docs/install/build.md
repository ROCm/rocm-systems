---
myst:
  html_meta:
    "description lang=en": "How to build AMD SMI from source."
    "keywords": "system, management, interface, contribute, contributing, ROCm, develop, testing"
---

# Build AMD SMI from source

To build AMD SMI as part of the ROCm Core SDK, see [TheRock build
instructions](https://github.com/ROCm/TheRock/blob/main/docs/development/README.md).
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build AMD SMI standalone using the following
instructions.

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

1. Clone the rocm-systems repository to your local Linux machine
   and sparse-checkout the AMD SMI project.

   ```shell
   git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
   git -C rocm-systems sparse-checkout set projects/amdsmi
   cd rocm-systems/projects/amdsmi
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
in the repository](https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi/tests).
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

The [C/C++ API reference](../reference/amdsmi-cpp-api/index.md) is generated
with [Doxygen 1.15.0](https://www.doxygen.nl/manual/changelog.html#log_1_15_0),
which must be installed separately and available on your PATH.

1. Create a Python virtual environment and install documentation dependencies.

   ```bash
   # From rocm-systems/projects/amdsmi
   python3.12 -m venv docs/.venv
   source docs/.venv/bin/activate
   pip install -r docs/sphinx/requirements.txt
   ```

2. Use the following command to build the documentation using
   [Sphinx](https://www.sphinx-doc.org/en/master/).

   ```bash
   python -m sphinx docs docs/_build -j auto -E -v
   ```

3. Open `docs/_build/index.html` in your web browser to view the
   documentation. To serve the site locally instead, run:

   ```bash
   python -m http.server -d docs/_build
   # Go to http://localhost:8000 in your browser
   ```

For related information, see [Building
documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html).
