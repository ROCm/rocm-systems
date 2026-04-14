# hipFile Python Bindings

Python bindings for the [hipFile](https://github.com/ROCm/rocm-systems/tree/develop/projects/hipFile) C library.

> [!CAUTION]
> These bindings in particular are *experimental* and the API will change.

## Building & Installing

1. Setup a Python virtual environment.
    ```bash
    $ python3 -m venv .venv
    ```
2. Activate the Python virtual environment.
    ```bash
    $ source .venv/bin/activate
    ```
3. Build or install the C hipFile library. See [INSTALL.md](../INSTALL.md).
4. Build the hipFile package.\
   The build script will automatically look for the hipFile C header
   and shared library in the CMake build directory first, and fallback
   to the default ROCm install location. The search path can be
   overridden by setting `HIPFILE_INCLUDE_DIR`, `HIPFILE_LIBRARY`,
   or `HIP_INCLUDE_DIR` via the build arg `-Ccmake.define.<KEY>=<VALUE>`
   or by setting these as an environment variable.
    ```bash
    (.venv) $ cd projects/hipfile/python
    (.venv) $ python -m build --wheel
    ```
    Or to install an editable version of the Python package:
    ```bash
    (.venv) $ cd projects/hipfile/python
    (.venv) $ pip install -e .
    ```

An editable Python package is a development package that allows for
live changes to the <ins>Python</ins> source code to be applied
immediately without needing to rebuild or reinstall the Python
package under development. Any changes to the <ins>Cython</ins>
source code will require a rebuild step.
