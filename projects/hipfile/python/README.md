# hipFile Python Bindings

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
3. Build the C hipFile library. See [INSTALL.md](../INSTALL.md).
4. Build & Install the hipFile package.
```bash
(.venv) $ pip install -e . -Ccmake.define.HIPFILE_INCLUDE_DIR=../include -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include
```

This will install an editable version of hipFile in your virtual environment.
It is editable in the sense that any changes you make to the hipFile <ins>Python</ins>
source code will be immediately available for any tests/scripts that use the hipFile library.
Any changes to the <ins>Cython</ins> source code will require a rebuild step.