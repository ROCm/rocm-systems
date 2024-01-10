# Python Support

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 3
```

[OmniTrace](https://github.com/AMDResearch/omnitrace) supports profiling Python code at the source-level and/or the script-level.
Python support is enabled via the `OMNITRACE_USE_PYTHON` and the `OMNITRACE_PYTHON_VERSIONS="<MAJOR>.<MINOR>` CMake options.
Alternatively, to build multiple python versions, use `OMNITRACE_PYTHON_VERSIONS="<MAJOR>.<MINOR>;[<MAJOR>.<MINOR>]"`,
and `OMNITRACE_PYTHON_ROOT_DIRS="/path/to/version;[/path/to/version]"` instead of `OMNITRACE_PYTHON_VERSION`.
When building multiple Python versions, the length of the `OMNITRACE_PYTHON_VERSIONS` and `OMNITRACE_PYTHON_ROOT_DIRS` lists must
be the same size.

> ***When using omnitrace for Python, the Python interpreter major and minor version (e.g. 3.7) must match the interpreter major and minor version***
> ***used when compiling the Python bindings, i.e. when building omnitrace, a `libpyomnitrace.<IMPL>-<VERSION>-<ARCH>-<OS>-<ABI>.so` will be generated***
> ***where `IMPL` is the Python implementation, `VERSION` is the major and minor version, `ARCH` is the architecture,***
> ***`OS` is the operating system, and `ABI` is the application binary interface; Example: `libpyomnitrace.cpython-38-x86_64-linux-gnu.so`.***

## Getting Started

The omnitrace Python package is installed in `lib/pythonX.Y/site-packages/omnitrace`. In order to ensure the Python interpreter can find the omnitrace package,
add this path to the `PYTHONPATH` environment variable, e.g.:

```bash
export PYTHONPATH=/opt/omnitrace/lib/python3.8/site-packages:${PYTHONPATH}
```

If using either the `share/omnitrace/setup-env.sh` script or the modulefile in `share/modulefiles/omnitrace`, prefixing the `PYTHONPATH`
environment variable is automatically handled.

## Running OmniTrace on a Python Script

OmniTrace provides an `omnitrace-python` helper bash script which effectively handles ensuring `PYTHONPATH` is properly set and the correct python interpreter is used.
Thus the following are effectively equivalent:

```bash
omnitrace-python --help

export PYTHONPATH=/opt/omnitrace/lib/python3.8/site-packages:${PYTHONPATH}
python3.8 -m omnitrace --help
```

> ***`omnitrace-python` / `python -m omnitrace` uses the same command-line syntax as the `omnitrace` executable (i.e. `omnitrace-python <OMNITRACE_ARGS> -- <SCRIPT> <SCRIPT_ARGS>`) and has similar options.***

### Command Line Options

Use `omnitrace-python --help` to view the available options:

```console
usage: omnitrace [-h] [-v VERBOSITY] [-b] [-c FILE] [-s FILE] [-F [BOOL]] [--label [{args,file,line} [{args,file,line} ...]]] [-I FUNC [FUNC ...]] [-E FUNC [FUNC ...]] [-R FUNC [FUNC ...]] [-MI FILE [FILE ...]] [-ME FILE [FILE ...]] [-MR FILE [FILE ...]] [--trace-c [BOOL]]

optional arguments:
  -h, --help            show this help message and exit
  -v VERBOSITY, --verbosity VERBOSITY
                        Logging verbosity
  -b, --builtin         Put 'profile' in the builtins. Use '@profile' to decorate a single function, or 'with profile:' to profile a single section of code.
  -c FILE, --config FILE
                        OmniTrace configuration file
  -s FILE, --setup FILE
                        Code to execute before the code to profile
  -F [BOOL], --full-filepath [BOOL]
                        Encode the full function filename (instead of basename)
  --label [{args,file,line} [{args,file,line} ...]]
                        Encode the function arguments, filename, and/or line number into the profiling function label
  -I FUNC [FUNC ...], --function-include FUNC [FUNC ...]
                        Include any entries with these function names
  -E FUNC [FUNC ...], --function-exclude FUNC [FUNC ...]
                        Filter out any entries with these function names
  -R FUNC [FUNC ...], --function-restrict FUNC [FUNC ...]
                        Select only entries with these function names
  -MI FILE [FILE ...], --module-include FILE [FILE ...]
                        Include any entries from these files
  -ME FILE [FILE ...], --module-exclude FILE [FILE ...]
                        Filter out any entries from these files
  -MR FILE [FILE ...], --module-restrict FILE [FILE ...]
                        Select only entries from these files
  --trace-c [BOOL]      Enable profiling C functions

usage: python3 -m omnitrace <OMNITRACE_ARGS> -- <SCRIPT> <SCRIPT_ARGS>
```

> ***The `--trace-c` option does not incorporate omnitrace's dynamic instrumentation support, rather it just enables profiling the underlying C function call within the Python interpreter.***

### Selective Instrumentation

Similar to the `omnitrace` executable, command-line options exist for restricting, including, and excluded the desired functions and modules, e.g. `--function-exclude "^__init__$"`.
Alternatively, adding `@profile` decorator to the primary function of interest in combination with the `-b` / `--builtin` option will narrow the scope of the
instrumentation to these function(s) and their children.

Consider the following Python code (`example.py`):

```python
import sys

def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


def inefficient(n):
    a = 0
    for i in range(n):
        a += i
        for j in range(n):
            a += j
    return a


def run(n):
    return fib(n) + inefficient(n)


if __name__ == "__main__":
    run(20)
```

Using `omnitrace-python ./example.py` with `OMNITRACE_PROFILE=ON` and `OMNITRACE_TIMEMORY_COMPONENTS=trip_count` would produce:

```console
|-------------------------------------------------------------------------------------------|
|                                COUNTS NUMBER OF INVOCATIONS                               |
|-------------------------------------------------------------------------------------------|
|                      LABEL                        | COUNT  | DEPTH  |   METRIC   |  SUM   |
|---------------------------------------------------|--------|--------|------------|--------|
| |0>>> run                                         |      1 |      0 | trip_count |      1 |
| |0>>> |_fib                                       |      1 |      1 | trip_count |      1 |
| |0>>>   |_fib                                     |      2 |      2 | trip_count |      2 |
| |0>>>     |_fib                                   |      4 |      3 | trip_count |      4 |
| |0>>>       |_fib                                 |      8 |      4 | trip_count |      8 |
| |0>>>         |_fib                               |     16 |      5 | trip_count |     16 |
| |0>>>           |_fib                             |     32 |      6 | trip_count |     32 |
| |0>>>             |_fib                           |     64 |      7 | trip_count |     64 |
| |0>>>               |_fib                         |    128 |      8 | trip_count |    128 |
| |0>>>                 |_fib                       |    256 |      9 | trip_count |    256 |
| |0>>>                   |_fib                     |    512 |     10 | trip_count |    512 |
| |0>>>                     |_fib                   |   1024 |     11 | trip_count |   1024 |
| |0>>>                       |_fib                 |   2026 |     12 | trip_count |   2026 |
| |0>>>                         |_fib               |   3632 |     13 | trip_count |   3632 |
| |0>>>                           |_fib             |   5020 |     14 | trip_count |   5020 |
| |0>>>                             |_fib           |   4760 |     15 | trip_count |   4760 |
| |0>>>                               |_fib         |   2942 |     16 | trip_count |   2942 |
| |0>>>                                 |_fib       |   1152 |     17 | trip_count |   1152 |
| |0>>>                                   |_fib     |    274 |     18 | trip_count |    274 |
| |0>>>                                     |_fib   |     36 |     19 | trip_count |     36 |
| |0>>>                                       |_fib |      2 |     20 | trip_count |      2 |
| |0>>> |_inefficient                               |      1 |      1 | trip_count |      1 |
|-------------------------------------------------------------------------------------------|
```

If the `inefficient` function were decorated with `@profile`:

```python
@profile
def inefficient(n):
    # ...
```

And executed with `omnitrace-python -b -- ./example.py`, omnitrace would produce:

```console
|-----------------------------------------------------------|
|                COUNTS NUMBER OF INVOCATIONS               |
|-----------------------------------------------------------|
|      LABEL        | COUNT  | DEPTH  |   METRIC   |  SUM   |
|-------------------|--------|--------|------------|--------|
| |0>>> inefficient |      1 |      0 | trip_count |      1 |
|-----------------------------------------------------------|
```

## OmniTrace Python Source Instrumentation

Starting from the unmodified `example.py` script above, we start by importing the `omnitrace` module:

```python
import sys
import omnitrace  # import omnitrace

def fib(n):
    # ... etc. ...
```

Then, we can add `@omnitrace.profile()` to the `run` function:

```python
@omnitrace.profile()
def run(n):
    # ...
```

Or we can use `omnitrace.profile()` as a context-manager around `run(20)`:

```python
if __name__ == "__main__":
    with omnitrace.profile():
        run(20)
```

The results for both of the source-level instrumentation modes are identical to the original `omnitrace-python ./example.py` results:

```console
|-------------------------------------------------------------------------------------------|
|                                COUNTS NUMBER OF INVOCATIONS                               |
|-------------------------------------------------------------------------------------------|
|                      LABEL                        | COUNT  | DEPTH  |   METRIC   |  SUM   |
|---------------------------------------------------|--------|--------|------------|--------|
| |0>>> run                                         |      1 |      0 | trip_count |      1 |
| |0>>> |_fib                                       |      1 |      1 | trip_count |      1 |
| |0>>>   |_fib                                     |      2 |      2 | trip_count |      2 |
| |0>>>     |_fib                                   |      4 |      3 | trip_count |      4 |
| |0>>>       |_fib                                 |      8 |      4 | trip_count |      8 |
| |0>>>         |_fib                               |     16 |      5 | trip_count |     16 |
| |0>>>           |_fib                             |     32 |      6 | trip_count |     32 |
| |0>>>             |_fib                           |     64 |      7 | trip_count |     64 |
| |0>>>               |_fib                         |    128 |      8 | trip_count |    128 |
| |0>>>                 |_fib                       |    256 |      9 | trip_count |    256 |
| |0>>>                   |_fib                     |    512 |     10 | trip_count |    512 |
| |0>>>                     |_fib                   |   1024 |     11 | trip_count |   1024 |
| |0>>>                       |_fib                 |   2026 |     12 | trip_count |   2026 |
| |0>>>                         |_fib               |   3632 |     13 | trip_count |   3632 |
| |0>>>                           |_fib             |   5020 |     14 | trip_count |   5020 |
| |0>>>                             |_fib           |   4760 |     15 | trip_count |   4760 |
| |0>>>                               |_fib         |   2942 |     16 | trip_count |   2942 |
| |0>>>                                 |_fib       |   1152 |     17 | trip_count |   1152 |
| |0>>>                                   |_fib     |    274 |     18 | trip_count |    274 |
| |0>>>                                     |_fib   |     36 |     19 | trip_count |     36 |
| |0>>>                                       |_fib |      2 |     20 | trip_count |      2 |
| |0>>> |_inefficient                               |      1 |      1 | trip_count |      1 |
|-------------------------------------------------------------------------------------------|
```

> ***When `omnitrace-python` is used without built-ins, the profiling results will likely be cluttered by***
> ***numerous functions called during the importing of more complex modules, e.g. `import numpy`.***

### OmniTrace Python Source Instrumentation Configuration

Within the Python source code, the profiler can be configured by directly modifying the `omnitrace.profiler.config` data fields.

```python
import sys

def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


def inefficient(n):
    a = 0
    for i in range(n):
        a += i
        for j in range(n):
            a += j
    return a


def run(n):
    return fib(n) + inefficient(n)


if __name__ == "__main__":
    from omnitrace.profiler import config
    from omnitrace import profile

    config.include_args = True
    config.include_filename = False
    config.include_line = False
    config.restrict_functions += ["fib", "run"]

    with profile():
        run(5)
```

Executing this script would produce:

```console
|------------------------------------------------------------------|
|                   COUNTS NUMBER OF INVOCATIONS                   |
|------------------------------------------------------------------|
|          LABEL           | COUNT  | DEPTH  |   METRIC   |  SUM   |
|--------------------------|--------|--------|------------|--------|
| |0>>> run(n=5)           |      1 |      0 | trip_count |      1 |
| |0>>> |_fib(n=5)         |      1 |      1 | trip_count |      1 |
| |0>>>   |_fib(n=4)       |      1 |      2 | trip_count |      1 |
| |0>>>     |_fib(n=3)     |      1 |      3 | trip_count |      1 |
| |0>>>       |_fib(n=2)   |      1 |      4 | trip_count |      1 |
| |0>>>         |_fib(n=1) |      1 |      5 | trip_count |      1 |
| |0>>>         |_fib(n=0) |      1 |      5 | trip_count |      1 |
| |0>>>       |_fib(n=1)   |      1 |      4 | trip_count |      1 |
| |0>>>     |_fib(n=2)     |      1 |      3 | trip_count |      1 |
| |0>>>       |_fib(n=1)   |      1 |      4 | trip_count |      1 |
| |0>>>       |_fib(n=0)   |      1 |      4 | trip_count |      1 |
| |0>>>   |_fib(n=3)       |      1 |      2 | trip_count |      1 |
| |0>>>     |_fib(n=2)     |      1 |      3 | trip_count |      1 |
| |0>>>       |_fib(n=1)   |      1 |      4 | trip_count |      1 |
| |0>>>       |_fib(n=0)   |      1 |      4 | trip_count |      1 |
| |0>>>     |_fib(n=1)     |      1 |      3 | trip_count |      1 |
|------------------------------------------------------------------|
```
