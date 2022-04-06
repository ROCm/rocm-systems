# Python Support

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 3
```

[Omnitrace](https://github.com/AMDResearch/omnitrace) supports profiling Python code at the source-level and/or the script-level.
Python support is enabled via the `OMNITRACE_USE_PYTHON` and `OMNITRACE_PYTHON_VERSION=<MAJOR>.<MINOR>` CMake options.

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

## Running Omnitrace on a Python Script

Omnitrace provides an `omnitrace-python` helper bash script which effectively handles ensuring `PYTHONPATH` is properly set and the correct python interpreter is used.
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
usage: omnitrace [-h] [-b] [-c FILE] [-s FILE] [--trace-c [BOOL]] [-a [BOOL]] [-l [BOOL]] [-f [BOOL]] [-F [BOOL]] [-I FUNC [FUNC ...]] [-E FUNC [FUNC ...]] [-R FUNC [FUNC ...]] [-MI FILE [FILE ...]] [-ME FILE [FILE ...]] [-MR FILE [FILE ...]] [-v VERBOSITY]

optional arguments:
  -h, --help            show this help message and exit
  -b, --builtin         Put 'profile' in the builtins. Use '@profile' to decorate a single function, or 'with profile:' to profile a single section of code.
  -c FILE, --config FILE
                        Omnitrace configuration file
  -s FILE, --setup FILE
                        Code to execute before the code to profile
  --trace-c [BOOL]      Enable profiling C functions
  -a [BOOL], --include-args [BOOL]
                        Encode the argument values
  -l [BOOL], --include-line [BOOL]
                        Encode the function line number
  -f [BOOL], --include-file [BOOL]
                        Encode the function filename
  -F [BOOL], --full-filepath [BOOL]
                        Encode the full function filename (instead of basename)
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
  -v VERBOSITY, --verbosity VERBOSITY
                        Logging verbosity

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

Using `omnitrace-python ./example.py` with `OMNITRACE_USE_TIMEMORY=ON` and `OMNITRACE_TIMEMORY_COMPONENTS=trip_count` would produce:

```console
|-------------------------------------------------------------------------------------------|
|                                COUNTS NUMBER OF INVOCATIONS                               |
|-------------------------------------------------------------------------------------------|
|                      LABEL                        | COUNT  | DEPTH  |   METRIC   |  SUM   |
|---------------------------------------------------|--------|--------|------------|--------|
| |0>>> run                                         |      1 |      0 | trip_count |      1 |
| |0>>> |_fib                                       |  10946 |      1 | trip_count |  10946 |
| |0>>>   |_fib                                     |   4181 |      2 | trip_count |   4181 |
| |0>>>     |_fib                                   |   2584 |      3 | trip_count |   2584 |
| |0>>>       |_fib                                 |   1597 |      4 | trip_count |   1597 |
| |0>>>         |_fib                               |    987 |      5 | trip_count |    987 |
| |0>>>           |_fib                             |    610 |      6 | trip_count |    610 |
| |0>>>             |_fib                           |    377 |      7 | trip_count |    377 |
| |0>>>               |_fib                         |    233 |      8 | trip_count |    233 |
| |0>>>                 |_fib                       |    144 |      9 | trip_count |    144 |
| |0>>>                   |_fib                     |     89 |     10 | trip_count |     89 |
| |0>>>                     |_fib                   |     55 |     11 | trip_count |     55 |
| |0>>>                       |_fib                 |     34 |     12 | trip_count |     34 |
| |0>>>                         |_fib               |     21 |     13 | trip_count |     21 |
| |0>>>                           |_fib             |     13 |     14 | trip_count |     13 |
| |0>>>                             |_fib           |      8 |     15 | trip_count |      8 |
| |0>>>                               |_fib         |      5 |     16 | trip_count |      5 |
| |0>>>                                 |_fib       |      3 |     17 | trip_count |      3 |
| |0>>>                                   |_fib     |      2 |     18 | trip_count |      2 |
| |0>>>                                     |_fib   |      1 |     19 | trip_count |      1 |
| |0>>>                                       |_fib |      1 |     20 | trip_count |      1 |
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

## Omnitrace Python Source Instrumentation

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
| |0>>> |_fib                                       |  10946 |      1 | trip_count |  10946 |
| |0>>>   |_fib                                     |   4181 |      2 | trip_count |   4181 |
| |0>>>     |_fib                                   |   2584 |      3 | trip_count |   2584 |
| |0>>>       |_fib                                 |   1597 |      4 | trip_count |   1597 |
| |0>>>         |_fib                               |    987 |      5 | trip_count |    987 |
| |0>>>           |_fib                             |    610 |      6 | trip_count |    610 |
| |0>>>             |_fib                           |    377 |      7 | trip_count |    377 |
| |0>>>               |_fib                         |    233 |      8 | trip_count |    233 |
| |0>>>                 |_fib                       |    144 |      9 | trip_count |    144 |
| |0>>>                   |_fib                     |     89 |     10 | trip_count |     89 |
| |0>>>                     |_fib                   |     55 |     11 | trip_count |     55 |
| |0>>>                       |_fib                 |     34 |     12 | trip_count |     34 |
| |0>>>                         |_fib               |     21 |     13 | trip_count |     21 |
| |0>>>                           |_fib             |     13 |     14 | trip_count |     13 |
| |0>>>                             |_fib           |      8 |     15 | trip_count |      8 |
| |0>>>                               |_fib         |      5 |     16 | trip_count |      5 |
| |0>>>                                 |_fib       |      3 |     17 | trip_count |      3 |
| |0>>>                                   |_fib     |      2 |     18 | trip_count |      2 |
| |0>>>                                     |_fib   |      1 |     19 | trip_count |      1 |
| |0>>>                                       |_fib |      1 |     20 | trip_count |      1 |
| |0>>> |_inefficient                               |      1 |      1 | trip_count |      1 |
|-------------------------------------------------------------------------------------------|
```

> ***When `omnitrace-python` is used without built-ins, the profiling results will likely be cluttered by***
> ***numerous functions called during the importing of more complex modules, e.g. `import numpy`.***

### Omnitrace Python Source Instrumentation Configuration

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
