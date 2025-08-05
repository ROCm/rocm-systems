.. meta::
   :description: ROCm Systems Profiler Python profiling documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, Python, profiling Python, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Profiling Python scripts
****************************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocprofiler-systems>`_ supports profiling Python code at the
source level and the script level.
Python support is enabled via the ``ROCPROFSYS_USE_PYTHON`` and the
``ROCPROFSYS_PYTHON_VERSIONS="<MAJOR>.<MINOR>`` CMake options.
Alternatively, to build multiple Python versions, use
``ROCPROFSYS_PYTHON_VERSIONS="<MAJOR>.<MINOR>;[<MAJOR>.<MINOR>]"``,
and ``ROCPROFSYS_PYTHON_ROOT_DIRS="/path/to/version;[/path/to/version]"`` instead of ``ROCPROFSYS_PYTHON_VERSION``.
When building multiple Python versions, the length of the ``ROCPROFSYS_PYTHON_VERSIONS``
and ``ROCPROFSYS_PYTHON_ROOT_DIRS`` lists must
be the same size.

.. note::

   When using ROCm Systems Profiler with Python programs, the Python interpreter major and minor version (e.g. 3.7)
   must match the interpreter major and minor version
   used when compiling the Python bindings. When building ROCm Systems Profiler,
   the shared object file ``libpyrocprofsys.<IMPL>-<VERSION>-<ARCH>-<OS>-<ABI>.so`` is generated
   where ``IMPL`` is the Python implementation, ``VERSION`` is the major and minor
   version, ``ARCH`` is the architecture,
   ``OS`` is the operating system, and ``ABI`` is the application binary interface,
   for example, ``libpyrocprofsys.cpython-38-x86_64-linux-gnu.so``.

.. note::

   ROCm Systems Profiler has limited support for Artificial Intelligence (AI) and Machine Learning (ML) workloads.
   Data from child threads is not captured. For other profiling options,
   see `rocprofV3 <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#using-rocprofv3>`_.

Getting started
========================================

The ROCm Systems Profiler Python package is installed in ``lib/pythonX.Y/site-packages/rocprofsys``.
To ensure the Python interpreter can find the ROCm Systems Profiler package,
add this path to the ``PYTHONPATH`` environment variable, as in the following example:

.. code-block:: shell

   export PYTHONPATH=/opt/rocprofiler-systems/lib/python3.8/site-packages:${PYTHONPATH}

Both the ``share/rocprofiler-systems/setup-env.sh`` script and the module file in
``share/modulefiles/rocprofiler-systems`` automatically handle the prefixing of the ``PYTHONPATH``
environment variable.

Running ROCm Systems Profiler on a Python script
================================================

ROCm Systems Profiler provides an ``rocprof-sys-python`` helper bash script which
ensures ``PYTHONPATH`` is properly set and the correct Python interpreter is used.
This means the following commands are effectively equivalent:

.. code-block:: shell

   rocprof-sys-python --help

and

.. code-block:: shell

   export PYTHONPATH=/opt/rocprofiler-systems/lib/python3.8/site-packages:${PYTHONPATH}
   python3.8 -m rocprofsys --help

.. note::

   ``rocprof-sys-python`` and ``python -m rocprofsys`` use the same command-line syntax
   as the other ``rocprof-sys`` executables (``rocprof-sys-python <ROCPROFSYS_ARGS> -- <SCRIPT> <SCRIPT_ARGS>``)
   and has similar options.

Command line options
-----------------------------------

Use ``rocprof-sys-python --help`` to view the available options:

.. code-block:: shell

   usage: rocprof-sys [-h] [-v VERBOSITY] [-b] [-c FILE] [-s FILE] [-F [BOOL]] [--label [{args,file,line} [{args,file,line} ...]]] [-I FUNC [FUNC ...]] [-E FUNC [FUNC ...]] [-R FUNC [FUNC ...]] [-MI FILE [FILE ...]] [-ME FILE [FILE ...]] [-MR FILE [FILE ...]] [--trace-c [BOOL]]

   optional arguments:
   -h, --help            show this help message and exit
   -v VERBOSITY, --verbosity VERBOSITY
                           Logging verbosity
   -b, --builtin         Put 'profile' in the builtins. Use '@profile' to decorate a single function, or 'with profile:' to profile a single section of code.
   -c FILE, --config FILE
                           ROCm Systems Profiler configuration file
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

   usage: python3 -m rocprofsys <ROCPROFSYS_ARGS> -- <SCRIPT> <SCRIPT_ARGS>

.. note::

   The ``--trace-c`` option does not incorporate ROCm Systems Profiler's dynamic instrumentation support.
   It only enables profiling the underlying C function call within the Python interpreter.

Selective instrumentation
-----------------------------------

Similar to the ``rocprof-sys-instrument`` executable, command-line options exist for restricting,
including, and excluding certain functions and modules, for example, ``--function-exclude "^__init__$"``.
Alternatively, add the ``@profile`` decorator to the primary function of interest
in your program and use the ``-b`` / ``--builtin`` command-line option to narrow the scope of the
instrumentation to this function and its children.

Consider the following Python code (``example.py``):

.. code-block:: python

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

Running ``rocprof-sys-python -- ./example.py`` with ``ROCPROFSYS_PROFILE=ON`` and
``ROCPROFSYS_TIMEMORY_COMPONENTS=trip_count`` produces the following:

.. code-block:: shell

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

If the ``inefficient`` function is decorated with ``@profile`` as follows:

.. code-block:: python

   @profile
   def inefficient(n):
      # ...

And then run using the command ``rocprof-sys-python -b -- ./example.py``, ROCm Systems Profiler produces this output:

.. code-block:: shell

   |-----------------------------------------------------------|
   |                COUNTS NUMBER OF INVOCATIONS               |
   |-----------------------------------------------------------|
   |      LABEL        | COUNT  | DEPTH  |   METRIC   |  SUM   |
   |-------------------|--------|--------|------------|--------|
   | |0>>> inefficient |      1 |      0 | trip_count |      1 |
   |-----------------------------------------------------------|

ROCm Systems Profiler Python source instrumentation
===================================================

Starting with the unmodified ``example.py`` script above, import the ``rocprofsys`` module:

.. code-block:: python

   import sys
   import rocprofsys  # import rocprofsys

   def fib(n):
      # ... etc. ...

Next, add ``@rocprofsys.profile()`` to the ``run`` function:

.. code-block:: python

   @rocprofsys.profile()
   def run(n):
      # ...

Alternatively, use ``rocprofsys.profile()`` as a context-manager around ``run(20)``:

.. code-block:: python

   if __name__ == "__main__":
      with rocprofsys.profile():
         run(20)

The results for both of the source-level instrumentation modes are identical to the
original ``rocprof-sys-python -- ./example.py`` results:

.. code-block:: shell

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

.. note::

   When ``rocprof-sys-python`` is used without built-ins, the profiling results can be cluttered by the
   numerous functions called when more complex modules are imported, such as ``import numpy``.

ROCm Systems Profiler Python source instrumentation configuration
-----------------------------------------------------------------

Within the Python source code, the profiler can be configured by directly
modifying the ``rocprof-sys.profiler.config`` data fields.

.. code-block:: python

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
      from rocprofsys.profiler import config
      from rocprofsys import profile

      config.include_args = True
      config.include_filename = False
      config.include_line = False
      config.restrict_functions += ["fib", "run"]

      with profile():
         run(5)

Executing this script produces the following:

.. code-block:: shell

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
