.. meta::
   :description: Use the rocsys command to launch ROCm Systems Profiler tools
   :keywords: rocsys, rocprof-sys, rocprofiler-systems, ROCm, profiler, command line, instrumentation, binary rewrite

.. _using-rocsys:

************************
Using the rocsys command
************************

``rocsys`` is the unified command-line entry point for ROCm Systems Profiler.
A subcommand selects the tool. Flags after the subcommand are forwarded to that
tool, including ``--help``.

The existing ``rocprof-sys-*`` executables remain available and behave as before.
Use them directly when a workflow page shows that command, or use the matching
``rocsys`` verb below.

Usage
=====

.. code-block:: shell

   rocsys -- ./app
   rocsys profile -- ./app
   rocsys <subcommand> [flags] [--] <app> [app-args]
   rocsys --help
   rocsys <subcommand> --help

``rocsys`` with no arguments prints the top-level help. ``rocsys --version``
prints the version.

Subcommands
===========

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Verb
     - Runs
     - See also
   * - ``profile``
     - A full trace profile through ``rocprof-sys-run``. This is also the default when the first argument is a flag or ``--``.
     - :doc:`using-preset-profiles`, :doc:`instrumenting-rewriting-binary-application`
   * - ``instrument``
     - Runtime instrumentation through ``rocprof-sys-instrument``.
     - :doc:`instrumenting-rewriting-binary-application`
   * - ``rewrite``
     - Binary rewrite through ``rocprof-sys-instrument``. ``rocsys`` inserts ``-o`` when you do not pass an output path.
     - :doc:`instrumenting-rewriting-binary-application`
   * - ``causal``
     - ``rocprof-sys-causal``
     - :doc:`performing-causal-profiling`
   * - ``avail``
     - ``rocprof-sys-avail``. An application argument is optional.
     - :doc:`general-tips-using-rocprof-sys`
   * - ``python``
     - ``rocprof-sys-python``
     - :doc:`profiling-python-scripts`
   * - ``attach``
     - ``rocprof-sys-attach``
     - :doc:`attaching-to-running-process`

These two forms collect the same full trace profile:

.. code-block:: shell

   rocsys -- ./app
   rocsys profile -- ./app

``rocsys profile`` with no application exits with an error. Pass ``--help`` to
read the ``rocprof-sys-run`` options, including presets:

.. code-block:: shell

   rocsys profile --help
   rocsys profile --preset=balanced -- ./app

Default output for rewrite
==========================

``rocsys rewrite -- ./app`` runs ``rocprof-sys-instrument -o -- ./app``. When
``-o`` has no filename, the rewritten file is written in the current directory:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Target
     - Default output
   * - ``./app``, already in the current directory and not a library
     - ``./app.inst``
   * - A name with no extension, such as ``./libfoo``
     - ``./libfoo.inst``
   * - A library already in the current directory, such as ``./libfoo.so``
     - ``./instrumented/libfoo.so``
   * - A target outside the current directory, such as ``/usr/bin/app``
     - ``./app``

A target is treated as a library only when its basename contains a dot and either
starts with ``lib``, contains ``.so``, or ends with ``.a``.
Pass ``-o``, ``--output``, or ``-o=`` to choose the path yourself:

.. code-block:: shell

   rocsys rewrite -o app.inst -- ./app

Call-stack sampling
===================

Call-stack sampling is not a ``rocsys`` subcommand. ``rocsys sample`` reports
an unknown subcommand. Use ``rocprof-sys-sample`` directly. See
:doc:`sampling-call-stack`.
