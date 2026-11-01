.. meta::
   :description: The rocsys command, a new unified command-line entry point for ROCm Systems Profiler
   :keywords: rocsys, rocprof-sys, rocprofiler-systems, ROCm, profiler, command line, instrumentation, binary rewrite, sampling, causal profiling, Python, attach

.. _using-rocsys:

***************************************
Using the rocsys command (experimental)
***************************************

The ``rocprof-sys-*`` executables are the original ROCm Systems Profiler tools.
The other pages in this documentation show those commands.

The ``rocsys`` command is a new unified command-line entry point for profiling.
Running ``rocsys -- ./app`` or ``rocsys profile -- ./app`` produces a full trace
profile, and ``rocsys rewrite`` runs ``rocprof-sys-instrument`` in binary-rewrite
mode (``-o``). A ``rocsys`` subcommand selects the tool, and any arguments after the subcommand are
forwarded to the selected tool. ``rocsys --help`` reports that the command is
experimental and that commands and workflows are subject to change.

Usage
=====

.. code-block:: shell

   rocsys -- ./app
   rocsys profile -- ./app
   rocsys <subcommand> [options] -- <app> [app-args]
   rocsys --help
   rocsys --version
   rocsys <subcommand> --help

``profile`` is the default. ``rocsys -- ./app`` and ``rocsys profile -- ./app``
both produce a full trace profile, the same way ``rocprof-sys-run`` launches the
application. With no other flags, that records a rocpd trace of the default
ROCm domains and GPU telemetry.
It does not rewrite the binary, and it does not enable call-stack sampling or
an aggregated function profile. You can change the data that is collected by using
``--preset`` or domain flags, as described in :doc:`using-preset-profiles`. Output paths follow
``rocprof-sys-run``; see :doc:`understanding-rocprof-sys-output`.

``rocsys`` with no arguments prints the top-level help. ``rocsys --version``
prints the version. Pass ``--help`` after ``profile`` to read the
``rocprof-sys-run`` options, including presets:

.. code-block:: shell

   rocsys profile --help
   rocsys profile --preset=balanced -- ./app

How arguments are parsed
========================

* The subcommand, when you pass one, is the first argument.
  ``rocsys profile --preset=balanced -- ./app`` and
  ``rocsys --preset=balanced -- ./app`` are profile runs.
  A flag placed before the subcommand does not select that subcommand.
* A first token that does not start with ``-`` must be a known verb.
  ``rocsys ./app`` exits with ``unknown subcommand './app'``.
  Use ``rocsys -- ./app`` or ``rocsys profile -- ./app``.
* Tool options go before ``--``. Tokens after ``--`` belong to the application.
  ``rocsys profile -- ./app --preset=balanced`` does not apply the preset.
* Every verb except ``avail`` requires another argument after the verb.
  ``rocsys profile``, ``rocsys instrument``, ``rocsys rewrite``,
  ``rocsys causal``, ``rocsys python``, and ``rocsys attach`` with no further
  arguments exit with ``missing application argument``.
  ``rocsys avail`` is valid on its own.
* No arguments, or ``-h``, ``-?``, or ``--help`` as the first token, prints
  ``rocsys`` help and does not forward it. ``rocsys --help=sampling`` is
  top-level help. Topic help for a profile run is
  ``rocsys profile --help=sampling``.
* ``rocsys <subcommand> --help`` forwards to that tool.
  ``rewrite`` inserts ``-o`` before the forwarded arguments, so
  ``rocsys rewrite --help`` is run as ``rocprof-sys-instrument -o --help``.
  Read the instrumenter options with ``rocsys instrument --help``.

Subcommands
===========

``rocsys --help`` lists the verbs in this order.

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - Verb
     - Description
     - See also
   * - ``profile``
     - The default. Produces a full trace profile, using the same run mode as ``rocprof-sys-run``. With no other flags, that is a rocpd trace of the default ROCm domains and GPU telemetry. Also launches a binary already produced by ``rewrite``.
     - :doc:`using-preset-profiles`, :doc:`understanding-rocprof-sys-output`
   * - ``avail``
     - Queries the settings, domains, hardware counters, and components available on this system through ``rocprof-sys-avail``. An application argument is optional.
     - :doc:`general-tips-using-rocprof-sys`
   * - ``attach``
     - Attaches to an already running process through ``rocprof-sys-attach``. Pass ``-p`` or ``--pid``. This verb does not take an application after ``--``.
     - :doc:`attaching-to-running-process`
   * - ``causal``
     - Estimates how much faster the application would run if a selected function or line were sped up, through ``rocprof-sys-causal``.
     - :doc:`performing-causal-profiling`
   * - ``instrument``
     - Instruments the application in memory and then runs it, through ``rocprof-sys-instrument``. Does not write a new binary.
     - :doc:`instrumenting-rewriting-binary-application`
   * - ``rewrite``
     - Writes an instrumented binary through ``rocprof-sys-instrument`` and exits. ``rocsys`` inserts ``-o`` when you do not pass an output path. The rewritten binary is not executed.
     - :doc:`instrumenting-rewriting-binary-application`
   * - ``python``
     - Profiles a Python script through the ``rocprof-sys-python`` helper. Its flags are the Python-script options, such as ``-b``, ``-I``, and ``-E``, not ``--preset``.
     - :doc:`profiling-python-scripts`

Examples
========

.. code-block:: shell

   rocsys instrument -- ./app
   rocsys rewrite -- ./app
   rocsys profile -- ./app.inst

   rocsys avail
   rocsys avail -d

   rocsys causal -l foo -- ./app
   rocsys python -- ./script.py
   rocsys attach -p $(pidof my_app)

``instrument`` performs runtime instrumentation and does not write a new binary.
``rewrite`` writes a new binary and exits. Profile that binary with
``rocsys profile -- ./app.inst``.

``python`` forwards to the ``rocprof-sys-python`` helper. Its flags are the
ones on :doc:`profiling-python-scripts`, such as ``-b``, ``-I``, and ``-E``.

Default output for rewrite
==========================

``rocsys rewrite -- ./app`` runs ``rocprof-sys-instrument -o -- ./app``. When
``-o`` has no filename, the rewritten file is written in the current directory.
"Already in the current directory" means the resolved path is the current
directory plus the file's basename. A relative path such as ``./subdir/app``
is handled like a target outside the current directory.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Target
     - Default output
   * - ``./app``, already in the current directory and not a library
     - ``./app.inst``
   * - A name with no extension, such as ``./libfoo``
     - ``./libfoo.inst``
   * - A local non-library whose name contains a dot, such as ``./my.app``
     - ``./my.app.inst``
   * - A library already in the current directory, such as ``./libfoo.so``
     - ``./instrumented/libfoo.so``
   * - A target outside the current directory, such as ``/usr/bin/app`` or ``./subdir/app``
     - ``./app``
   * - A library outside the current directory, such as ``/usr/lib/libfoo.so`` or ``./libs/libfoo.so``
     - ``./libfoo.so``

A target is treated as a library only when its basename contains a dot and either
starts with ``lib``, contains ``.so``, or ends with ``.a``.
The ``instrumented/`` directory is used only for a library whose resolved path
is the current directory plus its basename.

The default name is written in the current directory and can overwrite a file
that is already there. Rewriting ``/usr/bin/app`` writes ``./app``.
Pass ``-o``, ``--output``, ``-o=``, or ``--output=`` to choose the path yourself:

.. code-block:: shell

   rocsys rewrite -o app.inst -- ./app
   rocsys rewrite --output=app.inst -- ./app

Call-stack sampling
===================

There is no ``sample`` subcommand. ``rocsys sample -- ./app`` reports an unknown
subcommand.

Call-stack sampling without instrumentation is ``rocprof-sys-sample``.
See :doc:`sampling-call-stack`.

To turn call-stack sampling on during a profile run, pass ``--sample``
(``-S``) or ``--cpu``. ``--sample`` accepts a timer, ``cputime`` or
``realtime``. ``--cpu=50`` sets the sampling frequency in Hz:

.. code-block:: shell

   rocsys profile --sample -- ./my_app
   rocsys profile --sample realtime -- ./my_app
   rocsys profile --cpu=50 -- ./my_app

That keeps the run-tool collection and adds sampling. It does not switch to
``rocprof-sys-sample``.
