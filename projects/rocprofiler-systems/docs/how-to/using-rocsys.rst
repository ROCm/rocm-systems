.. meta::
   :description: The rocsys command, a new unified command-line entry point for ROCm Systems Profiler
   :keywords: rocsys, rocprof-sys, rocprofiler-systems, ROCm, profiler, command line, instrumentation, binary rewrite, sampling, causal profiling, Python, attach

.. _using-rocsys:

***************************************
Using the rocsys command (experimental)
***************************************

The ``rocsys`` command is a new unified command-line entry point for profiling.
Running ``rocsys -- ./app`` or ``rocsys profile -- ./app`` produces a full trace
profile, and ``rocsys rewrite`` runs ``rocprof-sys-instrument`` in binary-rewrite
mode (``-o``). A ``rocsys`` subcommand selects the tool, and any arguments after the subcommand are
forwarded to the selected tool.

Usage
=====

.. code-block:: shell

   rocsys -- ./app
   rocsys profile -- ./app
   rocsys <subcommand> [options] -- <app> [app-args]
   rocsys --help
   rocsys --version
   rocsys <subcommand> --help

``profile`` is the default subcommand if none is specified.
``rocsys -- ./app``, ``rocsys --preset=balanced -- ./app``, and
``rocsys profile -- ./app`` produce a full trace profile. With no other flags,
a profile records a rocpd trace of the default ROCm domains and default GPU
telemetry. You can change the data that is collected by using ``--preset`` or
domain flags, as described in :doc:`using-preset-profiles`.

``rocsys`` with no arguments prints the top-level help. ``rocsys --version``
prints the version. Print the top-level help, or run a profile with a preset:

.. code-block:: shell

   rocsys --help
   rocsys --preset=balanced -- ./app

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
     - The default subcommand. Produces a full trace profile, using ``rocprof-sys-run``. Also launches an instrumented binary produced by ``rewrite``.
     - :doc:`using-preset-profiles`, :doc:`understanding-rocprof-sys-output`
   * - ``avail``
     - Queries the settings, domains, hardware counters, and components available on the current system through ``rocprof-sys-avail``.
     - :doc:`general-tips-using-rocprof-sys`
   * - ``attach``
     - Attaches to an already running process through ``rocprof-sys-attach``. Pass ``-p`` or ``--pid``. This verb does not take an application after ``--``.
     - :doc:`attaching-to-running-process`
   * - ``causal``
     - Estimates how much faster the application would run if a selected function or line were sped up, through ``rocprof-sys-causal``.
     - :doc:`performing-causal-profiling`
   * - ``instrument``
     - Instruments the application in memory and then runs it, through ``rocprof-sys-instrument``.
     - :doc:`instrumenting-rewriting-binary-application`
   * - ``rewrite``
     - Writes an instrumented binary through ``rocprof-sys-instrument`` and exits. The rewritten binary is not executed.
     - :doc:`instrumenting-rewriting-binary-application`
   * - ``python``
     - Profiles a Python application through ``rocprof-sys-python``. Its flags are the Python-script options, such as ``-b``, ``-I``, and ``-E``, not ``--preset``.
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
``rocsys -- ./app.inst``.

Default output for rewrite
==========================

``rocsys rewrite -- ./app`` runs ``rocprof-sys-instrument -o -- ./app``. When
``-o`` has no filename, the output name and location follow the rules in
:doc:`instrumenting-rewriting-binary-application`. The default name is written in
the current directory and can overwrite a file that already exists.
Pass ``-o``, ``--output``, ``-o=``, or ``--output=`` to choose the path yourself:

.. code-block:: shell

   rocsys rewrite -o app.inst -- ./app
   rocsys rewrite --output=app.inst -- ./app

Call-stack sampling
===================

There is no ``sample`` subcommand. ``rocsys sample -- ./app`` reports an unknown
subcommand error.

To turn call-stack sampling on during a profile run, pass ``--sample``
(``-S``) or ``--cpu``. ``--sample`` accepts a timer, ``cputime`` or
``realtime``. ``--cpu`` sets the sampling frequency in Hz and defaults to
100 Hz. The same flags apply when ``profile`` is implicit:

.. code-block:: shell

   rocsys --sample -- ./my_app
   rocsys --sample realtime -- ./my_app
   rocsys --cpu=50 -- ./my_app

That keeps the run-tool collection and adds sampling. It does not switch to
``rocprof-sys-sample``. See :doc:`sampling-call-stack`.
