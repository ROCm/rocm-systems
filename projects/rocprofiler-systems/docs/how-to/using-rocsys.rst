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
mode (``-o``). A ``rocsys`` subcommand selects the tool. ``rocsys`` reads its own
options, described in :ref:`rocsys-options`, and forwards every other argument after the
subcommand, unchanged and in order, to the selected tool.

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

The application and its arguments always follow ``--``. ``rocsys`` reports an error
for a command without ``--``, such as ``rocsys python ./script.py`` or
``rocsys rewrite -o app.inst ./app``, because it cannot tell the application from a
tool option's value. A command that runs no application still ends with ``--``, as in
``rocsys --preset=balanced --export-config --``. ``avail`` and ``attach`` run no
application and take no ``--``.

``rocsys`` with no arguments prints the top-level help. ``rocsys --version``
prints the version. Print the top-level help, or run a profile with a preset:

.. code-block:: shell

   rocsys --help
   rocsys --preset=balanced -- ./app

.. _rocsys-options:

rocsys options
==============

``rocsys`` has three options of its own. They apply to ``profile``, ``instrument``,
``python``, and ``attach``, and can appear anywhere between the subcommand and ``--``.
``rewrite``, ``avail``, and ``causal`` pass them to their tool unchanged, like any other
option: ``rocsys causal --log-level debug -- ./app`` uses the ``--log-level`` of
``rocprof-sys-causal``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Option
     - Effect
   * - ``--traces <names>``
     - Selects what is traced: the ROCm APIs ``hip-api`` (or ``hip``), ``hsa-api``
       (or ``hsa``), ``roctx``, ``kernel-dispatch``, ``memory-copy``,
       ``memory-allocation``, and ``rocshmem``, and the runtimes ``mpi``, ``openmp``,
       ``rccl``, ``ucx``, and ``openshmem``.
   * - ``--output-path <dir>``
     - Directory for the output files.
   * - ``--log-level <level>``
     - How much the tool logs: ``trace``, ``debug``, ``info``, ``warn`` (or
       ``warning``), ``error`` (or ``err``), ``critical``, or ``off``.

.. code-block:: shell

   rocsys --traces hip mpi -- ./app
   rocsys --traces hip,mpi -- ./app
   rocsys --traces hip --traces mpi -- ./app
   rocsys instrument --traces kernel-dispatch -- ./app
   rocsys --output-path out --log-level debug -- ./app

Names are separated by spaces or commas, and ``--traces`` can be repeated. With ``=``,
separate names with commas only: ``--traces=hip,mpi``. An unknown name is an error.
``--traces`` is the whole selection: a name you leave out is not traced, so
``rocsys --traces hip -- ./app`` also turns MPI tracing off. Without ``--traces``, the
tool's own defaults apply.

.. _rocsys-precedence:

Setting precedence
==================

``rocsys`` uses the order below, and the highest source in this list wins:

#. A tool option forwarded by ``rocsys``, such as ``--rocm`` or ``-o``, or a preset
   named with ``--preset``. Where both set the same thing, ``--rocm`` overrides the
   preset, but the preset overrides ``--gpu`` and ``--cpu``, and overrides ``-o`` when
   the preset sets an output path.
#. A ``rocsys`` option: ``--traces``, ``--output-path``, or ``--log-level``.
#. An environment variable.
#. A configuration file (:ref:`creating-a-configuration-file`).
#. The tool's built-in default.

.. note::

   This order might change in a future release. Don't rely on a preset to override
   ``--traces``, ``--output-path``, or ``--log-level``.

``rocsys`` writes only an option you type, and that write replaces the environment
variable of the same name. The tool then applies the preset and its own command line, so
``rocsys --traces hip --rocm=hsa -- ./app`` keeps the domains from ``--rocm``.
With no ``rocsys`` option for a setting, the shell value is left unchanged.
For ``rewrite``, pass ``--traces`` when the rewritten binary runs, as in
``rocsys --traces hip -- ./app.inst``.

``rocsys --help`` lists these options after the subcommands. ``rocsys <subcommand> --help``,
``-h``, or ``-?`` names the tool whose own ``--help`` lists every other option, and also
lists these options for ``profile``, ``instrument``, ``python``, and ``attach``. A help topic
such as ``rocsys profile --help=tracing`` is passed to the tool. A ``--help`` after ``--``
belongs to the application.

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
     - Attaches to an already running process through ``rocprof-sys-attach``. Pass ``-p`` or ``--pid``. This verb takes no application and no ``--``.
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
