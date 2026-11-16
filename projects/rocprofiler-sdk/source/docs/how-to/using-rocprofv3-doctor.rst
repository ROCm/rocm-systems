.. meta::
  :description: Documentation of rocprofv3 --doctor, which checks whether a machine is ready for profiling
  :keywords: ROCprofiler-SDK tool, rocprofv3 --doctor, rocprofv3-doctor, diagnostics, troubleshooting, rocprofv3 not working, ROCprofiler-SDK troubleshooting, GPU profiling diagnostics

.. _using-rocprofv3-doctor:

===========================================
Checking your setup with rocprofv3 --doctor
===========================================

``rocprofv3 --doctor`` inspects your machine and ROCprofiler-SDK installation, and
reports what would prevent ``rocprofv3`` from working.

When ``rocprofv3`` fails with a message that doesn't clearly point to a cause, run
``rocprofv3 --doctor`` first. It checks the installation, the driver and device nodes, the
runtime library stack, counter-collection permissions, the environment, container device
passthrough, the Python environment, the output filesystem, and the companion tools --
then prints a copy-pasteable remediation for anything it finds.

It goes beyond checking that the ROCm libraries exist: it loads them, and when a command it
runs fails, it reads the error output and explains the cause.

The tool makes no persistent changes to your system and has no ``--fix`` mode — but it's not
purely read-only either: see :ref:`what-a-run-does` for the temporary files, child processes,
and GPU initialization a run involves.

Run it as the same user who runs ``rocprofv3``, without ``sudo``. It needs no elevated
privileges, and a root process sees a different system: root opens ``/dev/kfd``
regardless of group membership, holds every capability, and ``sudo`` resets variables
such as ``LD_LIBRARY_PATH``. The ``environ.run-as-user`` check warns when the tool runs
as root on behalf of another account. Running as root is expected inside a container
whose processes run as root; there, the group-membership checks don't apply, so they
raise no warning.

.. _what-a-run-does:

What a run does
---------------

Every check declares a probe category, shown by ``--list-checks`` and in JSON output as
``probe``:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Probe
     - What it does
     - When it runs
   * - ``passive``
     - Reads files, ``/sys``, ``/proc``, and the environment, and runs read-only system
       queries such as ``ldconfig -p``. The output-directory checks create and
       immediately delete one exclusively-created temporary file.
     - Every run
   * - ``process``
     - Runs code in a child process: loads the ROCm libraries
       (``runtime.libraries-load``), imports Python modules and opens an in-memory
       ``sqlite3`` database (the ``python.*`` checks), and runs
       ``rocprofv3-avail --help``. A crash or hang in the child fails only that one
       check, never the whole report.
     - Every run
   * - ``gpu``
     - Initializes the GPU runtime: ``counters.avail-enumeration`` runs
       ``rocprofv3-avail info --pmc``, which opens every GPU agent.
     - Every run; skip it with ``--skip counters.avail-enumeration``
   * - opt-in
     - The ``smoke`` checks run ``rocprofv3-avail info`` and ``rocprofv3`` itself.
     - Only with ``--run-smoke-test`` or when named with ``--only``

Every child process has a timeout, and on timeout the child's whole process tree is
killed. A default run takes about a second on a healthy machine.

Installation
------------

``--doctor`` is built into ``rocprofv3``, so there's nothing separate to install or
put on your ``PATH``: wherever ``rocprofv3`` runs, ``rocprofv3 --doctor`` does. The
checks themselves are installed under ``<prefix>/libexec/rocprofiler-sdk/`` and aren't
meant to be run directly. ``rocprofv3`` is in the ``bin`` directory of your ROCm
installation. The following table maps each installation method to where that directory
lives:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Installation method
     - ``bin`` directory location
   * - ROCm system packages (``apt``/``dnf``)
     - ``/opt/rocm/bin`` (a link to ``/opt/rocm-X.Y.Z/bin``)
   * - TheRock system packages (``amdrocm-*``)
     - ``/opt/rocm/core-X.Y/bin``, also reachable as ``/opt/rocm/bin``
   * - TheRock Python packages (``pip install rocm[...]``)
     - The virtual environment's ``bin``; ``rocm-sdk path --bin`` prints it
   * - TheRock tarball, source build, or custom prefix
     - ``<prefix>/bin``

If ``rocprofv3`` isn't already on your ``PATH``, add that directory as shown here:

.. code-block:: bash

   export PATH=$PATH:/opt/rocm/bin

Basic usage
-----------

Run it with no further options to check everything:

.. code-block:: bash

   rocprofv3 --doctor

Options after ``--doctor`` configure the check rather than profiling, for example
``rocprofv3 --doctor --format json``. ``rocprofv3 --doctor --help`` lists them. Don't
combine ``--doctor`` with profiling options: it checks the setup and exits without
running the application.

Understanding the output
------------------------

The report is grouped by subsystem. Each line shows a status tag, the check ID, and a
short title; failing or warning checks add an indented detail line underneath with more
specifics:

.. code-block:: shell

   rocprofv3 --doctor v1.4.0  (ROCm 10.0.0)
   Checking: /opt/rocm-10.0.0  (found via install prefix of rocprofv3; ROCm system packages)

     Installation
     ============
     [ PASS ]  install.rocm-root                   ROCm root directory found
     [ PASS ]  install.sdk-library                 librocprofiler-sdk.so present
     [ PASS ]  install.kokkosp-library             librocprofiler-sdk-tool-kokkosp.so present

     Driver / Device
     ===============
     [ PASS ]  driver.kfd-device                   GPU device node exists (/dev/kfd, or /dev/dxg on WSL2)
     [ FAIL ]  driver.kfd-readable                 GPU device node readable and writable by the current user
                /dev/kfd is not readable by the current user

     Summary: 41 passed, 6 warning(s), 2 failed, 3 skipped (52 total)

     How to Fix
     ==========

     [FAIL] driver.kfd-readable - GPU device node readable and writable by the current user
         /dev/kfd is not readable by the current user

           Add your account to the render and video groups, then re-login:
             sudo usermod -a -G render,video $USER

Each check reports one of five status tags, shown in brackets in the preceding example:

.. list-table::
   :header-rows: 1

   * - Status
     - Meaning
   * - ``PASS``
     - The condition is satisfied.
   * - ``WARN``
     - The condition might be intentional or might only affect an optional feature. Warnings
       don't change the exit code.
   * - ``FAIL``
     - The condition definitely prevents some form of profiling. Fix these first.
   * - ``SKIP``
     - The check didn't run because a check it depends on didn't pass, or because you
       excluded it with ``--skip``. When a dependency failed, the detail line names it,
       so a single root cause produces one failure rather than a cascade.
   * - ``ERR``
     - The check itself broke — a ``rocprofv3 --doctor`` bug, not a problem with your
       system. `Open an issue <https://github.com/ROCm/rocm-systems/issues>`_ with the
       traceback from ``--verbose``.

After the summary, as shown in the preceding example, a **How to Fix** section repeats
every failure and actionable warning together with its remediation command.

The tool colors its output when stdout is a terminal, and disables color automatically
when stdout is redirected, when you pass ``--no-color``, or when the ``NO_COLOR``
environment variable is set.

Exit codes
----------

``rocprofv3 --doctor``'s exit code reflects the overall result, which scripts and CI can
check:

.. list-table::
   :header-rows: 1

   * - Code
     - Meaning
   * - ``0``
     - No check failed. Warnings alone don't produce a non-zero exit.
   * - ``1``
     - At least one check produced ``FAIL``.
   * - ``2``
     - ``rocprofv3 --doctor`` itself couldn't run (for example, the ``rocprofv3`` Python
       package couldn't be imported), an ``--only`` pattern matched no check, or a check
       broke (``ERR``) and none failed. A ``FAIL`` outranks an ``ERR``: a real problem
       with your system matters more than a bug in the tool itself.

Exit code ``0`` on a warning-only run is intentional: many healthy environments
legitimately set ``ROCR_VISIBLE_DEVICES`` or omit the optional ATT libraries, and failing
CI in those cases would make the tool useless. If your CI needs to be strict about warnings,
parse the JSON output instead.

Machine-readable output
-----------------------

To emit a versioned JSON document suitable for CI assertions and for
attaching to bug reports, use ``--format json``. The output is a single JSON object with the following fields:

.. code-block:: bash

   rocprofv3 --doctor --format json

.. code-block:: json

   {
     "schema_version": 1,
     "tool_version": "1.4.0",
     "rocm_version": "10.0.0",
     "rocm_root": "/opt/rocm",
     "rocm_root_source": "install prefix of rocprofv3",
     "install_kind": "system-package",
     "timestamp_utc": "2026-09-15T12:34:56Z",
     "summary": { "pass": 41, "warn": 6, "fail": 2, "skip": 3, "error": 0, "total": 52 },
     "checks": [
       {
         "id": "install.sdk-library",
         "group": "install",
         "title": "librocprofiler-sdk.so present",
         "severity": "error",
         "probe": "passive",
         "depends": ["install.rocm-root"],
         "status": "pass",
         "detail": "found at /opt/rocm/lib/librocprofiler-sdk.so",
         "remediation": "",
         "data": { "resolved_path": "/opt/rocm/lib/librocprofiler-sdk.so" }
       }
     ]
   }

``schema_version`` is an integer that is incremented only on a backward-incompatible
change. CI scripts should assert on it:

.. code-block:: bash

   rocprofv3 --doctor --format json > doctor.json
   python3 -c "
   import json, sys
   d = json.load(open('doctor.json'))
   assert d['schema_version'] == 1
   sys.exit(1 if d['summary']['fail'] else 0)
   "

Filtering checks
----------------

To list every available check ID and group, use ``--list-checks``. The output is sorted by group, then by check ID:

.. code-block:: bash

   rocprofv3 --doctor --list-checks

Run only part of the catalog with ``--only``, or exclude part of it with ``--skip``. Both
flags are repeatable, match against either the check ID or the group name, and accept
glob patterns.

* ``--only`` also runs the prerequisites of the checks it selects, so a selected check
  never reports on a foundation nobody verified. A pattern that matches nothing is an
  error (exit code ``2``), so a typo can't pass as a clean run.
* ``--skip`` takes precedence over ``--only``, and a check whose prerequisite was
  skipped reports ``SKIP``.

For example:

.. code-block:: bash

   rocprofv3 --doctor --only driver              # only the driver group
   rocprofv3 --doctor --only driver.kfd-*        # only the KFD checks
   rocprofv3 --doctor --only driver --only python
   rocprofv3 --doctor --skip counters.avail-enumeration

Skipping a specific check is the right response when you know a check doesn't apply to
your environment -- for example ``counters.avail-enumeration`` on a deliberately
restricted system where enumerating counters is expected to fail.

Verbose and quiet modes
-----------------------

To show the detail line and the structured data for every check including passing ones, use ``--verbose``. This is the most useful form to attach to a bug report:

.. code-block:: bash

   rocprofv3 --doctor --verbose

``--quiet`` prints only failing and warning checks:

.. code-block:: bash

   rocprofv3 --doctor --quiet

``--output FILE`` writes the report to a file in addition to stdout. The file copy is
always written without color codes:

.. code-block:: bash

   rocprofv3 --doctor --output rocprofv3-doctor-report.txt

Smoke testing
-------------

By default, the checks don't launch an actual profiling run. Pass ``--run-smoke-test``,
or select the ``smoke`` checks by name with ``--only``, to include them:

.. code-block:: bash

   rocprofv3 --doctor --run-smoke-test
   rocprofv3 --doctor --only smoke

The smoke checks can take 30 seconds or more, which is why they are opt-in. What they
establish is deliberately narrow:

* ``smoke.avail-info`` shows whether the HSA runtime initializes and whether
  rocprofiler-sdk enumerates at least one GPU agent.
* ``smoke.rocprofv3-launcher`` shows whether ``rocprofv3`` starts, injects its tool
  library, and runs a program (``/bin/true``) to completion. That program launches no GPU work,
  so a pass does **not** verify kernel execution, callback delivery, or trace output.
  Its output goes to a private temporary directory that is removed on success and kept
  (and reported) on failure.

How failures are diagnosed
--------------------------

A library being present on disk doesn't mean it works. ``runtime.libraries-load`` loads
``librocprofiler-sdk.so``, the rocprofv3 tool library, the HSA runtime, AQLProfile, HIP,
and ROCTx, each in a separate child process with every symbol resolved immediately. This
catches the failures that otherwise appear only when ``rocprofv3`` starts: a missing
dependency, libraries mixed from two ROCm versions, an old ``libstdc++`` from a conda
environment, or a corrupt file. When a dependency is missing, ``ldd`` lists every missing
dependency, not only the first. The child loads the libraries with the library
search path that ``rocprofv3`` gives to a profiled application (``LD_LIBRARY_PATH`` plus
``<rocm-root>/lib``), and with ``LD_PRELOAD`` cleared and the SDK's load-time
initialization and teardown disabled, so loading has no side effects.

When a check runs a command -- ``rocprofv3-avail``, or ``rocprofv3`` in the smoke tests --
and it fails or crashes, the tool matches its output against known error messages, and
the report explains what the output shows and what most likely caused it. A matched message is
treated as evidence, not as a conclusion: where one message has several possible causes,
the tool checks facts on the machine before naming one, and labels the cause ``likely``
only when those facts support it, ``possible`` otherwise. For example, ROCr reports
``HSA_STATUS_ERROR_OUT_OF_RESOURCES`` both when ``/dev/kfd`` can't be opened and when a
genuine allocation failure occurs, so the tool looks at ``/dev/kfd`` (``/dev/dxg`` under
WSL2) first:

.. code-block:: shell

   [ WARN ]  counters.avail-enumeration          rocprofv3-avail enumerates counters
              rocprofv3-avail info --pmc exited 1: the HSA runtime reported
              HSA_STATUS_ERROR_OUT_OF_RESOURCES (likely cause: the current user cannot
              open /dev/kfd read-write)

The tool suggests a fix only for the mechanism actually implicated. When the output
doesn't match any known message, the report keeps the last line of output as evidence
and suggests re-running with ``ROCPROFILER_LOG_LEVEL=info``.

In JSON output, each diagnosis is under ``data.diagnoses`` with its ``id``, the
``observation``, the candidate ``cause`` and its ``confidence`` (``high`` or
``possible``), the ``evidence`` line that matched, and the ``related_checks`` worth
reading alongside it. The following table maps each recognised message substring to its
diagnosis:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Message
     - Diagnosis
   * - ``cannot open shared object file``
     - A dependency is missing: a broken ROCm install if it is a ROCm library,
       otherwise a missing system package.
   * - ``undefined symbol``
     - Libraries from different ROCm versions are being mixed. ``likely`` when
       ``LD_LIBRARY_PATH`` or ``LD_PRELOAD`` points at another ROCm installation.
   * - ``version `GLIBCXX_...' not found``
     - An older ``libstdc++``/glibc is being loaded, often from a conda environment.
   * - ``wrong ELF class``, ``invalid ELF header``
     - A 32-bit library shadows a 64-bit one, or a library file is corrupt.
   * - ``HSA_STATUS_ERROR_OUT_OF_RESOURCES``
     - If ``/dev/kfd`` (``/dev/dxg`` under WSL2) is missing or not accessible, the GPU
       couldn't be opened. Otherwise, possibly a genuine resource limit such as
       exhausted GPU memory, or under WSL2 a runtime that doesn't take the DXG path.
   * - ``no ROCm-capable device is detected``
     - No GPU is visible to HIP. ``likely`` caused by a ``*_VISIBLE_DEVICES`` mask that
       hides every GPU (see ``environ.gpu-visibility``); any other mask is only a
       ``possible`` cause, since it may select a GPU that exists.
   * - ``could not be locked for profiling due to lack of permissions``,
       ``Required permission (CAP_PERFMON) is not set``
     - The GPU couldn't be locked for exclusive profiling without ``CAP_PERFMON``.
       Dispatch counters still work but may be inaccurate if another process profiles
       the same GPU; device-wide counter collection is degraded. Running as root grants
       the capability. ``perf_event_paranoid`` doesn't govern this, and file
       capabilities on the ``rocprofv3`` script have no effect.
   * - ``has a profiler attached to it``
     - Another process holds the GPU's profiler lock.
   * - ``depends on a newer version of KFD``,
       ``does not support locking device``
     - The amdgpu kernel driver is too old for the requested feature.
   * - ``Context has a conflict with another context``
     - Another profiling tool is already active in the process. ``likely`` when
       ``LD_PRELOAD``, ``HSA_TOOLS_LIB``, or ``ROCP_TOOL_LIBRARIES`` injects one.
   * - ``No module named '...'``
     - A Python module is missing.

A crash with none of these messages is still reported, with the signal that killed the
process and a ``possible`` cause.

ROCm installation layouts
-------------------------

ROCm isn't always under ``/opt/rocm``. ``rocprofv3 --doctor`` inspects the first of these
that contains a ROCm installation:

#. The prefix ``rocprofv3`` itself is installed under (symlinks resolved).
#. ``ROCM_PATH``, ``ROCM_HOME``, or ``ROCM_DIR``.
#. The prefix of the ``rocprofv3`` found on ``PATH``.
#. The prefix containing TheRock Python package (``_rocm_sdk_core``) importable by
   the current interpreter.
#. ``/opt/rocm``, then the newest ``/opt/rocm/core-X.Y``.

A virtual-environment prefix is searched for TheRock Python packages too, so running the
tool from an activated venv finds the ROCm inside it.

The report header and ``install.rocm-root`` show which root was chosen, how it was found,
and how it appears to have been installed (``install_kind`` in JSON output:
``system-package``, ``therock-package``, ``python-wheel``, ``distro-package``, or
``standalone``). Remediation commands follow that layout -- for example, a missing
library in a Python-package installation suggests ``pip``, not ``apt``.

When no location contains a ROCm installation, ``install.rocm-root`` fails and the checks
that depend on it report ``SKIP``, so the report shows that one cause rather than a
missing file for every library.

Override the root explicitly when several installations exist side by side:

.. code-block:: bash

   rocprofv3 --doctor --rocm-root /opt/rocm-6.2.0
   rocprofv3 --doctor --rocm-root /opt/rocm/core-10.0
   rocprofv3 --doctor --rocm-root "$(rocm-sdk path --root)"
   rocprofv3 --doctor --rocm-root ~/therock/install

Common findings
---------------

The following table explains some of the most common findings in more detail:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Check
     - What it means
   * - ``driver.kfd-device``
     - Neither ``/dev/kfd`` nor ``/dev/dxg`` (its WSL2 equivalent) exists, so the
       amdgpu/KFD driver isn't loaded, no AMD GPU is present, or (in a container) the
       device wasn't passed through. Every GPU-dependent check downstream reports
       ``SKIP``.
   * - ``driver.kfd-readable``
     - The device node exists but your account can't open it. Add yourself to the
       ``render`` and ``video`` groups and start a new login session. Under WSL2,
       ``/dev/dxg`` is normally world-accessible; check its mode instead.
   * - ``driver.render-group``
     - When your account is already listed in the group but the session isn't, the
       remediation is to re-login or run ``newgrp render`` -- not to run ``usermod``
       again.
   * - ``runtime.libraries-load``
     - A ROCm library is present but can't be loaded. The detail names the cause --
       usually a missing dependency or libraries mixed from two ROCm installations.
   * - ``install.kokkosp-library``
     - ``librocprofiler-sdk-tool-kokkosp.so`` is missing. ``rocprofv3`` resolves it on
       every launch, not only with ``--kokkos-trace``, so every run fails.
   * - ``install.metrics-path``
     - The counter database ``config.yaml`` isn't where the SDK looks. Each
       ``:``-separated entry of ``ROCPROFILER_METRICS_PATH`` is searched first, then the
       installed copy; when the variable names no usable entry, the SDK falls back to
       the installed copy and the check warns.
   * - ``counters.cap-perfmon``
     - Informational. Without ``CAP_PERFMON``, dispatch counter collection works, but the
       GPU can't be locked for exclusive profiling and device-wide counter collection is
       degraded. ``kernel.perf_event_paranoid`` is deliberately not checked:
       rocprofiler-sdk doesn't use perf events.
   * - ``driver.amdgpu-module``
     - Passes whenever ``/sys/module/amdgpu`` exists. The check shows the driver version
       only when the driver publishes one -- the DKMS driver does, the in-tree driver of
       a distribution kernel doesn't.
   * - ``driver.gpu-topology`` (WSL2)
     - Under WSL2 there's no KFD topology. The check instead locates ``libdxcore.so``
       (provided by WSL under ``/usr/lib/wsl/lib``) and ``librocdxg.so``, and fails when
       ``HSA_ENABLE_DXG_DETECTION=0``. ``driver.dri-devices``, ``driver.amdgpu-module``,
       and the group checks report ``SKIP``: they don't apply there.
   * - ``environ.gpu-visibility``
     - A ``*_VISIBLE_DEVICES`` variable restricts the GPUs ``rocprofv3`` sees. ``-1``
       hides all of them, as does an empty ``ROCR_VISIBLE_DEVICES``; an empty
       ``HIP_VISIBLE_DEVICES`` or ``CUDA_VISIBLE_DEVICES`` restricts nothing, and a
       non-empty ``HIP_VISIBLE_DEVICES`` overrides ``CUDA_VISIBLE_DEVICES``. The
       remediation unsets every such variable that is set.
   * - ``python.sqlite3``
     - Warning only: the ``rocpd`` command needs Python ``sqlite3`` to convert, merge,
       or query databases, but ``rocprofv3 --output-format rocpd`` writes them without
       it.
   * - ``environ.run-as-user``
     - The tool runs as root on behalf of another account (through ``sudo``, ``su``, or
       similar), so its results describe root rather than the user who profiles.
       Re-run it without ``sudo``. If you also run ``rocprofv3`` with ``sudo``, the
       results apply; use ``sudo -E`` for both.
   * - ``environ.ld-preload-conflict``
     - ``LD_PRELOAD`` loads a v1/v2 profiler library. Only one profiler may intercept the
       ROCm runtime at a time; ``unset LD_PRELOAD``.
   * - ``install.no-mixed-rocm``
     - ``LD_LIBRARY_PATH``, ``ROCM_PATH``, or ``ROCM_HOME`` points at a different ROCm
       installation than the one being inspected, which risks an ABI mismatch. The check
       doesn't report paths that are links into the same installation, such as
       ``/opt/rocm`` and ``/opt/rocm/core-X.Y``.
   * - ``container.device-passthrough``
     - You are in a container and the GPU device nodes weren't mounted into it. The
       remediation prints the required ``docker run`` flags.
