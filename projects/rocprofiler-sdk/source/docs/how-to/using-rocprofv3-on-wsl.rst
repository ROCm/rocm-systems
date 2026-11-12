.. meta::
  :description: Using rocprofv3 under WSL2 (Windows Subsystem for Linux)
  :keywords: ROCprofiler-SDK, rocprofv3, WSL, WSL2, DXG, hardware counters, environment variables

.. _using-rocprofv3-on-wsl:

=========================
Using rocprofv3 on WSL2
=========================

ROCprofiler-SDK has experimental support for WSL2 (Windows Subsystem for
Linux). On WSL2, the GPU is exposed through the DirectX paravirtual driver
(``/dev/dxg``) instead of the KFD driver (``/dev/kfd``). ``rocprofv3`` handles
this automatically, and you run it with the same commands as on native Linux.

This support has been validated on an AMD Radeon 890M (gfx1150) GPU.

Set up the environment
======================

1. Use a ``librocdxg`` and HSA runtime from the same ROCm release. Mixing
   versions isn't supported and can crash the profiled application.

2. On ROCm 7.2.x, enable the DXG path in the HSA runtime. Later releases enable
   it by default.

   .. code-block:: bash

      export HSA_ENABLE_DXG_DETECTION=1

   Don't set ``HSA_ENABLE_DXG_DETECTION`` to ``0``. That disables GPU access on
   WSL2.

3. Make sure ``HSA_TOOLS_DISABLE_REGISTER`` isn't set to ``1``. That setting
   prevents ``rocprofv3`` from collecting any data on any platform.

.. note::

   Set environment variables in ``~/.profile`` or in your job script, not in
   ``~/.bashrc``. The default Ubuntu ``~/.bashrc`` exits early for
   non-interactive shells, so scripts and CI jobs don't see variables set there.

To check the setup, run ``rocprofv3 --doctor``. For details, see
:ref:`using-rocprofv3-doctor`.

Profile an application
======================

Run ``rocprofv3`` the same way as on native Linux:

.. code-block:: bash

   rocprofv3 --sys-trace -d ./out -o results -- ./my_application

The following options are supported on WSL2:

* ``--hip-trace``, ``--hsa-trace``, ``--kernel-trace``, and
  ``--memory-copy-trace``
* ``--marker-trace``. The application must link against
  ``librocprofiler-sdk-roctx``. For details, see
  :ref:`using-rocprofiler-sdk-roctx`.
* ``--pmc``. Read :ref:`wsl-counter-accuracy` before you interpret the results.
* ``--sys-trace``
* ``--att``. Decoding requires the separately distributed
  ``rocprof-trace-decoder`` library. Pass its directory with
  ``--att-library-path`` or ``ROCPROF_ATT_LIBRARY_PATH``. For details, see
  :ref:`using-thread-trace`.

Host and GPU timestamps share one timeline, so you can compare them directly.

Limitations
===========

* Process attachment (``rocprof-attach``) isn't supported on WSL2.
* PC sampling and SPM aren't supported on WSL2. Both require the KFD driver.
* Thread trace with ``--selected-regions`` has produced empty traces on WSL2 in
  testing.

.. _wsl-counter-accuracy:

Hardware counter accuracy
-------------------------

On WSL2, hardware counters measure all GPU activity, not just your application.
Windows shares the GPU among all its clients, including the desktop compositor,
and any of their work that runs during a dispatch is counted toward that
dispatch. ROCprofiler-SDK can't separate it out.

Apart from occasional zero readings, the extra activity only ever adds to a
count. To get usable values:

* Run the workload several times and use the smallest non-zero value.
* Keep the Windows desktop idle while you collect.
* Be most cautious with short kernels, where a small amount of extra activity
  is a large share of the count.

A single dispatch can occasionally read zero on an ``SQ_*`` counter. Discard
it.

GRBM counters can read zero at the start of a session
-----------------------------------------------------

In some sessions, ``GRBM_*`` counters read zero for up to the first 300 ms.
Metrics derived from them, such as ``GPU_UTIL`` and ``VALUBusy``, are also
wrong for those dispatches. If you need GRBM-based metrics, discard the first
300 ms of dispatches or add a short warm-up phase to the application. A clean
run doesn't mean later runs are unaffected.

Troubleshooting
===============

``rocprofv3`` writes no output files
   The HSA runtime didn't register ``rocprofv3``. Check that
   ``HSA_TOOLS_DISABLE_REGISTER`` isn't set to ``1``, and that the HSA runtime
   and ``librocdxg`` come from the same ROCm release.

Tools report ``no supported GPU devices``
   On ROCm 7.2.x, set ``HSA_ENABLE_DXG_DETECTION=1``. On any release, check
   that it isn't set to ``0``.

All counter values are zero
   Check whether ``WSLKMT_VENDOR_PACKET`` is set to ``0``. For details, see
   :ref:`wsl-environment-variables`.

``rocprofv3`` finds fewer GPUs than HIP does
   ROCprofiler-SDK skips a GPU when it can't read that GPU's topology.
   Profiling still works on the other GPUs. To see why a GPU was skipped, run
   with ``ROCPROFILER_LOG_LEVEL=info``. The usual cause is a ``librocdxg`` that
   doesn't match ROCprofiler-SDK; install matching packages.

The application crashes during profiling
   Check that ``librocdxg`` and the HSA runtime come from the same ROCm release.
   A mismatched pair can corrupt memory, and the crash can appear later in
   unrelated code.

.. _wsl-environment-variables:

Environment variables
=====================

``HSA_ENABLE_DXG_DETECTION``
   HSA runtime variable that enables the DXG path. Required as ``1`` on ROCm
   7.2.x; enabled by default in later releases. Setting it to ``0`` disables
   GPU access on WSL2.

``HSA_TOOLS_DISABLE_REGISTER``
   HSA runtime variable. When set to ``1``, profiling tools can't attach and
   ``rocprofv3`` writes no output. Leave it unset.

``WSLKMT_VENDOR_PACKET``
   Enables the packets that hardware counter collection uses. ROCprofiler-SDK
   sets it to ``1`` automatically on WSL2 unless you've already set it. If it's
   ``0``, every counter reads zero.

The following variables are for diagnosis only. You don't need them in normal
use.

``ROCPROFILER_FORCE_GFX``
   Overrides the ``gfx`` target used to look up counter definitions, for
   example ``gfx1150``. Takes precedence over ``HSA_OVERRIDE_GFX_VERSION``. An
   invalid value is ignored with a warning.

``ROCPROFILER_FORCE_PLATFORM``
   Overrides platform detection. Accepts ``gnulinux`` or ``wsl``.
