.. meta::
  :description: Using rocprofv3 and ROCprofiler-SDK under WSL2 (Windows Subsystem for Linux)
  :keywords: ROCprofiler-SDK, rocprofv3, WSL, WSL2, DXG, DXCore, hardware counters, environment variables

.. _using-rocprofv3-on-wsl:

=========================
Using rocprofv3 on WSL2
=========================

ROCprofiler-SDK has experimental support for running under WSL2 (Windows
Subsystem for Linux). On WSL2 the GPU is exposed through the DirectX paravirt
driver (``/dev/dxg``) and the DXCore user-mode library rather than the native
KFD driver (``/dev/kfd``). Because ``/dev/kfd`` is not present, ROCprofiler-SDK
enumerates agents differently and arms hardware counters through the DXG
vendor-packet path instead of the KFD profiling ioctl.

This page describes how to set up a working WSL2 profiling environment, what
``rocprofv3`` collects once it is set up, and which results should not be taken
at face value. The behavior described here was measured on a gfx1150
(AMD Radeon 890M) system running WSL2 with ``/dev/dxg`` and no ``/dev/kfd``.
Other GPUs and other WSL2 installations are expected to behave the same way but
have not been measured.

Setting up
==========

Matching the runtime and the DXG thunk
--------------------------------------

``librocdxg.so`` (the DXG thunk) and ``libhsa-runtime64.so`` must be built from
the same source. This is a requirement, not a recommendation.

.. important::

   ``HsaNodeProperties``, the record the thunk fills in for each KMT node, is an
   internal structure with no stable layout: it is 376 bytes in the ROCm 7.2.x
   packages and 396 bytes in current development sources.
   ``hsaKmtGetNodeProperties()`` carries no size argument, so the thunk writes
   ``sizeof()`` bytes as *it* knows the type, into storage sized as the caller
   knows the type. Pairing a newer thunk with an older runtime therefore
   overruns the older side's buffer and corrupts the heap, which typically
   surfaces later as a crash somewhere unrelated.

The read is bounded by its destination rather than by the interface.
``hsaKmtGetNodeProperties()`` copies the record whole and takes no size
parameter, so each node is read into storage deliberately longer than the
record this build expects, and a thunk that writes a longer one overruns that
slack instead of the rest of the stack frame. An optional ``DxgAbiCheck``
handshake, by which a thunk reports the ``sizeof(HsaNodeProperties)`` it was
built with, is called when the thunk exports it, so whether the layout can be
negotiated depends on which ``librocdxg`` is installed: the released packages
export it, and a thunk reporting a different size is refused rather than read,
while a thunk built from the in-tree sources does not, and its records are read
on the assumption that the layouts match. A layout rearranged without a change
of ``sizeof()`` cannot be detected either way. None of this substitutes for a
matched pair.

Late attach has an additional runtime requirement. Released external
``librocdxg`` packages do not reference-count the topology snapshot, so
ROCprofiler-SDK recognizes rocprofiler-register's explicit late-attach marker
and disables WSL agent enumeration before loading or calling ``librocdxg``.
Consequently, ``rocprof-attach`` is unsupported on WSL until a refcount-capable
runtime containing the `#10034 snapshot ownership fix
<https://github.com/ROCm/rocm-systems/pull/10034>`_ is installed and this
temporary restriction is removed. Ordinary ``rocprofv3`` launch remains safe
with an older matched package because its topology read completes in a library
constructor before ``hsa_init()``. This permits #7016 to land first and #10034
second. ``DxgAbiCheck`` is only the structure-size handshake described above;
its presence does not establish snapshot ownership.

Required environment variables
------------------------------

Export the following before running ``rocprofv3``:

.. code-block:: bash

   # Resolve the matched runtime, thunk and ROCprofiler-SDK libraries ahead of
   # any older system install under /opt/rocm
   export LD_LIBRARY_PATH=<rocm-build>/lib:${LD_LIBRARY_PATH}

   # Required with stock ROCm 7.2.x packages; already the default on current
   # builds. Never set this to 0 on WSL2.
   export HSA_ENABLE_DXG_DETECTION=1

Tool registration needs no variable of its own. A matched runtime offers its HSA
API table to rocprofiler-register on the DXG path exactly as it does on bare
metal, so nothing has to be exported to make a run profilable. A runtime that
still suppresses registration on the DXG path cannot be profiled at all: the
application runs to completion and ``rocprofv3`` writes no output files — not
empty files, no files. That failure mode is silent, so if a run produces
nothing, confirm the runtime is the matched one described above.

``HSA_TOOLS_DISABLE_REGISTER`` is the general HSA opt-out rather than something
WSL introduced, and it disables registration when it reads exactly ``1``. Leave
it unset; a ``1`` exported elsewhere in your environment for unrelated reasons
suppresses profiling output here just as it does anywhere else.

``HSA_ENABLE_DXG_DETECTION`` is an HSA runtime variable rather than a
ROCprofiler-SDK one. On a current build the runtime enables DXG detection unless
the variable is explicitly set to ``0``, so exporting it is unnecessary but
harmless. With stock ROCm 7.2.x packages it is still opt-in, and without it
``hsa_init()`` fails and every tool reports ``no supported GPU devices``, which
reads like absent hardware rather than a configuration gap. Setting it to ``0``
disables the DXG path entirely and nothing on this page works.

.. note::

   Export these from ``~/.profile`` or from the job script rather than from
   ``~/.bashrc``. The default Ubuntu ``.bashrc`` returns early for
   non-interactive shells, so scripts and CI would never see them.

Overriding ``librocdxg`` at run time
------------------------------------

A newer ``librocdxg`` can be placed ahead of the installed one on the loader
search path instead of replacing the system package, which is one way to get a
matched pair without touching ``/opt/rocm``. Both the HSA runtime and
ROCprofiler-SDK ask for the unversioned name ``librocdxg.so``, so a directory
holding that one symlink, placed first on ``LD_LIBRARY_PATH``, redirects both
consumers:

.. code-block:: bash

   mkdir -p ~/wsl-dxg-lib
   ln -sf <path-to-matched-build>/librocdxg.so ~/wsl-dxg-lib/librocdxg.so
   export LD_LIBRARY_PATH=~/wsl-dxg-lib:${LD_LIBRARY_PATH}

.. warning::

   Preloading by absolute path — ``LD_PRELOAD=/path/to/librocdxg.so.<version>``
   — does **not** work, even though the preload itself succeeds. glibc registers
   a preloaded object under the path it was given and under that object's
   ``SONAME``. A request for ``librocdxg.so`` matches neither string, so the
   loader falls through to a path search and quietly loads a second, stock copy
   alongside the preloaded one. Preloading by bare name
   (``LD_PRELOAD=librocdxg.so``) does work, because the loader resolves that
   name through an ordinary search.

What rocprofv3 collects on WSL2
===============================

The record counts in this section come from a HIP application that runs eight
iterations, each of which copies 1 MiB from host to device, launches a kernel,
and copies the result back. Run it under ``rocprofv3`` the same way as on native
Linux:

.. code-block:: bash

   rocprofv3 --sys-trace -d ./out -o results -- ./my_application

.. list-table:: Collection verified on WSL2
   :header-rows: 1

   * - Option
     - What was collected

   * - ``--hip-trace``
     - 52 HIP API records, including exactly 16 ``hipMemcpy`` calls and 8
       ``hipLaunchKernel`` calls, matching the workload.

   * - ``--hsa-trace``
     - Between 710 and 734 records across 38 distinct HSA functions, with no
       zero-length and no negative durations.

   * - ``--kernel-trace``
     - 8 dispatches, each with a kernel name, begin and end timestamps, VGPR and
       SGPR counts, and workgroup and grid dimensions.

   * - ``--memory-copy-trace``
     - 16 records: 8 host-to-device and 8 device-to-host.

   * - ``--marker-trace``
     - 8 ranges named ``iter0`` through ``iter7``. The application must link
       against ``librocprofiler-sdk-roctx``; see
       :ref:`using-rocprofiler-sdk-roctx`.

   * - ``--pmc``
     - Four hardware counters collected together in one pass, with no PM4
       command-buffer overflow.

   * - ``--sys-trace``
     - All of the above in one run, consistently with the individual options.

Tracing and counter collection were also exercised together in four different
combinations of options, and neither perturbed the other's output.

Timestamps
----------

Timestamps are usable for building a timeline. Rather than inspecting the values
directly, correctness was checked by containment: every GPU dispatch falls
entirely inside the host interval between its ``hipLaunchKernel`` and the
following device-to-host ``hipMemcpy`` that consumes its result, and every copy
falls entirely inside the ``hipMemcpy`` that issued it. A skewed or wrongly
scaled GPU clock would break that relationship.

The reference workload has no explicit ``hipDeviceSynchronize``; the blocking
copy is what synchronizes it. Where a workload does call one, the dispatch is
contained by the launch-to-synchronize interval instead, as a variant of the
same workload confirmed.

The host timeline is the WSL VM's ``CLOCK_MONOTONIC`` in nanoseconds.
``HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY`` reports 1 GHz, the frequency measures
999.9978 MHz, and the timeline agrees with ``CLOCK_MONOTONIC`` to within 175 ns.
The GPU agent's own wall clock runs at 100 MHz (``WallClockKHz = 100000``) and
the runtime converts it, so agent-side and host-side timestamps can be compared
directly.

.. _wsl-counter-accuracy:

Hardware counter accuracy
=========================

Counter values on WSL2 need to be interpreted with care. The SQ counters are
global to the shader engines with no per-process scoping, and WDDM time-slices
the GPU among all its clients, so any GPU work performed for another client
inside a dispatch's counting window is attributed to that dispatch. The Windows
desktop compositor is one such client.

Apart from the zero read-backs described under `Known issues`_ below, the error
is one-directional: measured values are never below the analytically expected
result, and the smallest value observed over repeated runs is the one closest to
the truth:

* A 262,144 work-item kernel (wave32, so 8,192 waves in theory) reported a
  minimum ``SQ_WAVES`` of exactly 8,192 in each of three different collection
  configurations. Maxima in the same runs reached 185,561.
* A 1,048,576 work-item kernel (32,768 waves in theory) reported 36,537,
  111,929, 162,155, 260,666, 315,977 and 394,836 across six runs. The smallest
  of those is 11 percent above theory; none is below it.

The collector itself is therefore behaving correctly — where the floor is
reached it lands exactly on the analytically derived value — and the excess is
contamination from concurrent GPU work, not a counting error. What varies is how
much foreign activity happened to land in the window.

To get usable numbers, sample the same workload several times and take the
smallest non-zero value, or collect with the Windows desktop idle so there is
little foreign work to absorb. Short-running kernels are affected
disproportionately, because a fixed amount of contamination is a much larger
fraction of a small count.
ROCprofiler-SDK cannot correct for this: attributing counts back to a single
process would require the hardware counters to be filtered per VMID, below the
SDK.

Known issues
============

GRBM counters read zero at the start of a session
-------------------------------------------------

``GRBM_*`` counters can read back zero on the opening dispatches of a profiling
session, on dispatches that demonstrably ran and produced waves. Such a zero is
invalid data rather than a measurement, and it matters because derived metrics
divide by these counters: ``GPU_UTIL`` and ``CU_UTILIZATION`` divide by
``GRBM_COUNT``, and ``VALUBusy``, ``SALUBusy``, ``MemUnitStalled``,
``LDSBankConflict`` and ``MeanOccupancyPerCU`` depend on ``GRBM_GUI_ACTIVE``.
Discard the affected dispatches rather than letting the zero propagate into a
metric.

When GRBM-derived metrics matter, discard the first 300 ms of dispatches in the
session, which covers the longest burst observed, or give the application a
short warm-up phase so that the affected window falls before the region of
interest.

The behavior is narrow and well characterized. It is confined to the start of a
session: across 354 sessions and 23,990 dispatches, 11 sessions (3.1 percent)
and 191 dispatches (0.80 percent) were affected, all 11 bursts began at the very
first dispatch of their session, and none began mid-session, including through
36 seconds of continuous profiling. Each burst is contiguous and does not
recur — once a GRBM counter reports a non-zero value it keeps reporting for the
rest of that session — and bursts ran from 9 ms to 291 ms, with a median around
54 ms. Every ``GRBM_*`` counter in a set fails together, so one GRBM counter
reading zero while another reads normally is not a case you have to handle.

The fault sits in the GRBM block itself: the counting window, the perfmon
arming, the packet execution and the read-back have all been verified working,
and during a burst the block is correctly armed and correctly programmed for the
counter requested, indistinguishable from a healthy dispatch. What is left is a
counter that does not advance, and why it does not is still unknown. It is at
least not a matter of some other consumer of GPU utilization telemetry claiming
the block — profiling alongside a Windows utilization monitor that samples
continuously produced no affected sessions at all, the opposite of what
competition for the counter slots would predict.

Nothing under your control changes the odds: the rate does not vary with the
number or combination of counters requested, with dispatch duration, with a
dispatch's position in the session, or with how long the GPU sat idle
beforehand. It does drift over time, though, and affected sessions arrive in
clusters rather than independently — across roughly an hour of continuous
profiling it moved between zero and about one session in ten. A short run that
sees no zeros therefore does not establish that a machine is unaffected, and one
that sees several does not mean it is unusually bad.

SQ counters are not completely immune to a zero read-back — 0.10 percent of
dispatches showed one — but those are always isolated single dispatches rather
than an opening burst, so the GRBM behavior is a distinct effect.

Behavior that is not a defect
=============================

Two observations are easy to mistake for WSL2 gaps and are worth stating
explicitly.

The memory copy CSV has no byte-count column. Its columns are ``Kind``,
``Direction``, ``Stream_Id``, ``Source_Agent_Id``, ``Destination_Agent_Id``,
``Correlation_Id``, ``Start_Timestamp`` and ``End_Timestamp``. That is the
upstream schema on every platform, not something WSL2 drops. Byte counts are
available from the rocpd output, in ``rocpd_memory_copy.size``:

.. code-block:: bash

   rocprofv3 --memory-copy-trace --output-format rocpd -d ./out -o results -- ./my_application

Device-to-device copies within the same device do not appear in the memory copy
trace. HIP implements them as a blit kernel rather than as an SDMA copy, so they
appear in the kernel trace as ``__amd_rocclr_copyBuffer``. Copies that cross
agents are unaffected.

Not covered on WSL2
===================

PC sampling, thread trace (ATT) and SPM are not covered by this page and were
not verified in this configuration. Those paths reach the GPU through KFD
interfaces that WSL2 does not expose, and ROCprofiler-SDK gates them on the
presence of the KFD device.

.. _wsl-agent-discovery:

How agents are discovered
=========================

Agent discovery on WSL2 combines two sources, both read before any agent record
becomes visible through ``rocprofiler_query_available_agents()``:

* **DXCore** (``libdxcore.so``) enumerates the GPU adapters and supplies the
  adapter identity, its marketing name and the dedicated VRAM size.
* **librocdxg** (the DXG thunk) supplies the KMT node topology — compute unit
  and SIMD counts, shader engines and arrays, wavefront size, clocks, family,
  firmware versions and the gfx target — through the same interface the HSA
  runtime itself uses on ``/dev/dxg`` systems.

ROCprofiler-SDK loads ``librocdxg.so`` at run time with ``dlopen``, reusing the
copy the HSA runtime already loaded when there is one, so no build-time
configuration and no link dependency are involved. Records are built once, at
enumeration time, and are never modified afterwards, so pre-HSA consumers such
as ``rocprofv3-avail`` see the same complete records as a profiling run.

Each adapter is paired with its KMT node by Windows LUID. The PCI device id is
only a fallback for nodes that report no LUID, and it has to identify exactly
one of them: two identical GPUs that report no LUID cannot be told apart, so
both are omitted with a diagnostic rather than one being guessed at. Each node
is claimed by at most one adapter. An adapter with no matching node, or whose
node reports an incomplete topology, is likewise logged and omitted rather than
published with placeholder values.

The read uses the thunk's ordinary KMT entry points — ``hsaKmtOpenKFD``,
``hsaKmtAcquireSystemProperties``, ``hsaKmtGetNodeProperties``,
``hsaKmtReleaseSystemProperties`` and ``hsaKmtCloseKFD``. All of them must
resolve before any of them is called, so a thunk missing one is refused, with
the missing symbol named, before anything is opened.

When agents are missing
-----------------------

A ``librocdxg`` that does not export what the read needs, or that offers the
``DxgAbiCheck`` handshake and reports a different record layout, yields **no GPU
agents at all** in ROCprofiler-SDK. The log separates the two: a missing entry
point is named as such, while a rejected layout reads ``wsl topology: ...
rejected the HsaNodeProperties layout``. Only a thunk offering the handshake can
be turned away for its layout; one that does not offer it is read on the
assumption that the layouts match. This is visible rather than silent, but it is
not harmless: the HSA runtime loads the
same thunk and still reports its GPUs, so ROCprofiler-SDK and HSA disagree about
which agents exist. Partially described topologies land in the same place — the
GPU that could be described is published and the other is omitted, while HSA
keeps reporting both.

Agent records are paired with HSA agents by the KMT node id the thunk reported,
never by a dense ordinal, so the GPUs that were published stay correctly paired
no matter which ones were omitted. HSA GPUs left without a counterpart are
reported as an unsupported-profiling condition naming their KMT node ids:
profiling is unavailable on those GPUs, the ones that did pair remain fully
profilable, and the application under test keeps running. The log names the
symbol or adapter that was rejected; the fix is to bring the WSL ROCm runtime
package and ROCprofiler-SDK back into a matched pair.

Environment variables
=====================

``HSA_TOOLS_DISABLE_REGISTER``
   An existing HSA runtime variable that disables tool registration when it
   reads ``1``, leaving rocprofiler-register without an HSA API table to
   intercept. It takes no WSL2-specific value and should be left unset; set to
   ``1``, here as anywhere else, ``rocprofv3`` produces no output files at all.

``HSA_ENABLE_DXG_DETECTION``
   An HSA runtime variable. Current builds enable DXG detection unless this is
   explicitly set to ``0``; stock ROCm 7.2.x packages require ``1`` to be set,
   and otherwise fail ``hsa_init()`` and report ``no supported GPU devices``.
   Setting it to ``0`` disables the DXG path on any build.

.. _wsl-vendor-packet:

``WSLKMT_VENDOR_PACKET``
   Gates whether the libhsakmt DXG path honors the vendor-specific PM4 IB
   packets that AQLprofile emits for hardware counter collection. When unset
   (the libhsakmt default), the embedded PM4 IB is silently dropped and every
   ``Counter_Value`` reads back zero. ROCprofiler-SDK sets it to ``1``
   automatically when it detects a WSL GPU environment, using a non-overwriting
   write so an explicit user setting always wins. You normally do not need to
   set it; export ``WSLKMT_VENDOR_PACKET=0`` to opt out.

``ROCPROFILER_FORCE_GFX``
   Overrides the ``gfx`` target name used to look up counter definitions
   (``config.yaml`` is keyed by gfx target). The target normally comes from the
   KMT node's engine id, which already honors the HSA runtime's own
   ``HSA_OVERRIDE_GFX_VERSION``; set this variable to override both. The value
   must be of the form ``gfx<NNN>`` with at least three digits, the last of
   which may be a lowercase hex digit (as in ``gfx90a``); a malformed value is
   ignored with a warning and the node-reported target is used.

``ROCPROFILER_FORCE_PLATFORM``
   Overrides platform autodetection, which otherwise selects the KFD enumerator
   when KFD sysfs is present and the WSL enumerator when ``/dev/dxg`` and
   ``libdxcore.so`` are. Accepts ``gnulinux`` or ``wsl`` on Linux; a value not
   built into the binary is logged and ignored. This is a diagnostic escape
   hatch and is not needed in normal use.
