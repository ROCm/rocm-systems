.. meta::
  :description: Using rocprofv3 and ROCprofiler-SDK under WSL2 (Windows Subsystem for Linux)
  :keywords: ROCprofiler-SDK, rocprofv3, WSL, WSL2, DXG, DXCore, hardware counters, environment variables

.. _using-rocprofv3-on-wsl:

=========================
Using rocprofv3 on WSL2
=========================

ROCprofiler-SDK provides experimental support for running under WSL2 (Windows
Subsystem for Linux). On WSL2 the GPU is exposed through the DirectX paravirt
driver (``/dev/dxg``) and the DXCore user-mode library rather than the native
KFD driver (``/dev/kfd``). Because ``/dev/kfd`` isn't present, ROCprofiler-SDK
enumerates agents differently and arms hardware counters through the DXG
vendor-packet path instead of the KFD profiling ioctl.

This topic describes how to set up a working WSL2 profiling environment, what
``rocprofv3`` collects once it is set up, and which results warrant caution.
These findings come from testing on a gfx1150 (AMD Radeon 890M) system running
WSL2 with ``/dev/dxg`` and no ``/dev/kfd``. Other GPUs and other WSL2
installations should behave the same way, but testing hasn't covered them.

Setting up
==========

A working WSL2 profiling environment needs a matched build of the HSA runtime
and the DXG thunk, a small set of environment variables, and, optionally, a way
to override ``librocdxg`` without touching the system install. The following
subsections cover each in turn.

Matching the runtime and the DXG thunk
--------------------------------------

``librocdxg.so`` (the DXG thunk) and ``libhsa-runtime64.so`` must be built from
the same source. This is a requirement, not a recommendation.

.. important::

   ``HsaNodeProperties``, the record the thunk fills in for each KMT node, is an
   internal structure with no stable layout: it is 376 bytes in the ROCm 7.2.x
   packages and 396 bytes in the current development sources.
   ``hsaKmtGetNodeProperties()`` takes no size argument, so the thunk writes
   ``sizeof(HsaNodeProperties)`` bytes as its own build defines the struct,
   into storage the caller sized as its own build defines it. Pairing a newer
   thunk with an older runtime therefore overruns the older runtime's buffer
   and corrupts the heap; the corruption doesn't crash the program until some
   later, unconnected allocation or free touches the clobbered memory.

The read is bounded by its destination rather than by the interface.
``hsaKmtGetNodeProperties()`` copies the record whole and takes no size
parameter, so each node lands in storage deliberately longer than the
record this build expects, and a thunk that writes a longer one overruns that
slack instead of the rest of the stack frame. An optional ``DxgAbiCheck``
handshake lets a thunk report the ``sizeof(HsaNodeProperties)`` it was built
with; ROCprofiler-SDK calls the handshake when the thunk exports it, so
whether ROCprofiler-SDK can negotiate the layout depends on which
``librocdxg`` is installed: the released packages export it, so
ROCprofiler-SDK refuses a thunk that reports a different size instead of
reading it, while a thunk built from the in-tree sources doesn't export the
handshake, so ROCprofiler-SDK reads its records on the assumption that the
layouts match. Neither path can detect a layout rearranged without a change
of ``sizeof()``. None of this substitutes for a matched pair.

Late attach has an additional runtime requirement. Released external
``librocdxg`` packages don't reference-count the topology snapshot, so
ROCprofiler-SDK recognizes rocprofiler-register's explicit late-attach marker
and disables WSL agent enumeration before loading or calling ``librocdxg``.
Consequently, ``rocprof-attach`` remains unsupported on WSL until a
refcount-capable runtime containing the `#10034 snapshot ownership fix
<https://github.com/ROCm/rocm-systems/pull/10034>`_ ships and lifts this
temporary restriction. Ordinary ``rocprofv3`` launch remains safe
with an older matched package because its topology read completes in a library
constructor before ``hsa_init()``. This permits #7016 to land first and #10034
second. ``DxgAbiCheck`` is only the structure-size handshake as described in this section; its presence doesn't establish snapshot ownership.

Required environment variables
------------------------------

Export the following environment variables before running ``rocprofv3``:

.. code-block:: bash

   # Resolve the matched runtime, thunk and ROCprofiler-SDK libraries ahead of
   # any older system install under /opt/rocm
   export LD_LIBRARY_PATH=<rocm-build>/lib:${LD_LIBRARY_PATH}

   # Required with the released ROCm 7.2.x packages; already the default on
   # current builds. Never set this to 0 on WSL2.
   export HSA_ENABLE_DXG_DETECTION=1

Tool registration needs no variable of its own. A matched runtime offers its HSA
API table to rocprofiler-register on the DXG path exactly as it does on bare
metal, so a profilable run requires no extra variable. A runtime that still
suppresses registration on the DXG path blocks profiling entirely: the
application runs to completion and ``rocprofv3`` writes no output files — not
empty files, just no files. That failure mode is silent, so if a run produces
no output files, confirm the runtime and the DXG thunk share the same source
build (see `Matching the runtime and the DXG thunk`_).

``HSA_TOOLS_DISABLE_REGISTER`` is the general HSA opt-out rather than something
WSL introduced, and it disables registration when it reads exactly ``1``. Leave
it unset; a ``1`` exported elsewhere in your environment for unrelated reasons
suppresses profiling output here just as it does anywhere else.

``HSA_ENABLE_DXG_DETECTION`` is an HSA runtime variable rather than a
ROCprofiler-SDK one. On a current build the runtime enables DXG detection unless
you explicitly set the variable to ``0``, so exporting it is unnecessary but
harmless. With the released ROCm 7.2.x packages (as opposed to a build from
current development sources), it is still opt-in, and without it
``hsa_init()`` fails and every tool reports ``no supported GPU devices``, which
reads like absent hardware rather than a configuration gap. Setting it to ``0``
disables the DXG path entirely, so none of the WSL2 profiling described here
works.

.. note::

   Export these from ``~/.profile`` or from the job script rather than from
   ``~/.bashrc``. The default Ubuntu ``.bashrc`` returns early for
   non-interactive shells, so scripts and CI would never see these variables.

Overriding ``librocdxg`` at run time
------------------------------------

You can place a newer ``librocdxg`` ahead of the installed one on the loader
search path instead of replacing the system package — one way to get a
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
   — **doesn't** work, even though the preload itself succeeds. glibc registers
   a preloaded object under the path given to it and under that object's
   ``SONAME``. A request for ``librocdxg.so`` matches neither string, so the
   loader falls through to a path search and quietly loads a second, stock copy
   alongside the preloaded one. Preloading by bare name
   (``LD_PRELOAD=librocdxg.so``) also works because the loader resolves that
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
     - What rocprofv3 collected

   * - ``--hip-trace``
     - 52 HIP API records, including exactly 16 ``hipMemcpy`` calls and eight
       ``hipLaunchKernel`` calls, matching the workload.

   * - ``--hsa-trace``
     - Between 710 and 734 records across 38 distinct HSA functions, with no
       zero-length and no negative durations.

   * - ``--kernel-trace``
     - Eight dispatches, each with a kernel name, begin and end timestamps, VGPR and
       SGPR counts, and workgroup and grid dimensions.

   * - ``--memory-copy-trace``
     - 16 records: eight host-to-device and eight device-to-host.

   * - ``--marker-trace``
     - Eight ranges named ``iter0`` through ``iter7``. The application must link
       against ``librocprofiler-sdk-roctx``; see
       :ref:`using-rocprofiler-sdk-roctx`.

   * - ``--pmc``
     - Four hardware counters collected together in one pass, with no PM4
       command-buffer overflow.

   * - ``--sys-trace``
     - All of the above in one run, consistently with the individual options.

Testing also exercised tracing and counter collection together in four
different combinations of options, and neither perturbed the other's output.

Thread trace (ATT)
------------------

``--att`` works on WSL2. On native Linux, ROCprofiler-SDK takes the thread
trace buffers and signals, and performs the copies, directly through KFD on
GPUs that support it; without ``/dev/kfd`` it takes them from the HSA runtime
instead, which needs no configuration. Restricting the capture to ROCTx-selected regions with
``--selected-regions`` produced empty traces on WSL2 in testing.

Decoding the captured data needs the ``rocprof-trace-decoder`` library, which
ships separately. Point ``rocprofv3`` at the directory that holds
``librocprof-trace-decoder.so`` with ``--att-library-path`` or the
``ROCPROF_ATT_LIBRARY_PATH`` environment variable. When building the
ROCprofiler-SDK tests and samples against a decoder missing from
``${ROCM_PATH}``, set ``ROCPROFILER_SDK_TRACE_DECODER_ROOT`` to it; the build
skips the thread trace tests and samples when it finds no decoder.

Timestamps
----------

Timestamps are accurate enough for building a timeline. Rather than inspecting the values
directly, a containment check established correctness: every GPU dispatch falls
entirely inside the host interval between its ``hipLaunchKernel`` and the
following device-to-host ``hipMemcpy`` that consumes its result, and every copy
falls entirely inside the ``hipMemcpy`` that issued it. A skewed or wrongly
scaled GPU clock would break that relationship.

The reference workload has no explicit ``hipDeviceSynchronize``; the blocking
copy synchronizes it. Where a workload calls one, the
launch-to-synchronize interval contains the dispatch instead, as a variant of
the same workload confirmed.

The host timeline is the WSL VM's ``CLOCK_MONOTONIC`` in nanoseconds.
``HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY`` reports 1 GHz, the frequency measures
999.9978 MHz, and the timeline agrees with ``CLOCK_MONOTONIC`` to within 175 ns.
The GPU agent's own wall clock runs at 100 MHz (``WallClockKHz = 100000``) and
the runtime converts it, letting you compare agent-side and host-side
timestamps directly.

.. _wsl-counter-accuracy:

Hardware counter accuracy
=========================

Interpret counter values on WSL2 with care. The SQ counters are global to the
shader engines with no per-process scoping, and WDDM time-slices the GPU
among all its clients, so any GPU work another client performs inside a
dispatch's counting window counts toward that dispatch. The Windows desktop
compositor is one such client.

Apart from the zero read-backs described under the `Known issues`_, the error
is one-directional: measured values are never below the analytically expected
result, and the smallest value observed over repeated runs is the one closest to
the truth:

* A 262,144 work-item kernel (wave32, so 8,192 waves in theory) reported a
  minimum ``SQ_WAVES`` of exactly 8,192 in each of three different collection
  configurations. Maxima in the same runs reached 185,561.
* A 1,048,576 work-item kernel (32,768 waves in theory) reported 36,537,
  111,929, 162,155, 260,666, 315,977 and 394,836 across six runs. The smallest
  of those is 11 percent above theory; none is below it.

The collector itself therefore behaves correctly — reaching the floor
lands exactly on the analytically derived value — and the excess is
contamination from concurrent GPU work, not a counting error. What varies is how
much foreign activity happens to land in the window.

To get usable numbers, sample the same workload several times and take the
smallest non-zero value, or collect with the Windows desktop idle, so a
dispatch's counting window picks up little foreign work. Contamination affects short-running kernels
disproportionately, because a fixed amount of it is a much larger fraction for
a small count.
ROCprofiler-SDK can't correct for this: attributing counts back to a single
process would require filtering the hardware counters per VMID — a capability
that lives below ROCprofiler-SDK, not within it.

Known issues
============

This section documents a defect distinct from the measurement noise described
under `Hardware counter accuracy`_: rather than merely overcounting from
contamination, GRBM counters can read back exactly zero.

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
session: in testing across 354 sessions and 23,990 dispatches, the issue
affected 11 sessions (3.1 percent) and 191 dispatches (0.80 percent); all 11
bursts began
at the very first dispatch of their session, and none began mid-session,
including through 36 seconds of continuous profiling. Each burst is contiguous and doesn't
recur — once a GRBM counter reports a non-zero value it keeps reporting for the
rest of that session — and bursts ran from 9 ms to 291 ms, with a median around
54 ms. Every ``GRBM_*`` counter in a set fails together, so one GRBM counter
reading zero while another reads normally isn't a case you have to handle.

The fault sits in the GRBM block itself: testing has verified that the
counting window, the perfmon arming, the packet execution and the read-back
all work, and during a burst, profiling still correctly arms and programs the
block for the requested counter, indistinguishable from a healthy dispatch.
What remains is a counter that doesn't advance, and why it doesn't is still
unknown. It is at
least not a matter of some other consumer of GPU utilization telemetry claiming
the block: in testing, profiling alongside a Windows utilization monitor that
samples continuously produced no affected sessions at all, the opposite of
what competition for the counter slots would predict.

Nothing under your control changes the odds: the rate doesn't vary with the
number or combination of counters requested, with dispatch duration, with a
dispatch's position in the session, or with how long the GPU sits idle
beforehand. It does drift over time, though, and affected sessions arrive in
clusters rather than independently — in one test, across roughly an hour of
continuous profiling, it moved between zero and about one session in ten. A short run that
sees no zeros therefore doesn't establish that a machine is unaffected, and one
that sees several doesn't mean it is unusually bad.

SQ counters aren't completely immune to a zero read-back — 0.10 percent of
those same dispatches showed one — but those are always isolated single
dispatches rather than an opening burst, so the GRBM behavior is a distinct
effect.

Behavior that isn't a defect
=============================

Two observations are easy to mistake for WSL2 gaps and are worth stating.

The memory copy CSV has no byte-count column. Its columns are ``Kind``,
``Direction``, ``Stream_Id``, ``Source_Agent_Id``, ``Destination_Agent_Id``,
``Correlation_Id``, ``Start_Timestamp`` and ``End_Timestamp``. That's the
upstream schema on every platform, not something WSL2 drops. Byte counts are
available from the rocpd output, in ``rocpd_memory_copy.size``:

.. code-block:: bash

   rocprofv3 --memory-copy-trace --output-format rocpd -d ./out -o results -- ./my_application

Device-to-device copies within the same device don't appear in the memory copy
trace. HIP implements them as a blit kernel rather than as an SDMA copy, so they
appear in the kernel trace as ``__amd_rocclr_copyBuffer``. Copies that cross
agents are unaffected.

Not covered on WSL2
===================

This topic doesn't cover PC sampling and SPM, and testing didn't verify them
in this configuration. Those paths reach the GPU through KFD interfaces that
WSL2 doesn't expose, and ROCprofiler-SDK gates them on the presence of the
KFD device.

.. _wsl-agent-discovery:

How agent discovery works
=========================

Agent discovery on WSL2 combines two sources; ROCprofiler-SDK reads both
before any agent record becomes visible through
``rocprofiler_query_available_agents()``:

* **DXCore** (``libdxcore.so``) enumerates the GPU adapters and supplies the
  adapter identity, its marketing name and the dedicated VRAM size.
* **librocdxg** (the DXG thunk) supplies the KMT node topology — compute unit
  and SIMD counts, shader engines and arrays, wavefront size, clocks, family,
  firmware versions and the gfx target — through the same interface the HSA
  runtime itself uses on ``/dev/dxg`` systems.

ROCprofiler-SDK loads ``librocdxg.so`` at run time with ``dlopen``, reusing the
copy the HSA runtime already loaded when there's one, so it needs no
build-time configuration and no link dependency. ROCprofiler-SDK builds
records once, at enumeration time, and never modifies them afterwards, so
pre-HSA consumers such as ``rocprofv3-avail`` see the same complete records as
a profiling run.

Windows LUID pairs each adapter with its KMT node. The PCI device ID is only a
fallback for nodes that report no LUID, and it has to identify exactly one of
them: two identical GPUs that report no LUID are indistinguishable, so
ROCprofiler-SDK omits both with a diagnostic instead of guessing at one. At
most one adapter claims each node. ROCprofiler-SDK likewise logs and omits an
adapter with no matching node, or whose node reports an incomplete topology,
rather than publishing it with placeholder values.

The read uses the thunk's ordinary KMT entry points — ``hsaKmtOpenKFD``,
``hsaKmtAcquireSystemProperties``, ``hsaKmtGetNodeProperties``,
``hsaKmtReleaseSystemProperties`` and ``hsaKmtCloseKFD``. All of them must
resolve before ROCprofiler-SDK calls any of them, so it refuses a thunk
missing one, naming the missing symbol, before opening anything.

Missing agents
--------------

A ``librocdxg`` that doesn't export what the read needs, or that offers the
``DxgAbiCheck`` handshake and reports a different record layout, yields **no GPU
agents at all** in ROCprofiler-SDK. The log separates the two: it names a
missing entry point as such, while a rejected layout reads ``wsl topology:
... rejected the HsaNodeProperties layout``. ROCprofiler-SDK can turn away
only a thunk that offers the handshake, for its layout; it reads one that
doesn't offer the handshake on the assumption that the layouts match. This
is visible rather than silent, but it still causes a real problem: the HSA
runtime loads the same thunk and still reports its GPUs, so ROCprofiler-SDK
and HSA disagree about which agents exist. Partially described topologies land in the
same place: ROCprofiler-SDK publishes the GPU it could describe and omits the
other, while HSA keeps reporting both.

The KMT node ID the thunk reported, never a dense ordinal, pairs agent
records with HSA agents, so the GPUs ROCprofiler-SDK published stay correctly
paired no matter which ones it omitted. ROCprofiler-SDK reports HSA GPUs left
without a counterpart as an unsupported-profiling condition naming their KMT
node IDs: profiling is unavailable on those GPUs, the ones that did pair
remain fully profilable, and the application under test keeps running. The
log names the rejected symbol or adapter; the fix is to bring the WSL ROCm
runtime package and ROCprofiler-SDK back into a matched pair.

Environment variables
=====================

``HSA_TOOLS_DISABLE_REGISTER``
   An existing HSA runtime variable that disables tool registration when set to ``1``, leaving rocprofiler-register without an HSA API table to
   intercept. It takes no WSL2-specific value; leave it unset. Setting it to
   ``1``, here as on any platform, makes ``rocprofv3`` produce no output
   files.

``HSA_ENABLE_DXG_DETECTION``
   An HSA runtime variable. Current builds enable DXG detection unless you
   explicitly set this to ``0``; the released ROCm 7.2.x packages require
   setting it to ``1``, and otherwise fail ``hsa_init()`` and report ``no
   supported GPU devices``. Setting it to ``0`` disables the DXG path on any
   build.

.. _wsl-vendor-packet:

``WSLKMT_VENDOR_PACKET``
   Gates whether the libhsakmt DXG path honors the vendor-specific PM4 IB
   packets that AQLprofile emits for hardware counter collection. When unset
   (the libhsakmt default), libhsakmt silently drops the embedded PM4 IB and
   every ``Counter_Value`` reads back zero. ROCprofiler-SDK sets it to ``1``
   automatically when it detects a WSL GPU environment, but only if you
   haven't already set it yourself, so your explicit setting always takes
   precedence. You don't need to set it to enable this behavior; export
   ``WSLKMT_VENDOR_PACKET=0`` only if you want to opt out.

``ROCPROFILER_FORCE_GFX``
   Overrides the ``gfx`` target name used to look up counter definitions
   (gfx target is the lookup key in ``config.yaml``). The target normally comes
   from the KMT node's engine ID, which already honors the HSA runtime's own
   ``HSA_OVERRIDE_GFX_VERSION``; set this variable to override both. The value
   must be of the form ``gfx<NNN>`` with at least three digits, the last of
   which may be a lowercase hex digit (as in ``gfx90a``); a malformed value
   triggers a warning, and ROCprofiler-SDK falls back to the node-reported
   target.

``ROCPROFILER_FORCE_PLATFORM``
   Overrides platform autodetection, which otherwise selects the KFD enumerator
   when KFD sysfs is present and the WSL enumerator when ``/dev/dxg`` and
   ``libdxcore.so`` are. Accepts ``gnulinux`` or ``wsl`` on Linux;
   ROCprofiler-SDK logs and ignores a value not built into the binary. This is
   a diagnostic escape hatch that normal use doesn't need.
