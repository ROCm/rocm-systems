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
takes a different path for agent enumeration and hardware counter collection.

This page documents how agents are discovered on WSL2 and the environment
variables that control that behavior.

What works on WSL2
==================

* **API tracing** (HIP, HSA, kernel dispatch, memory copy/allocation, marker
  / ROCTx): works as on native Linux.
* **Hardware counter collection**: performance counters are collected with
  correct, non-zero, per-instance values through the DXG vendor-packet path
  (see :ref:`wsl-vendor-packet` below). Both single-counter and multi-counter
  collection are supported. Multi-counter collection previously hit a per-queue
  PM4 command-buffer frame-size limit in libhsakmt; that limit is now computed
  from the device geometry instead of a fixed bound, so multi-counter passes
  fit. This was verified on a gfx11-class GPU under WSL2 with a four-counter
  collection (``SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE SQ_INSTS_VALU``): all four
  counters were reported with no PM4 command-buffer overflow.

Known limitations
==================

* **PC sampling and Advanced Thread Trace (ATT)**: the software/decode layers
  build and pass their unit tests, but the end-to-end paths require KFD
  features that are not available on WSL2 and are therefore disabled.
* **SPM (Streaming Performance Monitor)**: not available; the gfx11-class
  counter database does not expose SPM-capable counters.

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

ROCprofiler-SDK loads ``librocdxg.so`` at run time with ``dlopen`` (reusing the
copy the HSA runtime already loaded when there is one) and negotiates the ABI
with the thunk before reading anything, so no build-time configuration is
required and no separate link dependency exists. Pre-HSA consumers such as
``rocprofv3-avail`` and tool initialization get the same complete records as a
profiling run: agent records are built once, at enumeration time, and are never
modified afterwards.

Each adapter is paired with its KMT node by Windows LUID. The PCI device id is
only a fallback for the nodes a LUID cannot speak for, and it has to identify
exactly one of them: two identical GPUs that report no LUID cannot be told
apart, so both are omitted with a diagnostic rather than one being guessed at.
Each node is claimed by at most one adapter. An adapter with no matching node,
or whose node reports an incomplete topology, is likewise logged and omitted
rather than published with placeholder values.

Version requirement
-------------------

The topology ABI (``DxgAbiCheck``, ``DxgAcquireTopologySnapshot``,
``DxgGetNodeTopology``, ``DxgReleaseTopologySnapshot``) is a hard requirement,
not a preference. There is no fallback to the older ``hsaKmtGetNodeProperties``
path: that call exchanges a full ``HsaNodeProperties``, whose layout is not
version-stable, so calling it across a package boundary is exactly the
mismatch ``DxgAbiCheck`` exists to prevent.

A ``librocdxg`` that predates this ABI, or one that rejects the handshake,
therefore yields **no GPU agents at all**. That is not harmless: the HSA runtime
loads the same thunk and still reports its GPUs, so rocprofiler-SDK and HSA
disagree about which agents exist.

Partially described topologies land in the same place. If one of two adapters
reports no LUID, is ambiguous against its node, or describes an incomplete
topology, rocprofiler-SDK publishes the GPU it could describe and omits the
other, while HSA keeps reporting both.

On WSL, agent records are paired with HSA agents by the real KMT node id the
thunk reported, so the GPUs that were published stay correctly paired no matter
which ones were omitted — the mapping never falls back to a dense ordinal that
would silently pair a published GPU with the wrong HSA agent. HSA GPUs left
without a counterpart are reported as an unsupported-profiling condition naming
their KMT node ids: profiling is unavailable on those GPUs, the ones that did
pair remain fully profilable, and the application under test keeps running. It
is not treated as the fatal internal inconsistency the equivalent mismatch means
on bare metal, where rocprofiler-SDK and HSA read the same KFD sysfs tree and
cannot legitimately disagree. The log names the symbol, ABI version or adapter
that was rejected; the fix is to update the WSL ROCm runtime package to one
matching this rocprofiler-SDK.

Environment variables
======================

.. _wsl-vendor-packet:

``WSLKMT_VENDOR_PACKET``
   Gates whether the libhsakmt DXG path honors the vendor-specific PM4 IB
   packets that AQLprofile emits for hardware counter collection. When unset
   (the libhsakmt default), the embedded PM4 IB is silently dropped and every
   ``Counter_Value`` reads back zero. ROCprofiler-SDK **enables this
   automatically** (sets it to ``1`` when it detects a WSL GPU environment),
   using a non-overwriting write so an explicit user setting always wins. You
   normally do not need to set it yourself; export ``WSLKMT_VENDOR_PACKET=0``
   to opt out.

``ROCPROFILER_FORCE_GFX``
   Overrides the GPU's ``gfx`` target name used to look up the counter
   definitions (``config.yaml`` is keyed by gfx target). The target normally
   comes from the KMT node's engine id, which already honors the HSA runtime's
   own ``HSA_OVERRIDE_GFX_VERSION``; set this variable to override both. The
   value is validated and must be of the form ``gfx<NNN>`` with at least three
   decimal digits; a malformed value is ignored with a warning and the
   node-reported target is used.

.. note::

   On WSL2 you may also need to prepend the ROCprofiler-SDK build's ``lib/``
   directory to ``LD_LIBRARY_PATH`` so the loader resolves the matching
   ROCprofiler-SDK and HSA libraries instead of an older system install under
   ``/opt/rocm``.
