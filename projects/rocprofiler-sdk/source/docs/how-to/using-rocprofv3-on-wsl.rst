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

This page documents the build-time macro and the environment variables that
control that behavior.

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
* **Advanced Thread Trace (ATT)**: supported when the ``rocprof-trace-decoder``
  library is available. ATT capture uses the same DXG path as hardware counter
  collection; the captured shader-data stream is decoded by an
  externally-provided ``librocprof-trace-decoder.so`` that ROCprofiler-SDK does
  not build. Point rocprofv3 at it with ``--att-library-path`` or the
  ``ROCPROF_ATT_LIBRARY_PATH`` environment variable. The build locates the
  decoder for the ATT tests via ``ROCPROFILER_SDK_TRACE_DECODER_ROOT`` (an
  extra find hint), falling back to ``${ROCM_PATH}/lib``; when it is not found
  the ATT tests are skipped.

Known limitations
==================

* **PC sampling**: the software/decode layers build and pass their unit tests,
  but the end-to-end path requires KFD features that are not available on WSL2
  and is therefore disabled.
* **SPM (Streaming Performance Monitor)**: not available; the gfx11-class
  counter database does not expose SPM-capable counters.

.. _wsl-build-macro:

Build-time macro
================

``ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS``
   Defined when ROCprofiler-SDK is built against the ``libhsakmt-windows``
   (``librocdxg``) shim, which is the native-Windows build path (it links
   ``gdi32`` and a matching KMD). On the WSL2 / Linux build this macro is left
   **undefined**, so the shim calls compile out and the WSL agent enumerator
   seeds documented gfx11-class defaults for the gfx target name/version and the
   compute-unit topology (DXCore cannot read them, and tools such as
   ``rocprofv3-avail`` query agents before the HSA runtime is up). Once the HSA
   runtime is initialized, ``construct_agent_cache()`` refines those fields
   (topology, ``num_xcc``, ``domain``, ``family_id``, firmware versions,
   workgroup/grid limits, and — unless ``ROCPROFILER_FORCE_GFX`` is set — the gfx
   target name/version) from the HSA runtime, so GPUs whose target differs from
   the seeded default report correct values at runtime. Most WSL2 users build
   with the macro undefined.

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
   definitions (``config.yaml`` is keyed by gfx target). DXCore on WSL2 does not
   expose the gfx target, so the enumerator defaults to a documented gfx11-class
   target at agent-creation time and, once the HSA runtime is up, refines the
   name/version from the HSA runtime (``HSA_AGENT_INFO_NAME``) in
   ``construct_agent_cache()``. Set this variable to force a specific target; an
   explicit value wins over the HSA-derived one (and is the way to get a correct
   target for pre-HSA tools such as ``rocprofv3-avail`` on GPUs whose target
   differs from the default). The value is validated and must be of the form
   ``gfx<NNN>`` with at least three decimal digits; a malformed value is ignored
   with a warning and the default is used.

.. note::

   On WSL2 you may also need to prepend the ROCprofiler-SDK build's ``lib/``
   directory to ``LD_LIBRARY_PATH`` so the loader resolves the matching
   ROCprofiler-SDK and HSA libraries instead of an older system install under
   ``/opt/rocm``.
