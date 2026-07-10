.. meta::
   :description: Hotswap debug instrumentation architecture
   :keywords: ROCR, hotswap, instrumentation, debug

.. _hotswap-debug-instrumentation:

Hotswap debug instrumentation architecture
==========================================

This note describes a separate opt-in debug mode for observing hotswap
decisions and device-time hotswap mask workaround state. It is intentionally
separate from production hotswap behavior. Production hotswap should keep using
only the rewrite decisions and COMGR flags required for correctness.

Goals
-----

* Report when ROCR requests a COMGR hotswap mask workaround.
* Capture launch-time cluster metadata when it is available to ROCR before
  dispatch.
* Optionally capture device-time state from rewritten code without changing the
  user kernel ABI.
* Keep unsupported instrumentation cases explicit instead of silently implying
  complete coverage.

Non-goals
---------

* Do not use GPU ``printf``. It is intrusive, ABI-sensitive, and not reliable
  for arbitrary kernels.
* Do not change user kernarg layouts or require application recompilation.
* Do not promise per-instruction-site instrumentation for every kernel. That
  mode depends on scratch registers, valid insertion points, and safe control
  flow.

Mode overview
-------------

The debug feature should be split into escalating modes controlled by a ROCR
environment variable such as ``HSA_HOTSWAP_DEBUG``:

``dispatch``
  ROCR records host-side hotswap decisions and dispatch metadata. This mode is
  the least invasive mode and should be available without COMGR device
  instrumentation support. Dispatch packet coverage is limited to runtime paths
  where ROCR actually observes the packet; many application packets are written
  directly into the queue ring.

``entry``
  ROCR passes a debug record buffer to COMGR. COMGR emits a wrapper at kernel
  entry that records device-time fields once per kernel entry and then branches
  to the original kernel body.

``site``
  COMGR emits records near individual hotswap mask workaround sites. This is a
  best-effort diagnostic mode for targeted kernels, not a guaranteed mode for
  arbitrary production kernels.

Cluster metadata sources
------------------------

Cluster metadata should be reported from the least intrusive source that can
answer the question:

* If cluster size and cluster count are known before launch, ROCR should record
  the extended dispatch packet fields directly. The MI400 AQL packet
  definitions carry ``cluster_count_x``, ``cluster_count_y``,
  ``cluster_count_z``, ``cluster_size_x``, ``cluster_size_y``, and
  ``cluster_size_z`` fields.
* If a value is only meaningful at device time, ROCR should ask COMGR to emit
  instrumentation that reads the shader programming guide defined hardware
  field, such as the ``IB_STS2`` cluster ID status field, and writes it into
  the debug record. The production PLAT-204339 B0 workaround already uses this
  class of COMGR-emitted device read for correctness; debug mode should reuse
  the same documented source instead of inventing a host-side cluster ID.
* If neither source is available, the record should mark the field unknown
  instead of guessing.

ROCR responsibilities
---------------------

ROCR should own the debug mode policy and runtime resources:

* Parse the debug-mode environment variable during hotswap initialization.
* Assign a stable debug identifier to each rewritten code object and kernel.
* Allocate a GPU-visible debug buffer or ring buffer before invoking COMGR for
  device instrumentation modes.
* Pass the buffer address, buffer size, selected debug mode, and record schema
  version to COMGR through a versioned hotswap options structure.
* Retain the debug buffer for at least as long as the rewritten code object can
  execute.
* Record host-side events when hotswap selects a mask workaround, including the
  source ISA, target ISA, requested COMGR flags, and code object URI when
  available.
* Capture dispatch metadata available in ROCR, including extended dispatch
  packet fields such as ``cluster_count_x``, ``cluster_count_y``,
  ``cluster_count_z``, ``cluster_size_x``, ``cluster_size_y``, and
  ``cluster_size_z`` when the dispatch path exposes them.
* Drain records at explicit synchronization or lifetime boundaries, such as
  executable destruction, runtime shutdown, or a debug-only flush path.

ROCR should not try to synthesize device-time cluster IDs from host metadata.
If the value must be observed at device time, ROCR should request COMGR device
instrumentation and provide storage for the result.

Dispatch metadata capture should be described as opportunistic unless the
queue path being debugged routes packets through a ROCR-visible submission or
intercept path. Code-object-load decisions are always visible to hotswap; every
individual dispatch packet is not.

COMGR responsibilities
----------------------

COMGR should own all instruction-level instrumentation:

* Define the debug bits and any extended hotswap options needed to receive a
  debug buffer address, size, code object identifier, and record schema version.
* Emit entry-wrapper instrumentation using normal assembler/MC infrastructure
  instead of hard-coded opcodes.
* Preserve the original kernel ABI. The debug buffer address should be
  materialized from rewritten code or code object data produced by COMGR, not
  passed through user kernargs.
* Save and restore any registers used by instrumentation.
* Emit device-time reads for hardware state, such as the shader programming
  guide defined cluster ID field, in the wrapper or site instrumentation path.
  Cluster size should come from host packet metadata unless a documented
  device-time field is identified for the target.
* Emit per-site records only when scratch registers and insertion points are
  proven safe. If a requested site record cannot be emitted, return a clear
  debug-mode failure instead of emitting partial instrumentation silently.

Compatibility and failure policy
--------------------------------

Debug instrumentation should be negotiated separately from production strict
mode:

* Keep production hotswap flags minimal. Debug flags should not be required for
  PLAT-204339 correctness.
* Extend the COMGR hotswap options structure by using the existing ``size``
  field as a version boundary. New ROCR builds must continue to pass the base
  option size when no debug fields are needed.
* If ``dispatch`` mode is selected and COMGR lacks debug support, ROCR can
  still emit host-side decision records.
* If ``entry`` or ``site`` mode is selected and COMGR lacks the requested debug
  support, ROCR should fail the debug request explicitly or disable only that
  debug mode according to the environment policy. It should not imply that
  device-time coverage was collected.
* If production strict mode is required and COMGR cannot produce or load the
  strict rewrite, ROCR must fail the load instead of falling back to the
  original code object.

Record model
------------

The shared record schema should be fixed-size and versioned. A record should be
safe to write from device code with a single atomic index reservation followed
by plain stores. Fields should include:

* record kind: host decision, dispatch metadata, entry record, or site record
* code object ID and kernel ID
* requested hotswap flags
* source and target gfx identifiers, encoded as IDs or hashes
* dispatch cluster count and cluster size when ROCR knows them
* metadata source: host dispatch packet, device register read, or unknown
* device-time cluster ID or cluster size when COMGR instrumentation reads it
* code offset or patch-site identifier for site records
* overflow or truncation status

ROCR should tolerate dropped records and report overflow explicitly.

Expected coverage
-----------------

``dispatch`` mode is guaranteed only for information already visible to ROCR.
It can identify that a mask workaround was requested and can report cluster
metadata carried by a dispatch packet when that packet passes through a
ROCR-observed path. It cannot report per-wave device state, cannot guarantee
coverage for direct queue-ring writes, and cannot prove that a specific dynamic
instruction instance executed.

``entry`` mode is the preferred device-time mode. It has a single controlled
instrumentation point, avoids changing user ABI, and can record hardware state
before the original kernel body runs.

``site`` mode is useful for targeted diagnostics but is not universal. Kernels
with no safe scratch register, no valid insertion point, or unsupported control
flow should fail instrumentation in this mode.

Testing plan
------------

* Add ROCR unit tests for debug mode parsing, record-buffer lifetime, host
  decision records, and dispatch metadata capture.
* Add COMGR lit tests that disassemble entry-wrapper and site instrumentation
  for representative gfx1250 kernels.
* Add negative tests for no scratch register, unsupported insertion points, and
  debug buffer overflow.
* Add rocrtst coverage that runs with a COMGR build supporting the debug
  options and verifies that records are emitted and drained.
* Validate on gfx1250 A0 hardware with B0 code objects that exercise tensor and
  cluster-load mask workaround paths.

Suggested implementation phases
-------------------------------

* First add ``HSA_HOTSWAP_DEBUG=dispatch`` parsing and host-side decision
  records in ROCR. This phase does not require COMGR changes.
* Then add COMGR debug option validation and explicit unsupported-mode errors
  for ``entry`` and ``site``.
* Add entry-wrapper records once the device record writer, register allocation,
  and buffer-address materialization are proven on gfx1250.
* Add per-site records last, with negative tests for every case that cannot be
  instrumented safely.
