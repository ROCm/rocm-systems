.. meta::
   :description: How to provide an external one-sided RMA backend for RCCL
   :keywords: RCCL, ROCm, RMA, plugin, one-sided

.. _using-rccl-rma-plugin:

*********************************
Using the RCCL RMA plugin API
*********************************

The RMA plugin API lets an external library provide the host-side network
operations used by RCCL's one-sided communication path. RCCL loads the newest
supported interface exported by the library, trying ``ncclRmaPlugin_v16``,
``ncclRmaPlugin_v15``, ``ncclRmaPlugin_v14`` and then ``ncclRmaPlugin_v13``.

Implementing version 16
=======================

Implement ``ncclRma_v16_t`` from ``src/include/plugin/rma/rma_v16.h`` and
export the vtable as ``ncclRmaPlugin_v16``. The following callbacks are
required:

* ``init``, ``devices``, ``getRmaProperties``, ``getProperties``, ``listen``
  and ``connect``
* ``createContext``, ``regMrSym``, ``deregMrSym`` and ``destroyContext``
* ``closeColl`` and ``closeListen``
* ``iput``, ``iputSignal``, ``iget``, ``test`` and ``finalize``

``regMrSymDmaBuf`` is also required when ``getProperties`` advertises
``NCCL_PTR_DMABUF``. ``iflush``, ``rmaProgress`` and ``queryLastError`` are
optional.

Version 16 adds ``getRmaProperties``, which RCCL calls once per collective
communicator when it sets up the GIN proxy. It fills ``ncclRmaProperties_v16_t``:

* ``flushesAllPutsOnAnySignal``: set to ``true`` only when the backend makes
  every put already received on the communicator visible whenever any signal
  arrives, from any peer, on every NIC the communicator uses. RCCL's device
  barrier then skips its per-peer flush. Return ``false`` when unsure; that keeps
  the per-peer flush and is what RCCL reports for v13 to v15 plugins.

Version 15 added an ``optFlags`` argument to ``iput``, ``iputSignal`` and
``iget``, which v16 keeps. Plugins must accept these values:

* ``ncclRmaOptFlagsDefault`` (``0``): no optional behavior requested.
* ``ncclRmaOptFlagsAggregateRequests`` (``1 << 0``): the caller permits the
  backend to aggregate compatible requests. This is a hint; it must not change
  the operation's data or ordering semantics.

The GIN proxy sets the hint when that peer's next operation is already queued
and will be issued in the same progress iteration (see
``NCCL_GIN_PROXY_POLL_BATCH``). Each run ends with a data operation without the
hint or with an ``iflush``, so a backend that holds hinted requests, for example
by delaying a doorbell, should submit them on either.

Compatibility with older plugins
=================================

RCCL continues to load v15, v14 and v13 plugins. Their compatibility wrappers
report ``flushesAllPutsOnAnySignal`` as ``false``. The v14 and v13 wrappers also
adapt the older callback signatures and discard ``optFlags``, because those
versions cannot consume the hint. New plugins should export v16.

Loading a plugin
================

Set ``NCCL_RMA_PLUGIN`` to an absolute path, a shared-library file name or a
short name. RCCL passes the raw value to ``dlopen`` first. If that fails, it
retries a short name with the ``librccl-rma`` prefix:

.. code-block:: shell

   export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
   export NCCL_RMA_PLUGIN=mybackend

The command above loads a file literally named ``mybackend`` when one is on the
library search path, and otherwise resolves ``librccl-rma-mybackend.so``.
During communicator initialization, ``NCCL_DEBUG=INFO`` reports the selected
interface version and the backend assigned to the communicator.

``plugins/rma/example`` is a minimal v15 implementation that demonstrates the
lifecycle, registration and data-operation callback signatures. RCCL loads it
through the v15 compatibility wrapper.
