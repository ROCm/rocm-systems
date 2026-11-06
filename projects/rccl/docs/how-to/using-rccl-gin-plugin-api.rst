.. meta::
   :description: How to provide a GIN backend for RCCL
   :keywords: RCCL, ROCm, library, API, GIN, RMA, plugin

.. _using-rccl-gin-plugin:

**********************************
Using the RCCL GIN plugin API
**********************************

GIN (GPU-Initiated Networking) lets a kernel issue network operations itself,
through the device-side API described in :doc:`device-api-gin`. This topic covers the
other side of that interface: how to supply the *backend* that carries those
operations over your network.

RCCL supports two kinds of GIN backend, and which one you write determines which
plugin API you implement:

* A **proxy** backend, where RCCL implements the device side for you and converts
  each GIN request into a host-initiated network operation. You only implement a
  host-side API, and that API is the :ref:`RMA plugin API <using-rccl-rma-plugin>`.
* A **device-initiated** backend, where your code runs on the GPU and talks to the
  NIC directly. This requires both device-side and host-side work inside RCCL and
  is not a plugin-only deliverable.

.. _gin-proxy-backend:

Proxy backends use the RMA plugin API
=====================================

Proxy GIN is built in and enabled by default. RCCL registers an internal backend,
``ncclGinProxy``, which implements the GIN device API and drains each request through
a host progress thread. It does not move data itself: it forwards to whatever RMA
backend the communicator adopted, so the way to put a custom network behind proxy
GIN is to write an RMA plugin. See :ref:`using-rccl-rma-plugin` for the v15
interface, compatibility behavior and loading instructions.

When the RMA plugin is adopted, RCCL reports the first line below. If an external
proxy-type GIN plugin is also selected through ``NCCL_GIN_PLUGIN``, RCCL skips it
in favor of the built-in proxy and reports the second line:

.. code-block:: shell

   RMA/Plugin: Assigned plugin Example to comm
   GIN/Plugin: ... using NCCL GIN proxy over RMA backend Example

.. note::

   An external plugin loaded through ``NCCL_GIN_PLUGIN`` that reports
   ``NCCL_NET_DEVICE_GIN_PROXY`` is deliberately *not* used: the built-in
   ``ncclGinProxy`` already provides that backend type, and RCCL keeps it rather
   than the external one. This is why a custom proxy backend belongs behind
   ``NCCL_RMA_PLUGIN`` and not ``NCCL_GIN_PLUGIN``.

Device-initiated backends
=========================

A device-initiated backend cannot be supplied by an external plugin alone. It needs,
inside RCCL:

* a value in ``ncclGinType_t`` (``src/include/nccl_device/core.h``) matching the
  corresponding ``NCCL_NET_DEVICE_GIN_*`` device type,
* a ``case`` for that type, with its supported version list, in the backend
  dispatch in ``src/gin/gin_host.cc``, which rejects unknown types, and
* a device-side implementation of the GIN API, compiled per GPU architecture and
  linked into the device binary.

The in-tree device-initiated backends are built this way: ``ROCSHMEM_GDA`` and
``ANVIL_SDMA``, both enabled by configuring with ``--rocshmem-gin``.

What the GIN example plugin shows
=================================

``plugins/gin/example`` is a reference for the *shape* of the host-side GIN vtable
and for exporting several ``ncclGinPlugin_vNN`` versions from one object. Because it
reports the proxy device type, RCCL supersedes it as described above, so only
``init``, ``devices``, ``getProperties`` and ``finalize`` ever run. Treat it as an
API reference rather than a working backend, and use ``plugins/rma/example`` as the
starting point for a custom proxy backend.

Environment variables
=====================

.. list-table::
    :header-rows: 1
    :widths: 30,50,20

    * - Variable
      - Description
      - Default

    * - ``NCCL_GIN_PLUGIN``
      - Comma-separated list of GIN plugins to load, each a path or a short name
        resolved against the ``librccl-gin`` prefix.
      - Unset

    * - ``NCCL_GIN_ENABLE``
      - Set to ``0`` to register no GIN backend, so GIN reports as unsupported.
        RCCL-specific.
      - ``1``

    * - ``NCCL_GIN_TYPE``
      - Require a specific backend type, skipping backends of any other type:
        ``2`` proxy, ``3`` GDAKI, ``4`` GPI, ``5`` EFA GDA, ``6`` rocSHMEM GDA,
        ``7`` Anvil SDMA. ``-1`` disables the generic type filter, but does not
        auto-enable the in-tree device backends: rocSHMEM GDA requires ``6``,
        while Anvil SDMA accepts an unset value or ``7``.
      - Unset

    * - ``NCCL_RMA_PLUGIN``
      - Comma-separated list of RMA plugins to load, each a path or a short name
        resolved against the ``librccl-rma`` prefix. This is the backend proxy GIN
        forwards to.
      - Unset

For the consumer side of GIN, including how a kernel issues put, signal and flush
operations, see :doc:`device-api-gin`.
