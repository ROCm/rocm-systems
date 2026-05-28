.. meta::
   :description: How to use the NCCL Net API
   :keywords: RCCL, ROCm, library, API, NCCL Net, plugin

.. _using-nccl:

*****************************
Using the NCCL Net plugin API
*****************************

NCCL provides a way to use external plugins to let NCCL run on many network types. This 
topic describes the NCCL Net plugin API and explains how to implement a network plugin for NCCL.

Plugins implement the NCCL network API and decouple NCCL binary builds, which are built against a
particular version of the GPU stack (such as NVIDIA CUDA), from the network code, which is built against a
particular version of the networking stack. Using this method, you can easily integrate any CUDA version
with any network stack version.

NCCL network plugins are packaged as a shared library called ``librccl-net.so``. The shared library
contains one or more implementations of the NCCL Net API in the form of versioned structs,
which are filled with pointers to all required functions.

Plugin architecture
===================

When NCCL is initialized, it searches for a ``librccl-net.so`` library and dynamically loads it,
then searches for symbols inside the library.

The ``NCCL_NET_PLUGIN`` environment variable allows multiple plugins to coexist. If it's set, NCCL
looks for a library named ``librccl-net-${NCCL_NET_PLUGIN}.so``. It is therefore
recommended that you name the library according to that pattern, with a symlink pointing from ``librccl-net.so``
to ``librccl-net-${NCCL_NET_PLUGIN}.so``. This lets users select the correct plugin
if there are multiple plugins in the path.

Struct versioning
-----------------

After a library is found, NCCL looks for a symbol named ``ncclNet_vX``, with ``X`` increasing
over time. This versioning pattern ensures that the plugin and the NCCL core are compatible.

Plugins are encouraged to provide a number of these symbols, implementing many versions
of the NCCL Net API. This is so the same plugin can be compiled for and support a wide range of NCCL
versions.

Conversely, and to ease transition, NCCL can choose to support different plugin versions. It can look
for the latest ``ncclNet`` struct version but also search for older versions, so that older plugins
still work.

In-network collective operations (collNet)
----------------------------------------------

In addition to the ``ncclNet`` structure, network plugins can provide a ``collNet`` structure which
implements any supported in-network collective operations. This is an optional
structure provided by the network plugin,
but its versioning is tied to the ``ncclNet`` structure and many functions are common between the two to
ease implementation. The ``collNet`` structure can be used by the NCCL ``collNet``
algorithm to accelerate inter-node reductions in allReduce.

Header management
------------------

To help users effortlessly build plugins, plugins should copy the ``ncclNet_vX`` definitions
they support to their list of internal includes. An example is shown in ``ext-net/example/``, which stores
all headers in the ``nccl/`` directory and provides thin layers to implement old versions on top
of newer ones.

The ``nccl/`` directory is populated with ``net_vX.h`` files, which extract all relevant definitions
from the old API versions. It also provides error codes in ``err.h``.

API (v6)
=========

Here is the main ``ncclNet_v6`` struct. Each function is explained in later sections.

.. code:: shell

    typedef struct {
    // Name of the network (mainly for logs)
    const char* name;
    // Initialize the network.
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    // Return the number of adapters.
    ncclResult_t (*devices)(int* ndev);
    // Get various device properties.
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
    // Create a receiving object and provide a handle to connect to it. The
    // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
    // between ranks to create a connection.
    ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
    // Connect to a handle and return a sending comm object for that peer.
    // This call must not block for the connection to be established, and instead
    // should return successfully with sendComm == NULL with the expectation that
    // it will be called again until sendComm != NULL.
    ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
    // Finalize connection establishment after remote peer has called connect.
    // This call must not block for the connection to be established, and instead
    // should return successfully with recvComm == NULL with the expectation that
    // it will be called again until recvComm != NULL.
    ncclResult_t (*accept)(void* listenComm, void** recvComm);
    // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
    // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
    ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
    /* DMA-BUF support */
    ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
    ncclResult_t (*deregMr)(void* comm, void* mhandle);
    // Asynchronous send to a peer.
    // May return request == NULL if the call cannot be performed (or would block)
    ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
    // Asynchronous recv from a peer.
    // May return request == NULL if the call cannot be performed (or would block)
    ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
    // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
    // visible to the GPU
    ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
    // Test whether a request is complete. If size is not NULL, it returns the
    // number of bytes sent/received.
    ncclResult_t (*test)(void* request, int* done, int* sizes);
    // Close and free send/recv comm objects
    ncclResult_t (*closeSend)(void* sendComm);
    ncclResult_t (*closeRecv)(void* recvComm);
    ncclResult_t (*closeListen)(void* listenComm);
    } ncclNet_v6_t;

Error codes
-----------

All plugins functions use NCCL error codes as their return value. ``ncclSuccess`` should be returned upon
success. Otherwise, plugins can return one of the following codes:

* ``ncclSystemError`` is the most common error for network plugins. It should be returned when a call to the Linux kernel or a system library fails. This typically includes all network and hardware errors.
* ``ncclInternalError`` is returned when the NCCL core code is using the network plugin in an incorrect way, for example, allocating more requests than it should or passing an invalid argument in API calls.
* ``ncclInvalidUsage`` should be returned when the error is most likely due to user error. This can include misconfiguration, but also size mismatches.
* ``ncclInvalidArgument`` should not typically be used by plugins because arguments should be checked by the NCCL core layer.
* ``ncclUnhandledCudaError`` is returned when an error is received from NVIDIA CUDA. Network plugins should not need to rely on CUDA, so this error should not be common.

Operational overview
--------------------

NCCL first calls the ``init`` function, queries the number of network devices with the
``devices`` function, and retrieves the properties from each network device using ``getProperties``.

To establish a connection between two network devices, NCCL first calls ``listen`` on the
receiving side. It passes the returned handle to the sender side of the connection, and uses it to call ``connect``.
Finally, ``accept`` is called on the receiving side to finalize the establishment of the connection.

After the connection is established, communication is performed using the functions ``isend``,
``irecv``, and ``test``. Prior to calling ``isend`` or ``irecv``, NCCL calls the ``regMr`` function on
all buffers to allow RDMA NICs to prepare the buffers. ``deregMr`` is used to unregister buffers.

In certain conditions, ``iflush`` is called after a ``receive`` call completes to allow the network
plugin to flush data and ensure the GPU processes the newly written data.

To close the connections, NCCL calls ``closeListen`` to close the object returned by ``listen``,
``closeSend`` to close the object returned by ``connect``, and ``closeRecv`` to close the object returned
by ``accept``.

API Functions
-------------

The RCCL Tuner plugin API provides the following interface for initialization, connection management, and
communications.

Initialization
^^^^^^^^^^^^^^

*  ``name`` - The ``name`` field should point to a character string with the name of the network plugin. This name is used for all logging, especially when ``NCCL_DEBUG=INFO`` is set.

   .. note::

      Setting ``NCCL_NET=<plugin name>`` ensures a specific network implementation is used, with
      a matching ``name``. This is not to be confused with ``NCCL_NET_PLUGIN`` which defines a suffix for the
      ``librccl-net.so`` library name to load.

*  ``init`` - As soon as NCCL finds the plugin and the correct ``ncclNet`` symbol, it calls the ``init`` function. This allows the plugin to discover network devices and ensure they are usable.
   If the ``init`` function does not return ``ncclSuccess``, then NCCL does not use the plugin and falls back to internal ones.

   To allow the plugin logs to seamlessly integrate into the NCCL logs, NCCL provides a logging function to ``init``. This function is typically used to allow ``INFO`` and ``WARN`` macros within the plugin code by adding the following definitions:

   .. code:: shell

      #define WARN(...) logFunction(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
      #define INFO(FLAGS, ...) logFunction(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

*  ``devices`` - After the plugin is initialized, NCCL queries the number of devices available. 
   This should not be zero. Otherwise, NCCL initialization will fail. If no device is present or usable, the ``init`` function should not return ``ncclSuccess``.

*  ``getProperties`` - Right after retrieving the number of devices, NCCL queries the properties for each available network device. 
   These properties are necessary when multiple adapters are present to ensure NCCL uses each adapter in the optimal way.

   *  The ``name`` is only used for logging.

   *  The ``pciPath`` is the base for all topology detection and should point to the PCI device directory
      in ``/sys``. This is typically the directory pointed to by ``/sys/class/net/eth0/device`` or
      ``/sys/class/infiniband/mlx5_0/device``. If the network interface is virtual, then ``pciPath`` should
      be ``NULL``.

   *  The ``guid`` field is used to determine whether network adapters are connected to multiple PCI
      endpoints. For normal cases, this is set to the device number. If multiple network devices have
      the same ``guid``, then NCCL understands them to be sharing the same network port to the fabric. In this case,
      it will not use the port multiple times.

   *  The ``ptrSupport`` field indicates whether or not CUDA pointers are supported. If so, it should be
      set to ``NCCL_PTR_HOST|NCCL_PTR_CUDA``. Otherwise, it should be set to ``NCCL_PTR_HOST``. If the plugin
      supports ``dmabuf``, it should set ``ptrSupport`` to ``NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF`` and
      provide a ``regMrDmaBuf`` function.

   *  The ``regIsGlobal`` field allows NCCL to register buffers in advance, for example, using a loopback connection.
      Later, it also lets NCCL expect that a subsequent registration on a buffer from a previous registration
      will happen nearly immediately, because the buffer is already known by the network adapter. A typical
      implementation maintains a registration cache, with the call to ``ncclCommRegister`` creating the
      initial entry in the cache using ``regMr()`` on a loopback connection. Any later call to the NCCL
      system can call ``regMr()`` again on the real connection, with the real buffer (which could be at a
      different offset within the original buffer, with a smaller size, for example). It
      could then call ``deregMr()`` immediately afterwards.
      The ``ncclCommDeregister`` call should issue the final call to ``deregMr()`` and effectively remove the mapping
      on the network adapter.

   *  The ``speed`` field indicates the speed of the network port in Mbps (10^6 bits per second).
      This ensures proper optimization of flows within the node.

   *  The ``port`` field indicates the port number. This is important for topology detection and
      flow optimization within the node when a NIC with a single PCI connection is connected to the fabric through multiple ports.

   *  The ``latency`` field indicates the network latency in microseconds. This can be useful to
      improve the NCCL tuning and ensure NCCL switches from tree to ring at the correct size.

   *  The ``maxComms`` field indicates the maximum number of connections that can be created.

   *  The ``maxRecvs`` field indicates the maximum number for grouped receive operations (see grouped receive).

Connection establishment
^^^^^^^^^^^^^^^^^^^^^^^^

Connections are used in an unidirectional manner, with a sender side and a receiver
side.

*  ``listen`` - To create a connection, NCCL calls ``listen`` on the receiver side.
   This function accepts a device number as an input argument and returns a local ``listenComm`` object and a ``handle``
   to pass to the other side of the connection, so that the sender can connect to the receiver.
   The ``handle`` is a buffer of size ``NCCL_NET_HANDLE_MAXSIZE`` and is provided by NCCL.
   This call should never block, but unlike ``connect`` and ``accept``, ``listenComm`` should never be ``NULL``
   if the call succeeds.

*  ``connect`` - NCCL uses its bootstrap infrastructure to provide the ``handle`` to the sender side,
   then calls ``connect`` on the sender side on a given device index ``dev`` and provides the ``handle``.
   ``connect`` should not block either. Instead, it sets ``sendComm`` to ``NULL`` and returns ``ncclSuccess``.
   In that case, NCCL will keep calling ``accept`` again until it succeeds.

*  ``accept`` - To finalize the connection, the receiver side calls ``accept`` on the ``listenComm`` object
   previously returned by the ``listen`` call. If the sender did not connect yet, ``accept`` should not block.
   It should return ``ncclSuccess``, setting ``recvComm`` to ``NULL``. NCCL will keep calling ``accept``
   again until it succeeds.

*  ``closeListen`` / ``closeSend`` / ``closeRecv`` - When a ``listenComm``, ``sendComm``, or ``recvComm`` object is no longer
   needed, NCCL calls ``closeListen``, ``closeSend``, or ``closeRecv`` to free the associated resources.

Communication
^^^^^^^^^^^^^

Communication is handled using the asynchronous send and receive operations: ``isend``, ``irecv``, and ``test``.
To support RDMA capabilities, buffer registration and flush functions are provided.

To keep track of asynchronous send, receive, and flush operations, requests are returned to NCCL,
then queried using ``test``. Each ``sendComm`` or ``recvComm`` must be able to handle
``NCCL_NET_MAX_REQUESTS`` requests in parallel.

.. note::

   This value should be multiplied by the multi-receive capability of the plugin for the sender
   side, so the plugin can effectively have ``NCCL_NET_MAX_REQUESTS`` multi-receive operations happening
   in parallel. If ``maxRecvs`` is 8 and ``NCCL_NET_MAX_REQUESTS`` is 8, then each
   ``sendComm`` must be able to handle up to 64 (8x8) concurrent ``isend`` operations.

*  ``regMr`` - Prior to sending or receiving data, NCCL calls ``regMr`` with any buffers later used for communication.
   It provides a ``sendComm`` or ``recvComm`` object for the ``comm`` argument,
   the buffer pointer ``data``, the ``size``, and the ``type``. The type is either ``NCCL_PTR_HOST`` or ``NCCL_PTR_CUDA`` if
   the network supports CUDA pointers.

   The network plugin can use the output argument ``mhandle`` to store any reference to the memory registration, because
   ``mhandle`` is returned for all ``isend``, ``irecv``, ``iflush``, and ``deregMr`` calls.

*  ``regMrDmaBuf`` - If the plugin has set the ``NCCL_PTR_DMABUF`` property in ``ptrSupport``, 
   NCCL uses ``regMrDmaBuf`` instead of ``regMr``. If the property was not set, ``regMrDmaBuf`` can be set to ``NULL``.

*  ``deregMr`` - When buffers are no longer used for communication, NCCL calls ``deregMr`` to let the plugin
   free resources. This function is used to deregister handles returned by ``regMr`` and ``regMrDmaBuf``.

*  ``isend`` - Data is sent through the connection using ``isend``, passing the ``sendComm`` object previously created
   by ``connect``, the buffer described by ``data``, ``size``, and ``mhandle``. A ``tag`` must
   be used if the network supports multi-receive operations (see ``irecv``) to distinguish between different send requests
   matching the same multi-receive. Otherwise it can be set to ``0``.

   The ``isend`` operation returns a handle in the ``request`` argument for further calls to ``test``.
   If the ``isend`` operation cannot be initiated, ``request`` is set to ``NULL``. NCCL will call ``isend`` again later.

*  ``irecv`` - To receive data, NCCL calls ``irecv`` with the ``recvComm`` returned by ``accept``.
   The argument ``n`` configures NCCL for multi-receive, to allow grouping of multiple sends
   through a single network connection. Each buffer can be described by the ``data``, ``sizes``, and ``mhandles`` arrays.
   ``tags`` specify a tag for each receive so that each of the ``n`` independent ``isend`` operations is received
   into the right buffer.

   If all receive operations can be initiated, ``irecv`` returns a handle in the ``request`` pointer. Otherwise,
   it sets the pointer to ``NULL``. In the case of multi-receive, all ``n`` receive operations are handled by a single request handle.

   The sizes provided to ``irecv`` can (and will) be larger than the size of the ``isend`` operation.
   However, it is an error if the receive size is smaller than the send size.

   .. note::

      For a given connection, send and receive operations should always match in the order they were
      posted. Tags provided for receive operations are only used to assign a given send operation to one
      of the buffers of the first (multi-)receive operation in the queue, not to allow for out-of-order tag
      matching on any receive operation posted.

*  ``test`` - After an ``isend`` or ``irecv`` operation is initiated, NCCL calls ``test`` on the request handles until
   the operation completes. When that happens, ``done`` is set to ``1`` and ``sizes`` is set to the real size sent or received,
   the latter could potentially be lower than the size passed to ``irecv``.

   In the case of a multi-receive, all receives are considered as part of a single operation, the goal
   being to allow aggregation. Therefore, they share a single request and a single ``done`` status. However,
   they can have different sizes, so if ``done`` is non-zero, the ``sizes`` array should contain the ``n`` sizes
   corresponding to the buffers passed to ``irecv``.

   After ``test`` returns ``1`` in ``done``, the request handle can be freed. This means that NCCL will never
   call ``test`` again on that request, unless it is reallocated by another call to ``isend`` or ``irecv``.

*  ``iflush`` - After a receive operation completes, if the operation was targeting GPU memory and received
   a non-zero number of bytes, NCCL calls ``iflush``. This lets the network flush any buffer to ensure
   the GPU can read it immediately without seeing stale data. This flush operation is decoupled from
   the ``test`` code to improve the latency of ``LL*`` protocols, because those are capable of determining
   when data is valid or not.

   ``iflush`` returns a request which must be queried using ``test`` until it completes.
