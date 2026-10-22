.. meta::
   :description: Asynchronous I/O API reference for hipFile, covering stream-based non-blocking reads and writes and stream registration flags.
   :keywords: hipFile, async, asynchronous, I/O, stream, ROCm, GPU, API, reference

*******************************
Asynchronous I/O API reference
*******************************

The hipFile asynchronous API lets you enqueue non-blocking reads and writes on
HIP streams. Stream registration communicates hints about fixed offsets and
alignment to the driver.

For a step-by-step guide to using these functions, see
:doc:`/tutorials/async-multistream-io`.

How asynchronous I/O runs
===========================

``hipFileReadAsync()`` and ``hipFileWriteAsync()`` return once the request is
queued on the stream. The transfer runs later, in stream order, and its result
is written to the ``bytes_read_p`` or ``bytes_written_p`` argument: the number
of bytes transferred, or a negative value if the transfer failed. Synchronize
the stream before you read the result, and keep the pointer arguments valid
until then. If a call returns an error, nothing was queued.

How a transfer runs depends on the stream:

Registered streams
   On devices that support stream wait values, hipFile hands the transfer to an
   internal thread pool and makes the stream wait for it to finish, so later
   work on the stream still runs after the transfer. Transfers on different
   registered streams can run at the same time.

Unregistered streams
   The transfer runs in stream order, but the ROCm runtime runs stream host
   functions on a single thread, so transfers on unregistered streams run one
   at a time. While one runs, it also delays transfers queued on other streams.
   Registered streams on devices that don't support stream wait values behave
   the same way.

The null stream
   The call synchronizes the null stream, runs the transfer, and returns after
   the transfer finishes.

``hipFileStreamDeregister()`` synchronizes a registered stream before it frees
the stream's resources, so it waits for the stream's transfers in flight.

.. note::

   If the ``GPU_MAX_HW_QUEUES`` environment variable is set to ``1``, a transfer
   on a registered stream that uses the fallback path can hang. See
   :ref:`hipfile-gpu-max-hw-queues`.

Stream registration flags
===========================

Flags passed to ``hipFileStreamRegister`` to optimize stream processing when
transfer parameters are known at registration time.

.. doxygengroup:: stream_flags
   :content-only:

Asynchronous I/O and stream registration functions
====================================================

Functions to enqueue non-blocking reads and writes on HIP streams and to
register and deregister streams with the hipFile driver.

.. doxygengroup:: async
   :content-only:
