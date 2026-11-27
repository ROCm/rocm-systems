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

Registering a stream with ``hipFileStreamRegister()`` is optional. Asynchronous
I/O on an unregistered stream runs the transfer without registering it, on the
blocking path described below. By default a registered stream uses the same
path; registration selects the faster wait-value-gated path only when
``HIPFILE_FORCE_BLOCKING_ASYNC`` is ``false`` (see
:ref:`hipfile-force-blocking-async`).

How a transfer runs depends on the stream, the device, and that setting:

Registered streams on devices that support stream wait values, with ``HIPFILE_FORCE_BLOCKING_ASYNC=false``
   hipFile hands the transfer to an internal thread pool and makes the stream
   wait for it to finish, so later work on the stream still runs after the
   transfer. Transfers on different streams can run at the same time, and
   queuing a request never waits for the stream, so you can hold a stream with
   your own wait value, queue requests, and release it afterward. Do not
   deregister a registered stream on the same device that has processed I/O
   while the hold is in effect: deregistration calls ``hipFree`` on the
   stream's signal memory, which synchronizes the device and hangs until the
   held stream unblocks.

All other streams (the default)
   This covers every stream while ``HIPFILE_FORCE_BLOCKING_ASYNC`` is ``true``,
   unregistered streams, and registered streams on devices without stream wait
   values. The transfer runs in stream order inside a stream host function that
   waits for it to finish. The ROCm runtime runs host functions on a single
   thread, so these transfers run one at a time and delay host functions queued
   on other streams.

The null stream and ``hipStreamPerThread``
   The call runs synchronously: it synchronizes the stream, runs the transfer,
   and returns only after the transfer has finished, with the result already
   written to ``bytes_read_p`` or ``bytes_written_p``. This matches cuFile's
   documented behavior for the default stream. Even so, synchronize the stream
   and keep the pointer arguments valid until the call returns, so the same code
   works on any stream.

Deregister every stream you registered with ``hipFileStreamRegister()`` before
you destroy it. ``hipFileStreamDeregister()`` synchronizes the stream before it
frees the stream's resources, so it waits for the stream's transfers in flight.
If the stream processed any asynchronous I/O, deregistration also calls
``hipFree`` to release the stream's signal memory, which synchronizes all
streams on the device. Do not deregister a registered stream that has processed
I/O while another stream on the same device is blocked --- for example, one
held with a wait value you have not yet released --- because the ``hipFree``
call will hang until that stream unblocks.
A stream you used for asynchronous I/O without registering it holds no hipFile
state and needs no deregistration.

hipFile reserves the device's greatest stream priority for the streams its
fallback path copies data on. A stream at that priority could hold up those
copies while it waits for a transfer, so hipFile rejects it:
``hipFileStreamRegister()`` returns ``hipFileInvalidValue``, and
``hipFileReadAsync()`` and ``hipFileWriteAsync()`` return
``hipFileAsyncNotSupported``. Use a lower priority for streams you use for
asynchronous I/O. A device with only one stream priority level has no reserved
priority. See :ref:`hipfile-greatest-priority-streams` for a related case.

Asynchronous I/O accepts only device memory. ``hipFileReadAsync()`` and
``hipFileWriteAsync()`` return ``hipFileAsyncNotSupported`` for a buffer
registered from host memory; use the synchronous ``hipFileRead()`` and
``hipFileWrite()`` for host buffers.

.. _hipfile-force-blocking-async:

Forced blocking path and stream priorities
============================================

hipFile runs every asynchronous transfer on the blocking path by default, even
on a registered stream on a device that supports stream wait values. The
``HIPFILE_FORCE_BLOCKING_ASYNC`` environment variable controls this, and
defaults to ``true``.

The default avoids a deadlock. On the wait-value-gated path, the application's
stream is parked on a stream wait value until its transfer finishes. A transfer
on the fallback path finishes only after hipFile copies the data through the
GPU on streams of its own. hipFile creates those streams at the device's
greatest priority so that they get hardware queues that application streams
don't use. Since ROCm 10.2, the HIP runtime disables stream priorities and
gives every stream a normal-priority hardware queue, so a fallback copy can be
queued behind the parked application stream. The copy then waits for the
stream, the stream waits for the copy, and neither ever runs.

On the blocking path the stream is not parked on a wait value, so this deadlock
does not occur. The cost is that transfers run one at a time, as described in
the previous section.

You can set ``HIPFILE_FORCE_BLOCKING_ASYNC=false`` to restore the
wait-value-gated path, but only if you also restore stream priorities in the
HIP runtime by setting its ``DEBUG_HIP_IGNORE_STREAM_PRIORITY`` environment
variable to ``0``. Set both before the process makes its first HIP or hipFile
call.

.. warning::

   Do this at your own risk. The HIP runtime disables stream priorities to
   avoid known hardware queue priority problems in the GPU kernel driver, and
   ``DEBUG_HIP_IGNORE_STREAM_PRIORITY`` is a HIP debugging setting, not a
   supported interface. Setting ``HIPFILE_FORCE_BLOCKING_ASYNC=false`` without
   restoring stream priorities can deadlock asynchronous I/O that uses the
   fallback path.

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
