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

Registering a stream with ``hipFileStreamRegister()`` is optional and only an
optimization. Asynchronous I/O on an unregistered stream runs the transfer
without registering it, on the blocking path described below; register the
stream to get the faster wait-value-gated path.

How a transfer runs depends on the stream and the device:

Registered streams on devices that support stream wait values
   hipFile hands the transfer to an internal thread pool and makes the stream
   wait for it to finish, so later work on the stream still runs after the
   transfer. Transfers on different streams can run at the same time, and
   queuing a request never waits for the stream, so you can hold a stream with
   your own wait value, queue requests, and release it afterward. Do not
   deregister a registered stream on the same device that has processed I/O
   while the hold is in effect: deregistration calls ``hipFree`` on the
   stream's signal memory, which synchronizes the device and hangs until the
   held stream unblocks.

Unregistered streams, and registered streams on devices without stream wait values
   The transfer runs in stream order inside a stream host function that waits
   for it to finish. The ROCm runtime runs host functions on a single thread, so
   these transfers run one at a time and delay host functions queued on other
   streams. Register a stream on a wait-value-capable device for the faster path
   above.

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
