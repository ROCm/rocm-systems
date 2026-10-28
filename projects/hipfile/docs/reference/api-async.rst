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

Registering a stream with ``hipFileStreamRegister()`` is optional. The first
asynchronous request on a stream that isn't registered registers it implicitly,
with no flags. Registering a stream that was registered implicitly applies the
new flags to later requests.

How a transfer runs depends on the stream and the device:

Streams on devices that support stream wait values
   hipFile hands the transfer to an internal thread pool and makes the stream
   wait for it to finish, so later work on the stream still runs after the
   transfer. Transfers on different streams can run at the same time, and
   queuing a request never waits for the stream, so you can hold a stream with
   your own wait value, queue requests, and release it afterward.

Streams on devices that don't support stream wait values
   The transfer runs in stream order inside a stream host function that waits
   for it to finish. The ROCm runtime runs host functions on a single thread, so
   these transfers run one at a time and delay host functions queued on other
   streams.

The null stream and ``hipStreamPerThread``
   The call synchronizes the stream, runs the transfer, and returns after the
   transfer finishes.

Deregister every stream you use for asynchronous I/O, including streams that
were registered implicitly, before you destroy it. Otherwise a new stream that
reuses the handle could pick up the old stream's state.
``hipFileStreamDeregister()`` synchronizes the stream before it frees the
stream's resources, so it waits for the stream's transfers in flight.

hipFile reserves the device's greatest stream priority for the streams its
fallback path copies data on. A stream at that priority could hold up those
copies while it waits for a transfer, so hipFile rejects it:
``hipFileStreamRegister()`` returns ``hipFileInvalidValue``, and
``hipFileReadAsync()`` and ``hipFileWriteAsync()`` return
``hipFileAsyncNotSupported``. Use a lower priority for streams you use for
asynchronous I/O. A device with only one stream priority level has no reserved
priority. See :ref:`hipfile-greatest-priority-streams` for a related case.

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
