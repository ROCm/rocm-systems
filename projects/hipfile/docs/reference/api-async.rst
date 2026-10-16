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

Stream registration and the default stream
===========================================

Asynchronous reads and writes can be submitted on any non-default HIP stream,
whether or not it has been registered. Registering a stream with
``hipFileStreamRegister`` gives it a dedicated copy stream and completion-signal
memory, so its transfers are gated with a stream wait value and overlap with
work on other registered streams. An unregistered non-default stream still
submits asynchronously and preserves stream ordering, but its transfers are
serialized on a single driver host-function thread and do not overlap across
streams. Registration is therefore recommended for any stream used for repeated
or concurrent asynchronous I/O.

Operations submitted on the default (null) stream run synchronously: the call
blocks until the transfer completes. Use a registered non-default stream for
non-blocking, overlapping I/O.

An asynchronous submission can fail after its transfer has already been
committed to the stream. When this happens the call returns an error, but the
committed transfer may still execute, and the ordering guarantee between it and
the next operation submitted on that same stream no longer holds. Treat a stream
whose submission returned an error as errored: synchronize it with
``hipStreamSynchronize`` before reusing it, or stop submitting asynchronous I/O
on it.

Scalability considerations
===========================

Asynchronous transfers are dispatched through per-stream HIP hardware queues.
When many streams perform I/O concurrently, they contend for a limited pool of
hardware queues, and this contention can limit scalability as the number of
active streams grows.

Increasing the number of available hardware queues can mitigate some of this
contention. Set the ``GPU_MAX_HW_QUEUES`` environment variable to raise the
per-process hardware queue limit before running your application.

The compatibility (fallback) path requires at least two hardware queues. With
``GPU_MAX_HW_QUEUES=1``, a fallback asynchronous transfer can deadlock because
the stream's completion wait and the staging copy it waits on share the single
queue, so the copy cannot make progress behind the wait. The GPU-direct fast
path transfers storage to GPU memory without using a stream hardware queue and
is unaffected at any queue count.

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
