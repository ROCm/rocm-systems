.. meta::
   :description: Asynchronous I/O API reference for hipFile, covering stream-based non-blocking reads and writes and stream registration flags.
   :keywords: hipFile, async, asynchronous, I/O, stream, ROCm, GPU, API, reference

*******************************
Asynchronous I/O API reference
*******************************

The hipFile asynchronous API lets you enqueue non-blocking reads and writes on
HIP streams. Stream registration communicates hints about fixed offsets and
alignment to the driver. On the AMD backend, asynchronous operations are
serviced through the fallback path rather than the fastpath.

For a step-by-step guide to using these functions, see
:doc:`/tutorials/async-multistream-io`.

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
