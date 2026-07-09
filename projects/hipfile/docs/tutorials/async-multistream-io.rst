.. meta::
   :description: Tutorial walking through asynchronous multi-stream I/O with registered streams using hipFile.
   :keywords: hipFile, ROCm, async, multi-stream, GPU I/O, hipFileReadAsync, hipFileWriteAsync, hipFileStreamRegister, tutorial

******************************************************
Asynchronous multi-stream I/O with registered streams
******************************************************

`roundtrip-async-multi-stream-registered.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/async/roundtrip-async-multi-stream-registered.cpp>`_ reads a file into GPU memory,
writes it back out, and splits the work across multiple HIP streams that run
concurrently.

.. note::

  Asynchronous I/O does not currently support the fastpath backend and will perform fallback I/O using a CPU bounce buffer.

When to use this pattern
=========================

Use multi-stream registered asynchronous I/O when you need to:

- Transfer large files in parallel slices so reads and writes overlap on the
  GPU.
- Pipeline storage I/O with GPU compute on separate streams.

Prerequisites
===============

Verify you have:

- A working hipFile installation. See :doc:`/install/install`.
- An AMD GPU with ROCm and hipFile support.
- A file system that supports ``O_DIRECT``, for example, ext4 or XFS.
- The ``examples_common`` helper library shipped with hipFile under
  ``examples/common/``.

Step-by-step walkthrough
===========================

Select the GPU and seed the input file
--------------------------------------

.. code-block:: cpp

   hip_err = hipSetDevice(gpu_id);

   if (seed_read_file(read_path, TOTAL_SIZE))
       return EXIT_FAILURE;

``seed_read_file()`` allocates a buffer where byte ``i`` is ``i & 0xFF``. It
writes that buffer to ``read_path`` with POSIX ``write()``, replacing any prior
file contents. The output size is ``NUM_STREAMS * SLICE_SIZE``. The defaults
use four streams and 1 MiB per slice, so 4 MiB total.

Allocate per-stream GPU buffers and register them
-------------------------------------------------

For each of the ``NUM_STREAMS`` streams, the example calls ``hipMalloc()`` to
allocate a GPU buffer. It registers that buffer with hipFile, creates a
non-blocking HIP stream, zeros the buffer on that stream, and registers the
stream with the driver.

Each buffer is registered separately with ``hipFileBufRegister()`` with ``flags``
set to ``0``. Streams are created with the ``hipStreamNonBlocking`` flag so they
don't implicitly synchronize with the legacy default stream.

``hipFileStreamRegister()`` takes flags that mark the properties that stay fixed
for every later submission on that stream:

- ``HIPFILE_STREAM_FIXED_BUF_OFFSET``: the buffer offset cannot be changed after submission.
- ``HIPFILE_STREAM_FIXED_FILE_OFFSET``: the file offset cannot be changed after submission.
- ``HIPFILE_STREAM_FIXED_FILE_SIZE``: the transfer size cannot be changed after submission.
- ``HIPFILE_STREAM_PAGE_ALIGNED_INPUTS``: all offsets and sizes are 4 KiB
  aligned.

The ``slice_state`` struct holds sizes and offsets as named fields. The
asynchronous API reads those values by pointer and dereferences them at
completion time. You must keep them valid until ``hipStreamSynchronize()``
finishes.

Open input and output files
---------------------------

The ``open_file()`` helper from ``examples_common`` calls ``open()`` and
then ``hipFileHandleRegister()``. One pair of file descriptors is shared across
all streams.

Submit reads on all streams
---------------------------

Each ``hipFileReadAsync()`` call enqueues a read of ``SLICE_SIZE`` bytes at file
offset ``i * SLICE_SIZE`` into the matching GPU buffer. The streams are
independent, so those reads can run concurrently.

Submit writes on all streams
----------------------------

Each write uses the same stream as its matching read. HIP stream semantics run
operations on one stream in submission order. The write for slice ``i`` sees
the read for slice ``i`` finish without an extra host synchronization call
between them. Across streams, reads and writes can overlap freely.

Synchronize and verify byte counts
----------------------------------

After synchronization, each ``slice_state`` instance reports ``bytes_read`` and
``bytes_written``. The example checks that every slice moved ``SLICE_SIZE``
bytes.

Verify the output file hash
---------------------------

``verify_files_match()`` hashes the first ``TOTAL_SIZE`` bytes of both files
with FNV-1a and compares the digests. Matching hashes mean the round trip was
lossless.

Clean up resources
------------------

Teardown reverses setup for each slice:

1. ``hipFileStreamDeregister()``: remove the stream from hipFile.
2. ``hipStreamDestroy()``: destroy the HIP stream.
3. ``hipFileBufDeregister()``: remove the GPU buffer from hipFile.
4. ``hipFree()``: free the device memory.

The booleans ``stream_registered``, ``stream_created``, and ``buf_registered``
track partial setup. If the setup loop fails partway through, only resources
that were created get torn down.

Stream ordering
=================

This example depends on HIP stream ordering:

- On one stream, work runs in submission order. The ``hipFileWriteAsync()``
  call for slice ``i`` is submitted after ``hipFileReadAsync()`` for that slice
  on that stream, so the write sees the completed read without extra
  synchronization.
- Across ``hipStreamNonBlocking`` streams, there's no ordering guarantee.
  Slices can read and write in parallel.

Increasing ``NUM_STREAMS`` adds more concurrent slices. Each stream touches a
disjoint region of the file, so you don't need cross-stream coordination.

Running the example
=======================

.. code:: shell

   ./roundtrip-async-multi-stream-registered /path/to/readfile /path/to/writefile [GPUID]

On success, the program prints:

.. code-block:: text

   OK  /path/to/readfile == /path/to/writefile  (4194304 bytes across 4 registered streams, hash 0x...)



