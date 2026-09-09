.. meta::
   :description: Tutorial walking through batch I/O using hipFile.
   :keywords: hipFile, ROCm, batch, GPU I/O, hipFileBatchIOSubmit, hipFileBatchIOGetStatus, hipFileBatchIOSetUp, tutorial

*********
Batch I/O
*********

`batch-roundtrip.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/batch/batch-roundtrip.cpp>`_ reads a file into a
registered GPU buffer with a batch of 4 KiB requests, writes the buffer back out
in a second batch phase, and verifies the output matches the input. Instead of
submitting every request at once, it keeps a fixed number of requests in flight
and submits a replacement each time one completes.

When to use this pattern
=========================

Use batch I/O when you need to:

- Amortize submission overhead across many small transfers instead of paying it
  per request.
- Keep the storage device busy with a bounded number of outstanding requests.
- Process completions as they arrive, in whatever order the device returns them.

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

Parse arguments and seed the input file
---------------------------------------

The example takes the payload size and the batch capacity on the command line:

.. code-block:: cpp

   payload_size   = parse_integral<size_t>(argv[3]);
   batch_capacity = parse_integral<unsigned>(argv[4]);

``batch_capacity`` must be between ``1`` and ``128``.

``seed_read_file()`` then writes ``payload_size`` bytes where byte ``i`` is
``i & 0xFF`` to ``read_path`` with POSIX ``write()``, replacing any prior file
contents.

Allocate and register one GPU buffer
------------------------------------

.. code-block:: cpp

   const size_t buffer_size = align_up(payload_size, BATCH_IO_SIZE);

   hip_err     = hipMalloc(&device_buffer, buffer_size);
   hipfile_err = hipFileBufRegister(device_buffer, buffer_size, 0);

The buffer holds the whole file rounded up to a 4 KiB boundary, so every request
transfers a full ``BATCH_IO_SIZE`` block without a partial tail. All requests in
both phases share this one registered buffer and address disjoint regions of it
through ``devPtr_offset``.

Open input and output files
---------------------------

The ``open_file()`` helper from ``examples_common`` calls ``open()`` with
``O_DIRECT`` and then ``hipFileHandleRegister()``. Each phase uses one file
handle for all of its requests.

Create the batch context
------------------------

.. code-block:: cpp

   hipfile_err = hipFileBatchIOSetUp(&batch_handle, batch_capacity);

``batch_capacity`` is the maximum number of requests that can be in flight on
this handle at one time. The same handle is reused for the read phase and the
write phase. Each phase drains completely before the next one starts.

Describe a request
------------------

``configure_request()`` fills in one ``hipFileIOParams_t`` for chunk
``chunk_index``:

.. code-block:: cpp

   request.operation.mode                  = hipFileBatch;
   request.operation.u.batch.devPtr_base   = device_buffer;
   request.operation.u.batch.file_offset   = static_cast<int64_t>(operation_offset);
   request.operation.u.batch.devPtr_offset = static_cast<int64_t>(operation_offset);
   request.operation.u.batch.size          = BATCH_IO_SIZE;
   request.operation.fh                    = file_handle;
   request.operation.opcode                = opcode;
   request.operation.cookie                = request.cookie.get();

``mode`` must be ``hipFileBatch``.  The ``cookie`` is an opaque pointer
that hipFile hands back in the completion event. The example points it at a
``BatchCookie`` holding the chunk index and the byte count that chunk should
transfer. Because the cookie is the only link between an event and its request,
it must stay valid until that request completes.

Reads at the tail of the file expect fewer than ``BATCH_IO_SIZE`` bytes, so
``expected_bytes`` is clamped to the bytes remaining in the payload. Writes
always expect a full block because the padded buffer is written out in full and
truncated afterwards.

Fill the window and submit
--------------------------

``run_batch_phase()`` starts with up to ``batch_capacity`` configured requests
and submits every request that isn't already in flight:

.. code-block:: cpp

   hipfile_err = hipFileBatchIOSubmit(batch_handle, submissions.size(),
                                      submissions.data(), /*flags=*/0);

``hipFileBatchIOSubmit()`` takes an array of parameters, so one call can enqueue
many requests. The number of in-flight requests plus the number being submitted
must not exceed the capacity passed to ``hipFileBatchIOSetUp()``.

Wait for completions
--------------------

.. code-block:: cpp

   unsigned nr = batch_capacity;
   hipfile_err = hipFileBatchIOGetStatus(batch_handle, /*min_nr=*/1, &nr,
                                         events.data(), /*timeout=*/nullptr);

``min_nr`` is the minimum number of events to wait for. ``nr`` is an in/out
parameter that carries the event array capacity in and the number of events
returned out. A ``nullptr`` timeout blocks until at least ``min_nr`` events are
available. Passing ``min_nr`` of ``1`` returns as soon as any request finishes,
which is what lets the example refill the window promptly.

Check each event and refill the window
--------------------------------------

For every returned event, the example finds the matching request by cookie, then
checks both the status and the transferred byte count:

.. code-block:: cpp

   if (event.status != hipFileComplete) { /* request failed */ }
   if (event.ret != cookie->expected_bytes) { /* short transfer */ }

The ``hipFileComplete`` status will be returned for successful requests.
``hipFileFailed`` or ``hipFileCanceled`` will be returned for failed or
canceled events. ``event.ret`` is the byte count for that request, which
can be short even when the status is ``hipFileComplete``.

If chunks remain, the completed slot is reconfigured for the next chunk index
and resubmitted on the following loop iteration. Otherwise the slot is swapped
to the end of the request vector and popped, shrinking the window. The phase
ends when no request slots are left.

Finish the round trip
---------------------

All reads complete before any writes are submitted, so the write phase always
sees a fully populated buffer. After the write phase, ``ftruncate()`` cuts the
output file back from the 4 KiB-aligned size to ``payload_size``, and
``verify_files_match()`` hashes the first ``payload_size`` bytes of both files
with FNV-1a and compares the digests. Matching hashes mean the round trip was
lossless.

Clean up resources
------------------

Teardown reverses setup:

1. ``hipFileBatchIODestroy()``: destroy the batch context.
2. ``close_file()``: deregister and close each file handle.
3. ``hipFileBufDeregister()``: remove the GPU buffer from hipFile.
4. ``hipFree()``: free the device memory.

The flags ``read_handle_open``, ``write_handle_open``, and ``buffer_registered``
track partial setup. If setup fails partway through, only resources that were
created get torn down.

Completion ordering
====================

Batch requests carry no ordering guarantees:

- Events can come back in any order, and one ``hipFileBatchIOGetStatus()`` call
  can return anywhere from ``min_nr`` to ``nr`` events.
- Requests in a batch can overlap in time, so two requests should not modify the
  same file or buffer region.
- Ordering between phases has to be enforced by the application. This example
  drains the read phase entirely before submitting any writes.

Increasing ``BATCH_SIZE`` widens the window and raises the number of concurrent
requests. Each request touches a disjoint 4 KiB region of the file and of the
buffer, so no cross-request coordination is needed.

Running the example
=======================

.. code:: shell

   ./batch-roundtrip /path/to/readfile /path/to/writefile FILE_SIZE BATCH_SIZE [GPUID]

For example, a 4 MiB payload with 16 requests in flight:

.. code:: shell

   ./batch-roundtrip /path/to/readfile /path/to/writefile 4194304 16

On success, the program prints:

.. code-block:: text

   OK  /path/to/readfile == /path/to/writefile  (4194304 bytes, hash 0x...)
