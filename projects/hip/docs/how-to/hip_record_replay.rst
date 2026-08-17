.. meta::
   :description: How to record a HIP workload into an HRR archive and replay it on a GPU
                 with hrr-playback.
   :keywords: AMD, ROCm, HIP, HRR, record, replay, capture, playback, debugging

**********************************************************
Recording and replaying HIP workloads
**********************************************************

HIP Record & Replay (HRR) records the HIP API calls an application makes, together
with the host buffers and code objects it needs, into an archive that can be
replayed on a GPU later. The replay does not need the original application, its
source, or its input data, which makes an archive a practical way to hand a
failing workload to somebody else.

HRR has two halves, and both ship with ROCm:

* **Capture** lives in the HIP runtime. It is off unless you ask for it, and is
  turned on with a single environment variable.
* **Replay** is a separate program, ``hrr-playback``, installed alongside the
  other ROCm binaries.

This page takes you from an installed ROCm to a completed replay. For the archive
format, the full flag list and the known limitations, see the
`HRR README <https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/hipamd/src/hrr/README.md>`_
and ``DESIGN.md`` next to it in the ROCm sources.

Getting the replay tool
=======================================================

``hrr-playback`` is part of the ROCm HIP runtime component, so a normal ROCm
installation already provides it under the ROCm binary directory:

.. code-block:: bash

  ls "${ROCM_PATH:-/opt/rocm}/bin/hrr-playback"

The ROCm container images carry it as well, which is the shortest route when you
have been handed an archive and want to open it without touching the host:

.. code-block:: bash

  docker run --rm --device=/dev/kfd --device=/dev/dri \
    -v "$PWD/my_capture.hrr:/capture:ro" \
    <rocm-image> hrr-playback /capture --info

.. note::

  The tool loads the HIP runtime, the compiler libraries and the ROCm system
  dependency libraries at run time. Installing ROCm through its packages pulls
  all of them in. Unpacking the runtime package on its own is not enough, and
  fails at start-up on a missing library.

Recording a workload
=======================================================

Set ``HIP_HRR_CAPTURE_OUTPUT`` to the archive you want written, and run the
application as usual:

.. code-block:: bash

  HIP_HRR_CAPTURE_OUTPUT=./my_capture.hrr ./my_hip_app

Capture is off whenever the variable is unset, so nothing changes for an ordinary
run. The archive is a directory, not a single file, and it holds one
``pid-<pid>/`` subdirectory per recorded process:

.. code-block:: text

  my_capture.hrr/
    manifest.json       which processes the archive holds
    pid-<pid>/
      manifest.json     what this process recorded, and whether it ended cleanly
      events.bin        the recorded call sequence
      blobs/            host buffers, addressed by content hash
      code_objects/     the GPU code the run used

Expect the result to be large. It contains the buffers the workload moved, so a
short run of a big model produces gigabytes.

.. warning::

  The archive holds the recorded buffers verbatim, so anything the workload had
  in host memory is readable in it. Treat an archive as you would treat the input
  data of the run it came from.

Your first replay
=======================================================

Start with a summary. This reads the archive only, and needs no GPU:

.. code-block:: bash

  hrr-playback ./my_capture.hrr --info

The summary reports how many events, kernels and buffers the archive holds, and
whether the recorded process shut down cleanly. An archive from a run that
crashed is still usable: the reader recovers every complete record and says so.

Then replay it on a GPU:

.. code-block:: bash

  hrr-playback ./my_capture.hrr

Replay reissues the recorded calls in order. Where the original run copied data
back from the device, HRR compares what the GPU produces now against what was
recorded, and finishes with a summary:

.. code-block:: text

  [HRR] Device  : AMD Instinct MI350X (gfx950:sramecc+:xnack-)
  [HRR] -- Replay summary ------------------------------
  [HRR]   Kernels launched: 20
  [HRR]   D2H checks     : 1 pass, 0 fail, 0 skipped
  [HRR] PASS

The exit code is zero when every comparison passed. If the archive contains more
than one recorded process, point the tool at the process you want rather than at
the archive root:

.. code-block:: bash

  hrr-playback ./my_capture.hrr/pid-<pid>/

When a replay faults or hangs, ``--sync-after-launch`` makes the tool wait for
the GPU after every kernel, so the failure stops replay on the kernel that caused
it and that kernel is named. ``--sync-watchdog-ms`` gives up on a hang after a
time limit instead of blocking. The README covers the rest of the flags.

Matching the archive to the tool
=======================================================

The archive format carries a version number, and the reader accepts only its own
version. An archive and a ``hrr-playback`` from different ROCm releases therefore
do not necessarily work together, in either direction. The mismatch is reported
with both numbers:

.. code-block:: text

  [HRR] Version mismatch: file=3 reader=4

Read this as: the archive is written in format version 3, and this tool reads
version 4. The fix is to replay with a ``hrr-playback`` from the same ROCm
release that produced the archive, rather than to convert the archive.

Two practical consequences:

* When you ask somebody for an archive, ask which ROCm version produced it.
* When the archive root holds ``pid-<pid>/`` subdirectories, a tool too old to
  understand that layout reports a missing ``events.bin`` instead of a version
  mismatch. Point it at the process subdirectory to get the real diagnosis.

Recording and replaying on the same machine and the same ROCm installation always
matches, because both halves ship together.
