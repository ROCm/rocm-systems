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
source, or its input data. The archive holds the workload's code and data instead,
so it is confidential: share one only as agreed, after reading the ``NOTICE.txt``
that capture writes into it (see :ref:`hrr-sharing-an-archive`).

HRR has two halves:

* **Capture** is part of the HIP runtime library. It is off unless you ask for it
  with an environment variable, and records only once you have consented to it.
* **Replay** is a separate developer tool, ``hrr-playback``, installed into the
  ROCm binary directory by a ROCm build that includes the HRR project.

This page takes you from an installed ROCm to a completed replay. For the archive
format, the full flag list and the known limitations, see the
`HRR README <https://github.com/ROCm/rocm-systems/blob/develop/projects/hrr/README.md>`_
and ``DESIGN.md`` next to it in the ROCm sources.

Getting the replay tool
=======================================================

Look in the ROCm binary directory, and ask the tool what it is:

.. code-block:: bash

  "${ROCM_PATH:-/opt/rocm}/bin/hrr-playback" --version

``--version`` needs neither an archive nor a GPU. It reports the archive format
version this build reads, the revision it was built from, and the HIP runtime it
loaded. Those three facts identify a replay, so quote them in any report about
one.

A ROCm container image that carries the tool is the shortest route when you have
been handed an archive and want to open it without touching the host:

.. code-block:: bash

  docker run --rm --device=/dev/kfd --device=/dev/dri \
    -v "$PWD/my_capture.hrr:/capture:ro" \
    <rocm-image> hrr-playback /capture --info

.. note::

  The tool loads the HIP runtime, the compiler libraries and the ROCm system
  dependency libraries at run time. Installing ROCm through its packages pulls
  all of them in. Unpacking the runtime package on its own is not enough, and
  fails at start-up on a missing library.

If your installation does not carry the tool, the HRR README covers building it
from the ROCm sources, and the rule that comes with it: a tool built from the
sources reads archives recorded by the HIP runtime built from the same commit.

Recording a workload
=======================================================

Set ``HIP_HRR_CAPTURE_OUTPUT`` to the archive you want written, and run the
application as usual:

.. code-block:: bash

  HIP_HRR_CAPTURE_OUTPUT=./my_capture.hrr ./my_hip_app

Capture is off whenever the variable is unset, so nothing changes for an ordinary
run. It also needs your consent, given in one of two ways:

* Once per user, with the line ``yes`` in the file
  ``${XDG_CONFIG_HOME:-$HOME/.config}/hrr/capture-consent``. The file covers that user on
  every host that shares the home directory.
* With ``HIP_HRR_CAPTURE_CONSENT=yes``, where no such file can be kept, as in CI or
  a container. The variable applies to every process that inherits it, child
  processes and MPI ranks included. Any other value, such as ``no``, withholds
  consent even when the file says yes. An empty value counts as unset, so the
  file still decides: to turn capture off, unset ``HIP_HRR_CAPTURE_OUTPUT`` or
  set ``HIP_HRR_CAPTURE_CONSENT=no``.

Without either, the application runs unrecorded and says why in one line on
stderr. Whoever controls your environment or configuration directory can give
consent for you. The file counts only if it is a regular file you own, not
reached through a symbolic link and with no other hard link.

.. code-block:: bash

  d="${XDG_CONFIG_HOME:-$HOME/.config}/hrr" && mkdir -p "$d" && echo yes > "$d/capture-consent"

The archive is a directory, not a single file, and it holds one
``pid-<pid>/`` subdirectory per recorded process:

.. code-block:: text

  my_capture.hrr/
    manifest.json       which processes the archive holds
    pid-<pid>/
      manifest.json     what this process recorded, and whether it ended cleanly
      events.bin        the recorded call sequence
      blobs/            host buffers, addressed by content hash
      code_objects/     the GPU code the run used
      regions/          optional, and written by tooling outside the runtime:
                        memory a framework allocated without going through HIP

Expect the result to be large. It contains the buffers the workload moved, so a
short run of a big model produces gigabytes.

.. _hrr-sharing-an-archive:

Sharing an archive
-------------------------------------------------------

.. warning::

  Capture writes the notice below to ``NOTICE.txt`` at the archive root. Read it
  before an archive leaves the machine.

.. literalinclude:: ../../../hrr/include/hrr/hrr_archive_notice.h
   :language: text
   :start-after: /* hrr-archive-notice-begin */
   :end-before: /* hrr-archive-notice-end */

The HRR sources explain what the notice covers in
`Sharing an archive <https://github.com/ROCm/rocm-systems/blob/develop/projects/hrr/PRIVACY.md#sharing-an-archive>`_.

Your first replay
=======================================================

Start with a summary of what the archive holds. This reads the archive only, and
needs no GPU:

.. code-block:: bash

  hrr-playback ./my_capture.hrr --info

Pointed at the archive directory, ``--info`` lists the recorded processes:

.. code-block:: text

  HRR Archive Root: ./my_capture.hrr
  ========================================
  Capture Mode: in-tree
  Owner PID:    1180
  Processes:    2

    PID          Parent PID   Complete   Events       Blobs      Path
    ---          ----------   --------   ------       -----      ----
    1180         0            yes        412          6          ./my_capture.hrr/pid-1180
    1204         1180         NO         918274       1503       ./my_capture.hrr/pid-1204

``Complete`` reports whether that process shut down cleanly. An archive from a run
that crashed is still usable: the reader recovers every complete record. To leave
it in a tidy state, ``hrr-playback ./my_capture.hrr --repair`` rewrites every
process capture under the archive with a clean trailer and rebuilds the index.

Point the tool at one ``pid-<pid>/`` directory for the detail: the event and
kernel breakdown, and with ``--events`` the whole recorded sequence.

.. code-block:: bash

  hrr-playback ./my_capture.hrr/pid-1204/ --info

Then replay it on a GPU:

.. code-block:: bash

  hrr-playback ./my_capture.hrr/pid-1204/

An archive holding a single process can be replayed by naming the archive
directory itself; the tool resolves it. With more than one recorded process it
lists them and asks you to choose, because there is no sensible default.

Replay reissues the recorded calls in order. Where the original run copied data
back from the device, HRR compares what the GPU produces now against what was
recorded, and finishes with a summary:

.. code-block:: text

  [HRR] Archive : 918274 events, 1180 kernels, 1503 blobs, 3 code objects
  [HRR] Threads : 4 captured
  [HRR] Device  : AMD Instinct MI350X (gfx950:sramecc+:xnack-)
  [HRR] Runtime : /opt/rocm/lib/libamdhip64.so
  [HRR] Mode    : single-threaded
  [HRR] -- Replay summary ------------------------------
  [HRR]   Wall time      : 8421.3 ms
  [HRR]   Threads used   : 1
  [HRR]   Kernels launched: 1180
  [HRR]   Graphs launched : 0
  [HRR]   D2H checks     : 14 pass (12 exact, 2 within tol), 0 fail, 0 skipped
  [HRR] PASS

The exit code is zero when every comparison passed.

When a replay faults or hangs, ``--sync-after-launch`` makes the tool wait for
the GPU after every kernel, so the failure stops replay on the kernel that caused
it and that kernel is named. ``--sync-watchdog-ms`` gives up on a hang after a
time limit instead of blocking. The README covers the rest of the flags.

Matching the archive to the tool
=======================================================

The archive format carries a version number, and the reader accepts only its own
version, which ``--version`` reports. An archive and a ``hrr-playback`` from
different ROCm releases therefore do not necessarily work together, in either
direction. The mismatch is reported with both numbers:

.. code-block:: text

  [HRR] Version mismatch in ./my_capture.hrr/pid-1204/events.bin: file=4 reader=5

Read this as: the archive is written in format version 4, and this tool reads
version 5. The fix is to replay with a ``hrr-playback`` from the same ROCm
release that produced the archive, rather than to convert the archive.

The number changes when the recorded layout changes, which since version 5 no
longer happens merely because ROCm gained an API. So when you ask somebody for an
archive, ask which ROCm version produced it.

Recording and replaying on the same machine and the same ROCm installation always
matches, because both halves ship together.
