.. meta::
  :description: ROCm Data Center tool (RDC) glossary and terminology
  :keywords: ROCm Data Center tool, RDC, glossary, terminology, GPU group, field group, health watch, policy

.. _rdc-glossary:

********
Glossary
********

This topic explains the terminology used throughout the RDC documentation.

RDC
  The ROCm Data Center tool: a library, daemon, and CLI for monitoring and managing AMD GPUs
  (and AMD EPYC CPUs) across a cluster or data center. See :ref:`components`.

Compute node (CN)
  A node containing one or more GPUs, and optionally CPUs, on which compute jobs run, and on
  which ``rdcd`` typically runs.

Management node (MN) or main console
  A machine running system administration software that uses ``rdci`` or the RDC library to
  monitor and manage one or more compute nodes.

Standalone mode
  Running RDC with all its components installed, where ``rdcd`` runs as its own daemon and is
  queried locally or remotely by ``rdci``. See :ref:`using-RDC`.

Embedded mode
  Linking the RDC library directly into a management tool instead of running ``rdcd``, for
  fine-grained control with no gRPC dependency. See :ref:`using-RDC`.

GPU group
  A logical, user-defined grouping of one or more GPUs on a compute node, created with
  ``rdci group``, and used as the target of monitoring, health, and policy commands.

GPU or CPU partition
  A compute partition or CPU socket addressed individually, for example ``g0.1`` for GPU 0
  partition 1, or ``c0`` for CPU socket 0, when monitoring finer-grained than a whole device.
  See :ref:`rdc-features`.

Field
  A single metric RDC can read or watch, such as GPU temperature, memory usage, or power draw,
  identified by a numeric field ID, for example ``RDC_FI_POWER_USAGE``.

Field group
  A logical grouping of multiple fields, created with ``rdci fieldgroup``, so they can be
  watched or queried together, for example by ``rdci dmon``.

Job
  A workload submitted to one or more compute nodes whose GPU statistics RDC can record and
  summarize with ``rdci stats``. See :ref:`job-stats`.

Health watch
  A hardware subsystem, such as PCIe, XGMI, memory, EEPROM, or power/thermal, that
  ``rdci health`` monitors for a GPU group and reports as pass, warning, or fail.

Policy
  A threshold, such as page retirement, temperature, or power, registered on a GPU group with
  ``rdci policy``, which can trigger a notification or a GPU reset when exceeded.
