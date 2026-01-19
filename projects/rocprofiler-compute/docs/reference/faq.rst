.. meta::
    :description: ROCm Compute Profiler FAQ and troubleshooting
    :keywords: ROCm Compute Profiler, FAQ, troubleshooting, ROCm, profiler, tool, Instinct,
               accelerator, AMD, SSH, error, version, workaround, help

***
FAQ
***

Frequently asked questions and troubleshooting tips.

python ast error: 'Constant' object has no attribute 'kind'
===========================================================

This error arises from a bug in the default ``astunparse 1.6.3`` with
``python 3.8``. The error doesn't seem to occur with Python 3.7 or 3.9.

Workaround:

.. code-block:: shell

   $ pip3 uninstall astunparse
   $ pip3 astunparse

tabulate doesn't print properly
===============================

To get around this issue, set the following environment variables to update your
locale settings.

.. code-block:: shell

   $ export LC_ALL=C.UTF-8
   $ export LANG=C.UTF-8

Why does VALU utilization exceed the theoretical peak?
======================================================

In specific circumstances, the GPU can co-issue two VALU instructions in the same clock cycle. This may result in an observed VALU Utilization and FP64 VALU FLOP values above the theoretical peak. This is expected hardware behavior and not a measurement error.

This dual-issue capability can be further investigated via:

* **ROCm Compute Viewer**: The Instructions view shows when two instructions are issued to the VALU in the same cycle.
* **On MI350 and newer platforms**: Starting in ROCm 7.2.0, the ``Dual-issue VALU Utilization`` metric shows the % of time when VALU is executing dual-issued instructions.

When ROCm Compute Profiler detects values exceeding their theoretical peaks, it displays a warning message indicating this behavior.

What does "Counter variance corrected" mean?
=============================================

When profiling, you may see the following warning:

.. code-block:: text

   WARNING: Counter variance corrected: X value(s) adjusted (max Y% deviation from multi-pass collection).

This indicates that ROCm Compute Profiler detected and corrected negative values in derived metrics. This is expected behavior, not an error.

**Why does this happen?**

Hardware performance counters are collected across multiple profiling passes. When calculating derived metrics that involve subtraction (such as ``A - B``), small run-to-run variance can occasionally produce negative results. Since negative event counts are physically impossible, these values are automatically clamped to zero.

**When should I be concerned?**

* **Deviation < 1%**: Normal hardware variance. No action needed.
* **Deviation ≥ 1%**: The warning is displayed. Results are still valid, but variance was higher than typical.
* **Deviation > 5%**: Consider investigating profiling conditions (system load, thermal throttling, non-deterministic application behavior, etc.).

This correction primarily affects L2 cache metrics where counter subtraction is used to derive values like remote read/write traffic, but run-to-run variations may impact the accuracy of a number of derived metrics in ROCm Compute Profiler.

How can I SSH tunnel in MobaXterm?
==================================

1. Open MobaXterm.
2. In the top ribbon, select **Tunneling** to access tunneling options.

   .. image:: ../data/faq/tunnel_demo1.png
      :align: center
      :alt: MobaXterm Tunnel button
      :width: 800

   This pop-up should appear.

   .. image:: ../data/faq/tunnel_demo2.png
      :align: center
      :alt: MobaXterm pop-up
      :width: 800

3. Select **New SSH tunnel**.

   .. image:: ../data/faq/tunnel_demo3.png
      :align: center
      :alt: MobaXterm pop-up
      :width: 800

4. Configure the SSH tunnel.

   Local clients
     * ``<Forwarded port>``: ``[PORT]``

   Remote server
     * ``<Remote server>``: ``localhost``
     * ``<Remote port>``: ``[PORT]``

   SSH server
     * ``<SSH server>``: *name of the server to connect to*
     * ``<SSH login>``: *username to login to the server*
     * ``<SSH port>``: ``22``
