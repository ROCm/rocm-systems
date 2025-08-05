.. meta::
    :description: ROCm Compute Profiler FAQ and troubleshooting
    :keywords: ROCm Compute Profiler, FAQ, troubleshooting, ROCm, profiler, tool, Instinct,
               accelerator, AMD, SSH, error, version, workaround, help

***
FAQ
***

Frequently asked questions and troubleshooting tips.

How do I export profiling data I have already generated using ROCm Compute Profiler?
====================================================================================

To interact with the Grafana GUI, you must sync data with the MongoDB
backend. You can do this using :ref:`database <modes-database>` mode.

Pass in the directory of your desired workload as follows.

.. code-block:: shell

    $ rocprof-compute database --import -w <path-to-results> -H <hostname> -u <username> -t <team-name>

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
