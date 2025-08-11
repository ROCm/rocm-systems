.. meta::
  :description: The ROCm Data Center tool (RDC) addresses key infrastructure challenges regarding AMD GPUs in cluster and data center environments and simplifies their administration
  :keywords: ROCm Data Center tool API, RDC API

.. _api-intro:

*************************
Introduction to RDC API
*************************

.. note::
  This is the alpha version of RDC API and is subject to change without notice. The primary purpose of this API is to solicit feedback. AMD accepts no responsibility for any software breakage caused by API changes.

RDC API
========

RDC API is the core library that provides all the RDC features.

RDC API includes the following libraries:

* ``librdc_bootstrap.so``: Loads one of the following two libraries during runtime, depending on the mode.

  - ``rdci`` mode: Loads ``librdc_client.so``
  - ``rdcd`` mode: Loads ``librdc.so``

* ``librdc_client.so``: Exposes RDC functionality using ``gRPC`` client.

* ``librdc.so``: RDC API. This depends on ``libamd_smi.so``.

* ``libamd_smi.so``: Stateless low overhead access to GPU data.

.. figure:: ../data/api_libs.png

    Different libraries and how they are linked.
