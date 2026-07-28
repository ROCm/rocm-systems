.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API

.. _index:

******************
RCCL documentation
******************

The ROCm Communication Collectives Library (RCCL) (pronounced "rickle") is an open-source, host-initiated library enabling collective communications executed via GPU -- it also
supports direct send/receive operations. RCCL supports collective algorithms across multiple processes/nodes via networking using Infiniband Verbs or TCP/IP sockets. RCCL is forked from
the NVIDIA Collective Communication Library (NCCL); RCCL maintains an indentical API library to NCCL. 

The RCCL public repository is located within the rocm-systems repo at `<https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install RCCL <./install/installation>`
    * :doc:`Build from source <./install/building-installing>`
    * :doc:`Running RCCL using Docker <./install/docker-install>`

  .. grid-item-card:: How to

    * :doc:`Using the RCCL Tuner plugin <./how-to/using-rccl-tuner-plugin-api>`
    * :doc:`Using the NCCL Net plugin <./how-to/using-nccl>`
    * :doc:`Fault tolerance in RCCL <./how-to/fault-tolerance>`
    * :doc:`Troubleshoot RCCL <./how-to/troubleshooting-rccl>`
    * :doc:`RCCL usage tips <./how-to/rccl-usage-tips>`

  .. grid-item-card:: Conceptual

    * :doc:`Collective operations in RCCL <./conceptual/collective-operations>`

  .. grid-item-card:: Examples

    * `RCCL Tuner plugin examples <https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl/plugins/tuner/example>`_
    * `NCCL Net plugin examples <https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl/plugins/net/example>`_

  .. grid-item-card:: API reference

    * :ref:`Library specification<library-specification>`
    * :ref:`api-library`
    * :doc:`Precision support <./api-reference/data-type-support>`
    * :ref:`Environment variables<env-variables>`

To contribute to the documentation, see
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
