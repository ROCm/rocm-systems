.. meta::
   :description: RCCL documentation hub for AMD's collective communication library — install, build, tutorials, how-to guides, conceptual articles, and API reference.
   :keywords: RCCL, ROCm, collective communication, multi-GPU, multi-node, AllReduce, AllGather, ReduceScatter, AllToAll, AMD Instinct, distributed training, HIP, MPI, InfiniBand

.. _index:

******************
RCCL documentation
******************

The ROCm Communication Collectives Library (RCCL) (pronounced "rickle") is an open-source, host-initiated library enabling collective communications executed via GPU -- it also
supports direct send/receive operations. RCCL supports collective algorithms across multiple processes and nodes using networking through Infiniband Verbs or TCP/IP sockets. RCCL is forked from
the NVIDIA Collective Communication Library (NCCL); RCCL maintains an identical API library to NCCL. 

The RCCL public repository is located within the rocm-systems repo at `<https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install RCCL <./install/installation>`
    * :doc:`Build from source <./install/building-installing>`
    * :doc:`Run RCCL using Docker <./install/docker-install>`

  .. grid-item-card:: Conceptual

    * :doc:`Collective operations in RCCL <./conceptual/collective-operations>`
    * :doc:`Collective algorithms in RCCL <./conceptual/collective-algorithms>`
    * :doc:`Collective protocols in RCCL <./conceptual/collective-protocols>`
    * :doc:`Hardware-specific optimizations <./conceptual/hardware-specific-optimizations>`
    * :doc:`Fault tolerance <./conceptual/fault-tolerance>`

  .. grid-item-card:: How to

    * :doc:`Run RCCL-Tests <./how-to/running-rccl-tests>`
    * :doc:`Use the RCCL Recorder and Replayer <./how-to/rccl-recorder-replayer>`
    * :doc:`Use the RCCL Tuner plugin <./how-to/using-rccl-tuner-plugin-api>`
    * :doc:`Use the NCCL Net plugin <./how-to/using-nccl>`
    * :doc:`RCCL usage tips <./how-to/rccl-usage-tips>`

  .. grid-item-card:: Tutorials

    * :doc:`Create a RCCL program <./tutorials/create-a-rccl-program>`

  .. grid-item-card:: Reference

    * :ref:`Library specification<library-specification>`
    * :ref:`api-library`
    * :doc:`Precision support <./reference/data-type-support>`
    * :ref:`Environment variables<env-variables>`
    * :doc:`Troubleshooting <./reference/troubleshooting-rccl>`

To contribute to the documentation, see
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
