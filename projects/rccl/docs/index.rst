.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API

.. _index:

******************
RCCL documentation
******************

The ROCm Communication Collectives Library (RCCL) is a stand-alone library
that provides multi-GPU and multi-node collective communication primitives
optimized for AMD GPUs. It uses PCIe and xGMI high-speed interconnects.
To learn more, see :doc:`what-is-rccl`

The RCCL public repository is located at `<https://github.com/ROCm/rccl>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installing RCCL using the install script <./install/installation>`
    * :doc:`Running RCCL using Docker <./install/docker-install>`
    * :doc:`Building and installing RCCL from source code <./install/building-installing>`

  .. grid-item-card:: How to

    * :doc:`Using the RCCL Tuner plugin <./how-to/using-rccl-tuner-plugin-api>`
    * :doc:`Using the NCCL Net plugin <./how-to/using-nccl>`
    * :doc:`Troubleshoot RCCL <./how-to/troubleshooting-rccl>`
    * :doc:`RCCL usage tips <./how-to/rccl-usage-tips>`


  .. grid-item-card:: Examples

    * `RCCL Tuner plugin examples <https://github.com/ROCm/rccl/tree/develop/ext-tuner/example>`_
    * `NCCL Net plugin examples <https://github.com/ROCm/rccl/tree/develop/ext-net/example>`_

  .. grid-item-card:: API reference

    * :ref:`Library specification<library-specification>`
    * :ref:`api-library`
    * :ref:`Environment variables<env-variables>`

To contribute to the documentation, see
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
