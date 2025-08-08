.. meta::
  :description: ROCm SMI
  :keywords: install, SMI, library, api, AMD, ROCm

****************************************************
ROCm System Management Interface (ROCm SMI) library
****************************************************

The ROCm SMI library, is part of the ROCm software stack. It is a C++ library for Linux that provides a user space interface for applications to monitor and control GPU applications.

For more information, refer to `<https://github.com/ROCm/rocm_smi_lib>`__.

.. note::

   The AMD System Management Interface Library (AMD SMI) is a successor to ROCm SMI. It is a unified system management
   interface tool that provides a user space interface for applications to monitor and control GPU applications and gives
   users the ability to query information about drivers and GPUs on the system.

   AMD SMI will replace ``rocm_smi_lib`` over time. We recommend that users transition to AMD SMI.

   For more information, refer to `<https://github.com/ROCm/amdsmi>`__ and the :doc:`AMD SMI documentation <amdsmi:index>`.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

     * :doc:`ROCm SMI installation <./install/install>`

  .. grid-item-card:: API Reference

      * :doc:`Files <../doxygen/html/files>`
      * :doc:`Globals <../doxygen/html/globals>`
      * :doc:`Data structures <../doxygen/html/annotated>`
      * :doc:`Modules <../doxygen/html/modules>`
      * :doc:`Python API <reference/python_api>`

  .. grid-item-card:: How to

     * :doc:`Use C++ in ROCm SMI <how-to/use-cpp>`
     * :doc:`Use Python in ROCm SMI <how-to/use-python>`
   

  .. grid-item-card:: Tutorials    

     * :doc:`C++ <tutorials/cpp_tutorials>`
     * :doc:`Python <tutorials/python_tutorials>`  


To contribute to the documentation, refer to `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

