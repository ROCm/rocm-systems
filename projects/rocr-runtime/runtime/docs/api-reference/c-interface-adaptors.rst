.. meta::
   :description: HSA runtime implementation
   :keywords: ROCR, ROCm, library, tool, runtime

.. _c-interface-adaptors:

C interface adaptors
=====================

The C interface layer is the :ref:`top layer in ROCR <runtime-design>` that provides C++ APIs as defined in the `HSA Runtime Specification 1.2 <https://hsafoundation.com/wp-content/uploads/2021/02/HSA-Runtime-1.2.pdf>`_. The C interface layer also consists of the interfaces and default definitions for the standard extensions. The interface functions simply forward to a function pointer table defined here. The table is initialized to point to default definitions, which simply returns an appropriate error code. If available, the extension library is loaded as part of runtime initialization and the table is updated to point to the extension library.

Files present in this layer:

- ``hsa.h`` (cpp)

- ``hsa_ext_interface.h`` (cpp)