.. meta::
   :description: A library that can be loaded by ROCr to print the AMDGPU wavefront states
   :keywords: ROCdebug-agent documentation, ROCR Debug Agent documentation, rocr, ROCR

.. _index:

===============================
ROCR Debug Agent documentation
===============================

The ROCR Debug Agent (ROCdebug-agent) is a library that can be loaded by the ROCm software runtime :doc:`(ROCR) <rocr-runtime:index>` to provide the following functionality:

- Print the state of all AMD GPU wavefronts that cause a queue error (such as a memory violation, executing a ``s_trap 2``, or executing an illegal instruction).

- Print the state of all AMD GPU wavefronts by sending a SIGQUIT signal to the process using ``kill -s SIGQUIT <pid>`` command or by pressing ``Ctrl-\``, while the program is executing.

This functionality is provided for all AMD GPUs supported by the ROCm Debugger API Library :doc:`(ROCdbgapi) <rocdbgapi:index>`.

The code is open source and hosted at https://github.com/ROCm/rocr_debug_agent

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`Installation <debug-agent-installation>`

  .. grid-item-card:: How to

    * :ref:`User guide <debug-agent-user-guide>`

To contribute to the documentation, refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
