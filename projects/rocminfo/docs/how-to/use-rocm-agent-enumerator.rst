.. meta::
  :description: agent, enumerator ROCmInfo
  :keywords: install, rocminfo, AMD, ROCm, ROCmInfo


Using ROCm agent enumerator
-----------------------------

The rocm_agent_enumerator tool prints the list of available AMD GCN ISA or acthitecture names. With the option ‘-name’, it prints out available architecture names that can be used by third-party scripts to determine which ISAs are needed to execute code on all GPUs in the system.

See the following example output of the rocm_agent_enumerator command on a system with an MI-300X installation,

.. code-block::

    gfx000
    gfx941


.. Note:: 

The gfx000 represents the CPU agent.

