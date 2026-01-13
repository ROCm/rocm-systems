.. meta::
  :description: Install rocminfo
  :keywords: install, rocminfo, AMD, ROCm


Building rocminfo
*****************

Use the standard CMake build procedure to build rocminfo. The ROCm root directory (the parent directory containing ROCM headers and libraries) must be provided as a CMake argument using the standard ``CMAKE_PREFIX_PATH`` variable.

After cloning the `rocminfo Git repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocminfo>`_, run ``git-fetch --tags`` to retrieve the repository tags, which are used for versioning.

For example:

.. code-block::

    $ git fetch --tags origin

    Building from the CMakeLists.txt directory might look like this:

    mkdir -p build

    cd build

    cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..

    make

    cd ..

Upon a successful build, the rocminfo binary and the Python script ``rocm_agent_enumerator`` will present in the ``build`` directory.

rocminfo execution
-------------------

rocminfo reports information about HSA system attributes and agents.

``rocm_agent_enumerator`` prints a list of AMD GCN ISA or architecture names. With the option ``-name``, it prints out the available architecture names obtained from rocminfo. Otherwise, it generates the ISA in one of the following ways:

* ``ROCM_TARGET_LST``: A user-defined environment variable that specifies the path to the ``target.lst`` file. This can be used in a sandboxed install environment where executing rocminfo is not possible.

* ``target.lst``: A user-supplied text file in the same folder as ``rocm_agent_enumerator``. This can be used in a container environment where the ROCm stack is not available.

* HSA topology: Gathers information from the HSA node topology under ``/sys/class/kfd/kfd/topology/nodes/``.

* ``lspci``: Enumerates the PCI bus and locates supported devices from a hard-coded lookup table.

* rocminfo: A tool shipped with this script to enumerate GPU agents available on a working ROCm stack.
