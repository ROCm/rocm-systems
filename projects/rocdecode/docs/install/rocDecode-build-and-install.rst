.. meta::
  :description: Build and install rocDecode with the source code
  :keywords: install, building, rocDecode, AMD, ROCm, source code, developer

********************************************************************
Building and installing rocDecode from source code
********************************************************************

rocDecode is delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_. For TheRock installation details, refer to the `TheRock documentation <https://github.com/ROCm/TheRock#readme>`_.

To build rocDecode standalone from source, :doc:`clone the rocDecode project <./rocDecode-clone-project>` and change to the project directory:

.. code:: shell

  cd rocm-systems/projects/rocdecode

Build and install rocDecode using the following commands:

.. code:: shell

  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocDecode libraries will be copied to ``/opt/rocm/lib`` and the rocDecode header files will be copied to ``/opt/rocm/include/rocdecode``.

To run the installed CTest-based verification:

.. code:: shell

  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV

Run ``make test`` to test your build. To run the test with the verbose option, run ``make test ARGS="-VV"``.

