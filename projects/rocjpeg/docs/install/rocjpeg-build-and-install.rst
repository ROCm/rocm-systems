.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

********************************************************************
Building and installing rocJPEG from source code
********************************************************************

rocJPEG is delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_. For TheRock installation details, refer to the `TheRock documentation <https://github.com/ROCm/TheRock#readme>`_.

To build rocJPEG standalone from source, :doc:`clone the rocJPEG project <./rocjpeg-clone-repo>` and change to the project directory:

.. code:: shell

  cd rocm-systems/projects/rocjpeg

Build and install rocJPEG using the following commands:

.. code:: shell

  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocJPEG libraries will be copied to ``/opt/rocm/lib`` and the rocJPEG header files will be copied to ``/opt/rocm/include/rocjpeg``.

To run the installed CTest-based verification:

.. code:: shell

  mkdir rocjpeg-test && cd rocjpeg-test
  cmake /opt/rocm/share/rocjpeg/test/
  ctest -VV

To test your build, run ``make test``. To run the test with the verbose option, run ``make test ARGS="-VV"``.
