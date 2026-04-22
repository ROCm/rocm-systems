.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

********************************************************************
Building and installing rocJPEG from source code
********************************************************************

If you will be contributing to the rocJPEG code base, or if you want to preview new features, build rocJPEG from its source code.

:doc:`Clone the rocJPEG project <./rocjpeg-clone-repo>`. Change directory to ``projects/rocjpeg``:

.. code:: shell

  cd rocm-systems/projects/rocjpeg

Use `rocJPEG-setup.py <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg/rocJPEG-setup.py>`_ to install prerequisites:

.. code:: shell

  python rocJPEG-setup.py  --rocm_path [ ROCm Installation Path - optional (default:/opt/rocm)]

Build and install rocJPEG using the following commands:

.. code:: shell

  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocJPEG libraries will be copied to ``/opt/rocm/lib`` and the rocJPEG header files will be copied to ``/opt/rocm/include/rocjpeg``.

Build the CTest module:

.. code:: shell

  mkdir rocjpeg-test && cd rocjpeg-test
  cmake /opt/rocm/share/rocjpeg/test/
  ctest -VV

To test your build, run ``make test``. To run the test with the verbose option, run ``make test ARGS=\"-VV\"``. 
