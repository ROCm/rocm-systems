.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

********************************************************************
Building and installing rocJPEG from source code
********************************************************************

These instructions are for building rocJPEG from its source code. If you will not be contributing to the rocJPEG code base or previewing features, :doc:`package installers <./rocjpeg-package-install>` are available.

.. note::

  ROCm must be installed before installing rocJPEG. See `Quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for detailed ROCm installation instructions.

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

Install the CTest module:

.. code:: shell

  mkdir rocjpeg-test && cd rocjpeg-test
  cmake /opt/rocm/share/rocjpeg/test/
  ctest -VV

To test your build, run ``make test``. To run the test with the verbose option, run ``make test ARGS=\"-VV\"``. 
