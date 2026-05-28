.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

********************************************************************
Building and installing rocJPEG from source code
********************************************************************

These instructions are for building rocJPEG from its source code. If you will not be contributing to the rocJPEG code base or previewing features,  `package installers <https://rocm.docs.amd.com/projects/rocJPEG/en/latest/install/rocjpeg-package-install.html>`_ are available.

.. note::

  ROCm must be installed before installing rocJPEG. See `Quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for detailed ROCm installation instructions.

Use `rocJPEG-setup.py <https://github.com/ROCm/rocJPEG/blob/develop/rocJPEG-setup.py>`_ available from the rocJPEG GitHub repo to install the prerequisites:

.. code:: shell

  python rocJPEG-setup.py  --rocm_path [ ROCm Installation Path - optional (default:/opt/rocm)]

Build and install rocJPEG using the following commands:

.. code:: shell

  git clone https://github.com/ROCm/rocJPEG.git
  cd rocJPEG
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
