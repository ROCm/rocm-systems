.. meta::
  :description: Build and install rocDecode with the source code
  :keywords: install, building, rocDecode, AMD, ROCm, source code, developer

********************************************************************
Building and installing rocDecode from source code
********************************************************************

If you will be contributing to the rocDecode code base, or if you want to preview new features, build rocDecode from its source code.

If you will not be previewing features or contributing to the code base, use the :doc:`package installers <./rocDecode-package-install>` to install rocDecode. 

Before building rocDecode, use `rocDecode-setup.py <https://github.com/ROCm/rocDecode/blob/develop/rocDecode-setup.py>`_ to install all the required prerequisites:

.. code:: shell

  python3 rocDecode-setup.py  [--rocm_path ROCM_INSTALLATION_PATH; default=/opt/rocm]
                              [--runtime {ON|OFF}; default=ON]
                              [--developer {ON|OFF}; default=OFF]

.. note:: 

  Never run ``rocDecode-setup.py`` with ``--runtime OFF``.  
  
  ``--developer ON`` is required to use the code samples.

Build and install rocDecode using the following commands:

.. code:: shell

  git clone https://github.com/ROCm/rocDecode.git
  cd rocDecode
  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocDecode libraries will be copied to ``/opt/rocm/lib`` and the rocDecode header files will be copied to ``/opt/rocm/include/rocdecode``.

Build and install the rocDecode test module. This module is required if you'll be using the rocDecode samples, and can only be installed if ``rocDecode-setup.py`` was run with ``--developer ON``.

.. code:: shell

  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV

Run ``make test`` to test your build. To run the test with the verbose option, run ``make test ARGS="-VV"``. 

To create a package installer for rocDecode, run:

.. code:: shell

  sudo make package

