.. meta::
  :description: Build and install rocDecode with the source code
  :keywords: install, building, rocDecode, AMD, ROCm, source code, developer

********************************************************************
Build rocDecode from source code
********************************************************************

Only build rocDecode from source code if you're contributing to the rocDecode project or want to preview new features. 

Use sparse checkout to clone the rocDecode project:

.. code::

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
  cd rocm-systems
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocdecode
  git checkout develop

The default develop branch is intended for users who want to preview new features or contribute to the rocDecode codebase.

Change directory to ``projects/rocdecode``:

.. code:: shell

  cd rocm-systems/projects/rocdecode

Build and install rocDecode using the following commands:

.. code:: shell

  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocDecode libraries will be copied to ``/opt/rocm/lib`` and the rocDecode header files will be copied to ``/opt/rocm/include/rocdecode``.

.. note::

  FFmpeg development libraries must be installed to build and run samples:

  ``sudo apt install libavcodec-dev libavformat-dev libavutil-dev``

Build the CTest-based verification:

.. code:: shell

  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV

Run ``make test`` to test your build. To run the test with the verbose option, run ``make test ARGS="-VV"``.

