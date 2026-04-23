.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

********************************************************************
Build rocJPEG from source code
********************************************************************

Only build rocJPEG from its source code if you're contributing to the rocJPEG project or want to preview new features. 

.. code::

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
  cd rocm-systems
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocjpeg
  git checkout develop

The default develop branch is intended for users who want to preview new features or contribute to the rocJPEG codebase.

Change directory to ``projects/rocjpeg``:

.. code:: shell

  cd rocm-systems/projects/rocjpeg


Build and install rocJPEG using the following commands:

.. code:: shell

  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocJPEG libraries will be copied to ``/opt/rocm/lib`` and the rocJPEG header files will be copied to ``/opt/rocm/include/rocjpeg``.

Build the CTest-based verification:

.. code:: shell

  mkdir rocjpeg-test && cd rocjpeg-test
  cmake /opt/rocm/share/rocjpeg/test/
  ctest -VV

To test your build, run ``make test``. To run the test with the verbose option, run ``make test ARGS="-VV"``.

