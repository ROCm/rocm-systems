fio
===

A hipfile engine has been added to the ROCm fork of fio:
https://github.com/ROCm/fio/tree/hipFile

To build fio with hipfile support use the following:

.. code-block:: console

    $ cd ~
    $ git clone git@github.com:ROCm/hipFile.git
    $ mkdir hipFile/build; cd hipFile/build
    $ cmake .. && cmake --build . -j

    $ cd ~
    $ git clone git@github.com:ROCm/fio.git
    $ cd fio && git checkout origin/hipFile
    $ HIPFILE=$HOME/hipFile \
            CFLAGS="-I/opt/rocm/include -I$HIPFILE/include" \
            LDFLAGS="-L/opt/rocm/lib -L$HIPFILE/build/src/amd_detail -Wl,-rpath,$HIPFILE/build/src/amd_detail" \
            ./configure --enable-libhipfile
    $ make -j

Use the example workload file from the fio repository:

.. code-block:: console

    $ GPU_DEV_IDS=0 FIO_DIR=PATH_TO_EXT4_OR_XFS_DIRECTORY ~/fio/fio ~/fio/examples/libhipfile-hipfile.fio
