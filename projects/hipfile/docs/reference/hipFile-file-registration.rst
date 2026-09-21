.. meta::
   :description: File registration in hipFile
   :keywords: hipFile, ROCm, registration, file handle, buffer, hipFileHandleRegister

**************************
hipFile file registration
**************************

Before performing any I/O on a file, the file must be registered with hipFile through ``hipFileHandleRegister()``.

File registration is required before any call to ``hipFileRead()`` or ``hipFileWrite()``. ``hipFileHandleRegister()`` takes a descriptor of type ``hipFileDescr_t`` that specifies the handle type. It returns a file handle that is then passed to ``hipFileRead()`` and ``hipFileWrite()``.

For example, the |roundtrip_verify|_  example first calls ``open_file()`` from |examples_common|_ before calling ``hipFileWrite()``:

.. code:: cpp

  if (open_file(created_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &fd, &handle)) {
      goto deregister_buf;

      [...]

      nbytes = hipFileWrite(handle, devbuf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);

``open_file()`` calls ``hipFileHandleRegister()``:

.. code:: cpp

  int open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
  {
    *fd = open(path, flags, mode);
    if (-1 == *fd) {
        fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
        return 1;
    }

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = *fd;

    hipFileError_t hipfile_err = hipFileHandleRegister(handle, &descr);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Could not register %s (%s)\n", path, hipFileGetOpErrorString(hipfile_err.err));
        close(*fd);
        return 1;
    }

    return 0;
  }

Two file descriptors, a buffered file descriptor and an unbuffered file descriptor, are kept for the registered file. The unbuffered descriptor is used by the fastpath backend and the buffered descriptor is used by the fallback backend.

If the file doesn't support direct I/O and was not opened with ``O_DIRECT``, the unbuffered descriptor will not be available and fastpath will decline the I/O request. For more information on backend selection, see :doc:`/reference/hipFile-io-backends`.

When the file handle is no longer needed, ``hipFileHandleDeregister()`` must be called to release the associated resources.



.. |roundtrip_verify| replace:: ``roundtrip-verify.cpp``
.. _roundtrip_verify: https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile/examples/basics/roundtrip-verify.cpp

.. |examples_common| replace:: ``examples_common.cpp``
.. _examples_common: https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile/examples/common/examples_common.cpp
