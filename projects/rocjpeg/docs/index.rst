.. meta::
  :description: rocJPEG documentation and API reference library
  :keywords: rocJPEG, ROCm, API, documentation

********************************************************************
rocJPEG documentation
********************************************************************

rocJPEG provides APIs and samples that you can use to easily access the JPEG decoding
features of your media engines (VCNs). It also allows interoperability with other compute engines on
the GPU using Video Acceleration API (VA-API)/HIP.

rocJPEG is delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_ and doesn't require separate build or installation. 

The rocJPEG source code is located at https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg and is intended for users who want to preview new features or contribute to the rocJPEG project.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: How to

    * :doc:`Use rocJPEG <how-to/using-rocjpeg>`
    * :doc:`Build rocJPEG from source code <how-to/rocjpeg-build-and-install>`
    * :doc:`Retrieve image information with rocJPEG <how-to/rocjpeg-retrieve-image-info>`
    * :doc:`Decode a JPEG stream with rocJPEG <how-to/rocjpeg-decoding-a-jpeg-stream>`

  .. grid-item-card:: Samples

    * `rocJPEG samples on GitHub <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg/samples>`_

  .. grid-item-card:: Reference

    * :doc:`rocJPEG environment variables <./reference/rocJPEG-env-vars>`
    * :doc:`rocJPEG subsampling and hardware capabilities <./reference/rocjpeg-formats-and-architectures>`
    * :doc:`rocJPEG API library <../doxygen/html/files>`
    * :doc:`rocJPEG Functions <../doxygen/html/globals>`
    * :doc:`rocJPEG Data structures <../doxygen/html/annotated>`


To contribute to the documentation, refer to
`Contributing to ROCm documentation <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.