.. meta::
  :description: retrieving image information with rocJPEG
  :keywords: rocJPEG, ROCm, API, documentation, image information, jpeg


********************************************************************
Retrieving image information with rocJPEG
********************************************************************

Retrieving image information is done using ``rocJpegGetImageInfo()``. 

.. code:: cpp

    RocJpegStatus rocJpegGetImageInfo(
      RocJpegHandle handle,
      RocJpegStreamHandle jpeg_stream_handle,
      uint8_t *num_components,
      RocJpegChromaSubsampling *subsampling,
      uint32_t *widths,
      uint32_t *heights);

``rocJpegGetImageInfo()`` takes the ``RocJpegHandle`` and a ``RocJpegStreamHandle`` as inputs, and returns the subsampling, number of components, and widths and heights of the components. These are passed to the ``subsampling``, ``num_components``, and ``widths`` and ``heights`` output parameters.

The ``subsampling`` output parameter is a ``RocJpegChromaSubsampling`` enum. 

.. code:: cpp

    typedef enum {
      ROCJPEG_CSS_444 = 0,
      ROCJPEG_CSS_440 = 1,
      ROCJPEG_CSS_422 = 2,
      ROCJPEG_CSS_420 = 3,
      ROCJPEG_CSS_411 = 4,
      ROCJPEG_CSS_400 = 5,
      ROCJPEG_CSS_UNKNOWN = -1
    } RocJpegChromaSubsampling;

Its value is set to the chroma subsampling retrieved from the image. 

For example:

.. code:: cpp

  // Get the image info
  uint8_t num_components;
  RocJpegChromaSubsampling subsampling;
  uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
  uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};

  status = rocJpegGetImageInfo(handle, rocjpeg_stream_handle, &num_components, &subsampling, widths, heights);
  if (status != ROCJPEG_STATUS_SUCCESS) {
    std::cerr << "Failed to get image info with error code: " << rocJpegGetErrorName(status) << std::endl;
    rocJpegStreamDestroy(rocjpeg_stream_handle);
    rocJpegDestroy(handle);
    return EXIT_FAILURE;
  }


``rocJpegGetImageInfo()`` is thread safe.

.. note::

  The VCN hardware-accelerated JPEG decoder in AMD GPUs only supports decoding JPEG images with ``ROCJPEG_CSS_444``, ``ROCJPEG_CSS_440``, ``ROCJPEG_CSS_422``, ``ROCJPEG_CSS_420``, and ``ROCJPEG_CSS_400`` chroma subsampling.
