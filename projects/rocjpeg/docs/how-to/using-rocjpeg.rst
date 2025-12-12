.. meta::
  :description: Using rocJPEG
  :keywords: parse JPEG, parse, decode, JPEG decoder, JPEG decoding, rocJPEG, AMD, ROCm

********************************************************************
Using rocJPEG
********************************************************************

This document provides a high-level overview of how to do common operations using the rocJPEG APIs exposed in the ``rocjpeg.h`` header file. 

Creating handles 
==================

Handles need to be created to decode and parse JPEG streams.

``rocJpegCreate()`` returns an instance of a ``RocJpegHandle`` based on the specified backend and GPU device ID. The ``RocJpegHandle`` instance must be retained for the entire decode session. 

.. code:: cpp

    RocJpegStatus rocJpegCreate(
      RocJpegBackend backend,
      int device_id,
      RocJpegHandle *handle);


``rocJpegStreamCreate()`` returns a ``rocJpegStreamHandle``, which is a pointer used to represent an instance of a JPEG stream. The instance of ``rocJpegStreamHandle`` is used to parse the JPEG stream and to store the JPEG stream parameters.

.. code:: cpp

    RocJpegStatus rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle);

For example:

.. code:: cpp

  // Read the JPEG image file
  std::ifstream input("mug_420.jpg", std::ios::in | std::ios::binary | std::ios::ate);

  // Get the JPEG image file size
  std::streamsize file_size = input.tellg();
  input.seekg(0, std::ios::beg);

  std::vector<char> file_data;
  // resize if buffer is too small
  if (file_data.size() < file_size) {
    file_data.resize(file_size);
  }
  // Read the JPEG stream
  if (!input.read(file_data.data(), file_size)) {
    std::cerr << "ERROR: cannot read from file: " << std::endl;
    return EXIT_FAILURE;
  }

  // Initialize rocJPEG
  RocJpegHandle handle;
  RocJpegStatus status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, 0, &handle);
  if (status != ROCJPEG_STATUS_SUCCESS) {
    std::cerr << "Failed to create rocJPEG handle with error code: " << rocJpegGetErrorName(status) << std::endl;
    return EXIT_FAILURE;
  }

  // Create a JPEG stream
  RocJpegStreamHandle rocjpeg_stream_handle;
  status = rocJpegStreamCreate(&rocjpeg_stream_handle);
  if (status != ROCJPEG_STATUS_SUCCESS) {
    std::cerr << "Failed to create JPEG stream with error code: " << rocJpegGetErrorName(status) << std::endl;
    rocJpegDestroy(handle);
    return EXIT_FAILURE;
  }

``rocJpegGetErrorName()`` returns error codes in text format from rocJPEG APIs.


Parsing a stream
=================

``rocJpegStreamParse()`` is used to parse a JPEG stream.

The stream data buffer is passed through the ``data`` input parameter. The length of the buffer is passed through the ``length`` input parameter. The parsed stream is returned through the ``jpeg_stream_handle`` provided. ``jpeg_stream_handle`` must have already been created with ``rocJpegStreamCreate()``.

.. code:: cpp

    RocJpegStatus rocJpegStreamParse(const unsigned char *data, 
                                      size_t length, 
                                      RocJpegStreamHandle jpeg_stream_handle);


For example:

.. code:: cpp

  // Parse the JPEG stream
  status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(file_data.data()), file_size, rocjpeg_stream_handle);
  if (status != ROCJPEG_STATUS_SUCCESS) {
    std::cerr << "Failed to parse JPEG stream with error code: " << rocJpegGetErrorName(status) << std::endl;
    rocJpegStreamDestroy(rocjpeg_stream_handle);
    rocJpegDestroy(handle);
    return EXIT_FAILURE;
  }


Getting image information
===========================

``rocJpegGetImageInfo()`` is used to retrieve the number of components, the chroma subsampling, and the width and height of the JPEG image. 

.. code:: cpp

    RocJpegStatus rocJpegGetImageInfo(
      RocJpegHandle handle,
      RocJpegStreamHandle jpeg_stream_handle,
      uint8_t *num_components,
      RocJpegChromaSubsampling *subsampling,
      uint32_t *widths,
      uint32_t *heights);


For more information on ``rocJpegGetImageInfo()``, see `Retrieving image information with rocJPEG <./rocjpeg-retrieve-image-info.html>`_.

Decoding a stream
====================

``rocJpegDecode()`` takes the image passed to it through the ``jpeg_stream_handle`` input parameter and decodes it based on the backend used to create ``handle`` input parameter. 

The ``decode_params`` input parameter is used to specify the decoding parameters. Memory needs to be allocated for each channel of the destination image.

.. code:: cpp

    RocJpegStatus rocJpegDecode(
      RocJpegHandle handle,
      RocJpegStreamHandle jpeg_stream_handle,
      const RocJpegDecodeParams *decode_params,
      RocJpegImage *destination);

For more information on decoding streams, see `Decoding a JPEG stream with rocJPEG <./rocjpeg-decoding-a-jpeg-stream.html>`_.


Destroying handles and freeing resources
==========================================

Once the JPEG stream is decoded, resources need to be freed. 

Use |hipfree|_ to release the memory previously allocated  by ``hipMalloc()`` for each channel of the destination ``rocJpegImage``.

.. |hipfree| replace:: ``hipFree()``
.. _hipfree: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/virtual_memory.html

Use ``rocJpegStreamDestroy()`` to release the ``rocJpegStreamHandle`` and its resources, and use ``rocJPegDestroy()`` to release ``RocJpegHandle`` and destroy the session. 

.. code:: cpp

  RocJpegStatus rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle)

  RocJpegStatus rocJpegDestroy(RocJpegHandle handle)

For example:

.. code:: cpp
  
  hipFree((void *)output_image.channel[0]);
  hipFree((void *)output_image.channel[1]);
  rocJpegStreamDestroy(rocjpeg_stream_handle);
  rocJpegDestroy(handle);