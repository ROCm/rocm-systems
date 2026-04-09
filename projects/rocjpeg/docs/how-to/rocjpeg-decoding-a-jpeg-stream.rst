.. meta::
  :description: decoding a jpeg stream with rocJPEG
  :keywords: rocJPEG, ROCm, API, documentation, decoding, jpeg


********************************************************************
Decoding a JPEG stream with rocJPEG
********************************************************************

rocJPEG provides two functions, ``rocJpegDecode()`` and ``rocJpegDecodeBatched()``, for decoding JPEG image. 

.. code:: cpp

  RocJpegStatus rocJpegDecode(
    RocJpegHandle handle,
    RocJpegStreamHandle jpeg_stream_handle,
    const RocJpegDecodeParams *decode_params,
    RocJpegImage *destination);

  RocJpegStatus rocJpegDecodeBatched(
    RocJpegHandle handle,
    RocJpegStreamHandle *jpeg_stream_handles,
    int batch_size,
    const RocJpegDecodeParams *decode_params,
    RocJpegImage *destinations);

``rocJpegDecode()`` is used for decoding single images and ``rocJpegDecodeBatched()`` is used for decoding batches of JPEG images. ``rocJpegDecode()`` and ``rocJpegDecodeBatched()`` copy decoded images to a ``RocJpegImage`` struct.

.. code:: cpp

    typedef struct {
      uint8_t* channel[ROCJPEG_MAX_COMPONENT];
      uint32_t pitch[ROCJPEG_MAX_COMPONENT];
    } RocJpegImage;

``rocJpegDecodeBatched()`` behaves the same way as ``rocJpegDecode()`` except that ``rocJpegDecodeBatched()`` takes an array of stream handles and an array of decode parameters as input, decodes the batch of JPEG images, and stores the decoded images in an output array of destination images. 

``rocJpegDecodeBatched()`` is suited for use on ASICs with multiple JPEG cores and is more efficient than multiple calls to ``rocJpegDecode()``. Choosing a batch size that is a multiple of available JPEG cores is recommended. 

Memory has to be allocate to each channel of ``RocJpegImage``, including every channel of every ``RocJpegImage`` in the destination image array passed to ``rocJpegDecodeBatched()``. Use |hipmalloc|_ to allocate memory.

.. |hipmalloc| replace:: ``hipMalloc()``
.. _hipmalloc: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/virtual_memory.html

For example:

.. code:: cpp

  // Allocate device memory for the decoded output image
  RocJpegImage output_image = {};
  RocJpegDecodeParams decode_params = {};
  decode_params.output_format = ROCJPEG_OUTPUT_NATIVE;

  // For this sample assuming the input image has a YUV420 chroma subsampling.
  // For YUV420 subsampling, the native decoded output image would be NV12 (i.e., the rocJPegDecode API copies Y to first channel and UV (interleaved) to second channel of RocJpegImage)
  output_image.pitch[1] = output_image.pitch[0] = widths[0];
  hipError_t hip_status;
  hip_status = hipMalloc(&output_image.channel[0], output_image.pitch[0] * heights[0]);
  if (hip_status != hipSuccess) {
    std::cerr << "Failed to allocate device memory for the first channel" << std::endl;
    rocJpegStreamDestroy(rocjpeg_stream_handle);
    rocJpegDestroy(handle);
    return EXIT_FAILURE;
  }

  hip_status = hipMalloc(&output_image.channel[1], output_image.pitch[1] * (heights[0] >> 1));
  if (hip_status != hipSuccess) {
    std::cerr << "Failed to allocate device memory for the second channel" << std::endl;
    hipFree((void *)output_image.channel[0]);
    rocJpegStreamDestroy(rocjpeg_stream_handle);
    rocJpegDestroy(handle);
    return EXIT_FAILURE;
  }

  // Decode the JPEG stream
  status = rocJpegDecode(handle, rocjpeg_stream_handle, &decode_params, &output_image);
  if (status != ROCJPEG_STATUS_SUCCESS) {
    std::cerr << "Failed to decode JPEG stream with error code: " << rocJpegGetErrorName(status) << std::endl;
    hipFree((void *)output_image.channel[0]);
    hipFree((void *)output_image.channel[1]);
    rocJpegStreamDestroy(rocjpeg_stream_handle);
    rocJpegDestroy(handle);
    return EXIT_FAILURE;
  }


The behaviors of ``rocJpegDecode()`` and ``rocJpegDecodeBatched()`` depend on ``RocJpegOutputFormat`` and ``RocJpegDecodeParms``. 

``RocJpegOutputFormat`` specifies the output format to be used to decode the JPEG image. It can be set to any one of these output formats:

.. csv-table::
  :header: "Output format", "Meaning"

  "ROCJPEG_OUTPUT_NATIVE", "Return native unchanged decoded YUV image from the VCN JPEG deocder."
  "ROCJPEG_OUTPUT_YUV_PLANAR", "Return in the YUV planar format."
  "ROCJPEG_OUTPUT_Y", "Return the Y component only."
  "ROCJPEG_OUTPUT_RGB", "Convert to interleaved RGB."
  "ROCJPEG_OUTPUT_RGB_PLANAR", "Convert to planar RGB."

``RocJpegOutputFormat`` is a member of the ``RocJpegDecodeParams`` struct. ``RocJpegDecodeParams`` defines the output format, crop rectangle, and target dimensions to use when decoding the image.

.. code:: cpp

  typedef struct {
    RocJpegOutputFormat output_format; /**< Output data format. See RocJpegOutputFormat for description. */
    struct {
        int16_t left; /**< Left coordinate of the crop rectangle. */
        int16_t top; /**< Top coordinate of the crop rectangle. */
        int16_t right; /**< Right coordinate of the crop rectangle. */
        int16_t bottom; /**< Bottom coordinate of the crop rectangle. */
    } crop_rectangle; /**< Defines the region of interest (ROI) to be copied into the RocJpegImage output buffers. */
    struct {
        uint32_t width; /**< Target width of the picture to be resized. */
        uint32_t height; /**< Target height of the picture to be resized. */
    } target_dimension; /**< (future use) Defines the target width and height of the picture to be resized. Both should be even.
                            If specified, allocate the RocJpegImage buffers based on these dimensions. */
  } RocJpegDecodeParams;


For example, consider a situation where ``RocJpegOutputFormat`` is set to ``ROCJPEG_OUTPUT_NATIVE``. Based on the chroma subsampling of the input image, ``rocJpegDecode()`` does one of the following:

* For ``ROCJPEG_CSS_444`` and ``ROCJPEG_CSS_440``: writes Y, U, and V to the first, second, and third channels of ``RocJpegImage``.
* For ``ROCJPEG_CSS_422``: writes YUYV (packed) to the first channel of ``RocJpegImage``.
* For ``ROCJPEG_CSS_420``: writes Y to the first channel and UV (interleaved) to the second channel of ``RocJpegImage``.
* For ``ROCJPEG_CSS_400``: writes Y to the first channel of ``RocJpegImage``.

If ``RocJpegOutputFormat`` is set to ``ROCJPEG_OUTPUT_Y`` or   ``ROCJPEG_OUTPUT_RGB``, then ``rocJpegDecode()`` copies the output to the first channel of ``RocJpegImage``.

If ``RocJpegOutputFormat`` is set to ``ROCJPEG_OUTPUT_YUV_PLANAR`` or ``ROCJPEG_OUTPUT_RGB_PLANAR``, the data is written to the corresponding channels of the ``RocJpegImage`` destination structure.

The destination images must be large enough to store the output.

Use |rocjpegimageinfo|_ to extract information and calculate the required memory sizes for the destination image following these guidelines:.

.. |rocjpegimageinfo| replace:: ``rocJpegGetImageInfo()``
.. _rocjpegimageinfo: ./rocjpeg-retrieve-image-info.html

.. csv-table::
  :header: "Output format", "Chroma subsampling", "Minimum size of destination.pitch[c]", "Minimum size of destination.channel[c]"

  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_444", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[0] for c = 0, 1, 2"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_440", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[0] = destination.pitch[0] * heights[0], destination.channel[c] = destination.pitch[c] * heights[0] / 2 for c = 1, 2"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_422", "destination.pitch[0] = widths[0] * 2", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_420", "destination.pitch[1] = destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0], destination.channel[1] = destination.pitch[1] * (heights[0] >> 1)"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_400", "destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_YUV_PLANAR", "ROCJPEG_CSS_444, ROCJPEG_CSS_440, ROCJPEG_CSS_422, ROCJPEG_CSS_420", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[c] for c = 0, 1, 2"
  "ROCJPEG_OUTPUT_YUV_PLANAR", "ROCJPEG_CSS_400", "destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_Y", "Any of the supported chroma subsampling", "destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_RGB", "Any of the supported chroma subsampling", "destination.pitch[0] = widths[0] * 3", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_RGB_PLANAR", "Any of the supported chroma subsampling", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[c] for c = 0, 1, 2"

