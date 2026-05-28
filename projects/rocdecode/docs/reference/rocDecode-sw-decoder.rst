.. meta::
  :description: The rocDecode software decoder 
  :keywords: decode, video decoder, video decoding, rocDecode, core APIs, AMD, ROCm

********************************************************************
The rocDecode software decoder API
********************************************************************

The rocDecode software decoder API exposed in |rocdecodehost|_ is used to decode frames that have been demultiplexed (demuxed) by :doc:`the FFmpeg demuxer <../how-to/using-rocDecode-ffmpeg>`.

Decoding parameters are stored in the ``RocDecoderHostCreateInfo`` struct and passed to ``rocDecCreateDecoderHost()`` to create a new software decoder. ``rocDecCreateDecoderHost()`` returns a handle to the decoder. For example:

.. code:: cpp

  RocDecoderHostCreateInfo create_info = {};
  create_info.codec_type = rocdec_codec_id;
  create_info.num_decode_threads = 0;     // default
  create_info.max_width = DEFAULT_WIDTH;
  create_info.max_height = DEFAULT_HEIGHT;
  create_info.chroma_format = rocDecVideoChromaFormat_420;
  create_info.output_format = rocDecVideoSurfaceFormat_P016;
  create_info.bit_depth_minus_8 = 2;
  create_info.num_output_surfaces = 1;
  create_info.user_data = &dec_info;
  rocDecCreateDecoderHost(&dec_info.decoder, &create_info);

``rocDecGetDecoderCapsHost()`` queries the capabilities of the underlying software video decoder. Decoder capabilities usually include supported codecs, maximum resolution, and
bit depth.

``rocDecDecodeFrameHost()`` is used to submit frames for software decoding. ``rocDecDecodeFrameHost()`` takes the decoder handle and the pointer to the ``RocdecPicParamsHost`` struct and initiates the video decoding. ``RocdecPicParamsHost`` is populated with the decoded frame information.

The ``pfn_sequence_callback`` callback is triggered when a format change occurs or when a new sequence header is encountered. This callback must be registered in any application that uses the software decoder and its implementation must call ``rocDecReconfigureDecoderHost()`` to reconfigure the decoder to handle the new sequence or format. 

``rocDecGetDecodeStatusHost()`` can be called to query the decoding status of a frame. The result of the query is either ``rocDecodeStatus_Success``, if decoding is complete, or ``rocDecodeStatus_InProgress``, if decoding is still in progress.

The ``pfn_display_picture`` callback is triggered when a frame has been decoded. This callback must be registered by any application that uses the software decoder and its implementation must call ``rocDecGetVideoFrameHost()``. ``rocDecGetVideoFrameHost()`` returns the decoded frame's host memory pointer. The decoded frame can then be further processed using this pointer.

``rocDecGetVideoFrameHost()`` provides a way to access the decoded frame in host memory. This is a blocking call that only returns once both frame decoding and memory mapping are done. It returns the host memory pointer as well as information about the :doc:`output surface type <../conceptual/rocDecode-memory-types>`. 

If the output surface type is ``OUT_SURFACE_MEM_DEV_INTERNAL``, meaning intermediate GPU memory, the direct pointer to the decoded surface is provided. If the requested surface
type is ``OUT_SURFACE_MEM_DEV_COPIED`` or ``OUT_SURFACE_MEM_HOST_COPIED``, the internal decoded frame is copied to another buffer, either in device memory or host memory. 

Once decoding is complete, ``rocDecDestroyDecoderHost()`` must be called to destroy the decoder and free resources.

.. |apifolder| replace:: ``api/rocdecode``
.. _apifolder: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode

.. |rocparser| replace:: ``api/rocdecode/rocparser.h``
.. _rocparser: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/rocparser.h

.. |rocdecode| replace:: ``api/rocDecode/rocdecode.h``
.. _rocdecode: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/rocdecode.h

.. |rocdecodehost| replace:: ``api/rocDecode/rocdecode_host.h``
.. _rocdecodehost: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/rocdecode_host.h

.. |bitstreamreader| replace:: ``api/rocDecode/roc_bitstream_reader.h``
.. _bitstreamreader: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/roc_bitstream_reader.h

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocDecode/tree/develop/utils
