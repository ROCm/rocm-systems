.. meta::
  :description: The rocDecode hardware decoder 
  :keywords: decode, video decoder, video decoding, rocDecode, core APIs, AMD, ROCm

********************************************************************
The rocDecode hardware decoder API
********************************************************************

The rocDecode hardware decoder API exposed in |rocdecode|_ is used to decode frames that were parsed by :doc:`the rocDecode parser <./rocDecode-parser>`.

Parsing parameters are stored in the ``RocDecoderCreateInfo`` struct and passed to ``rocDecCreateDecoder()`` to create a new decoder. ``rocDecCreateDecoder()`` returns a handle to the decoder. For example:

.. code:: cpp

  RocDecoderCreateInfo create_info = {};
  create_info.codec_type = dec_info.rocdec_codec_id;     // user specified codec_type for raw files
  create_info.max_width = DEFAULT_WIDTH;
  create_info.max_height = DEFAULT_HEIGHT;
  create_info.width = DEFAULT_WIDTH;
  create_info.height = DEFAULT_HEIGHT;
  create_info.num_decode_surfaces = 6;
  create_info.num_output_surfaces = 1;
  rocDecCreateDecoder(&decoder_handle, &create_info);


``rocDecGetDecoderCaps()`` queries the capabilities of the underlying hardware video decoder. Decoder capabilities usually include supported codecs, maximum resolution, and
bit depth.

``rocDecDecodeFrame()`` is used to submit frames for hardware decoding. This function must be called when the ``pfn_decode_picture`` callback is triggered in the ``rocDecParseVideoData()`` call. See :doc:`The rocDecode parser API <./rocDecode-parser>` for details about this call.

``rocDecDecodeFrame()`` takes the decoder handle and the pointer to the ``RocdecPicParams()`` struct and initiates the video decoding using VA-API. ``RocdecPicParams`` is populated with the decoded frame information.

The ``pfn_sequence_callback`` callback is triggered when a format change occurs or when a new sequence header is encountered. The implementation of ``pfn_sequence_callback`` must call ``rocDecReconfigureDecoder()`` to reconfigure the decoder to handle the new sequence or format. See :doc:`The rocDecode parser API <./rocDecode-parser>` for details about this callback.

``rocDecGetDecodeStatus()`` can be called to query the decoding status of a frame. The result of the query is either ``rocDecodeStatus_Success``, if decoding is complete, or ``rocDecodeStatus_InProgress``, if decoding is still in progress.

The ``pfn_display_picture`` callback is triggered when a frame has been decoded. The decoded frame can then be further processed in device memory. The implementation for this callback must call ``rocDecGetVideoFrame()`` to obtain the decoded frame's HIP device pointer.

``rocDecGetVideoFrame()`` provides a way to access the decoded frame in HIP. This is a blocking call that only returns once frame decoding and memory mapping is complete. It returns the HIP device pointer as well as information about the :doc:`output surface type <../conceptual/rocDecode-memory-types>`. 

If the output surface type is ``OUT_SURFACE_MEM_DEV_INTERNAL``, meaning intermediate GPU memory, the direct pointer to the decoded surface is provided. If the requested surface
type is ``OUT_SURFACE_MEM_DEV_COPIED`` or ``OUT_SURFACE_MEM_HOST_COPIED``, the internal decoded frame is copied to another buffer, either in device memory or host memory. 

Once decoding is complete, ``rocDecDestroyVideoParser()`` and ``rocDecDestroyDecoder()`` must be called to destroy the parser and the decoding session, and free resources.

.. |apifolder| replace:: ``api/rocdecode``
.. _apifolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode

.. |rocparser| replace:: ``api/rocdecode/rocparser.h``
.. _rocparser: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocparser.h

.. |rocdecode| replace:: ``api/rocDecode/rocdecode.h``
.. _rocdecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocdecode.h

.. |rocdecodehost| replace:: ``api/rocDecode/rocdecode_host.h``
.. _rocdecodehost: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocdecode_host.h

.. |bitstreamreader| replace:: ``api/rocDecode/roc_bitstream_reader.h``
.. _bitstreamreader: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/roc_bitstream_reader.h

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils
