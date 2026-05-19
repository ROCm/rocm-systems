.. meta::
  :description: The rocDecode parser API
  :keywords: parse video, parser, decode, video decoder, video decoding, rocDecode, core APIs, AMD, ROCm

********************************************************************
The rocDecode parser API
********************************************************************

The rocDecode parser API, exposed in |rocparser|_, is used to decode bitstreams and organize them in a structured format that can be consumed by the hardware decoder.

The parser parameters are stored in the ``RocdecParserParams`` struct and passed to ``rocDecCreateVideoParser()`` to create a new parser. ``rocDecCreateVideoParser()`` returns a handle to the parser. For example:

.. code:: cpp

    RocdecParserParams params = {};
    params.codec_type = rocdec_codec_id;
    params.max_num_decode_surfaces = 6;
    params.max_display_delay = 1;
    params.user_data = &dec_info;
    rocDecCreateVideoParser(&parser_handle, &params);

Elementary stream video packets extracted from the demultiplexer (demuxer) are passed to the parser using the ``RocdecSourceDataPacket`` struct. Packet information in ``RocdecSourceDataPacket`` is passed to ``rocDecParseVideoData()``. For example:

.. code:: cpp

    RocdecSourceDataPacket packet = {};
    packet.payload_size = frames[i].size();
    packet.payload = frames[i].data();
    rocDecParseVideoData(parser_handle, &packet);

Three callbacks must be registered when the parser is used: ``pfn_decode_picture``, ``pfn_sequence_callback``, and ``pfn_display_picture``. These callbacks are triggered in the ``rocDecParseVideoData()`` call.

``pfn_decode_picture`` is triggered when a picture is ready for decoding. Its implementation must call ``rocDecDecodeFrame()`` from the hardware decoder API. 

``pfn_sequence_callback`` is triggered when a new sequence header is encountered or when there's a format change. Its implementation handles reconfiguring the decoder to handle the new frame format. Its implementation must call ``rocDecReconfigureDecoder()`` from the hardware decoder API.

``pfn_display_picture`` is triggered when a frame has been decoded.  Its implementation must call ``rocDecGetVideoFrame()`` from the hardware decoder API. 

A fourth callback, ``pfn_get_sei_msg``, is optional. ``pfn_get_sei_msg`` is triggered when a Supplementation Enhancement Information (SEI) message is parsed and returned to the caller. 

If any of the callbacks return an error, the error is propagated back to the application.

Once the stream is fully decoded, ``rocDecDestroyVideoParser()`` must be called to destroy the parser object and free all allocated resources.



.. |rocparser| replace:: ``rocparser.h``
.. _rocparser: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocparser.h

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils

.. |rocdecdecode| replace:: ``rocdecdecode.cpp``
.. _rocdecdecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples/rocdecDecode/rocdecdecode.cpp
