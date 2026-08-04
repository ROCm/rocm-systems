.. meta::
  :description: rocDecode decode pipelines
  :keywords: rocDecode, video pipeline, demux, bitstream reader, hardware decode, software decode, VCN, FFMpeg

********************************************************************
rocDecode decode pipelines
********************************************************************

The rocDecode video decoder pipeline takes video data, extracts it, parses it, and decodes it.

The FFMpeg demultiplexer (demuxer) extracts a segment of video data and sends it to a decoder. 

.. note::

  The bitstream reader utility class can be used to extract and parse coded picture data from an elementary video stream. It can only be used with elementary video streams and IVF container files.

When hardware decoding is used, the parser extracts information such as picture and slice parameters, and sends it to the hardware decoder to consume. The hardware decoder decodes the frame using the Video Acceleration API (VA-API). 

When FFMpeg-based software decoding is used, no separate parsing step is required and the data that was extracted by the demuxer is sent directly to the decoder to consume.

.. note:: 

  The FFMpeg development libraries must be installed to use the FFMpeg utilities:

  ``sudo apt install libavcodec-dev libavformat-dev libavutil-dev``

