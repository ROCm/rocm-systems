.. meta::
  :description: rocDecode FFMpeg utility decoder
  :keywords: decode, video decoder, video decoding, ffmpeg, rocDecode, utility functions, AMD, ROCm

***********************************************
The rocDecode FFMpeg decoder utility class
***********************************************

The rocDecode FFMpeg decoder utility class, `FFMpegVideoDecoder <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/ffmpegvideodecode/ffmpeg_video_dec.h>`_, is used to decode demultiplexed (demuxed) frames on the CPU. 

.. note:: 

  The ``rocdecode-host`` package must be installed to use FFMpegVideoDecoder.


FFMpegVideoDecoder uses the FFMpeg video decoding backend to decode frame packets output by the FFMpeg demultiplexer (demuxer). It inherits from :doc:`the RocVideoDecode utility class <./rocDecode-util-decoder>` and uses many of the same functions. 

The ``FFMpegVideoDecoder()`` constructor takes the same parameters as ``RocVideoDecoder`` except for the first parameter, which is the number of CPU threads rather than the device ID, and the force zero latency parameter, which isn't supported with the FFMpeg decoder.

Six functions are implemented in FFMpegVideoDecoder: ``DecodeFrame()``, ``GetFrame()``, ``GetOutputSurfaceInfo()``, ``SaveFrameToFile()``, ``ReleaseFrame()``, and ``ReconfigureDecoder()``.

``DecodeFrame()`` submits a frame for host-based decoding using the FFMpeg decoder and returns the number of available decoded frames.

Once decoding is done, ``GetFrame()`` returns the decoded frame and its timestamp and ``GetOutputSurfaceInfo()`` returns a pointer to the metadata associated with the decoded frame.

``SaveFrameToFile()`` saves the decoded output surface to a file. 

Once processing is complete, ``ReleaseFrame()`` is used to release the frame.

``ReconfigureDecoder`` is used to reconfigure the decoder when the video stream resolution changes.

For information about using the FFMpeg decoder, see :doc:`Understanding the rocDecode videodecode sample
<../how-to/using-rocDecode-videodecode-sample>`.

.. |ffmpeg| replace:: ``utils/ffmpegvideodecode/ffmpeg_video_dec.h``
.. _ffmpeg: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/ffmpegvideodecode/ffmpeg_video_dec.h


.. |rocdecode| replace:: ``utils/rocvideodecode/roc_video_dec.h``
.. _rocdecode: ttps://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/rocvideodecode/roc_video_dec.h
