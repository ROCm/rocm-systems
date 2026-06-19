.. meta::
  :description: The rocDecode RocVideoDecoder utility class
  :keywords: parse video, parse, decode, video decoder, video decoding, rocDecode, RocVideoDecoder, core APIs, AMD, ROCm, utility class

**********************************************
The rocDecode RocVideoDecoder utility class
**********************************************

The `RocVideoDecoder class <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/rocvideodecode/roc_video_dec.h>`_ provides high-level calls to :doc:`the core APIs <../reference/rocDecode-core-APIs>`.

The RocVideoDecoder class includes get functions that return information about the input frame, the output frame, the codec, the decoder, and the surface attributes. 

* ``GetCodecId()``: returns the codec ID.
* ``GetWidth()``: returns the output frame width
* ``GetDecodeWidth()``: returns the decode width
* ``GetHeight()``: returns the output frame height
* ``GetChromaHeight()``: returns the chroma height
* ``GetNumChromaPlanes()``: returns the number of chroma planes
* ``GetFrameSize()``: returns the frame size based on pixel format
* ``GetBitDepth()``: returns the bit depth
* ``GetBytePerPixel()``: returns the bytes per pixel
* ``GetSurfaceSize()``: returns the surface size
* ``GetSurfaceStride()``: returns the surface stride
* ``GetCodecFmtName()``: returns the name of the output codec format
* ``GetSurfaceFmtName()``: returns the name of the surface format 
* ``GetDeviceinfo()``: gets device information for the current device.
* ``GetOutputSurfaceInfo()``: gets the pointer to the output surface information

Utility functions are also provided:

* ``CodecSupported()`` returns whether the codec is supported by the current device
* ``AddDecoderSessionOverHead()`` adds to decoder initialization and deinitialization time to the total time for the decoding session
* ``GetDecoderSessionOverHead()`` returns the total decoder initialization and deinitialization time for the decoding session

The ``RocVideoDecoder`` constructor instantiates a parser and a decoder, and initializes HIP on the device.

.. code:: cpp

  RocVideoDecoder(int device_id,  OutputSurfaceMemoryType out_mem_type, rocDecVideoCodec codec, bool force_zero_latency = false,
                  const Rect *p_crop_rect = nullptr, bool extract_user_SEI_Message = false, uint32_t disp_delay = 0, int max_width = 0, 
                  int max_height = 0, uint32_t clk_rate = 1000, bool skip_init = false);

``DecodeFrame()`` decodes a frame and returns the number of frames available for display.

``GetFrame()`` returns a decoded frame and its timestamp.

``ReleaseFrame`` releases the frame after processing is complete. It can only be used with ``OUT_SURFACE_MEM_DEV_INTERNAL``.

``DecodeFrame()``, ``GetFrame()``, and ``ReleaseFrame()`` should be called in a loop that fetches the available frames for decoding.

The decoder instance is reused when there's a video resolution change, bit depth change, display size change, or decode buffer pool size change, without a change in the codec.

The decoder maintains a decode buffer pool of images that haven't yet been displayed or processed. When the the decoder is reconfigured, the current decode buffer pool is deleted and a new decode buffer pool is created. 

``WaitForDecodeCompletion()`` waits for the last submitted picture to be decoded.

``ReconfigureDecoder()`` reconfigures the decoder.

``SetReconfigParams()`` sets the reconfiguration parameters.

``GetNumOfFlushedFrames()`` returns the number of frames that were flushed during the reconfiguration. 

``SaveFrameToFile()`` saves a flushed frame to a file.

For more information about using the utility decoder, see :doc:`Using the rocDecode RocVideoDecoder <../how-to/using-rocDecode-video-decoder>` and :doc:`Understanding the rocDecode videodecode sample
<../how-to/using-rocDecode-videodecode-sample>`.

.. |ffmpeg| replace:: ``utils/ffmpegvideodecode/ffmpeg_video_dec.h``
.. _ffmpeg: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/ffmpegvideodecode/ffmpeg_video_dec.h

.. |rocdecode| replace:: ``utils/rocvideodecode/roc_video_dec.h``
.. _rocdecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/rocvideodecode/roc_video_dec.h

.. |apifolder| replace:: ``api`` folder
.. _apifolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api
