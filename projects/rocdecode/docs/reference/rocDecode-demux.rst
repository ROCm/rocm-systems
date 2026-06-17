.. meta::
  :description: The rocDecode VideoDemuxer utility class
  :keywords: video demuxing, video demultiplexing, rocDecode, core APIs, AMD, ROCm, utility class

********************************************************************
The rocDecode VideoDemuxer utility class
********************************************************************

The rocDecode FFMpeg demuxer utility class, `VideoDemuxer <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/video_demuxer.h>`_, is used to demultiplex (demux) frames that can then be consumed by the :doc:`FFMpeg decoder <./rocDecode-ffmpeg-decoder>` or the :doc:`hardware decoder <./rocDecode-hw-decoder>`. 

The ``VideoDemuxer()`` constructor takes either a file path or a StreamProvider object. The StreamProvider class is an abstract class that feeds a stream of packets to the demuxer. The StreamProvider ``GetData()`` and ``GetBufferSize()`` functions must be implemented to use a StreamProvider.

The VideoDemuxer ``Demux()`` function extracts the next video packet from the stream, and returns a pointer to the packet data, the packet size, and optionally the presentation timestamp. 

``Demux()`` demuxes frames sequentially starting at the beginning of the stream. The ``seek()`` function is used to start demuxing from a different frame.

A ``VideoSeekContext`` is passed to ``seek()``. The seek context specifies a seek criteria and a seek mode. The seek criteria describe whether the demuxer needs to seek to a specific frame or seek to a specific timestamp. The seek mode indicates whether the demuxer should seek to the exact frame or to the previous keyframe. 

.. note:: 

  The FFmpeg development libraries must be installed to use the FFMpeg demuxer:

  ``sudo apt install libavcodec-dev libavformat-dev libavutil-dev``
