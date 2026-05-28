.. meta::
  :description: Using rocDecode with the FFMpeg demultiplexer
  :keywords: parse video, parse, rocDecode, AMD, ROCm, FFmpeg demuxer

********************************************************************
Using the rocDecode FFmpeg demultiplexer
********************************************************************

The rocDecode FFmpeg demultiplexer (demuxer) extracts coded picture data from digital media files.

To use the rocDecode FFmpeg demuxer , import the ``video_demuxer.h`` header file. 

.. code:: C++ 

  #include "video_demuxer.h"

Instantiate a ``VideoDemuxer`` with the path to the video file. The ``GetCodecId`` and ``GetBitDepth`` functions can be used to obtain the video stream's codec ID and bit depth. The ``AVCodec2RocDecVideoCodec`` utility function converts the codec ID returned from the demuxer to its corresponding ``rocDecVideoCodec_enum`` value.

.. code:: C++

  VideoDemuxer *demuxer;
  demuxer = new VideoDemuxer(input_file_path.c_str());
  rocdec_codec_id = AVCodec2RocDecVideoCodec(demuxer->GetCodecID());
  bit_depth = demuxer->GetBitDepth();

Call ``Demux`` to extract frame data from the stream:

.. code:: C++

  demuxer->Demux(&pvideo, &n_video_bytes, &pts);
  
The demuxer will demultiplex frames sequentially starting at the beginning of the stream. To start the demultiplexing and decoding process from a different frame, create a seek context that specifies a seek criteria and a seek mode. 

The seek criteria describes whether the demuxer needs to seek to a specific frame or seek to a specific timestamp. The seek mode indicates whether the demuxer should seek to the exact frame or to the previous keyframe. 

The seek criteria is defined by the ``SeekCriteriaEnum`` enum and the seek mode is defined by the ``SeekModeEnum`` enum. Both the ``SeekCriteriaEnum`` and the ``SeekModeEnum`` are defined in ``video_demuxer.h``.

Set the seek criteria to ``SEEK_CRITERIA_FRAME_NUM`` to seek to a frame or to ``SEEK_CRITERIA_TIME_STAMP`` to seek to a timestamp. Set the seek mode to ``SEEK_MODE_EXACT_FRAME`` to seek to the exact frame or to ``SEEK_MODE_PREV_KEY_FRAME`` to seek to the previous keyframe.

From |videodecode|_:

.. code:: C++

  VideoSeekContext video_seek_ctx;
  [...]
  do {
    [...] 
    if (seek_criteria == 1 && first_frame) {
      // use VideoSeekContext class to seek to given frame number
      video_seek_ctx.seek_frame_ = seek_to_frame;
      video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;            
      video_seek_ctx.seek_mode_ = (seek_mode ? SEEK_MODE_EXACT_FRAME : SEEK_MODE_PREV_KEY_FRAME);
      demuxer->Seek(video_seek_ctx, &pvideo, &n_video_bytes);
      pts = video_seek_ctx.out_frame_pts_;
      std::cout << "info: Number of frames that were decoded during seek - " << video_seek_ctx.num_frames_decoded_ << std::endl;
      first_frame = false;
    } else if (seek_criteria == 2 && first_frame) {
      // use VideoSeekContext class to seek to given timestamp
      video_seek_ctx.seek_frame_ = seek_to_frame;
      video_seek_ctx.seek_crit_ = SEEK_CRITERIA_TIME_STAMP;
      video_seek_ctx.seek_mode_ = (seek_mode ? SEEK_MODE_EXACT_FRAME : SEEK_MODE_PREV_KEY_FRAME);
      demuxer->Seek(video_seek_ctx, &pvideo, &n_video_bytes);
      pts = video_seek_ctx.out_frame_pts_;
      std::cout << "info: Duration of frame found after seek - " << video_seek_ctx.out_frame_duration_ << " ms" << std::endl;
      first_frame = false;
    } else {
      demuxer->Demux(&pvideo, &n_video_bytes, &pts);
    }    
    [...]
  } while (n_video_bytes);

Delete the demuxer once demultiplexing is complete.

.. code:: C++

  delete demuxer;

.. |videodecode| replace:: ``videodecode.cpp``
.. _videodecode: https://github.com/ROCm/rocDecode/tree/develop/samples/videoDecode/videodecode.cpp

.. |videodecoderaw| replace:: ``videodecoderaw.cpp``
.. _videodecoderaw: https://github.com/ROCm/rocDecode/tree/develop/samples/videoDecodeRaw

.. |common| replace:: ``common.h``
.. _common: https://github.com/ROCm/rocDecode/blob/develop/samples/common.h

.. |apifolder| replace:: ``api`` folder
.. _apifolder: https://github.com/ROCm/rocDecode/tree/develop/api

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocDecode/tree/develop/utils


.. |reconfig_struct| replace:: ``ReconfigParams_t``
.. _reconfig_struct: https://rocm.docs.amd.com/projects/rocDecode/en/latest/doxygen/html/structReconfigParams__t.html

