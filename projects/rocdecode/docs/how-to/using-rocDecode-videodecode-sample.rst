.. meta::
  :description: Understanding the rocDecode videodecode sample
  :keywords: parse video, parse, decode, video decoder, video decoding, rocDecode, AMD, ROCm, sample, walkthrough

********************************************************************
Understanding the rocDecode videodecode sample
********************************************************************

The |videodecode|_ sample in the rocDecode GitHub repository |samplefolder|_ demonstrates how to decode a video stream.

As with the other rocDecode samples, ``videodecode.cpp`` uses the utility classes in the rocDecode repository's |utilsfolder|_.

rocDecode provides two ways to decode a video stream: using the rocDecode RocVideoDecoder on GPU or using the FFMpeg video decoder on CPU.

The ``videodecode.cpp`` sample lets the user choose which method to use through the ``--backend`` argument.

``videodecode.cpp`` takes the following arguments:

.. list-table:: 
    :widths: 10 30 60 
    :header-rows: 1

    * - Argument      
      - Description 
      - Note

    * - ``-i``
      - Input file path
      - Required. The path to the input video stream.
    
    * - ``-o`` 
      - Output file path
      - Optional. The file to which to write the decoded frames, including those that remain in the decoded frame buffer pool when the RocVideoDecoder is being reconfigured.

    * - ``-d`` 
      - GPU device ID 
      - Optional. Set it to 0 for the first device, 1 for the second device, 2 for the third device, and so on for each subsequent device. Set to 0 by default.
  
    * - ``-backend`` 
      - The backend to use for decoding
      - Optional. Set it to 0 to use RocVideoDecode on GPU, 1 to use the FFMpeg decoder on CPU, or 2 to use the FFMpeg decoder with no multithreading on CPU. Uses RocVideoDecode on GPU by default.

    * - ``-f`` 
      - Number of frames to decode 
      - Optional. Decodes the entire stream by default.

    * - ``-z`` 
      - Force zero latency 
      - Optional. When set to ``true`` forces decoded frames to be flushed out for display immediately. ``false`` by default.

    * - ``-disp_delay`` 
      - Display delay
      - Optional. The number of frames to decode before displaying the results. Set to 1 by default.
      
    * - ``-sei`` 
      - Extract Supplemental Enhancement Information (SEI)
      - Optional. Set to ``true`` to extract SEI. ``false`` by default.

    * - ``-md5`` 
      - Generate MD5 message digest
      - Optional. Set to ``true`` to generate the MD5 message digest for the decoded YUV image sequence. ``false`` by default.

    * - ``-md5_check``
      - Compare the generated MD5 with a provided MD5 string
      - Optional. When a file containing an MD5 string is passed to this argument, the MD5 message is compared to the string in the file.

    * - ``-crop`` 
      - Crop rectangle 
      - Optional. Takes four integers defining the crop rectangle to use with the output. This argument is ignored when using interopped decoded frame. See the documentation for the `Rect struct <https://rocm.docs.amd.com/projects/rocDecode/en/latest/doxygen/html/structRect.html>`_ for more information. There is no cropping by default.

    * - ``-m`` 
      - The output surface memory type
      - Optional. The memory type where the surface data, such as the decoded frames, resides. Set this to 0 for intermediate GPU memory, to 1 for GPU memory, and to 2 for CPU memory. See :doc:`Surface data memory locations <../conceptual/rocDecode-memory-types>` for more information. Uses intermediate GPU memory by default. 

    * - ``-seek_criteria`` 
      - Seek criteria and seek starting point
      - Optional. Set to 1 and the frame number to start demultiplexing from that specific frame. Set to 2 and the timestamp to start demultiplexing from that specific timestamp. The seek criteria and starting point must be comma-separated (``,``). Demultiplexing begins at the first frame by default. 

    * - ``-seek_mode``
      - Seek mode 
      - Optional. Set to 0 to seek to the previous keyframe. Set to 1 to seek to the exact frame. Seeks to previous keyframe by default.
    
    * - ``-no_ffmpeg_demux`` 
      - Don't use the FFMpeg demultiplexer
      - Optional. Set to ``true`` to use the RocDecode bitstream reader to obtain picture data. The bitstream reader can only be used with an elementary stream. The FFmpeg demultiplexer is used by default.

Because the ``videodecode.cpp`` example can use the RocDecode RocVideoDecoder, the FFMpeg decoder, the FFmpeg demultiplexer (demuxer), or the RocDecode bitstream reader, it imports the ``roc_video_dec.h``, ``video_demuxer.h``, and ``ffmpeg_video_dec.h`` header files. These headers contain the convenience classes and functions for decoding and demultiplexing video.


The FFMpeg demuxer is used to demultiplex the input stream unless the ``-no_ffmpeg_demux`` argument was set to ``true``.

.. code:: C++

  VideoDemuxer *demuxer;
  demuxer = new VideoDemuxer(input_file_path.c_str());

The ``GetCodecId`` and ``GetBitDepth`` functions are used to obtain the video stream's codec and bit depth. The ``AVCodec2RocDecVideoCodec`` utility function converts the codec returned from the demuxer to its corresponding ``rocDecVideoCodec_enum`` value.

.. code:: C++

  rocdec_codec_id = AVCodec2RocDecVideoCodec(demuxer->GetCodecID());
  bit_depth = demuxer->GetBitDepth();

The codec ID and bit depth are used to instantiate the video decoder. If the GPU backend was selected, the RocVideoDecoder is instantiated:

.. code:: C++

  RocVideoDecoder *viddec;
  viddec = new RocVideoDecoder(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay);

For more information about the rocDecode RocVideoDecoder, see :doc:`Using the rocDecode RocVideoDecoder <./using-rocDecode-video-decoder>`. 

If the CPU backend was selected, the FFMpeg decoder is instantiated:

.. code:: C++

  viddec = new FFMpegVideoDecoder(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay);

The decoder instance is reused when there is a change to the video resolution without a change in the codec. When the video stream resolution changes, the decoder is reconfigured for the new resolution and the pool of frame buffers that the decoder maintains is deleted.

The |reconfig_struct|_ struct is used to store information on how to handle the frames that remain in the buffers at the time of  reconfiguration. A callback, a user-defined flush mode, and a user-defined struct are passed to ``ReconfigParams_t``. The reconfiguration parameters are then passed to the decoder using ``SetReconfigParams``.

The reconfiguration structs are defined in |common|_ in the rocDecode samples. Three possibilities for the remaining frames in the decoded frame buffer pool are provided:

* ``RECONFIG_FLUSH_MODE_NONE``: delete the frames along with the buffers.
* ``RECONFIG_FLUSH_MODE_DUMP_TO_FILE``: write the frames to the specified output file before deleting the buffers.
* ``RECONFIG_FLUSH_MODE_CALCULATE_MD5``: calculate the MD5 of the frames before deleting the buffers.

.. code:: C++

  typedef enum ReconfigFlushMode_enum {
    RECONFIG_FLUSH_MODE_NONE = 0x0,                      /**<  Just flush to get the frame count */
    RECONFIG_FLUSH_MODE_DUMP_TO_FILE = 0x1,              /**<  The remaining frames will be dumped to file in this mode */
    RECONFIG_FLUSH_MODE_CALCULATE_MD5 = (0x1 << 1),      /**<  Calculate the MD5 of the flushed frames */
  } ReconfigFlushMode;

  typedef struct ReconfigDumpFileStruct_t {
    bool b_dump_frames_to_file;
    std::string output_file_name;
    void *md5_generator_handle;
  } ReconfigDumpFileStruct;


If the ``-o`` output file path argument was set, the remaining frames in the decoded frame buffer pool will be written to the output file upon reconfiguration. If the ``-md5`` argument was set to ``true``, the MD5 of the frames in the decoded frame buffer pool will be calculated before they're flushed or written to file. If neither option was selected, the frames in the decoded frame buffer pool will be deleted along with the buffers without being saved or processed. 

.. code:: C++

  reconfig_params.p_fn_reconfigure_flush = ReconfigureFlushCallback;
  reconfig_user_struct.b_dump_frames_to_file = dump_output_frames;
  reconfig_user_struct.output_file_name = output_file_path;
  reconfig_params.reconfig_flush_mode = RECONFIG_FLUSH_MODE_NONE;
  if (dump_output_frames) {
      reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_DUMP_TO_FILE;
  }
  if (b_generate_md5) {
      reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_CALCULATE_MD5;
  }
  reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;

The reconfiguration parameters need to be defined prior to entering the decoding loop.

In the decode loop, the video stream is demultiplexed before being decoded. 

The demuxer will demultiplex frames sequentially starting at the beginning of the stream unless ``-seek_criteria`` was set to either 1 or 2. 

If the ``-seek_criteria`` argument was set to 1 and ``-seek_mode`` was set to 1, the demuxer will start demultiplexing the video at the frame provided. 

If the ``-seek_criteria`` argument was set to 1 and ``-seek_mode`` wasn't set or was set to 0, the demuxer will start demultiplexing the video at the first keyframe before the frame provided.

If the ``-seek_criteria`` argument was set to 2 the demuxer will start demultiplexing the video at the timestamp provided.

The seek criteria is defined by the ``SeekCriteriaEnum`` enum and the seek mode is defined by the ``SeekModeEnum`` enum. Both the ``SeekCriteriaEnum`` and the ``SeekModeEnum`` are defined in ``video_demuxer.h``.

From ``videodecode.cpp``:

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

The video can now be decoded using the ``DecodeFrame`` function.

If the ``-md5`` argument was set to ``true``, MD5 is calculated for the file. If an output file path was provided, the decoded frames will be written to file. 

The frame is released with ``ReleaseFrame`` once processing is complete.

.. code:: C++

  n_frame_returned = viddec->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts, &decoded_pics);

  [...]

  for (int i = 0; i < n_frame_returned; i++) {
    pframe = viddec->GetFrame(&pts);
    if (b_generate_md5) {
      md5_generator->UpdateMd5ForFrame(pframe, surf_info);
    }
    if (dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED) {
      viddec->SaveFrameToFile(output_file_path, pframe, surf_info);
    }
  
  viddec->ReleaseFrame(pts);


The demuxer is deleted once decoding is done.

.. code:: C++

  delete demuxer;

.. |videodecode| replace:: ``videodecode.cpp``
.. _videodecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples/videoDecode/videodecode.cpp

.. |videodecoderaw| replace:: ``videodecoderaw.cpp``
.. _videodecoderaw: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples/videoDecodeRaw

.. |common| replace:: ``common.h``
.. _common: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples/common.h

.. |apifolder| replace:: ``api`` folder
.. _apifolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils

.. |samplefolder| replace:: ``samples`` folder
.. _samplefolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples

.. |reconfig_struct| replace:: ``ReconfigParams_t``
.. _reconfig_struct: https://rocm.docs.amd.com/projects/rocDecode/en/latest/doxygen/html/structReconfigParams__t.html

.. |br| raw:: html

      </br>   