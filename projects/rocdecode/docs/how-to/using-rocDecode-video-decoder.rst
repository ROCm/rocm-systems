.. meta::
  :description: UUsing the rocDecode RocVideoDecoder
  :keywords: parse video, parse, decode, video decoder, video decoding, rocDecode, AMD, ROCm, RocVideoDecoder

********************************************************************
Using the rocDecode RocVideoDecoder
********************************************************************

rocDecode provides two methods fpr decoding a video stream: using the rocDecode RocVideoDecoder on the GPU or using the FFmpeg decoder on the CPU. 


This topic covers how to decode a video stream using the RocVideoDecoder class in |roc_video_dec|_. The RocVideoDecode class provides high-level calls to the core APIs in the |apifolder|_ of the rocDecode GitHub repository. For information about the core APIs, see :doc:`Using the rocDecode core APIs <../reference/rocDecode-core-APIs>`.

The RocVideoDecoder takes a demultiplexed coded picture as input. The picture can be demultiplexed from a video stream using the :doc:`FFmpeg demultiplexer <./using-rocDecode-ffmpeg>`.

To use the rocDecode video decoder, import the ``roc_video_dec.h`` header file and instantiate ``RocVideoDecoder``.

The ``RocVideoDecoder`` constructor takes the following parameters:

.. list-table:: 
    :widths: 15 70 15
    :header-rows: 1

    *   - Parameter
        - Description 
        - Default

    *   - ``device_id``
        - ``int`` |br| |br| The GPU device ID. |br| |br| Set it to 0 for the first device, 1 for the second device, 2 for the third device, and so on for each subsequent device. 
        - 0
    
    *   - ``out_mem_type``
        - |OutputSurfaceMemoryType|_ |br| |br| The memory type where the surface data, such as the decoded frames, resides. |br| |br| 0: ``OUT_SURFACE_MEM_DEV_INTERNAL``. The surface data is stored internally on memory shared by the GPU and CPU. |br| |br| 1: ``OUT_SURFACE_MEM_DEV_COPIED``. The surface data resides on the GPU. |br| |br| 2: ``OUT_SURFACE_MEM_HOST_COPIED``. The surface data resides on the CPU. |br| |br| See :doc:`Surface data memory locations <../conceptual/rocDecode-memory-types>` for more information.
        - 0, OUT_SURFACE_MEM_DEV_INTERNAL

    *   - ``codec``
        - |rocDecVideoCodec|_ |br| |br| The video file's codec ID converted to ``rocDecVideoCodec`` using ``AVCodec2RocDecVideoCodec``.
        - No default, a value must be provided

    *   - ``force_zero_latency``
        - ``bool`` |br| |br| Set to ``true`` to flush decoded frames for immediate display.
        - ``false``

    *   - ``p_crop_rect``
        - ``const Rect *`` |br| |br| The rectangle to use for cropping.
        - No cropping

    *   - ``extract_user_SEI_Message``
        - ``bool`` |br| |br| Set to ``true`` to extract Supplemental Enhancement Information (SEI) from the video stream.
        - ``false``, no SEI will be extracted

    *   - ``disp_delay``
        - ``uint32_t`` |br| |br| Delay the display by this number of frames.
        - 0, no delay in displaying the frames

    *   - ``max_width``
        - ``int`` |br| |br| Max width.
        - 0

    *   - ``max_height``
        - ``int`` |br| |br| Max height. 
        - 0

    *   - ``clk_rate``
        - ``uint32_t`` |br| |br| Clock rate. 
        - 1000


.. |br| raw:: html

      </br>


For example, from |videodecode|_:

.. code:: C++

    RocVideoDecoder viddec(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay);

``RocVideoDecoder`` will create a parser and a decoder, and initialize HIP on the device. 

The same decoder instance is reused when there's a change to the video resolution without a change in the codec. 

The decoder maintains a pool of frame buffers for decoded images that haven't yet been displayed or processed. When the video stream resolution changes, the existing frame buffers in the buffer pool are deleted. The decoder is then reconfigured for the new resolution and new buffers are created.

To prevent the remaining frames in the buffers from being deleted along with the buffers, a callback function can be defined to consume the remaining frames. 

The |reconfig_struct|_ struct stores information on how to handle the reconfiguration. A callback, a user-defined flush mode, and a user-defined struct are passed to ``ReconfigParams_t``. The reconfiguration parameters are then passed to the decoder using ``SetReconfigParams``.

The reconfiguration parameters need to be defined prior to entering the decoding loop. For example, the reconfiguration structs are defined in |common|_ in the rocDecode samples and then used in ``videodecode.cpp``

.. code:: cpp
    
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
    viddec.SetReconfigParams(&reconfig_params);

In the decode loop, the demultiplexed coded picture is passed to ``DecodeFrame``. Once the frame is decoded and processed, it is released with ``ReleaseFrame``. 

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

.. |roc_video_dec| replace:: ``roc_video_dec.h``
.. _roc_video_dec: https://github.com/ROCm/rocDecode/tree/develop/utils/rocvideodecode/roc_video_dec.h

.. |reconfig_struct| replace:: ``ReconfigParams_t``
.. _reconfig_struct: https://rocm.docs.amd.com/projects/rocDecode/en/latest/doxygen/html/structReconfigParams__t.html

.. |OutputSurfaceMemoryType| replace:: ``OutputSurfaceMemoryType``
.. _OutputSurfaceMemoryType: https://rocm.docs.amd.com/projects/rocDecode/en/latest/doxygen/html/roc__video__dec_8h.html

.. |rocDecVideoCodec| replace:: ``rocDecVideoCodec``
.. _rocDecVideoCodec: https://rocm.docs.amd.com/projects/rocDecode/en/latest/doxygen/html/rocdecode_8h.html