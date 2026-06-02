.. meta::
  :description: The rocDecode utility classes
  :keywords: rocDecode, utility classes, AMD, ROCm

********************************************************************
The rocDecode utility classes
********************************************************************

The rocDecode utility classes provide high-level calls to the :doc:`rocDecode core APIs <./rocDecode-core-APIs>`.

The rocDecode utility classes are exposed in header files in the |utilsfolder|_ folder of the `rocDecode project <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode>`_. 

:doc:`The rocDecode decoder utility class  <./rocDecode-util-decoder>` is exposed in |rocdecode|_. It contains functions that create, control, and destroy the decoder, as well as functions that decode the parsed frames on the GPU.

:doc:`The FFMpeg decoder utility class <./rocDecode-ffmpeg-decoder>` is exposed in |ffmpeg|_. It contains the same functionality as ``rocdecode.h``, but all the operations are run on the CPU rather than the GPU.

:doc:`The video demultiplexer utility class <./rocDecode-demux>` is exposed in |demux|_. It contains functions for demultiplexing a video stream. 

.. |apifolder| replace:: ``api/rocdecode``
.. _apifolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode

.. |ffmpeg| replace:: ``utils/ffmpegvideodecode/ffmpeg_video_dec.h``
.. _ffmpeg: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/ffmpegvideodecode/ffmpeg_video_dec.h

.. |rocdecode| replace:: ``utils/rocvideodecode/roc_video_dec.h``
.. _rocdecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/rocvideodecode/roc_video_dec.h


.. |demux| replace:: ``utils/video_demuxer.h``
.. _demux: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils/video_demuxer.h

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils
