.. meta::
  :description: Using rocDecode
  :keywords: parse video, parse, decode, video decoder, video decoding, rocDecode, core APIs, AMD, ROCm

********************************************************************
The rocDecode core APIs
********************************************************************

The rocDecode core APIs are intended for users who want to have full control of the decoding pipeline and interact with the core components instead of the utility classes. The :doc:`Using the rocDecode videodecode sample <../how-to/using-rocDecode-videodecode-sample>` provides an introduction to using the utility classes.

The rocDecode core APIs are exposed in header files in the |apifolder|_ folder of the `rocDecode GitHub repository <https://github.com/ROCm/rocDecode>`_. 

:doc:`The rocDecode parser API <./rocDecode-parser>` is exposed in |rocparser|_. It contains functions that create and destroy the parser, as well as functions that parse the bitstream.

:doc:`The hardware decoder API <./rocDecode-hw-decoder>` is exposed in |rocdecode|_. It contains functions that create, control, and destroy the decoder, as well as functions that decode the parsed frames on the GPU.

:doc:`The software decoder API <./rocDecode-sw-decoder>` is exposed in |rocdecodehost|_. It contains the same functionality as ``rocdecode.h``, but all the operations are run on the host rather than the GPU.

:doc:`The bitstream reader API <../how-to/using-rocDecode-bitstream>` is exposed in |bitstreamreader|_. It provides an alternative to the FFMpeg demuxer and contains a simple stream file parser that can read elementary files and IVF container files.

.. |apifolder| replace:: ``api/rocdecode``
.. _apifolder: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode

.. |rocparser| replace:: ``api/rocdecode/rocparser.h``
.. _rocparser: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/rocparser.h

.. |rocdecode| replace:: ``api/rocDecode/rocdecode.h``
.. _rocdecode: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/rocdecode.h

.. |rocdecodehost| replace:: ``api/rocDecode/rocdecode_host.h``
.. _rocdecodehost: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/rocdecode_host.h

.. |bitstreamreader| replace:: ``api/rocDecode/roc_bitstream_reader.h``
.. _bitstreamreader: https://github.com/ROCm/rocDecode/tree/develop/api/rocdecode/roc_bitstream_reader.h

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocDecode/tree/develop/utils
