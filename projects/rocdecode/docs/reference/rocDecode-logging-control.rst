.. meta::
  :description: rocDecode logging controls
  :keywords: rocDecode, core APIs, logging, AMD, ROCm

********************************************************************
rocDecode logging control
********************************************************************

rocDecode core components can be configured to output different levels of log messages during decoding.

The log level can be changed by either setting the log level through the ``ROCDEC_LOG_LEVEL`` environment variable, or by calling the ``RocDecLogger::SetLogLevel()`` function in |commons|_.

The logging levels are:

| 0: Critical (Default level)
| 1: Error
| 2: Warning
| 3: Info
| 4: Debug

The log level defines the maximum severity of log messages to output. For example, to output warning and error messages as well as critical messages, ``ROCDEC_LOG_LEVEL`` would need to be set to 2:

.. code:: shell

    ROCDEC_LOG_LEVEL = 2

or

.. code:: cpp

    SetLogLevel(2);


.. |apifolder| replace:: ``api/rocdecode``
.. _apifolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode

.. |rocparser| replace:: ``api/rocdecode/rocparser.h``
.. _rocparser: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocparser.h

.. |rocdecode| replace:: ``api/rocDecode/rocdecode.h``
.. _rocdecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocdecode.h

.. |rocdecodehost| replace:: ``api/rocDecode/rocdecode_host.h``
.. _rocdecodehost: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/rocdecode_host.h

.. |bitstreamreader| replace:: ``api/rocDecode/roc_bitstream_reader.h``
.. _bitstreamreader: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode/roc_bitstream_reader.h

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils

.. |commons| replace:: ``commons.h``
.. _commons: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/src/commons.h
