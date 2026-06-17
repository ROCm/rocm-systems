.. meta::
  :description: rocJPEG logging controls
  :keywords: rocJPEG, core APIs, logging, AMD, ROCm

********************************************************************
rocJPEG logging control
********************************************************************

rocJPEG can be configured to output different levels of log messages using the ``ROCJPEG_LOG_LEVEL`` environment variable.

The logging levels are:

| 0: Critical (Default level; output critical messages only)
| 1: Error (Output critical and error messages)
| 2: Warning (Output critical, error, and warning messages)
| 3: Info (Output critical, error, warning, and info messages)
| 4: Debug (Output critical, error, warning, info, and debug messages)

The log level defines the maximum severity of log messages to output. For example, to output warning and error messages as well as critical messages, set ``ROCJPEG_LOG_LEVEL`` to 2:

.. code:: shell

    ROCJPEG_LOG_LEVEL=2
