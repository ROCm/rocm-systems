.. meta::
   :description: rocDecode environment variables
   :keywords: rocDecode, environment variables, ROCm, AMD

********************************************************************
rocDecode environment variables
********************************************************************

The following environment variables affect the rocDecode runtime.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Values
   * - ``ROCDEC_LOG_LEVEL``
     - | Sets the maximum severity of log messages during decoding. The log level defines the maximum severity of log messages to output.
       | ``0``: Output critical messages only (default)
       | ``1``: Output critical and error messages
       | ``2``: Output critical, error and warning messages
       | ``3``: Output critical, error, warning and info messages
       | ``4``: Output critical, error, warning, info and debug messages
   * - ``ROCR_VISIBLE_DEVICES``
     - | Takes a comma-separated list of GPU indices.
       | When set, the decoder parses this list before it falls back to ``HIP_VISIBLE_DEVICES``.
   * - ``HIP_VISIBLE_DEVICES``
     - | Comma-separated list of GPU indices. Parsed when ``ROCR_VISIBLE_DEVICES`` isn't set.
   * - ``LIBVA_DRIVERS_PATH``
     - | Optional. Path to the directory containing the VA-API driver.
       | You don't normally need to set this. libva finds the driver on its own: on Windows it looks in ``%ROCM_PATH%\lib\rocm_sysdeps\bin``, and on Linux in the driver directory that was configured when libva was built.
       | Set it only to load a driver from outside the ROCm installation. It replaces libva's default search directory rather than adding to it, so an incorrect value makes VA-API initialization fail.

