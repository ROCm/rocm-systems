.. meta::
  :description: The device counting service reference page.

.. _device_counting_service_reference:

*******************************************************************************
Device counting service
*******************************************************************************

Queue priority
--------------

ROCprofiler-SDK uses an internal HSA queue for the device counting service. You can control the priority using the environment variable ``ROCPROFILER_DEVICE_COUNTING_QUEUE_PRIORITY``.

- Accepted values (case-insensitive): ``low``, ``normal``, ``high``
- Default: ``high``
- Applied value is logged at INFO level when enabled.

.. doxygengroup:: device_counting_service
   :content-only:
   :project: rocprofiler-sdk
