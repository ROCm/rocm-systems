.. meta::
   :description: Linux interposer activation API reference for rocJITsu Python and C bindings.
   :keywords: rocJITsu, Linux, interposer, Python, activation, GPU simulation

.. provenance:
..   mode: NEW
..   sources:
..     - lib/rocjitsu/include/rocjitsu/kmd/rj_interposer.h
..     - lib/rocjitsu/include/rocjitsu/kmd/api.h

=============================================
Linux interposer activation API reference
=============================================

The preloaded Linux ``librocjitsu.so`` provides startup-only activation for
language bindings. Configured CLI launches retain their original mode and
configuration. Exec descendants of global activation can repeat the same
effective configuration, including its CPU thread budget.

Activation and status
*********************

.. doxygenfunction:: rj_interposer_enable_v1
.. doxygenfunction:: rj_interposer_is_enabled_v1
