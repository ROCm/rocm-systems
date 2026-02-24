.. meta::
  :description: Clone the rocDecode project  
  :keywords: install, rocDecode, AMD, ROCm, clone 

*********************************
Cloning the rocDecode project  
*********************************

The rocDecode source code is available from the `ROCm systems GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode>`_. Use sparse checkout when cloning the rocDecode project:

.. code::

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
  cd rocm-systems
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocdecode

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the rocDecode codebase.

If you don't intend to contribute to the rocDecode codebase and won't be previewing features, use a branch that matches the version of ROCm installed on your system.


