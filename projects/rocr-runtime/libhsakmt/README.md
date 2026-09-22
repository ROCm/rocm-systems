# ROCt Library

This repository includes the user-mode API interfaces used to interact with the ROCk driver.

Starting at 1.7 release, ROCt uses drm render device. This requires the user to belong to video group. Add the user account to video group with "sudo usermod -a -G video _username_" command if the user if not part of video group yet.
NOTE: Users of Ubuntu 20.04 will need to add the user to the new "render" group, as Ubuntu has changed the owner:group of /dev/kfd to render:render as of that release

## ROCk Driver

The ROCt library is not a standalone product and requires that you have the correct ROCk driver installed, or are using a compatible upstream kernel.
Please refer to <https://rocm.docs.amd.com> under "Getting Started Guide" for a list of supported Operating Systems and kernel versions, as well as supported hardware.

## Building the Thunk

A simple cmake-based system is available for building thunk. To build the thunk from the the ROCT-Thunk-Interface directory, execute:

```bash
    mkdir -p build
    cd build
    cmake ..
    make
```

If the hsakmt-roct and hsakmt-roct-dev packages are desired:

```bash
    mkdir -p build
    cd build
    cmake ..
    make package
```

If you choose not to build and install packages, manual installation of the binaries and header files can be done via:

```bash
    make install
```

NOTE: For older versions of the thunk where hsakmt-dev.txt is present, "make package-dev" and "make install-dev" are required to generate/install the developer packages. Currently, these are created via the "make package" and "make install" commands

## Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2023 Advanced Micro Devices, Inc. All rights reserved.
