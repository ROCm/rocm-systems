# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""Pack AIE kernels into an hsaco for loading from ROCR.

Vendored from the public mlir-aie repo's ``python/compiler/hsaco`` (as
``hsaco_packer``, to avoid clashing with any installed ``aie.compiler.hsaco``)
so this test suite can build AIE4 (``aie4``) hsacos before that arch is
supported upstream. The only functional difference from upstream is ``aie4``
added to ``format.ARCHES``; this is meant to be backported once AIE4 support
lands there.

The entry points are the ``pack`` and ``dump`` submodules; import those
directly.
"""
