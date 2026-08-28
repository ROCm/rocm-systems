# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The wheel bundles a prebuilt libamd_smi_python.so and loads it via ctypes.
# It carries no Python C-extension, so it is ABI-independent across CPython
# versions (py3-none) but platform-specific (the .so is glibc/arch bound).
# Setuptools would otherwise tag it py3-none-any, which lets pip install it on
# any platform and then fail at load time. Force a non-pure, py3-none wheel so
# the platform tag reflects the bundled .so; auditwheel later stamps the
# concrete manylinux tag.
from setuptools import Distribution, setup

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # wheel >= 0.44 relocated the module
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    # Report a binary distribution so the bundled .so lands in platlib, which
    # auditwheel requires to inspect and stamp the manylinux tag.
    def has_ext_modules(self):
        return True


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": bdist_wheel}, distclass=BinaryDistribution)
