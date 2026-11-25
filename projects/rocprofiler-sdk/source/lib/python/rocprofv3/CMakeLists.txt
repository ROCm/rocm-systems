# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# NOTE: rocprofiler_rocprofv3_python() installs to a flat destination, so every source
# listed here must live directly in this directory. The rocprofv3-doctor modules therefore
# use a doctor_ filename prefix rather than a subpackage.
set(rocprofv3_PYTHON_SOURCES
    __init__.py
    avail.py
    doctor.py
    doctor_diagnose.py
    doctor_env.py
    doctor_layout.py
    doctor_registry.py
    doctor_result.py
    doctor_checks_container.py
    doctor_checks_counters.py
    doctor_checks_driver.py
    doctor_checks_environ.py
    doctor_checks_filesystem.py
    doctor_checks_install.py
    doctor_checks_python_env.py
    doctor_checks_runtime.py
    doctor_checks_smoke.py
    doctor_checks_tools.py)
rocprofiler_rocprofv3_python(${rocprofv3_PYTHON_SOURCES})

rocprofiler_reset_python3_cache()
