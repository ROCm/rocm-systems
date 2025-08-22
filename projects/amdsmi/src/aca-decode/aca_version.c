/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "aca_version.h"

/* Implementation of version functions */

int aca_get_version_major(void)
{
    return ACA_VERSION_MAJOR;
}

int aca_get_version_minor(void)
{
    return ACA_VERSION_MINOR;
}

int aca_get_version_patch(void)
{
    return ACA_VERSION_PATCH;
}

const char *aca_get_version_string(void)
{
    return ACA_VERSION_STRING;
}

aca_version_info_t aca_get_version_info(void)
{
    aca_version_info_t info;

    info.major = ACA_VERSION_MAJOR;
    info.minor = ACA_VERSION_MINOR;
    info.patch = ACA_VERSION_PATCH;
    info.string = ACA_VERSION_STRING;

    return info;
}
