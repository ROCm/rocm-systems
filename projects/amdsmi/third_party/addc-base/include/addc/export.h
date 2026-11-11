// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ADDC_EXPORT_H
#define ADDC_EXPORT_H

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef ADDC_BUILDING_LIB
#define ADDC_API __declspec(dllexport)
#else
#define ADDC_API __declspec(dllimport)
#endif
#elif __GNUC__ >= 4
#define ADDC_API __attribute__((visibility("default")))
#else
#define ADDC_API
#endif

#endif /* ADDC_EXPORT_H */
