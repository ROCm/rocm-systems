/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Restores TEST_F after the base transport's test file; see derived_transport_tests_begin.h.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#undef RCCL_DERIVED_TEST_F_
#pragma pop_macro("TEST_F")
