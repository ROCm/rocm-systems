// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dbt_translate_cli.h"

int main(int argc, char **argv) {
  return rocjitsu::tools::detail::run_dbt_translate_cli(argc, argv,
                                                        rocjitsu::tools::translate_code_object);
}
